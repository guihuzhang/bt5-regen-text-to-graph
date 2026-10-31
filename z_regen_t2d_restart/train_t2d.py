import argparse
import json
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.webnlg_dataset import WebNLGDataset
from model.model import get_model, get_tokenizer
from utils.sequence import shift_target_inputs_to_labels
from z_regen_t2d_restart.eval_t2d import ensure_nltk_resources, evaluate_t2g
from z_regen_t2d_restart.topk import load_topk, update_topk


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="HF-style T2G training loop with top-k checkpoint selection.")
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--train_prepare", type=str, default="webnlg-lex1-kbpm")
    parser.add_argument("--eval_prepare", type=str, default="webnlg")
    parser.add_argument("--select_split", type=str, default="valA", choices=["testA", "valA"])
    parser.add_argument(
        "--final_eval_split",
        type=str,
        default="",
        choices=["", "testA", "valA"],
        help="optional final evaluation split; defaults to select_split",
    )
    parser.add_argument("--src", type=str, default="A", choices=["A"])
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--t0", type=int, default=2, help="restart period in epochs when scheduler=cosine_restart")
    parser.add_argument("--tmul", type=int, default=2, help="restart multiplier when scheduler=cosine_restart")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--selection_beam_size", type=int, default=2)
    parser.add_argument("--final_beam_size", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_on_start", action="store_true")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=197402)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output_root", type=str, default=str(ROOT / "z_regen_t2d_restart" / "runs"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    return parser.parse_args()


def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s(%(name)s): %(message)s")
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode="w")]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_rank_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    return local_rank, rank, world_size, distributed


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(split: str, tokenizer, prepare: str, prepare_permutation: str, src: str):
    prepare_args = SimpleNamespace(prepare_permutation=prepare_permutation)
    return WebNLGDataset(split, tokenizer, src, prepare=prepare, prepare_args=prepare_args)


def build_loader(dataset, batch_size: int, num_workers: int, training: bool, distributed: bool, rank: int, world_size: int):
    sampler = None
    shuffle = training
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=training,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=dataset._collate_fn,
        num_workers=num_workers,
        drop_last=training,
    )
    return loader, sampler


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.eta_min)
    return CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.tmul, eta_min=args.eta_min)


def build_optimizer(model, args):
    base_model = model.module if isinstance(model, DDP) else model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in base_model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in base_model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


def save_checkpoint(model, ckpt_dir: Path, epoch: int, args):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    model_path = ckpt_dir / "model.pt"
    torch.save(state_dict, model_path)
    meta = {
        "epoch": epoch,
        "model": args.model,
        "train_prepare": args.train_prepare,
        "eval_prepare": args.eval_prepare,
        "select_split": args.select_split,
        "final_eval_split": args.final_eval_split or args.select_split,
    }
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return model_path


def append_metrics(metrics_path: Path, payload: dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def remove_dir_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def run_named_eval(
    *,
    model,
    eval_loader,
    tokenizer,
    device,
    split: str,
    run_dir: Path,
    label: str,
    beam_size: int,
    max_length: int,
):
    rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    pred_dir = run_dir / "predictions" / split / label
    metrics = evaluate_t2g(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        split=split,
        output_dir=pred_dir,
        beam_size=beam_size,
        max_length=max_length,
    )
    if not rank_zero:
        return None, pred_dir
    return metrics, pred_dir


def run_epoch_eval(
    *,
    model,
    eval_loader,
    tokenizer,
    device,
    split: str,
    run_dir: Path,
    epoch_label: str,
    epoch_value: int,
    beam_size: int,
    max_length: int,
    train_loss: float,
    lr_value: float,
    ckpt_dir: Path,
    metrics_path: Path,
    topk_path: Path,
    top_k: int,
):
    rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    metrics, pred_dir = run_named_eval(
        model=model,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        split=split,
        run_dir=run_dir,
        label=epoch_label,
        beam_size=beam_size,
        max_length=max_length,
    )
    if not rank_zero:
        return None, False, []
    exact_f1 = metrics["exact_f1"]

    payload = {
        "epoch": epoch_value,
        "epoch_label": epoch_label,
        "train_loss": train_loss,
        "exact_f1": exact_f1,
        "lr": lr_value,
        "checkpoint_dir": str(ckpt_dir),
        "prediction_dir": str(pred_dir),
    }
    append_metrics(metrics_path, payload)
    candidate = {
        "epoch": epoch_value,
        "exact_f1": exact_f1,
        "checkpoint_dir": str(ckpt_dir),
        "prediction_dir": str(pred_dir),
        "score_out": metrics["score_out"],
    }
    keep, evicted, kept = update_topk(topk_path, candidate, k=top_k)

    for entry in evicted:
        remove_dir_if_exists(Path(entry["checkpoint_dir"]))
        remove_dir_if_exists(Path(entry["prediction_dir"]))

    return exact_f1, kept, keep


def main():
    args = parse_args()
    local_rank, rank, world_size, distributed = get_rank_info()
    rank_zero = rank == 0

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{args.model.replace('-', '')}_bsz{args.per_device_train_batch_size}_lr{str(args.learning_rate).replace('.', '')}_{args.select_split}_{timestamp}"
    run_name = args.run_name or default_name
    run_dir = Path(args.output_root) / run_name

    if rank_zero:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=False)
        with open(run_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    if distributed:
        dist.barrier()

    setup_logging(run_dir / "logs" / f"train.rank{rank}.log")
    seed_everything(args.seed)
    if rank_zero:
        ensure_nltk_resources()
    if distributed:
        dist.barrier()

    log.info(f"run_dir={run_dir}")
    log.info(
        "train setup: model=%s train_prepare=%s eval_prepare=%s select_split=%s final_eval_split=%s epochs=%d train_bsz=%d eval_bsz=%d grad_accum=%d lr=%g precision=%s scheduler=%s eta_min=%g t0=%d tmul=%d selection_beam=%d final_beam=%d world_size=%d",
        args.model,
        args.train_prepare,
        args.eval_prepare,
        args.select_split,
        args.final_eval_split or args.select_split,
        args.num_epochs,
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.precision,
        args.scheduler,
        args.eta_min,
        args.t0,
        args.tmul,
        args.selection_beam_size,
        args.final_beam_size,
        world_size,
    )

    tokenizer, _ = get_tokenizer(args.model)
    train_dataset = build_dataset("train", tokenizer, args.train_prepare, "internal", args.src)
    final_eval_split = args.final_eval_split or args.select_split
    selection_eval_dataset = build_dataset(args.select_split, tokenizer, args.eval_prepare, "none", args.src)
    final_eval_dataset = (
        selection_eval_dataset
        if final_eval_split == args.select_split
        else build_dataset(final_eval_split, tokenizer, args.eval_prepare, "none", args.src)
    )
    train_loader, train_sampler = build_loader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        training=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    eval_loader, _ = build_loader(
        selection_eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        training=False,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    final_eval_loader = eval_loader
    if final_eval_dataset is not selection_eval_dataset:
        final_eval_loader, _ = build_loader(
            final_eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_workers,
            training=False,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

    model, _ = get_model(args.model)
    model.resize_token_embeddings(len(tokenizer))
    if args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint).expanduser()
        if not init_checkpoint.is_absolute():
            init_checkpoint = (ROOT / init_checkpoint).resolve()
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"missing init checkpoint: {init_checkpoint}")
        state_dict = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        log.info("loaded init checkpoint: %s", init_checkpoint)
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.precision == "fp16")

    use_amp = args.precision in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    metrics_path = run_dir / "metrics.jsonl"
    topk_path = run_dir / "topk.json"

    if args.eval_on_start:
        init_ckpt_dir = run_dir / "checkpoints" / "epoch_00"
        if rank_zero:
            save_checkpoint(model, init_ckpt_dir, 0, args)
        if distributed:
            dist.barrier()
        init_lr = optimizer.param_groups[0]["lr"]
        init_f1, init_kept, init_keep = run_epoch_eval(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            device=device,
            split=args.select_split,
            run_dir=run_dir,
            epoch_label="epoch_00",
            epoch_value=0,
            beam_size=args.selection_beam_size,
            max_length=args.max_length,
            train_loss=0.0,
            lr_value=init_lr,
            ckpt_dir=init_ckpt_dir,
            metrics_path=metrics_path,
            topk_path=topk_path,
            top_k=args.top_k,
        )
        if rank_zero:
            log.info(
                "epoch=%s train_loss=%.6f exact_f1=%.6f lr=%g kept=%s topk=%s",
                "00",
                0.0,
                init_f1,
                init_lr,
                init_kept,
                [entry["epoch"] for entry in init_keep],
            )

    if distributed:
        dist.barrier()

    epoch_iter = range(1, args.num_epochs + 1)
    if rank_zero:
        epoch_iter = tqdm(epoch_iter, desc="epochs", leave=True, dynamic_ncols=True)

    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        step_iter = train_loader
        if rank_zero:
            step_iter = tqdm(
                train_loader,
                desc=f"train epoch {epoch}/{args.num_epochs}",
                leave=True,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(step_iter, start=1):
            source_ids, source_mask, target_ids, target_mask, _ = (x.to(device) for x in batch)
            labels = shift_target_inputs_to_labels(target_ids, tokenizer.pad_token_id, device)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                output = model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    decoder_input_ids=target_ids,
                    decoder_attention_mask=target_mask,
                    labels=labels,
                )
                loss = output.loss
                loss_for_backward = loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if args.max_grad_norm and args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if rank_zero:
                step_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        avg_train_loss = running_loss / max(len(train_loader), 1)

        ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch:02d}"
        if rank_zero:
            save_checkpoint(model, ckpt_dir, epoch, args)
        if distributed:
            dist.barrier()
        exact_f1, kept, keep = run_epoch_eval(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            device=device,
            split=args.select_split,
            run_dir=run_dir,
            epoch_label=f"epoch_{epoch:02d}",
            epoch_value=epoch,
            beam_size=args.selection_beam_size,
            max_length=args.max_length,
            train_loss=avg_train_loss,
            lr_value=current_lr,
            ckpt_dir=ckpt_dir,
            metrics_path=metrics_path,
            topk_path=topk_path,
            top_k=args.top_k,
        )

        if rank_zero:
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(f1=f"{exact_f1:.4f}", loss=f"{avg_train_loss:.4f}")
            log.info(
                "epoch=%d train_loss=%.6f exact_f1=%.6f lr=%g kept=%s topk=%s",
                epoch,
                avg_train_loss,
                exact_f1,
                current_lr,
                kept,
                [entry["epoch"] for entry in keep],
            )

        if distributed:
            dist.barrier()

        if scheduler is not None:
            scheduler.step()

    topk_entries = load_topk(topk_path) if rank_zero else None
    if distributed:
        gathered_topk = [topk_entries]
        dist.broadcast_object_list(gathered_topk, src=0)
        topk_entries = gathered_topk[0]

    if topk_entries:
        best_entry = topk_entries[0]
        best_epoch = best_entry["epoch"]
        best_ckpt_dir = Path(best_entry["checkpoint_dir"])
        best_model_path = best_ckpt_dir / "model.pt"

        model_to_eval, _ = get_model(args.model)
        model_to_eval.resize_token_embeddings(len(tokenizer))
        state_dict = torch.load(best_model_path, map_location=device)
        model_to_eval.load_state_dict(state_dict)
        model_to_eval.to(device)
        if distributed:
            model_to_eval = DDP(model_to_eval, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        final_metrics, final_pred_dir = run_named_eval(
            model=model_to_eval,
            eval_loader=final_eval_loader,
            tokenizer=tokenizer,
            device=device,
            split=final_eval_split,
            run_dir=run_dir,
            label=f"best_epoch_{best_epoch:02d}_beam{args.final_beam_size}",
            beam_size=args.final_beam_size,
            max_length=args.max_length,
        )

        if rank_zero:
            final_payload = {
                "epoch": best_epoch,
                "exact_f1": final_metrics["exact_f1"],
                "beam_size": args.final_beam_size,
                "select_split": args.select_split,
                "final_eval_split": final_eval_split,
                "checkpoint_dir": str(best_ckpt_dir),
                "prediction_dir": str(final_pred_dir),
                "score_out": final_metrics["score_out"],
            }
            with open(run_dir / "final_top1.json", "w", encoding="utf-8") as f:
                json.dump(final_payload, f, ensure_ascii=False, indent=2)
            log.info(
                "final rerun: best_epoch=%d beam=%d exact_f1=%.6f",
                best_epoch,
                args.final_beam_size,
                final_metrics["exact_f1"],
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
