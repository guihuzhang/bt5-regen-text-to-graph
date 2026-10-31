import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer


ANNA_GRAPH_SPECIAL_TOKENS = ["<S>", "<P>", "<O>"]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from w20bt5reproduce.dataset_6in1 import (
    CANONICAL_TASK_NAMES,
    Seq2SeqCollator,
    build_eval_dataset,
    build_train_datasets,
)
from w20bt5reproduce.eval_parsing import ensure_nltk_resources, evaluate_parsing
from w20bt5reproduce.topk import load_topk, update_topk


log = logging.getLogger(__name__)


def normalize_hf_env() -> None:
    for var_name in ("HF_ENDPOINT", "HUGGINGFACE_CO_RESOLVE_ENDPOINT", "HUGGINGFACE_CO_PREFIX"):
        value = os.environ.get(var_name)
        if value and not value.startswith(("http://", "https://")):
            os.environ.pop(var_name, None)
    hf_home = Path(os.environ.get("HF_HOME", str(ROOT / "caches" / "huggingface"))).expanduser()
    transformers_cache = Path(
        os.environ.get("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    ).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)


def parse_args():
    parser = argparse.ArgumentParser(description="Anna-style 6-in-1 training loop.")
    parser.add_argument("--model_name_or_path", type=str, default="cointegrated/rut5-base-multitask")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="")
    parser.add_argument(
        "--selection_split_english",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="english eval split",
    )
    parser.add_argument(
        "--selection_split_russian",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="russian eval split",
    )
    parser.add_argument(
        "--train_tasks",
        type=str,
        default="all",
        help="comma-separated train tasks from CANONICAL_TASK_NAMES, or 'all'",
    )
    parser.add_argument(
        "--eval_languages",
        type=str,
        default="english,russian",
        help="comma-separated eval languages from {english,russian}",
    )
    parser.add_argument("--sampling_strategy", type=str, default="task_equal", choices=["task_equal", "sample_proportional"])
    parser.add_argument("--use_wiki_entities", type=int, default=1, choices=[0, 1])
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--log_steps", type=int, default=200)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--adafactor_clip_threshold", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_restart",
        choices=["none", "cosine_restart", "warmup_cosine_restart"],
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler_chunk_steps", type=int, default=3000)
    parser.add_argument("--t0", type=int, default=1, help="number of chunks in the first restart cycle")
    parser.add_argument("--tmul", type=int, default=2, help="restart cycle multiplier in chunk units")
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--selection_beam_size", type=int, default=2)
    parser.add_argument("--final_beam_size", type=int, default=5)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=197402)
    parser.add_argument("--eval_on_start", action="store_true")
    parser.add_argument("--output_root", type=str, default=str(ROOT / "w20bt5reproduce" / "runs"))
    parser.add_argument("--run_name", type=str, default="")
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


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def build_optimizer(model, args):
    base_model = model.module if isinstance(model, DDP) else model
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [
                p
                for n, p in base_model.named_parameters()
                if p.requires_grad and not any(key in n for key in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in base_model.named_parameters()
                if p.requires_grad and any(key in n for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer == "adafactor":
        return Adafactor(
            grouped,
            lr=args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=args.adafactor_clip_threshold,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False,
        )
    return AdamW(
        grouped,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )


def build_train_loader(dataset, batch_size, num_workers, distributed, rank, world_size, seed, collate_fn):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        shuffle = False
    else:
        shuffle = True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return loader, sampler


def build_eval_loader(dataset, batch_size, num_workers, collate_fn, distributed, rank, world_size):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )


def save_checkpoint(model, ckpt_dir: Path, step: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_model = model.module if isinstance(model, DDP) else model
    base_model.save_pretrained(ckpt_dir)
    with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"step": step}, f, ensure_ascii=False, indent=2)


def append_metrics(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def remove_dir_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def pick_task_index(num_tasks: int, generator: torch.Generator, device, distributed: bool):
    if distributed:
        if dist.get_rank() == 0:
            task_idx = torch.randint(0, num_tasks, (1,), generator=generator, device=device)
        else:
            task_idx = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(task_idx, src=0)
        return int(task_idx.item())
    return int(torch.randint(0, num_tasks, (1,), generator=generator).item())


def resolve_dataset_key(
    task_name: str,
    generator: torch.Generator,
    device,
    distributed: bool,
    use_wiki_entities: bool,
) -> str:
    if task_name not in {"translate_entities_english", "translate_entities_russian"}:
        return task_name
    if not use_wiki_entities:
        return f"{task_name}__base"

    variant_idx = pick_task_index(2, generator, device, distributed)
    variant_name = "base" if variant_idx == 0 else "wiki"
    return f"{task_name}__{variant_name}"


def build_task_sample_counts(train_datasets, task_names):
    counts = {}
    for task_name in task_names:
        if task_name in train_datasets:
            counts[task_name] = len(train_datasets[task_name])
        else:
            counts[task_name] = len(train_datasets[f"{task_name}__base"])
    return counts


def pick_weighted_task_index(task_probs: torch.Tensor, generator: torch.Generator, device, distributed: bool):
    if distributed:
        if dist.get_rank() == 0:
            task_idx = torch.multinomial(task_probs, 1, replacement=True, generator=generator)
        else:
            task_idx = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(task_idx, src=0)
        return int(task_idx.item())
    return int(torch.multinomial(task_probs, 1, replacement=True, generator=generator).item())


def get_next_batch(task_name: str, loaders, iterators, samplers, sampler_epochs):
    iterator = iterators.get(task_name)
    if iterator is None:
        iterator = iter(loaders[task_name])
    try:
        batch = next(iterator)
    except StopIteration:
        sampler_epochs[task_name] += 1
        sampler = samplers.get(task_name)
        if sampler is not None:
            sampler.set_epoch(sampler_epochs[task_name])
        iterator = iter(loaders[task_name])
        batch = next(iterator)
    iterators[task_name] = iterator
    return batch


def get_autocast_context(args):
    if args.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    return autocast(dtype=dtype)


def compute_seq2seq_loss(outputs, labels, label_smoothing: float):
    if label_smoothing <= 0:
        return outputs.loss
    logits = outputs.logits
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def compute_grad_norm(parameters) -> float:
    grads = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), 2).item())


def parse_eval_languages(raw_value: str):
    allowed = {"english", "russian"}
    languages = []
    for part in raw_value.split(","):
        language = part.strip().lower()
        if not language:
            continue
        if language not in allowed:
            raise ValueError(f"unsupported eval language: {language}")
        if language not in languages:
            languages.append(language)
    if not languages:
        raise ValueError("eval_languages must contain at least one of: english,russian")
    return tuple(languages)


def parse_train_tasks(raw_value: str):
    if raw_value.strip().lower() == "all":
        return tuple(CANONICAL_TASK_NAMES)

    allowed = set(CANONICAL_TASK_NAMES)
    task_names = []
    for part in raw_value.split(","):
        task_name = part.strip()
        if not task_name:
            continue
        if task_name not in allowed:
            raise ValueError(f"unsupported train task: {task_name}")
        if task_name not in task_names:
            task_names.append(task_name)
    if not task_names:
        raise ValueError("train_tasks must contain at least one canonical task name")
    return tuple(task_names)


def resolve_eval_splits(args, eval_languages):
    split_overrides = {
        "english": args.selection_split_english,
        "russian": args.selection_split_russian,
    }
    return {language: split_overrides[language] for language in eval_languages}


def format_eval_split_tag(eval_splits):
    unique_splits = sorted(set(eval_splits.values()))
    if len(unique_splits) == 1:
        return unique_splits[0]
    ordered_pairs = [f"{language[:2]}-{eval_splits[language]}" for language in sorted(eval_splits)]
    return "_".join(ordered_pairs)


def evaluate_all(
    *,
    model,
    tokenizer,
    device,
    eval_loaders,
    eval_sizes,
    eval_splits,
    eval_languages,
    run_dir: Path,
    step_label: str,
    beam_size: int,
    max_target_length: int,
):
    rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    metrics = {}
    for language in eval_languages:
        out_dir = run_dir / "predictions" / eval_splits[language] / language / step_label
        metrics[language] = evaluate_parsing(
            model=model,
            dataloader=eval_loaders[language],
            tokenizer=tokenizer,
            device=device,
            language=language,
            split=eval_splits[language],
            output_dir=out_dir,
            beam_size=beam_size,
            max_target_length=max_target_length,
        )
    if not rank_zero:
        return None
    exact_en = metrics["english"]["exact_f1"] if "english" in metrics else float("nan")
    exact_ru = metrics["russian"]["exact_f1"] if "russian" in metrics else float("nan")
    en_count = eval_sizes["english"] if "english" in metrics else 0
    ru_count = eval_sizes["russian"] if "russian" in metrics else 0
    total_count = en_count + ru_count
    exact_values = [metrics[language]["exact_f1"] for language in eval_languages]
    weighted_sum = sum(metrics[language]["exact_f1"] * eval_sizes[language] for language in eval_languages)
    return {
        "exact_en": exact_en,
        "exact_ru": exact_ru,
        "exact_avg": weighted_sum / total_count,
        "exact_mean": sum(exact_values) / len(exact_values),
        "en_count": en_count,
        "ru_count": ru_count,
        "eval_languages": list(eval_languages),
        "eval_splits": dict(eval_splits),
        "metrics_by_language": metrics,
    }


def main():
    args = parse_args()
    train_task_names = parse_train_tasks(args.train_tasks)
    eval_languages = parse_eval_languages(args.eval_languages)
    eval_splits = resolve_eval_splits(args, eval_languages)
    normalize_hf_env()
    local_rank, rank, world_size, distributed = get_rank_info()
    rank_zero = rank == 0

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    if distributed and rank_zero:
        warmup_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        warmup_tokenizer.add_tokens(ANNA_GRAPH_SPECIAL_TOKENS)
        warmup_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        warmup_model.resize_token_embeddings(len(warmup_tokenizer))
    if distributed:
        dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.add_tokens(ANNA_GRAPH_SPECIAL_TOKENS)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_split_tag = format_eval_split_tag(eval_splits)
    default_name = (
        f"u3t5_bsz{args.per_device_train_batch_size}_"
        f"acc{args.gradient_accumulation_steps}_"
        f"lr{args.learning_rate}_"
        f"{eval_split_tag}_{timestamp}"
    )
    run_name = args.run_name or default_name
    run_dir = Path(args.output_root) / run_name

    if rank_zero:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=False)
        (run_dir / "tokenizer").mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(run_dir / "tokenizer")
        with (run_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    if distributed:
        dist.barrier()

    setup_logging(run_dir / "logs" / f"train.rank{rank}.log")
    seed_everything(args.seed)
    if rank_zero:
        ensure_nltk_resources()
    if distributed:
        dist.barrier()

    use_wiki_entities = bool(args.use_wiki_entities)

    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_datasets = build_train_datasets(task_names=train_task_names, use_wiki_entities=use_wiki_entities)
    task_names = list(train_task_names)
    train_loaders = {}
    train_samplers = {}
    for dataset_key, dataset in train_datasets.items():
        loader, sampler = build_train_loader(
            dataset=dataset,
            batch_size=args.per_device_train_batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            collate_fn=collator,
        )
        train_loaders[dataset_key] = loader
        train_samplers[dataset_key] = sampler

    eval_loaders = {}
    eval_sizes = {}
    for language in eval_languages:
        eval_dataset = build_eval_dataset(language=language, split=eval_splits[language])
        eval_sizes[language] = len(eval_dataset)
        eval_loader = build_eval_loader(
            dataset=eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_workers,
            collate_fn=collator,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        eval_loaders[language] = eval_loader

    optimizer = build_optimizer(model, args)
    scaler = GradScaler(enabled=args.precision == "fp16")
    scheduler = None
    warmup_scheduler = None
    restart_scheduler = None
    scheduler_step_mode = "none"
    if args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t0,
            T_mult=args.tmul,
            eta_min=args.eta_min,
        )
        scheduler_step_mode = "chunk"
    elif args.scheduler == "warmup_cosine_restart":
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t0,
            T_mult=args.tmul,
            eta_min=args.eta_min,
        )
        if args.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=args.warmup_steps,
            )
        scheduler = restart_scheduler
        scheduler_step_mode = "warmup_chunk"

    iterators = {}
    sampler_epochs = {dataset_key: 0 for dataset_key in train_datasets.keys()}
    task_counts = {task_name: 0 for task_name in task_names}
    task_source_counts = {
        "translate_entities_english": {"base": 0, "wiki": 0},
        "translate_entities_russian": {"base": 0, "wiki": 0},
    }
    task_generator = torch.Generator(device=device if distributed else "cpu")
    task_generator.manual_seed(args.seed + 17)

    task_sample_counts = build_task_sample_counts(train_datasets, task_names)
    task_sampling_probs = None
    if args.sampling_strategy == "sample_proportional":
        total_count = float(sum(task_sample_counts.values()))
        task_sampling_probs = torch.tensor(
            [task_sample_counts[name] / total_count for name in task_names],
            dtype=torch.float32,
            device=device if distributed else torch.device("cpu"),
        )
    if rank_zero:
        log.info(
            "train_tasks=%s sampling_strategy=%s use_wiki_entities=%s optimizer=%s scheduler=%s warmup_steps=%d label_smoothing=%.3f adafactor_clip_threshold=%.2f eval_languages=%s eval_splits=%s task_sample_counts=%s",
            train_task_names,
            args.sampling_strategy,
            use_wiki_entities,
            args.optimizer,
            args.scheduler,
            args.warmup_steps,
            args.label_smoothing,
            args.adafactor_clip_threshold,
            eval_languages,
            eval_splits,
            task_sample_counts,
        )

    metrics_path = run_dir / "metrics.jsonl"
    topk_path = run_dir / "topk.json"
    global_step = 0
    train_loss_acc = 0.0
    loss_count = 0
    last_grad_norm = None

    if args.eval_on_start:
        if distributed:
            dist.barrier()
        eval_summary = evaluate_all(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eval_loaders=eval_loaders,
            eval_sizes=eval_sizes,
            eval_splits=eval_splits,
            eval_languages=eval_languages,
            run_dir=run_dir,
            step_label="step_000000",
            beam_size=args.selection_beam_size,
            max_target_length=args.max_target_length,
        )
        if rank_zero:
            ckpt_dir = run_dir / "checkpoints" / "step_000000"
            save_checkpoint(model, ckpt_dir, step=0)
            candidate = {
                "step": 0,
                "exact_en": eval_summary["exact_en"],
                "exact_ru": eval_summary["exact_ru"],
                "exact_avg": eval_summary["exact_avg"],
                "checkpoint_dir": str(ckpt_dir),
            }
            keep, evicted, kept = update_topk(topk_path, candidate, k=args.save_top_k)
            for entry in evicted:
                remove_dir_if_exists(Path(entry["checkpoint_dir"]))
            append_metrics(
                metrics_path,
                {
                    "step": 0,
                        "lr": current_lr(optimizer),
                        "eval_splits": eval_summary["eval_splits"],
                        "exact_en": eval_summary["exact_en"],
                        "exact_ru": eval_summary["exact_ru"],
                        "exact_avg": eval_summary["exact_avg"],
                    "exact_mean": eval_summary["exact_mean"],
                    "en_count": eval_summary["en_count"],
                    "ru_count": eval_summary["ru_count"],
                },
            )
            log.info(
                "step=%d exact_en=%.4f exact_ru=%.4f exact_avg=%.4f exact_mean=%.4f counts=(%d,%d) lr=%.2e kept=%s topk=%s",
                0,
                eval_summary["exact_en"],
                eval_summary["exact_ru"],
                eval_summary["exact_avg"],
                eval_summary["exact_mean"],
                eval_summary["en_count"],
                eval_summary["ru_count"],
                current_lr(optimizer),
                kept,
                [round(entry["exact_avg"], 4) for entry in keep],
            )
        if distributed:
            dist.barrier()

    pbar = None
    if rank_zero:
        pbar = tqdm(total=args.total_steps, desc="train 6in1", leave=True, dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    while global_step < args.total_steps:
        if task_sampling_probs is None:
            task_idx = pick_task_index(len(task_names), task_generator, device, distributed)
        else:
            task_idx = pick_weighted_task_index(task_sampling_probs, task_generator, device, distributed)
        task_name = task_names[task_idx]
        task_counts[task_name] += 1
        dataset_key = resolve_dataset_key(
            task_name,
            task_generator,
            device,
            distributed,
            use_wiki_entities=use_wiki_entities,
        )
        if dataset_key != task_name:
            variant_name = dataset_key.rsplit("__", 1)[1]
            task_source_counts[task_name][variant_name] += 1

        batch = get_next_batch(dataset_key, train_loaders, iterators, train_samplers, sampler_epochs)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with get_autocast_context(args):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = compute_seq2seq_loss(outputs, labels, args.label_smoothing)
            loss = raw_loss / args.gradient_accumulation_steps

        next_micro_step = loss_count + 1
        should_step = (next_micro_step % args.gradient_accumulation_steps) == 0
        sync_context = model.no_sync() if distributed and not should_step else nullcontext()
        with sync_context:
            if args.precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        train_loss_acc += float(loss.item()) * args.gradient_accumulation_steps
        loss_count += 1

        if not should_step:
            continue

        if args.precision == "fp16":
            scaler.unscale_(optimizer)
        if args.optimizer == "adafactor":
            last_grad_norm = compute_grad_norm(model.parameters())
        elif args.max_grad_norm > 0:
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
        else:
            last_grad_norm = None

        if args.precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if scheduler is not None:
            if scheduler_step_mode == "chunk":
                if global_step % args.scheduler_chunk_steps == 0:
                    scheduler.step(global_step // args.scheduler_chunk_steps)
            elif scheduler_step_mode == "warmup_chunk":
                if warmup_scheduler is not None and global_step <= args.warmup_steps:
                    warmup_scheduler.step()
                elif restart_scheduler is not None:
                    post_warmup_steps = global_step - args.warmup_steps
                    if post_warmup_steps > 0 and post_warmup_steps % args.scheduler_chunk_steps == 0:
                        restart_scheduler.step(post_warmup_steps // args.scheduler_chunk_steps)

        avg_loss = train_loss_acc / loss_count
        if pbar is not None:
            pbar.update(1)
            postfix = {
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr(optimizer):.2e}",
            }
            if last_grad_norm is not None:
                postfix["gn"] = f"{last_grad_norm:.2f}"
            pbar.set_postfix(**postfix)

        if global_step % args.eval_steps == 0 or global_step == args.total_steps:
            if distributed:
                dist.barrier()
            step_label = f"step_{global_step:06d}"
            eval_summary = evaluate_all(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_loaders=eval_loaders,
                eval_sizes=eval_sizes,
                eval_splits=eval_splits,
                eval_languages=eval_languages,
                run_dir=run_dir,
                step_label=step_label,
                beam_size=args.selection_beam_size,
                max_target_length=args.max_target_length,
            )
            if rank_zero:
                ckpt_dir = run_dir / "checkpoints" / step_label
                save_checkpoint(model, ckpt_dir, step=global_step)
                candidate = {
                    "step": global_step,
                    "exact_en": eval_summary["exact_en"],
                    "exact_ru": eval_summary["exact_ru"],
                    "exact_avg": eval_summary["exact_avg"],
                    "checkpoint_dir": str(ckpt_dir),
                }
                keep, evicted, kept = update_topk(topk_path, candidate, k=args.save_top_k)
                for entry in evicted:
                    remove_dir_if_exists(Path(entry["checkpoint_dir"]))
                append_metrics(
                    metrics_path,
                    {
                        "step": global_step,
                        "task_counts": task_counts,
                        "task_source_counts": task_source_counts,
                        "lr": current_lr(optimizer),
                        "avg_train_loss": avg_loss,
                        "eval_splits": eval_summary["eval_splits"],
                        "exact_en": eval_summary["exact_en"],
                        "exact_ru": eval_summary["exact_ru"],
                        "exact_avg": eval_summary["exact_avg"],
                        "exact_mean": eval_summary["exact_mean"],
                        "en_count": eval_summary["en_count"],
                        "ru_count": eval_summary["ru_count"],
                    },
                )
                log.info(
                    "step=%d avg_train_loss=%.4f exact_en=%.4f exact_ru=%.4f exact_avg=%.4f exact_mean=%.4f counts=(%d,%d) lr=%.2e kept=%s topk=%s",
                    global_step,
                    avg_loss,
                    eval_summary["exact_en"],
                    eval_summary["exact_ru"],
                    eval_summary["exact_avg"],
                    eval_summary["exact_mean"],
                    eval_summary["en_count"],
                    eval_summary["ru_count"],
                    current_lr(optimizer),
                    kept,
                    [round(entry["exact_avg"], 4) for entry in keep],
                )
            if distributed:
                dist.barrier()
        elif global_step % args.log_steps == 0:
            if rank_zero:
                payload = {
                    "step": global_step,
                    "avg_train_loss": avg_loss,
                    "lr": current_lr(optimizer),
                }
                if last_grad_norm is not None:
                    payload["grad_norm"] = last_grad_norm
                append_metrics(metrics_path, payload)
                log.info(
                    "step=%d avg_train_loss=%.4f lr=%.2e grad_norm=%s",
                    global_step,
                    avg_loss,
                    current_lr(optimizer),
                    f"{last_grad_norm:.2f}" if last_grad_norm is not None else "n/a",
                )

    if pbar is not None:
        pbar.close()

    topk_entries = load_topk(topk_path) if rank_zero else None
    if distributed:
        gathered_topk = [topk_entries]
        dist.broadcast_object_list(gathered_topk, src=0)
        topk_entries = gathered_topk[0]

    if topk_entries:
        best_entry = topk_entries[0]
        best_step = best_entry["step"]
        best_ckpt_dir = Path(best_entry["checkpoint_dir"])

        best_model = AutoModelForSeq2SeqLM.from_pretrained(best_ckpt_dir)
        best_model.to(device)

        final_step_label = f"best_step_{best_step:06d}_beam{args.final_beam_size}"
        final_summary = evaluate_all(
            model=best_model,
            tokenizer=tokenizer,
            device=device,
            eval_loaders=eval_loaders,
            eval_sizes=eval_sizes,
            eval_splits=eval_splits,
            eval_languages=eval_languages,
            run_dir=run_dir,
            step_label=final_step_label,
            beam_size=args.final_beam_size,
            max_target_length=args.max_target_length,
        )

        if rank_zero:
            final_payload = {
                "step": best_step,
                "exact_en": final_summary["exact_en"],
                "exact_ru": final_summary["exact_ru"],
                "exact_avg": final_summary["exact_avg"],
                "exact_mean": final_summary["exact_mean"],
                "en_count": final_summary["en_count"],
                "ru_count": final_summary["ru_count"],
                "eval_splits": final_summary["eval_splits"],
                "beam_size": args.final_beam_size,
                "checkpoint_dir": str(best_ckpt_dir),
                "prediction_dirs": {
                    language: str(run_dir / "predictions" / eval_splits[language] / language)
                    for language in eval_languages
                },
            }
            with (run_dir / "final_top1.json").open("w", encoding="utf-8") as f:
                json.dump(final_payload, f, ensure_ascii=False, indent=2)
            log.info(
                "final rerun: best_step=%d beam=%d exact_en=%.4f exact_ru=%.4f exact_avg=%.4f",
                best_step,
                args.final_beam_size,
                final_summary["exact_en"],
                final_summary["exact_ru"],
                final_summary["exact_avg"],
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
