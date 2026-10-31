import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils_webnlg import save_webnlg_rdf


log = logging.getLogger(__name__)


def ensure_nltk_resources() -> None:
    nltk_data_dir = Path(os.environ.get("NLTK_DATA", str(Path.home() / "nltk_data"))).expanduser()
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nltk_data_dir)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_dir))

    required_resources = [
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info("NLTK resource '%s' missing. Downloading to %s", package_name, nltk_data_dir)
            try:
                nltk.download(package_name, download_dir=str(nltk_data_dir), quiet=True)
            except FileExistsError:
                log.info("NLTK resource '%s' is being created concurrently; re-checking.", package_name)


def get_dist_info() -> Tuple[int, int, bool]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def get_reference_xml(split: str) -> Path:
    if split == "testA":
        return ROOT / "corpora" / "webnlg-references" / "release_v3.0" / "en" / "references.rdf.test" / "test.xml"
    if split == "valA":
        return ROOT / "corpora" / "webnlg-references" / "release_v3.0" / "en" / "references.rdf.val" / "val.xml"
    raise ValueError(f"unsupported split for t2g evaluation: {split}")


def parse_exact_f1(score_text: str) -> float:
    in_total = False
    in_exact = False
    for raw_line in score_text.splitlines():
        line = raw_line.strip()
        if line == "Total scores":
            in_total = True
            continue
        if in_total and line == "Scores per tag":
            break
        if in_total and line == "Exact":
            in_exact = True
            continue
        if in_exact and line.startswith("F1:"):
            return float(line.split("F1:", 1)[1].strip())
    raise RuntimeError("failed to parse Exact F1 from evaluator output")


@torch.no_grad()
def generate_predictions(
    model,
    dataloader,
    tokenizer,
    device,
    beam_size: int,
    max_length: int,
    progress_desc: str = "",
) -> Tuple[List[int], List[str], List[str], List[Dict]]:
    was_training = model.training
    model.eval()
    gen_model = model.module if hasattr(model, "module") else model

    sample_ids: List[int] = []
    hyps: List[str] = []
    targets: List[str] = []
    metas: List[Dict] = []

    rank, _, distributed = get_dist_info()
    batch_iter = dataloader
    if progress_desc and (not distributed or rank == 0):
        batch_iter = tqdm(dataloader, desc=progress_desc, leave=True, dynamic_ncols=True)

    for batch in batch_iter:
        source_ids, source_mask, target_ids, _, sample_idx = batch
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        output_ids = gen_model.generate(
            source_ids,
            attention_mask=source_mask,
            num_beams=beam_size,
            max_length=max_length,
        )

        for seq_ids in output_ids.cpu().tolist():
            hyps.append(
                tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        for seq_ids in target_ids.cpu().tolist():
            targets.append(
                tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        sample_idx_list = sample_idx.tolist()
        sample_ids.extend(sample_idx_list)
        metas.extend([dataloader.dataset.get_meta(idx) for idx in sample_idx_list])

    model.train(was_training)
    return sample_ids, hyps, targets, metas


def gather_predictions(
    sample_ids: List[int],
    hyps: List[str],
    targets: List[str],
    metas: List[Dict],
    dataset_size: int,
) -> Tuple[List[str], List[str], List[Dict]]:
    rank, world_size, distributed = get_dist_info()
    if not distributed:
        return hyps, targets, metas

    local_records = list(zip(sample_ids, hyps, targets, metas))
    gathered_records = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_records, local_records)

    if rank != 0:
        return [], [], []

    merged = {}
    for records in gathered_records:
        for sample_id, hyp, target, meta in records:
            if sample_id not in merged:
                merged[sample_id] = (hyp, target, meta)

    ordered_ids = sorted(merged.keys())
    if len(ordered_ids) != dataset_size:
        raise RuntimeError(
            f"distributed eval gathered {len(ordered_ids)} unique samples, expected {dataset_size}"
        )

    ordered_hyps = [merged[sample_id][0] for sample_id in ordered_ids]
    ordered_targets = [merged[sample_id][1] for sample_id in ordered_ids]
    ordered_metas = [merged[sample_id][2] for sample_id in ordered_ids]
    return ordered_hyps, ordered_targets, ordered_metas


def evaluate_t2g(model, dataloader, tokenizer, device, split: str, output_dir: Path, beam_size: int, max_length: int) -> Dict:
    rank, _, distributed = get_dist_info()
    rank_zero = rank == 0

    if distributed:
        if rank_zero:
            ensure_nltk_resources()
        dist.barrier()
    else:
        ensure_nltk_resources()

    if rank_zero:
        output_dir.mkdir(parents=True, exist_ok=True)
    progress_desc = f"eval {split} {output_dir.name}"
    sample_ids, hyps, targets, metas = generate_predictions(
        model, dataloader, tokenizer, device, beam_size, max_length, progress_desc=progress_desc
    )
    hyps, targets, metas = gather_predictions(sample_ids, hyps, targets, metas, len(dataloader.dataset))

    if distributed and not rank_zero:
        dist.barrier()
        return {"exact_f1": None, "score_out": "", "hyps_xml": ""}

    save_webnlg_rdf(hyps, targets, metas, str(output_dir), prepare="webnlg")

    eval_dir = ROOT / "eval" / "WebNLG-Text-to-triples"
    eval_script = eval_dir / "Evaluation_script.py"
    reference_xml = get_reference_xml(split)
    hypothesis_xml = output_dir / "hyps.xml"
    score_out = output_dir / "score.out"

    cmd = [
        sys.executable,
        str(eval_script),
        str(reference_xml),
        str(hypothesis_xml),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    score_text = completed.stdout
    if completed.stderr:
        log.info(completed.stderr.strip())
    with open(score_out, "w", encoding="utf-8") as f:
        f.write(score_text)

    exact_f1 = parse_exact_f1(score_text)
    payload = {
        "exact_f1": exact_f1,
        "score_out": str(score_out),
        "hyps_xml": str(hypothesis_xml),
    }
    if distributed:
        dist.barrier()
    return payload
