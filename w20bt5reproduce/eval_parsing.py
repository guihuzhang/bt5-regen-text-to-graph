import json
import logging
import math
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

from utils.utils_webnlg import create_xml
from utils.utils_xml import xml_prettify


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
            try:
                nltk.data.find(resource_path)
            except LookupError as exc:
                raise RuntimeError(
                    f"failed to prepare NLTK resource '{package_name}' in {nltk_data_dir}"
                ) from exc


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


def get_dist_info() -> Tuple[int, int, bool]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def anna_texts_to_triples(texts: List[str]) -> List[List[str]]:
    all_triples: List[List[str]] = []
    for text in texts:
        sample_triples: List[str] = []
        for raw_triple in text.split("<S>"):
            raw_triple = raw_triple.strip()
            if not raw_triple:
                continue
            if raw_triple.count("<P>") != 1 or raw_triple.count("<O>") != 1:
                continue
            parts = raw_triple.replace("<P>", " | ").replace("<O>", " | ").split(" | ")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            subj = subj.strip()
            pred = pred.strip()
            obj = obj.strip()
            if not subj or not pred or not obj:
                continue
            sample_triples.append(f"{subj} | {pred} | {obj}")
        all_triples.append(sample_triples)
    return all_triples


def dedupe_triples_preserve_order(samples: List[List[str]]) -> List[List[str]]:
    deduped_samples: List[List[str]] = []
    for triples in samples:
        seen = set()
        deduped = []
        for triple in triples:
            if triple in seen:
                continue
            seen.add(triple)
            deduped.append(triple)
        deduped_samples.append(deduped)
    return deduped_samples


def build_meta_infos(num_examples: int) -> List[Dict[str, str]]:
    return [
        {
            "category": "anna",
            "eid": str(i + 1),
            "shape": "unknown",
            "shape_type": "unknown",
            "size": "1",
        }
        for i in range(num_examples)
    ]


def save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def generate_predictions(model, dataloader, tokenizer, device, beam_size: int, max_target_length: int, progress_desc: str):
    was_training = model.training
    model.eval()
    gen_model = model.module if hasattr(model, "module") else model

    sample_ids: List[int] = []
    hypotheses: List[str] = []
    targets: List[str] = []
    rank, _, distributed = get_dist_info()
    batch_iter = dataloader
    if progress_desc and (not distributed or rank == 0):
        batch_iter = tqdm(dataloader, desc=progress_desc, leave=True, dynamic_ncols=True)

    for batch in batch_iter:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output_ids = gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            max_new_tokens=max_target_length,
        )
        for seq_ids in output_ids.cpu().tolist():
            hypotheses.append(
                tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        targets.extend(batch["target_texts"])
        sample_ids.extend([meta["sample_id"] for meta in batch["metas"]])

    model.train(was_training)
    return sample_ids, hypotheses, targets


def gather_predictions(
    sample_ids: List[int],
    hypotheses: List[str],
    targets: List[str],
    dataset_size: int,
) -> Tuple[List[str], List[str]]:
    rank, world_size, distributed = get_dist_info()
    if not distributed:
        return hypotheses, targets

    local_records = list(zip(sample_ids, hypotheses, targets))
    gathered_records = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_records, local_records)

    if rank != 0:
        return [], []

    merged = {}
    for records in gathered_records:
        for sample_id, hypothesis, target in records:
            if sample_id not in merged:
                merged[sample_id] = (hypothesis, target)

    ordered_ids = sorted(merged.keys())
    if len(ordered_ids) != dataset_size:
        raise RuntimeError(
            f"distributed eval gathered {len(ordered_ids)} unique samples, expected {dataset_size}"
        )

    ordered_hypotheses = [merged[sample_id][0] for sample_id in ordered_ids]
    ordered_targets = [merged[sample_id][1] for sample_id in ordered_ids]
    return ordered_hypotheses, ordered_targets


def evaluate_parsing(model, dataloader, tokenizer, device, language: str, split: str, output_dir: Path, beam_size: int, max_target_length: int) -> Dict:
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

    error = None
    payload = None
    try:
        sample_ids, hyps_text, refs_text = generate_predictions(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            device=device,
            beam_size=beam_size,
            max_target_length=max_target_length,
            progress_desc=f"eval {language} {split} {output_dir.name}",
        )
        hyps_text, refs_text = gather_predictions(sample_ids, hyps_text, refs_text, len(dataloader.dataset))

        if rank_zero:
            hyps_triples = anna_texts_to_triples(hyps_text)
            refs_triples = anna_texts_to_triples(refs_text)
            meta_infos = build_meta_infos(len(refs_triples))
            max_eval_hyp_triples = int(os.environ.get("MAX_EVAL_HYP_TRIPLES", "0"))

            refs_text_json = output_dir / "references.text.json"
            hyps_text_json = output_dir / "hypotheses.text.json"
            refs_json = output_dir / "references.triples.json"
            hyps_json = output_dir / "hypotheses.triples.json"
            refs_xml = output_dir / "refs.xml"
            hyps_xml = output_dir / "hyps.xml"
            score_out = output_dir / "score.out"

            save_json(refs_text_json, refs_text)
            save_json(hyps_text_json, hyps_text)
            save_json(refs_json, refs_triples)
            save_json(hyps_json, hyps_triples)

            hyps_triples_for_eval = dedupe_triples_preserve_order(hyps_triples)
            deduped_samples = sum(
                1 for raw_triples, deduped_triples in zip(hyps_triples, hyps_triples_for_eval)
                if len(deduped_triples) < len(raw_triples)
            )
            if deduped_samples:
                log.warning(
                    "deduplicated hypothesis triples for %d/%d samples before evaluator",
                    deduped_samples,
                    len(hyps_triples),
                )
            if max_eval_hyp_triples > 0:
                truncated_samples = sum(
                    1 for triples in hyps_triples_for_eval if len(triples) > max_eval_hyp_triples
                )
                hyps_triples_for_eval = [triples[:max_eval_hyp_triples] for triples in hyps_triples_for_eval]
                if truncated_samples:
                    max_raw_triples = max((len(triples) for triples in hyps_triples), default=0)
                    log.warning(
                        "clipped %d/%d hypothesis samples to first %d triples for evaluator (max raw triples=%d)",
                        truncated_samples,
                        len(hyps_triples),
                        max_eval_hyp_triples,
                        max_raw_triples,
                    )

            refs_xml_tree = create_xml(refs_triples, meta_infos, "modifiedtripleset", "mtriple")
            hyps_xml_tree = create_xml(hyps_triples_for_eval, meta_infos, "generatedtripleset", "gtriple")

            refs_xml.write_text(xml_prettify(refs_xml_tree), encoding="utf-8")
            hyps_xml.write_text(xml_prettify(hyps_xml_tree), encoding="utf-8")

            eval_dir = ROOT / "eval" / "WebNLG-Text-to-triples"
            eval_script = eval_dir / "Evaluation_script.py"
            eval_python = os.environ.get("EVAL_PYTHON", sys.executable)
            completed = subprocess.run(
                [eval_python, str(eval_script), str(refs_xml), str(hyps_xml)],
                cwd=str(eval_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                log.error(
                    "Evaluation_script.py failed (exit %d)\nstdout:\n%s\nstderr:\n%s",
                    completed.returncode,
                    completed.stdout,
                    completed.stderr,
                )
                raise RuntimeError(
                    f"Evaluation_script.py failed with exit code {completed.returncode}"
                )

            score_text = completed.stdout
            score_out.write_text(score_text, encoding="utf-8")
            payload = {
                "language": language,
                "split": split,
                "exact_f1": parse_exact_f1(score_text),
                "score_out": str(score_out),
                "hyps_xml": str(hyps_xml),
                "refs_xml": str(refs_xml),
            }
    except Exception as exc:
        error = exc
        if rank_zero:
            log.exception("evaluate_parsing failed for %s/%s", language, split)
    finally:
        if distributed:
            dist.barrier()

    if error is not None:
        raise error
    if not rank_zero:
        return {
            "language": language,
            "split": split,
            "exact_f1": None,
            "score_out": "",
            "hyps_xml": "",
            "refs_xml": "",
        }
    return payload
