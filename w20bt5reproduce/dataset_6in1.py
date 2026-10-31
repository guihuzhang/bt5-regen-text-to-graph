import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parents[1]
PARSING_REDO_DIR = ROOT / "corpora" / "data6in1" / "train_dev_test_redo"
AUX_TRAIN_DIR = ROOT / "corpora" / "data6in1" / "train_anna_aux"

TRAIN_TASK_FILES = {
    "parsing_english": PARSING_REDO_DIR / "parsing_english_train.tsv",
    "parsing_russian": PARSING_REDO_DIR / "parsing_russian_train.tsv",
    "translate_english": AUX_TRAIN_DIR / "translate_english.tsv",
    "translate_russian": AUX_TRAIN_DIR / "translate_russian.tsv",
}

ENTITY_TRAIN_TASK_FILES = {
    "translate_entities_english": {
        "base": AUX_TRAIN_DIR / "translate_entities_english.tsv",
        "wiki": AUX_TRAIN_DIR / "translate_entities_english_wiki.tsv",
    },
    "translate_entities_russian": {
        "base": AUX_TRAIN_DIR / "translate_entities_russian.tsv",
        "wiki": AUX_TRAIN_DIR / "translate_entities_russian_wiki.tsv",
    },
}

CANONICAL_TASK_NAMES = [
    "parsing_english",
    "parsing_russian",
    "translate_english",
    "translate_russian",
    "translate_entities_english",
    "translate_entities_russian",
]

EVAL_TASK_FILES = {
    ("english", "dev"): PARSING_REDO_DIR / "parsing_english_dev.tsv",
    ("english", "test"): PARSING_REDO_DIR / "parsing_english_test.tsv",
    ("russian", "dev"): PARSING_REDO_DIR / "parsing_russian_dev.tsv",
    ("russian", "test"): PARSING_REDO_DIR / "parsing_russian_test.tsv",
}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"inputs": row["inputs"], "targets": row["targets"]})
    return rows


@dataclass
class SampleMeta:
    task_name: str
    sample_id: int


class AnnaTSVDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, task_name: str):
        self.path = path
        self.task_name = task_name
        self.rows = _read_tsv(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {
            "input_text": row["inputs"],
            "target_text": row["targets"],
            "meta": {"task_name": self.task_name, "sample_id": idx},
        }


class Seq2SeqCollator:
    def __init__(self, tokenizer, max_source_length: int, max_target_length: int):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        inputs = [item["input_text"] for item in batch]
        targets = [item["target_text"] for item in batch]
        metas = [item["meta"] for item in batch]

        source_batch = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        # tokenize targets as plain text — text_target= kwarg unreliable on
        # slow T5Tokenizer; passing positional works across all HF versions
        target_batch = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = target_batch["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source_batch["input_ids"],
            "attention_mask": source_batch["attention_mask"],
            "labels": labels,
            "target_texts": targets,
            "metas": metas,
        }


def build_train_datasets(task_names=None, use_wiki_entities: bool = True) -> Dict[str, AnnaTSVDataset]:
    selected = task_names or CANONICAL_TASK_NAMES
    out = {}
    for task_name in selected:
        if task_name in TRAIN_TASK_FILES:
            path = TRAIN_TASK_FILES[task_name]
            if not path.exists():
                raise FileNotFoundError(f"missing training TSV: {path}")
            out[task_name] = AnnaTSVDataset(path, task_name)
            continue

        if task_name in ENTITY_TRAIN_TASK_FILES:
            variant_names = ["base", "wiki"] if use_wiki_entities else ["base"]
            for variant_name in variant_names:
                path = ENTITY_TRAIN_TASK_FILES[task_name][variant_name]
                if not path.exists():
                    raise FileNotFoundError(f"missing training TSV: {path}")
                dataset_key = f"{task_name}__{variant_name}"
                out[dataset_key] = AnnaTSVDataset(path, task_name)
            continue

        raise KeyError(f"unknown training task: {task_name}")
    return out


def build_eval_dataset(language: str, split: str) -> AnnaTSVDataset:
    key = (language, split)
    path = EVAL_TASK_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"missing eval TSV: {path}")
    return AnnaTSVDataset(path, f"parsing_{language}_{split}")
