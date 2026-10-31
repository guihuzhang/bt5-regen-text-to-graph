import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_topk(topk_path: Path) -> List[Dict]:
    if not topk_path.exists():
        return []
    with open(topk_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_topk(topk_path: Path, entries: List[Dict]) -> None:
    topk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(topk_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def update_topk(topk_path: Path, candidate: Dict, k: int = 3) -> Tuple[List[Dict], List[Dict], bool]:
    current = load_topk(topk_path)
    combined = {entry["epoch"]: entry for entry in current}
    combined[candidate["epoch"]] = candidate

    ranked = sorted(
        combined.values(),
        key=lambda x: (x["exact_f1"], x["epoch"]),
        reverse=True,
    )
    keep = ranked[:k]
    keep_epochs = {entry["epoch"] for entry in keep}
    evicted = ranked[k:]

    save_topk(topk_path, keep)
    is_kept = candidate["epoch"] in keep_epochs
    return keep, evicted, is_kept
