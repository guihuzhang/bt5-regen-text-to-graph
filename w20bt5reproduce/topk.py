import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_topk(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_topk(path: Path, candidate: Dict, k: int = 3) -> Tuple[List[Dict], List[Dict], bool]:
    current = load_topk(path)
    combined = {entry["step"]: entry for entry in current}
    combined[candidate["step"]] = candidate
    ranked = sorted(
        combined.values(),
        key=lambda entry: (
            entry["exact_avg"],
            entry["exact_en"],
            entry["exact_ru"],
            -entry["step"],
        ),
        reverse=True,
    )
    keep = ranked[:k]
    evicted = ranked[k:]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    kept = any(entry["step"] == candidate["step"] for entry in keep)
    return keep, evicted, kept
