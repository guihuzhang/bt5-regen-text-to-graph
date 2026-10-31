import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
SAVED_RESULTS = ("bt5-en", "bt5-ru", "regen-en")


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def new_output_directory(value):
    path = project_path(value)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_scores(text):
    total = text.split("Total scores", 1)[1].split("Scores per tag", 1)[0]
    scores = {}
    for name in ("Ent_type", "Partial", "Strict", "Exact"):
        match = re.search(
            rf"(?:^|\n){name}\s*\n.*?Precision:\s*([\d.eE+-]+)"
            rf"\s+Recall:\s*([\d.eE+-]+)\s+F1:\s*([\d.eE+-]+)",
            total,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"Missing {name} scores in evaluator output")
        scores[name] = dict(zip(("precision", "recall", "f1"), map(float, match.groups())))
    return scores


def score_predictions(predictions_path, reference_path, output_dir, deduplicate):
    for path in (predictions_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation input: {path}")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    reference_entries = ElementTree.parse(reference_path).findall("./entries/entry")
    if not predictions or len(predictions) != len(reference_entries):
        raise ValueError(
            f"Predictions: {len(predictions)}; reference entries: {len(reference_entries)}"
        )
    benchmark = ElementTree.Element("benchmark")
    entries = ElementTree.SubElement(benchmark, "entries")
    for index, (triples, reference) in enumerate(zip(predictions, reference_entries)):
        if not isinstance(triples, list) or not all(isinstance(triple, str) for triple in triples):
            raise ValueError(f"Prediction {index} must be a list of triple strings")
        if deduplicate:
            triples = list(dict.fromkeys(triples))
        entry = ElementTree.SubElement(entries, "entry", reference.attrib)
        generated = ElementTree.SubElement(entry, "generatedtripleset")
        for triple in triples:
            if len(triple.split(" | ")) != 3:
                raise ValueError(f"Malformed triple at prediction {index}: {triple!r}")
            ElementTree.SubElement(generated, "gtriple").text = triple
    output_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_dir / "hyps.xml"
    ElementTree.indent(benchmark)
    ElementTree.ElementTree(benchmark).write(hypothesis_path, encoding="utf-8", xml_declaration=True)
    command = [
        sys.executable,
        str(ROOT / "eval/WebNLG-Text-to-triples/Evaluation_script.py"),
        str(reference_path),
        str(hypothesis_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (output_dir / "score.out").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr or "WebNLG evaluator failed")
    scores = read_scores(completed.stdout)
    result = {"samples": len(predictions), "scores": scores}
    (output_dir / "scores.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Recompute WebNLG scores from saved triple predictions.")
    parser.add_argument("--result", choices=(*SAVED_RESULTS, "all"), default="all")
    parser.add_argument("--output", default=f"runs/scores-{datetime.now():%Y%m%d-%H%M%S}")
    args = parser.parse_args()
    output = new_output_directory(args.output)
    selected = SAVED_RESULTS if args.result == "all" else (args.result,)
    print("| Result | Samples | Exact P | Exact R | Exact F1 |", flush=True)
    print("| --- | ---: | ---: | ---: | ---: |", flush=True)
    for name in selected:
        saved = ROOT / "predictions" / name
        result = score_predictions(
            saved / "hypotheses.triples.json",
            saved / "refs.xml",
            output / name,
            deduplicate=name.startswith("bt5"),
        )
        exact = result["scores"]["Exact"]
        print(
            f"| {name} | {result['samples']} | {100 * exact['precision']:.2f} | "
            f"{100 * exact['recall']:.2f} | {100 * exact['f1']:.2f} |",
            flush=True,
        )
    print(f"\nFull scores: {output}")


if __name__ == "__main__":
    main()
