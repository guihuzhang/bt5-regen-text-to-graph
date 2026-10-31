import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.webnlg.benchmark_reader import Benchmark, select_files


DEFAULT_INPUTS = {
    "en": PROJECT_ROOT / "corpora" / "webnlg-dataset" / "release_v3.0" / "en",
    "ru": PROJECT_ROOT / "corpora" / "webnlg-dataset" / "release_v3.0" / "ru",
}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "corpora" / "data6in1" / "train_dev_test_redo"

PREFIXES = {
    "en": "Parse English",
    "ru": "Parse Russian",
}


def normalize_piece(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\^\^.+$", "", text)
    text = re.sub(r"@[\w-]+$", "", text)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_predicate(text: str) -> str:
    text = normalize_piece(text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def linearize_modified_triples(entry) -> str:
    pieces = []
    for triple in entry.modifiedtripleset.triples:
        subj = normalize_piece(triple.s)
        pred = normalize_predicate(triple.p)
        obj = normalize_piece(triple.o)
        pieces.append(f"<S> {subj} <P> {pred} <O> {obj}")
    return " ".join(pieces)


def load_benchmark_from_dir(split_dir: Path) -> Benchmark:
    benchmark = Benchmark()
    benchmark.fill_benchmark(select_files(str(split_dir)))
    return benchmark


def load_benchmark_from_file(xml_file: Path) -> Benchmark:
    benchmark = Benchmark()
    benchmark.fill_benchmark([(str(xml_file.parent), xml_file.name)])
    return benchmark


def build_rows(benchmark: Benchmark, lang: str):
    prefix = PREFIXES[lang]
    rows = []
    for entry in benchmark.entries:
        target = linearize_modified_triples(entry)
        for lex in entry.lexs:
            source = f"{prefix} {lex.lex.strip()}"
            rows.append((source, target))
    return rows


def write_tsv(output_file: Path, rows):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(
            fp,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(["inputs", "targets"])
        writer.writerows(rows)


def main():
    output_dir = DEFAULT_OUTPUT_DIR

    jobs = [
        (
            "parsing_english_train.tsv",
            load_benchmark_from_dir(DEFAULT_INPUTS["en"] / "train"),
            "en",
        ),
        (
            "parsing_english_dev.tsv",
            load_benchmark_from_dir(DEFAULT_INPUTS["en"] / "dev"),
            "en",
        ),
        (
            "parsing_english_test.tsv",
            load_benchmark_from_file(
                DEFAULT_INPUTS["en"] / "test" / "semantic-parsing-test-data-with-refs-en.xml"
            ),
            "en",
        ),
        (
            "parsing_russian_train.tsv",
            load_benchmark_from_dir(DEFAULT_INPUTS["ru"] / "train"),
            "ru",
        ),
        (
            "parsing_russian_dev.tsv",
            load_benchmark_from_dir(DEFAULT_INPUTS["ru"] / "dev"),
            "ru",
        ),
        (
            "parsing_russian_test.tsv",
            load_benchmark_from_file(
                DEFAULT_INPUTS["ru"] / "test" / "semantic-parsing-test-data-with-refs-ru.xml"
            ),
            "ru",
        ),
    ]

    for filename, benchmark, lang in jobs:
        rows = build_rows(benchmark, lang)
        output_file = output_dir / filename
        write_tsv(output_file, rows)
        print(f"{filename}\t{len(rows)}")


if __name__ == "__main__":
    main()
