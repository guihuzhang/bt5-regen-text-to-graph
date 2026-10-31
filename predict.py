import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from score import new_output_directory, score_predictions


def main():
    parser = argparse.ArgumentParser(description="Run a BT5 or ReGen checkpoint on WebNLG.")
    parser.add_argument("--method", choices=("bt5", "regen"), required=True)
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    parser.add_argument("--language", choices=("english", "russian"), default="english")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--output", default=f"runs/inference-{datetime.now():%Y%m%d-%H%M%S}")
    args = parser.parse_args()
    if args.method == "regen" and args.language != "english":
        parser.error("The ReGen checkpoint is trained for English text-to-graph generation.")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive.")

    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer

    output = new_output_directory(args.output)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = (
        AutoTokenizer.from_pretrained(args.model, use_fast=False)
        if args.method == "bt5"
        else T5Tokenizer.from_pretrained(args.model, legacy=True)
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.precision))
    model.to(device).eval()
    if args.method == "bt5":
        from w20bt5reproduce.dataset_6in1 import Seq2SeqCollator, build_eval_dataset
        from w20bt5reproduce.eval_parsing import anna_texts_to_triples, build_meta_infos, generate_predictions
        from utils.utils_webnlg import create_xml

        dataset = build_eval_dataset(args.language, args.split)
        selected = Subset(dataset, range(min(args.limit, len(dataset)))) if args.limit else dataset
        loader = DataLoader(selected, batch_size=args.batch_size, collate_fn=Seq2SeqCollator(tokenizer, 256, 256))
        _, hypotheses, targets = generate_predictions(model, loader, tokenizer, device, 5, 256, "BT5")
        triples = anna_texts_to_triples(hypotheses)
        references = anna_texts_to_triples(targets)
        reference_tree = create_xml(references, build_meta_infos(len(references)), "modifiedtripleset", "mtriple")
        ElementTree.ElementTree(reference_tree).write(output / "refs.xml", encoding="utf-8", xml_declaration=True)
        inputs = [selected[index]["input_text"] for index in range(len(selected))]
    else:
        from dataset.webnlg_dataset import WebNLGDataset
        from z_regen_t2d_restart.eval_t2d import generate_predictions, get_reference_xml
        from utils.utils_webnlg_parse import get_triples

        split = "testA" if args.split == "test" else "val"
        dataset = WebNLGDataset(split, tokenizer, src="A", prepare="webnlg")
        if args.limit:
            dataset.samples = dataset.samples[:args.limit]
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset._collate_fn)
        _, hypotheses, _, _ = generate_predictions(model, loader, tokenizer, device, 5, 256, "ReGen")
        triples = get_triples(hypotheses)
        reference = get_reference_xml("testA" if args.split == "test" else "valA")
        shutil.copy2(reference, output / "refs.xml")
        if args.limit:
            tree = ElementTree.parse(output / "refs.xml")
            entries = tree.find("./entries")
            for entry in list(entries)[len(dataset):]:
                entries.remove(entry)
            tree.write(output / "refs.xml", encoding="utf-8", xml_declaration=True)
        inputs = ["Text to Graph: " + dataset[index]["text"] for index in range(len(dataset))]
    for name, values in (("inputs.json", inputs), ("hypotheses.text.json", hypotheses), ("hypotheses.triples.json", triples)):
        (output / name).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_score:
        result = score_predictions(output / "hypotheses.triples.json", output / "refs.xml", output, args.method == "bt5")
        print(f"Exact F1: {100 * result['scores']['Exact']['f1']:.2f}")
    print(f"Predictions: {output}")


if __name__ == "__main__":
    main()
