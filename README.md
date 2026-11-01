# BT5 and ReGen Reproduction for WebNLG Text-to-Graph

The first author of the [XQDT paper](https://aclanthology.org/2026.emnlp-main.397/) reproduced both BT5 and ReGen for WebNLG text-to-graph parsing, reimplementing the BT5 parser in PyTorch with Hugging Face Transformers and adapting [IBM's original code](https://github.com/IBM/regen) for ReGen. This repository includes training and inference code, saved predictions, and scripts for scoring the predicted subject–relation–object triples.

For BT5, [Anna Nikiforovskaya](https://github.com/Javanochka) shared her six-task setup for English and Russian semantic parsing in [Google's T5 library](https://github.com/google-research/text-to-text-transfer-transformer), along with the processed training data and bilingual title pairs. She also provided example scripts for training and running an RDF-to-text model and explained how to fine-tune a pretrained multilingual model for parsing.

- **BT5** supports English and Russian semantic parsing, following [Agarwal et al. (2020)](https://aclanthology.org/2020.webnlg-1.13/).
- **ReGen** implements the English WebNLG **T2G.CE** setting of [Dognin et al. (2021)](https://aclanthology.org/2021.emnlp-main.83/), using cross-entropy training.

## Results

Scores below are recomputed from the saved predictions on the WebNLG 2020 semantic-parsing test sets. Precision, recall, and F1 use the evaluator's **Exact** criterion and are reported as percentages.

| Model | Language | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| BT5 | English | 67.03 | 68.53 | 67.62 |
| BT5 | Russian | 93.34 | 93.65 | 93.46 |
| ReGen T2G.CE | English | 70.35 | 72.18 | 71.07 |

This reproduction uses ReGen’s cross-entropy model (CE) because it had the highest Exact F1 on WebNLG text-to-graph in [Table 2 of the paper](https://aclanthology.org/2021.emnlp-main.83.pdf#page=7): 72.3 for CE versus 72.0 after reinforcement learning (RL). The [IBM training guide](https://github.com/IBM/regen/blob/main/docs/train_t2g_webnlg.md#overview) reports lower scores for its example run: 70.11 for CE and 70.91 for RL (SCST). The reproduced model scored 71.07, between the two reported CE scores.

## Setup

Use Python 3.11 and install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

## Recompute the saved results

```bash
python score.py
```

This reads the per-example triple predictions and reference XML, rebuilds the candidate XML, and runs the WebNLG evaluator. It prints the result table and saves all four evaluation criteria—Exact, Strict, Partial, and Ent_type—in a new directory under `runs/`.

To score one result:

```bash
python score.py --result bt5-en
python score.py --result bt5-ru
python score.py --result regen-en
```

## Run inference

Use the [BT5 English–Russian checkpoint](https://huggingface.co/Loria-MosAIk/bt5-webnlg-en-ru) or the [ReGen English T2G.CE checkpoint](https://huggingface.co/Loria-MosAIk/regen-webnlg-t2g-ce) from Hugging Face. `--model` also accepts a local model directory.

```bash
python predict.py --method bt5 \
  --model Loria-MosAIk/bt5-webnlg-en-ru --language english

python predict.py --method bt5 \
  --model Loria-MosAIk/bt5-webnlg-en-ru --language russian

python predict.py --method regen \
  --model Loria-MosAIk/regen-webnlg-t2g-ce
```

Each command writes the inputs, generated text, parsed triples, reference XML, and scores to a new directory under `runs/`. Use `--split dev` for development data, `--batch-size` to change the batch size, or `--limit 10` for a short run. `--output` sets the destination directory.

## Train models

The training scripts read the WebNLG splits and BT5 translation pairs from `corpora/`. Set training options through environment variables.

**BT5:**

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash w20bt5reproduce/scripts/run6tasks_mt5base_adamw_equal10w.sh
```

**ReGen T2G.CE:**

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash z_regen_t2d_restart/scripts/run_t5large.sh
```

For two GPUs, set `CUDA_VISIBLE_DEVICES=0,1`; the launchers start distributed training automatically. Their defaults are:

| Setting | BT5 | ReGen T2G.CE |
| --- | --- | --- |
| Initial model | `cointegrated/rut5-base-multitask` | `t5-large` |
| Training batch size per GPU | 8 | 4 |
| Gradient accumulation | 1 | 1 |
| Learning rate | `1e-4` | `5e-5` |
| Optimizer | AdamW | AdamW |
| Training duration | 102,000 updates | 15 epochs |
| Evaluation and checkpoint saving | Every 3,000 updates | Every epoch |
| Precision | bfloat16 | bfloat16 |
| Seed | 197402 | 197402 |

BT5 uses 6,000 warm-up steps followed by cosine restarts. ReGen also uses cosine restarts. The launchers expose batch size, accumulation, learning rate, precision, and evaluation settings. For example, `PER_DEVICE_TRAIN_BATCH_SIZE=4 GRAD_ACCUM=2` changes the BT5 batch configuration.

To continue ReGen training from an existing `model.pt` checkpoint:

```bash
INIT_CHECKPOINT=/path/to/checkpoints/epoch_08/model.pt \
  CUDA_VISIBLE_DEVICES=0 \
  bash z_regen_t2d_restart/scripts/run_t5large_continue.sh
```

The continuation launcher uses four epochs, learning rate `2e-5`, batch size 4 per GPU, and gradient accumulation 2.

## Implementation choices

### BT5 initialization and task sampling

Agarwal et al. first train on translation tasks. This reproduction approximates this stage by starting from [`cointegrated/rut5-base-multitask`](https://huggingface.co/cointegrated/rut5-base-multitask), which already includes English–Russian translation training.

In the initial runs, sampling tasks in proportion to their dataset sizes did not match the scores reported by Agarwal et al. The training script instead samples one of six tasks per batch, with probability **1/6** per task:

- **English semantic parsing:** convert English text into triples.
- **Russian semantic parsing:** convert Russian text into triples.
- **English-to-Russian sentence translation:** translate English sentences into Russian.
- **Russian-to-English sentence translation:** translate Russian sentences into English.
- **English-to-Russian entity translation:** translate English entity names into Russian.
- **Russian-to-English entity translation:** translate Russian entity names into English.

Within each entity-translation task, the base and Wiki entity lists are sampled with equal probability. Thus, each of these four entity data streams receives probability 1/12 overall.

With this setup, the saved BT5 predictions give English Exact F1 of **67.62**, close to the **68.19** reported in [Agarwal et al., Table 8](https://aclanthology.org/2020.webnlg-1.13.pdf). The corresponding Russian scores are **93.46** and **91.11**.

### ReGen cross-entropy training

The implementation uses T5-large, the `Text to Graph:` prefix, and the graph markers `__subject__`, `__predicate__`, and `__object__`. Training uses individual text–graph pairs with graph-order permutations.

### From generated text to triples

BT5 uses `<S>`, `<P>`, and `<O>` to mark the subject, relation, and object. Its preprocessing replaces underscores with spaces and converts relation names such as `followedBy` to `followed by`.

For example, one saved English prediction is:

**Input**

```text
Parse English Turn Me On is a 35.1 minute long album produced by Wharton Tiers that was followed by the album entitled Take it Off.
```

**Generated text**

```text
<S> Turn Me On <P> length <O> 35.1 (minutes) <S> Turn Me On <P> creator <O> Wharton Tiers <S> Turn Me On <P> followed by <O> Take It Off
```

**Parsed triples**

```text
Turn Me On | length | 35.1 (minutes)
Turn Me On | creator | Wharton Tiers
Turn Me On | followed by | Take It Off
```

This shows the model's output, including its predicted relation names. The parser splits on the graph markers and keeps well-formed triples. BT5 removes exact duplicate predictions before scoring; ReGen uses its original parser and serialization.

## Data and evaluation

**Inputs**

- `corpora/webnlg-dataset/release_v3.0/` contains the English and Russian WebNLG training, development, and semantic-parsing test sets in XML format.
- `corpora/data6in1/train_dev_test_redo/` contains the BT5 parsing inputs and target outputs in TSV format.
- [BT5 translation pairs](corpora/data6in1/train_anna_aux/) cover sentences (`translate_english.tsv`, `translate_russian.tsv`), entity names (`translate_entities_english.tsv`, `translate_entities_russian.tsv`), and Wiki titles (`translate_entities_english_wiki.tsv`, `translate_entities_russian_wiki.tsv`). The language in each filename is the input language.
- `corpora/webnlg-references/` contains the ReGen evaluation references.

**Saved predictions**

- `predictions/bt5-en/` and `predictions/bt5-ru/` contain the generated text, parsed triples, and references for English and Russian BT5, respectively.
- `predictions/regen-en/` contains the parsed triples and references for English ReGen.

**Scoring**

`eval/WebNLG-Text-to-triples/` contains the WebNLG scoring implementation.

Evaluation uses the same example order in the predictions and references. The evaluator normalizes case, underscores, and relation-name spacing, aligns predicted and reference triples, and reports its Exact, Strict, Partial, and Ent_type scores. **Exact** here names the WebNLG evaluator's criterion; the separate `Full triple scores` block is a different measure. See `score.out` in each new scoring directory for all results.

## Citation

Please cite the original paper for the model you use. If you use the reproduction code or checkpoints, please also cite [XQDT](https://aclanthology.org/2026.emnlp-main.397/).

```bibtex
@inproceedings{agarwal-etal-2020-machine,
  title = {Machine Translation Aided Bilingual Data-to-Text Generation and Semantic Parsing},
  author = {Agarwal, Oshin and Kale, Mihir and Ge, Heming and Shakeri, Siamak and Al-Rfou, Rami},
  booktitle = {Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)},
  year = {2020},
  publisher = {Association for Computational Linguistics},
  pages = {125--130},
  url = {https://aclanthology.org/2020.webnlg-1.13/}
}

@inproceedings{dognin-etal-2021-regen,
  title = {{ReGen}: Reinforcement Learning for Text and Knowledge Base Generation using Pretrained Language Models},
  author = {Dognin, Pierre and Padhi, Inkit and Melnyk, Igor and Das, Payel},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year = {2021},
  publisher = {Association for Computational Linguistics},
  pages = {1084--1099},
  doi = {10.18653/v1/2021.emnlp-main.83},
  url = {https://aclanthology.org/2021.emnlp-main.83/}
}

@inproceedings{efimov-zhang-etal-2026-xqdt,
  title = {{XQDT}: eXplainable and Quantitative Data-Text Alignment Metric with Feedback Signals},
  author = {Efimov-Zhang, Kun and Song, Yifei and Gardent, Claire},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  publisher = {Association for Computational Linguistics},
  pages = {9121--9157},
  doi = {10.18653/2026.emnlp-main.397},
  url = {https://aclanthology.org/2026.emnlp-main.397/}
}
```

## Sources and licenses

- The ReGen code is adapted from [IBM/regen](https://github.com/IBM/regen) and retains its Apache-2.0 license in `LICENSE`.
- WebNLG 3.0 comes from the [WebNLG dataset repository](https://gitlab.com/shimorina/webnlg-dataset), whose data are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The evaluator and corpus reader retain their GPL-3.0 and MIT licenses in their respective directories.
