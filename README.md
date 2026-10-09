# Fingerprinting LLM Families from Their Text

**CMPSC 448 · Penn State · Fall 2026 — midterm project**

Can a small neural classifier tell which *family* of large language model generated a
response without self-identification response hints? We constructed balanced
prompt/response datasets for six LLM families from
[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), trained a
multi-kernel **TextCNN** and a **bidirectional LSTM** on them, and used the two models to
answer four research questions.

| | Research question |
|---|---|
| **RQ1** | How detectable is model identity from the **response alone**? |
| **RQ2** | Do **prompts** leak model identity, and does adding the prompt to the response improve classification? |
| **RQ3** | Do classifiers learn model-wide fingerprints or **domain-specific** patterns? (train on one task category, test zero-shot on others) |
| **RQ4** | Is the signal mainly **structural** (markdown formatting) or **linguistic**? |

The full write-up is in [`report/REPORT.md`](report/REPORT.md); the auto-generated results
table is in [`results/results_summary.md`](results/results_summary.md).

---

## Repository layout

```
.
├── config.py                  # paths, LLM-family map, sequence lengths, seed
├── dataset_authorization.py   # checks Hugging Face login + gated-dataset access
├── load_data.py               # downloads LMSYS-Chat-1M to ./local_datasets (git-ignored)
├── data_cleaning.py           # Phase 1: filter families, sample, format, strip self-ID
├── data_processing.py         # Phase 1: tokenize, vocab, pad/truncate, 70/15/15 split
├── models.py                  # Phase 2: TextCNN and BiLSTM (shared embedding config)
├── model_training.py          # Phase 2: training loop, metrics, grid search
├── research_questions.py      # Phase 3/4: RQ1-RQ4 experiments, statistics, figures
├── visualization.py           # all report figures
├── data/
│   ├── cleaned/lmsys_clean.csv        # (LLM_name, LLM_input, LLM_output) + metadata
│   └── processed/                     # token-id arrays per configuration + vocab + split
├── models/                    # best checkpoint per experiment (<rq>_<model>_<mode>.pt)
├── results/
│   ├── eda/                   # dataset overview figures
│   ├── rq1/ rq2/ rq3/ rq4/    # per-experiment metrics, grid results, curves, confusion matrices
│   └── results_summary.md
└── report/REPORT.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LMSYS-Chat-1M is a **gated** dataset: request access on its Hugging Face page and log in
with `hf auth login` (or let `dataset_authorization.py` prompt you).

A GPU is strongly recommended for the BiLSTM grid search (Google Colab's free T4 works).
`model_training.get_device()` picks CUDA → Apple MPS → CPU automatically.

## Reproducing everything

```bash
# Phase 1 — data
python load_data.py                     # ~2.5 GB download into ./local_datasets
python data_cleaning.py                 # -> data/cleaned/lmsys_clean.csv (6 families x 4,000)
python data_processing.py               # -> data/processed/ + results/eda/

# Phase 2/3 — required experiments (grid search + test evaluation)
python research_questions.py --rq 1 2

# Phase 4 — extra credit (re-uses the hyper-parameters selected in RQ1)
python research_questions.py --rq 3 4
```

Useful options:

| Command | What it does |
|---|---|
| `python data_cleaning.py --families GPT Claude LLaMA-2 Vicuna --n-per-family 3000` | choose 4-6 families / sample size |
| `python data_processing.py --tokenizer spacy` | spaCy tokenizer instead of NLTK |
| `python data_processing.py --input other.csv` | process an externally cleaned file (columns `LLM_name`, `LLM_input`, `LLM_output`) |
| `python research_questions.py --rq 1 2 3 4 --grid tiny --epochs 3` | quick end-to-end smoke test |
| `python research_questions.py --rq 2 --grid full` | larger grid |
| `python research_questions.py --rq 1 --force` | retrain even if results already exist |
| `python model_training.py --model rnn --mode both` | one grid search outside the RQ driver |

Finished experiments are cached in `results/`, so an interrupted run resumes where it stopped.

## Method summary

**Families.** GPT (gpt-3.5-turbo, gpt-4), Claude (claude-1, claude-2, claude-instant-1),
LLaMA-2 (7b/13b-chat), Vicuna (7b/13b/33b), PaLM (palm-2), MPT (7b/30b-chat).
The classifier predicts the **family**; the output layer size is the number of families.

**Cleaning** (`data_cleaning.py`). English conversations only; first user prompt and first
assistant reply; drop moderation-flagged turns and responses under 5 words; keep one
response per (family, prompt) to remove LMSYS's many repeated test prompts; random-sample
4,000 pairs per family (seed 42). Self-identification is removed at the sentence level
(a sentence that names a model/vendor *and* refers to the speaker, e.g. *"I am Claude, made
by Anthropic"*), and any remaining model names in prompts or responses are masked as
`[MODEL]`. Generic disclaimers that name no model (*"As an AI language model, …"*) are kept
because they are style, not identity. Each prompt also gets a heuristic task category
(coding / math / creative / general) for RQ3.

**Processing** (`data_processing.py`). Stratified 70/15/15 split *before* building the
vocabulary, so the vocabulary (min frequency 2, max 30k, lower-cased NLTK tokens) only
sees training text. One vocabulary is shared by all three sequence configurations:

| Configuration | Sequence | Length |
|---|---|---|
| `input` | prompt tokens | 128 |
| `output` | response tokens | 256 |
| `both` | prompt + `<sep>` + response | 385 |

Sequences are truncated from the end and right-padded; batches are trimmed to their longest
sequence at training time.

**Models** (`models.py`). Both use a 128-d embedding with `padding_idx=0`.
*TextCNN*: parallel `Conv1d` layers (100 filters each, kernel sizes from the grid) → ReLU →
global max-pool over time → concat → Dropout(0.5) → Linear.
*BiLSTM*: packed bidirectional LSTM (hidden 128 or 256) → concat final forward and backward
hidden states → Dropout(0.5) → Linear.

**Training** (`model_training.py`). `nn.CrossEntropyLoss`, AdamW (weight decay 0.01),
gradient clipping at 1.0, up to 10 epochs with early stopping (patience 3) on validation
macro-F1; the best epoch's weights are kept. Accuracy, macro-F1 and the confusion matrix are
recorded on the validation set every epoch (`history.json`).

| Grid (`small`, default) | CNN | RNN |
|---|---|---|
| learning rate | 1e-3, 3e-4 | 1e-3, 3e-4 |
| batch size | 32, 64 | 32, 64 |
| kernel sizes | (3,4,5), (2,3,4,5) | — |
| hidden dim | — | 128, 256 |

**Evaluation** (`research_questions.py`). Test accuracy and macro-F1 with 95% bootstrap
confidence intervals, row-normalised confusion matrices, per-class F1, exact McNemar tests
between paired models (CNN vs RNN; response-only vs prompt+response), and a binomial test of
prompt-only accuracy against chance.

## Data note

LMSYS-Chat-1M is released under a license agreement that every user accepts on Hugging Face.
Check its redistribution terms before pushing `data/cleaned/` or `data/processed/` to a
**public** repository; if in doubt, keep the repo private or add those folders to
`.gitignore` (everything can be regenerated with the commands above).
