"""
Phase 1 - Data Processing
=========================
Turns the cleaned CSV into padded integer tensors that the CNN/RNN can consume.

Steps
-----
1. Stratified 70 / 15 / 15 train / validation / test split (by LLM family)
2. Tokenize LLM_input and LLM_output (NLTK by default; spaCy or regex optional)
3. Build ONE fixed vocabulary from the *training* split only (no test leakage),
   shared by all three sequence configurations so results are comparable
4. Encode + pad/truncate each configuration:
       input   -> LLM_input tokens                          (MAX_INPUT_LEN)
       output  -> LLM_output tokens                         (MAX_OUTPUT_LEN)
       both    -> LLM_input + <sep> + LLM_output tokens     (MAX_INPUT_LEN + 1 + MAX_OUTPUT_LEN)
5. Save to data/processed/:
       meta.csv            one row per observation: split, label, family, task category
       {mode}.npz          X (token ids), lengths, y   for mode in input/output/both
       vocab.json, label_map.json, processing_config.json

Usage
-----
    python data_processing.py                          # uses data/cleaned/lmsys_clean.csv
    python data_processing.py --input my_clean.csv --tokenizer spacy
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CLEAN_CSV, FAMILY_MAP, MAX_INPUT_LEN, MAX_OUTPUT_LEN, MODES,
                    PAD, PAD_IDX, PROCESSED_DIR, SEED, SEP, SEP_IDX, UNK, UNK_IDX)

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_COLUMN_ALIASES = {
    "llm_name": "LLM_name", "model_family": "LLM_name", "family": "LLM_name", "label": "LLM_name",
    "llm_input": "LLM_input", "prompt": "LLM_input", "input": "LLM_input",
    "llm_output": "LLM_output", "response": "LLM_output", "output": "LLM_output",
    "task_category": "task_category",
}


def load_clean(path=CLEAN_CSV) -> pd.DataFrame:
    """Load the cleaned dataset (CSV, Parquet or JSONL) and normalise column names."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={c: _COLUMN_ALIASES[c.lower()] for c in df.columns
                            if c.lower() in _COLUMN_ALIASES})
    missing = {"LLM_name", "LLM_input", "LLM_output"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["LLM_input"] = df["LLM_input"].fillna("").astype(str)
    df["LLM_output"] = df["LLM_output"].fillna("").astype(str)
    if "task_category" not in df:
        from data_cleaning import categorize_prompt
        df["task_category"] = df["LLM_input"].apply(categorize_prompt)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
_REGEX_TOKEN_RE = re.compile(r"```|\*\*|__|#+|\w+|[^\w\s]")


def get_tokenizer(backend: str = "nltk"):
    """Return a function str -> list[str]."""
    if backend == "nltk":
        import nltk
        for res, path in (("punkt_tab", "tokenizers/punkt_tab"), ("punkt", "tokenizers/punkt")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(res, quiet=True)
        from nltk.tokenize import word_tokenize
        try:
            word_tokenize("warm up")
        except LookupError:
            print("[!] NLTK punkt data unavailable - falling back to the regex tokenizer")
            return lambda s: _REGEX_TOKEN_RE.findall(s)
        return word_tokenize
    if backend == "spacy":
        import spacy
        nlp = spacy.blank("en")            # rule-based tokenizer, no model download
        return lambda s: [t.text for t in nlp.tokenizer(s) if not t.is_space]
    if backend == "regex":
        return lambda s: _REGEX_TOKEN_RE.findall(s)
    raise ValueError(f"Unknown tokenizer backend: {backend}")


def tokenize_texts(texts, backend="nltk", lowercase=True, max_tokens=None):
    """Tokenize an iterable of strings. `max_tokens` caps work on very long texts."""
    tok = get_tokenizer(backend)
    out = []
    for t in texts:
        if lowercase:
            t = t.lower()
        # cheap pre-truncation by characters so a 20k-char response doesn't
        # dominate runtime; ~8 chars/token leaves plenty of slack
        if max_tokens:
            t = t[: max_tokens * 8]
        toks = tok(t)
        out.append(toks[:max_tokens] if max_tokens else toks)
    return out


# --------------------------------------------------------------------------- #
# Split, vocabulary, encoding
# --------------------------------------------------------------------------- #
def stratified_split(df, train=0.70, val=0.15, test=0.15, seed=SEED) -> pd.Series:
    """Return a Series of 'train' / 'val' / 'test' aligned with df.index."""
    assert abs(train + val + test - 1.0) < 1e-9
    idx = np.arange(len(df))
    y = df["LLM_name"].values
    tr, rest = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=test / (val + test), stratify=y[rest],
                              random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    return pd.Series(split, index=df.index, name="split")


def build_vocab(token_lists, min_freq=2, max_size=30000) -> dict:
    counts = Counter(t for toks in token_lists for t in toks)
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX, SEP: SEP_IDX}
    for tok, c in counts.most_common():
        if c < min_freq or len(vocab) >= max_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    """Map tokens to ids, truncate to max_len (keep the beginning), right-pad."""
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    length = max(len(ids), 1)                      # LSTM packing needs length >= 1
    return ids + [PAD_IDX] * (max_len - len(ids)), length


def encode_mode(in_toks, out_toks, vocab, mode, max_in, max_out):
    if mode == "input":
        pairs = [encode(t, vocab, max_in) for t in in_toks]
    elif mode == "output":
        pairs = [encode(t, vocab, max_out) for t in out_toks]
    elif mode == "both":
        total = max_in + 1 + max_out
        pairs = [encode(i[:max_in] + [SEP] + o[:max_out], vocab, total)
                 for i, o in zip(in_toks, out_toks)]
    else:
        raise ValueError(mode)
    X = np.asarray([p[0] for p in pairs], dtype=np.int32)
    lengths = np.asarray([p[1] for p in pairs], dtype=np.int32)
    return X, lengths


def label_order(families):
    """Label ids follow FAMILY_MAP order so plots are consistent across runs."""
    known = [f for f in FAMILY_MAP if f in set(families)]
    extra = sorted(set(families) - set(known))
    return {f: i for i, f in enumerate(known + extra)}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def process(df, out_dir=PROCESSED_DIR, backend="nltk", max_in=MAX_INPUT_LEN,
            max_out=MAX_OUTPUT_LEN, min_freq=2, max_vocab=30000, modes=MODES,
            split=None, output_transform=None, verbose=True):
    """Tokenize, build vocab, encode and save. Returns a dict of summary stats.

    `split` lets other experiments (RQ4) reuse the exact same split.
    `output_transform` is applied to LLM_output before tokenizing (RQ4 stripping).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["split"] = stratified_split(df) if split is None else np.asarray(split)
    label_map = label_order(df["LLM_name"].unique())
    df["label"] = df["LLM_name"].map(label_map)

    outputs = df["LLM_output"]
    if output_transform is not None:
        outputs = outputs.apply(output_transform)

    if verbose:
        print(f"Tokenizing {len(df):,} prompts and responses with '{backend}' ...")
    # keep a generous cap so length statistics are still meaningful
    in_toks = tokenize_texts(df["LLM_input"], backend, max_tokens=max(4 * max_in, 1024))
    out_toks = tokenize_texts(outputs, backend, max_tokens=max(4 * max_out, 2048))

    train_mask = (df["split"] == "train").values
    vocab = build_vocab([t for t, m in zip(in_toks, train_mask) if m] +
                        [t for t, m in zip(out_toks, train_mask) if m],
                        min_freq=min_freq, max_size=max_vocab)

    in_len = np.array([len(t) for t in in_toks])
    out_len = np.array([len(t) for t in out_toks])
    stats = {
        "n_rows": int(len(df)),
        "split_sizes": df["split"].value_counts().to_dict(),
        "tokenizer": backend,
        "vocab_size": len(vocab),
        "max_input_len": max_in,
        "max_output_len": max_out,
        "min_freq": min_freq,
        "input_len_percentiles": {p: float(np.percentile(in_len, p)) for p in (50, 75, 90, 95, 99)},
        "output_len_percentiles": {p: float(np.percentile(out_len, p)) for p in (50, 75, 90, 95, 99)},
        "pct_inputs_truncated": float((in_len > max_in).mean()),
        "pct_outputs_truncated": float((out_len > max_out).mean()),
    }
    # OOV rate on validation output tokens (how well the vocab generalises)
    val_out = [t for t, s in zip(out_toks, df["split"]) if s == "val"]
    n_val = sum(len(t) for t in val_out)
    stats["val_output_oov_rate"] = float(
        sum(tok not in vocab for t in val_out for tok in t) / max(n_val, 1))

    for mode in modes:
        X, lengths = encode_mode(in_toks, out_toks, vocab, mode, max_in, max_out)
        np.savez_compressed(out_dir / f"{mode}.npz", X=X, lengths=lengths,
                            y=df["label"].values.astype(np.int64))

    meta_cols = [c for c in ("conversation_id", "LLM_name", "label", "task_category", "split")
                 if c in df]
    meta = df[meta_cols].copy()
    meta["input_tokens"], meta["output_tokens"] = in_len, out_len
    meta.to_csv(out_dir / "meta.csv", index=False)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(out_dir / "processing_config.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    if verbose:
        print(f"Vocabulary size: {len(vocab):,} (min_freq={min_freq})")
        print(f"Split sizes: {stats['split_sizes']}")
        print(f"Output tokens  p50={stats['output_len_percentiles'][50]:.0f}  "
              f"p95={stats['output_len_percentiles'][95]:.0f}  "
              f"-> {stats['pct_outputs_truncated']:.1%} truncated at {max_out}")
        print(f"Input tokens   p50={stats['input_len_percentiles'][50]:.0f}  "
              f"p95={stats['input_len_percentiles'][95]:.0f}  "
              f"-> {stats['pct_inputs_truncated']:.1%} truncated at {max_in}")
        print(f"Saved processed data to {out_dir}")
    return stats


def load_processed(mode, processed_dir=PROCESSED_DIR):
    """Load arrays for one mode plus the shared metadata.

    Returns dict with X, lengths, y, meta (DataFrame), label_map, vocab_size.
    """
    processed_dir = Path(processed_dir)
    arr = np.load(processed_dir / f"{mode}.npz")
    meta = pd.read_csv(processed_dir / "meta.csv")
    with open(processed_dir / "label_map.json") as f:
        label_map = json.load(f)
    with open(processed_dir / "vocab.json") as f:
        vocab_size = len(json.load(f))
    return {"X": arr["X"], "lengths": arr["lengths"], "y": arr["y"], "meta": meta,
            "label_map": label_map, "vocab_size": vocab_size}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(CLEAN_CSV), help="cleaned CSV / Parquet / JSONL")
    p.add_argument("--out-dir", default=str(PROCESSED_DIR))
    p.add_argument("--tokenizer", default="nltk", choices=["nltk", "spacy", "regex"])
    p.add_argument("--max-input-len", type=int, default=MAX_INPUT_LEN)
    p.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab", type=int, default=30000)
    args = p.parse_args()

    df = load_clean(args.input)
    print(f"Loaded {len(df):,} rows, families: {df['LLM_name'].value_counts().to_dict()}")
    process(df, args.out_dir, args.tokenizer, args.max_input_len, args.max_output_len,
            args.min_freq, args.max_vocab)

    # dataset description figures for the report
    from visualization import plot_dataset_overview
    plot_dataset_overview(Path(args.out_dir) / "meta.csv")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
