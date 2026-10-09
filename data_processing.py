import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CLEAN_CSV, MAX_INPUT_LEN, MAX_OUTPUT_LEN,
                    PAD, PAD_IDX, PROCESSED_DIR, SEED, SEP, SEP_IDX,
                    UNK, UNK_IDX)

# --------------------------------------------------------------------------- #
# Simple whitespace + punctuation tokeniser (no external deps)
# --------------------------------------------------------------------------- #
_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # Lowercase whitespace-split with punctuation separated
    if not isinstance(text, str):
        return []
    return _TOK_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def load_clean(path=CLEAN_CSV) -> pd.DataFrame:
    # Read the cleaned CSV produced by data_cleaning.py
    return pd.read_csv(path)


def _build_vocab(token_lists, min_freq=2):
    # Build word → index mapping from an iterable of token lists
    counts = Counter(tok for tl in token_lists for tok in tl)
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX, SEP: SEP_IDX}
    idx = len(vocab)
    for tok, cnt in counts.most_common():
        if cnt < min_freq:
            break
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1
    return vocab


def _encode(tokens, vocab, max_len):
    # Map tokens to indices and right-pad / truncate to max_len
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    length = len(ids)
    ids += [PAD_IDX] * (max_len - length)
    return ids, length


def _make_split(df, seed=SEED):
    # Stratified 70/15/15 train/val/test split; returns an array of labels.
    labels = df["LLM_name"].values
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30,
                                           stratify=labels, random_state=seed)
    rest_labels = labels[rest_idx]
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50,
                                         stratify=rest_labels, random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def process(df, out_dir=PROCESSED_DIR, tokenizer="simple",
            max_input_len=MAX_INPUT_LEN, max_output_len=MAX_OUTPUT_LEN,
            min_freq=2, modes=("input", "output", "both"), split=None,
            output_transform=None, seed=SEED):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- label map ----
    families = sorted(df["LLM_name"].unique())
    label_map = {fam: i for i, fam in enumerate(families)}
    y = np.array([label_map[f] for f in df["LLM_name"]])

    # ---- tokenise ----
    inp_tokens = df["LLM_input"].apply(tokenize).tolist()
    out_text = df["LLM_output"]
    if output_transform is not None:
        out_text = out_text.apply(output_transform)
    out_tokens = out_text.apply(tokenize).tolist()

    # ---- vocab (built on training rows only) ----
    if split is None:
        split = _make_split(df, seed)
    train_mask = (split == "train")
    train_tokens = ([inp_tokens[i] for i in range(len(df)) if train_mask[i]] +
                    [out_tokens[i] for i in range(len(df)) if train_mask[i]])
    vocab = _build_vocab(train_tokens, min_freq)
    vocab_size = len(vocab)

    # ---- encode per mode ----
    for mode in modes:
        rows_X, rows_len = [], []
        for i in range(len(df)):
            if mode == "input":
                ids, length = _encode(inp_tokens[i], vocab, max_input_len)
            elif mode == "output":
                ids, length = _encode(out_tokens[i], vocab, max_output_len)
            else:  # both
                inp_ids = [vocab.get(t, UNK_IDX) for t in inp_tokens[i][:max_input_len]]
                out_ids = [vocab.get(t, UNK_IDX) for t in out_tokens[i][:max_output_len]]
                combined = inp_ids + [SEP_IDX] + out_ids
                max_total = max_input_len + 1 + max_output_len
                length = len(combined)
                combined = combined[:max_total]
                length = min(length, max_total)
                combined += [PAD_IDX] * (max_total - len(combined))
                ids = combined
            rows_X.append(ids)
            rows_len.append(length)

        X = np.array(rows_X, dtype=np.int32)
        lengths = np.array(rows_len, dtype=np.int32)

        np.savez_compressed(
            out_dir / f"{mode}.npz",
            X=X, lengths=lengths, y=y, split=split,
            LLM_name=df["LLM_name"].values,
            task_category=df["task_category"].values,
            label_names=np.array(families),
            vocab_size=np.array(vocab_size),
        )
        print(f"Saved {out_dir / f'{mode}.npz'}  "
              f"(X shape {X.shape}, vocab {vocab_size:,})")

    # ---- save config + vocab ----
    cfg = {"tokenizer": tokenizer, "max_input_len": max_input_len,
           "max_output_len": max_output_len, "min_freq": min_freq,
           "vocab_size": vocab_size, "n_rows": len(df), "seed": seed,
           "label_map": label_map}
    (out_dir / "processing_config.json").write_text(json.dumps(cfg, indent=2))

    # Save vocab as JSON for possible inspection / reuse
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=1))

    return vocab, label_map


def load_processed(mode: str, processed_dir=PROCESSED_DIR) -> dict:
    path = Path(processed_dir) / f"{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.  Run `python data_processing.py` first.")
    d = np.load(path, allow_pickle=True)
    families = list(d["label_names"])
    label_map = {fam: i for i, fam in enumerate(families)}
    meta = pd.DataFrame({
        "split": d["split"],
        "LLM_name": d["LLM_name"],
        "task_category": d["task_category"],
    })
    return {
        "X": d["X"],
        "lengths": d["lengths"],
        "y": d["y"].astype(np.int64),
        "meta": meta,
        "label_map": label_map,
        "vocab_size": int(d["vocab_size"]),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-csv", default=str(CLEAN_CSV))
    p.add_argument("--out-dir", default=str(PROCESSED_DIR))
    p.add_argument("--modes", nargs="+", default=["input", "output", "both"],
                   choices=["input", "output", "both"])
    p.add_argument("--min-freq", type=int, default=2)
    args = p.parse_args()

    df = load_clean(args.clean_csv)
    print(f"Loaded {len(df):,} rows from {args.clean_csv}")
    process(df, args.out_dir, min_freq=args.min_freq, modes=tuple(args.modes))


if __name__ == "__main__":
    main()