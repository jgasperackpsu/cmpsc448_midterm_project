"""
Phase 1 - Data Cleaning
=======================
Turns the raw LMSYS-Chat-1M dump (downloaded by load_data.py) into a balanced,
de-identified dataset of (LLM_name, LLM_input, LLM_output) observations.

Steps
-----
1. Load the raw Arrow shards from ./local_datasets/lmsys_chat_1m/train
2. Keep only English conversations from the model versions in FAMILY_MAP
3. Take the first user prompt and the first assistant reply of each conversation
4. Quality filters: non-empty, not flagged by OpenAI moderation, de-duplicated
5. Random-sample N prompt/response pairs per family (default 4,000)
6. Strip self-identification (e.g. "I am Claude, an AI assistant made by Anthropic")
   and mask any remaining model / vendor names so the classifier has to rely on style
7. Tag each prompt with a heuristic task category (used for RQ3)

Usage
-----
    python data_cleaning.py                         # all 6 families, 4000 each
    python data_cleaning.py --n-per-family 3000 --families GPT Claude LLaMA-2 Vicuna
"""
import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLEAN_CSV, CLEAN_DIR, FAMILY_MAP, RAW_DIR, SEED

# --------------------------------------------------------------------------- #
# Self-identification stripping
# --------------------------------------------------------------------------- #
# Names of models / vendors. A sentence that contains one of these AND a
# self-reference cue is treated as self-identification and removed entirely.
IDENTITY_TERMS = [
    r"chat\s?gpt", r"gpt[\s-]?4", r"gpt[\s-]?3(\.5)?(-turbo)?", r"\bgpt\b", r"open\s?ai",
    r"claude", r"anthropic",
    r"llama[\s-]?2?", r"meta\s?ai", r"\bmeta\b", r"facebook",
    r"vicuna", r"lmsys", r"large model systems organi[sz]ation",
    r"palm[\s-]?2?", r"\bbard\b", r"google", r"deepmind",
    r"\bmpt\b", r"mpt-\d+b", r"mosaic\s?ml", r"databricks",
]
IDENTITY_RE = re.compile("|".join(IDENTITY_TERMS), re.IGNORECASE)

# Cues that the sentence is the model talking about itself.
SELF_CUE_RE = re.compile(
    r"\b(i\s*am|i'm|i’m|my name|call me|me\b|myself|this model|"
    r"(an?|the) (ai|artificial intelligence|language model|large language model|chatbot|assistant)|"
    r"(developed|created|trained|built|made|designed|programmed|fine-tuned|finetuned|released) by)\b",
    re.IGNORECASE,
)

# Names that are unambiguous model/vendor identifiers.
MASK_TERMS = [
    r"chat\s?gpt", r"gpt[\s-]?4", r"gpt[\s-]?3(\.5)?(-turbo)?", r"open\s?ai",
    r"claude(?:[\s-]instant)?(?:[\s-]?v?\d(?:\.\d)?)?", r"anthropic",
    r"llama[\s-]?2", r"vicuna", r"lmsys", r"large model systems organi[sz]ation",
    r"palm[\s-]?2", r"\bbard\b", r"\bmpt(-\d+b)?(-chat)?\b", r"mosaic\s?ml",
]
MASK_RE = re.compile(r"\b(?:" + "|".join(MASK_TERMS) + r")\b", re.IGNORECASE)
MASK_TOKEN = "[MODEL]"

# Sentence splitter that keeps newlines (markdown structure matters for RQ4)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+|(?=\n)")


def strip_self_identification(text: str) -> tuple[str, int]:
    """Remove sentences in which the model identifies itself.

    Returns the cleaned text and the number of sentences removed.
    """
    if not isinstance(text, str) or not text:
        return "", 0
    pieces = _SENT_SPLIT_RE.split(text)
    kept, removed = [], 0
    for piece in pieces:
        if IDENTITY_RE.search(piece) and SELF_CUE_RE.search(piece):
            removed += 1
            # keep the newline that started this piece so formatting survives
            if piece.startswith("\n"):
                kept.append("\n")
            continue
        kept.append(piece)
    # re-join: pieces split on spaces lost the space, pieces split on \n kept it
    out = ""
    for piece in kept:
        if out and not piece.startswith("\n") and not out.endswith("\n"):
            out += " "
        out += piece
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, removed


def mask_model_names(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return MASK_RE.sub(MASK_TOKEN, text)


# --------------------------------------------------------------------------- #
# Heuristic task categories (RQ3)
# --------------------------------------------------------------------------- #
_CODE_RE = re.compile(
    r"```|\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|regex|"
    r"bash|shell script|code|function|script|program|compile|debug|bug|stack ?trace|"
    r"api|json|class|method|variable|algorithm|leetcode|pandas|numpy|react|django)\b",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"\b(solve|calculate|compute|equation|integral|derivative|probability|algebra|"
    r"geometry|math|arithmetic|percent|sum of|how many|x\s*=)\b|\d+\s*[\+\-\*/\^=]\s*\d+",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"\b(story|poem|poetry|haiku|lyrics|song|essay|fiction|novel|role-?play|pretend|"
    r"character|screenplay|script for a|joke|limerick|rap|creative|imagine)\b",
    re.IGNORECASE,
)


def categorize_prompt(prompt: str) -> str:
    """Assign one coarse task category to a prompt (first match wins)."""
    if not isinstance(prompt, str):
        return "general"
    if _CODE_RE.search(prompt):
        return "coding"
    if _MATH_RE.search(prompt):
        return "math"
    if _CREATIVE_RE.search(prompt):
        return "creative"
    return "general"


# --------------------------------------------------------------------------- #
# Loading and filtering
# --------------------------------------------------------------------------- #
def load_raw(raw_dir=RAW_DIR):
    """Load all Arrow shards regardless of their file names using pyarrow IPC."""
    import pyarrow.ipc as ipc

    # skip cache-*.arrow files that datasets writes next to the shards after .filter()
    raw_path = Path(raw_dir)
    train_dir = raw_path / "train" if (raw_path / "train").exists() else raw_path
    shards = sorted(p for p in glob(str(train_dir / "*.arrow"))
                    if not Path(p).name.startswith("cache-"))
    if not shards:
        raise FileNotFoundError(
            f"No .arrow files found in {train_dir}. Run load_data.py first."
        )
    print(f"Loading {len(shards)} Arrow shard(s) from {train_dir}")
    dataframes = []
    for p in shards:
        with open(p, "rb") as f:
            reader = ipc.RecordBatchStreamReader(f)
            dataframes.append(reader.read_all().to_pandas())
    return pd.concat(dataframes, ignore_index=True)


def first_turn(conversation):
    """Return (prompt, response) from the first user->assistant exchange."""
    prompt = response = None
    for msg in conversation:
        role, content = msg.get("role"), msg.get("content")
        if prompt is None and role == "user":
            prompt = content
        elif prompt is not None and role == "assistant":
            response = content
            break
    return prompt, response


def first_turn_flagged(moderation) -> bool:
    """True if either of the first two messages was flagged by OpenAI moderation."""
    if not moderation:
        return False
    return any(bool(m.get("flagged", False)) for m in moderation[:2] if isinstance(m, dict))


def build_clean_dataset(families, n_per_family, min_output_words=5, seed=SEED,
                        raw_dir=RAW_DIR):
    rng = np.random.default_rng(seed)
    version_to_family = {v: fam for fam in families for v in FAMILY_MAP[fam]}

    df = load_raw(raw_dir)
    print(f"Raw conversations: {len(df):,}")

    keep_models = set(version_to_family)
    mask = df["model"].isin(keep_models)
    if "language" in df.columns:
        mask = mask & (df["language"] == "English")
    df = df[mask].copy()
    print(f"English conversations from selected models: {len(df):,}")

    pairs = df["conversation"].apply(lambda c: first_turn(list(c) if c is not None else []))
    df["LLM_input"] = pairs.str[0]
    df["LLM_output"] = pairs.str[1]
    if "openai_moderation" in df:
        df["flagged"] = df["openai_moderation"].apply(
            lambda m: first_turn_flagged(list(m) if m is not None else []))
    else:
        df["flagged"] = False
    df["LLM_name"] = df["model"].map(version_to_family)

    # ---- quality filters ----
    before = len(df)
    df = df.dropna(subset=["LLM_input", "LLM_output"])
    df = df[~df["flagged"]]
    df = df[df["LLM_output"].str.split().str.len() >= min_output_words]
    # duplicate prompts are common in LMSYS (people re-test the same prompt)
    # keep one response per (family, prompt) so splits can't share near-copies
    df["_norm_prompt"] = df["LLM_input"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["LLM_name", "_norm_prompt"])
    print(f"After quality filters + de-duplication: {len(df):,} (dropped {before - len(df):,})")

    # ---- per-family sampling, stripping, masking ----
    out = []
    for fam in families:
        sub = df[df["LLM_name"] == fam]
        if sub.empty:
            print(f"  [!] {fam}: no rows found - skipping")
            continue
        # oversample a little because stripping may empty some responses
        take = min(len(sub), int(n_per_family * 1.5))
        sub = sub.iloc[rng.permutation(len(sub))[:take]].copy()

        stripped = sub["LLM_output"].apply(strip_self_identification)
        sub["LLM_output"] = stripped.str[0].apply(mask_model_names)
        sub["selfid_sentences_removed"] = stripped.str[1]
        sub["LLM_input"] = sub["LLM_input"].apply(mask_model_names)
        sub = sub[sub["LLM_output"].str.split().str.len() >= min_output_words]

        sub = sub.iloc[:n_per_family]
        if len(sub) < n_per_family:
            print(f"  [!] {fam}: only {len(sub):,} usable pairs (< {n_per_family:,})")
        out.append(sub)

    clean = pd.concat(out, ignore_index=True)
    clean["task_category"] = clean["LLM_input"].apply(categorize_prompt)
    clean = clean[["conversation_id", "LLM_name", "model", "task_category",
                   "LLM_input", "LLM_output", "selfid_sentences_removed"]]
    clean = clean.rename(columns={"model": "model_version"})
    return clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", nargs="+", default=list(FAMILY_MAP),
                   choices=list(FAMILY_MAP), help="LLM families to keep (4-6)")
    p.add_argument("--n-per-family", type=int, default=4000)
    p.add_argument("--min-output-words", type=int, default=5)
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--out", default=str(CLEAN_CSV))
    args = p.parse_args()

    clean = build_clean_dataset(args.families, args.n_per_family,
                                args.min_output_words, raw_dir=Path(args.raw_dir))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False)

    print("\nRows per family:")
    print(clean["LLM_name"].value_counts().to_string())
    print("\nResponses with self-identification removed, per family:")
    print((clean.groupby("LLM_name")["selfid_sentences_removed"].apply(lambda s: (s > 0).mean())
           .map("{:.1%}".format)).to_string())
    print("\nTask categories:")
    print(pd.crosstab(clean["task_category"], clean["LLM_name"]).to_string())
    print(f"\nSaved {len(clean):,} rows to {args.out}")


if __name__ == "__main__":
    main()
