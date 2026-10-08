"""Shared paths and constants for the LLM-fingerprinting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Directories -----------------------------------------------------------
RAW_DIR = ROOT / "local_datasets" / "lmsys_chat_1m"     # output of load_data.py (git-ignored)
TRAIN_DIR = RAW_DIR / "train"
CLEAN_DIR = ROOT / "data" / "cleaned"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLEAN_CSV = CLEAN_DIR / "lmsys_clean.csv"

# ---- Reproducibility --------------------------------------------------------
SEED = 42

# ---- LLM families -----------------------------------------------------------
# Model strings as they appear in the `model` column of LMSYS-Chat-1M, grouped
# into families. The classifier predicts the family (LLM_name), not the version.
FAMILY_MAP = {
    "GPT":     ["gpt-3.5-turbo", "gpt-4"],
    "Claude":  ["claude-1", "claude-2", "claude-instant-1"],
    "LLaMA-2": ["llama-2-7b-chat", "llama-2-13b-chat"],
    "Vicuna":  ["vicuna-7b", "vicuna-13b", "vicuna-33b"],
    "PaLM":    ["palm-2"],
    "MPT":     ["mpt-7b-chat", "mpt-30b-chat"],
}

# ---- Sequence lengths (tokens) ---------------------------------------------
MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 256

# ---- Special tokens ---------------------------------------------------------
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
PAD_IDX, UNK_IDX, SEP_IDX = 0, 1, 2

# The three sequence configurations used in RQ1/RQ2
MODES = ("input", "output", "both")
