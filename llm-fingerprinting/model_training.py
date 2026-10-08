"""
Phase 2 - Training, evaluation and grid search
==============================================
* Loss: nn.CrossEntropyLoss()     * Optimizer: AdamW
* Lightweight grid search over learning rate, batch size, and
  kernel sizes (CNN) / hidden dimension (RNN)
* Per-epoch validation tracking of accuracy, macro-F1 and the confusion matrix
* Early stopping on validation macro-F1; the best epoch's weights are kept

Usage (single experiment; research_questions.py drives the full set)
-----
    python model_training.py --model cnn --mode output --grid small
    python model_training.py --model rnn --mode both --grid tiny --epochs 3
"""
import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from data_processing import load_processed
from models import build_model, count_parameters

# --------------------------------------------------------------------------- #
# Hyper-parameter grids
# --------------------------------------------------------------------------- #
GRIDS = {
    "cnn": {
        "tiny":  {"lr": [1e-3], "batch_size": [64], "kernel_sizes": [(3, 4, 5)]},
        "small": {"lr": [1e-3, 3e-4], "batch_size": [32, 64],
                  "kernel_sizes": [(3, 4, 5), (2, 3, 4, 5)]},
        "full":  {"lr": [3e-3, 1e-3, 3e-4], "batch_size": [32, 64, 128],
                  "kernel_sizes": [(2, 3, 4), (3, 4, 5), (3, 5, 7), (2, 3, 4, 5)]},
    },
    "rnn": {
        "tiny":  {"lr": [1e-3], "batch_size": [64], "hidden_dim": [128]},
        "small": {"lr": [1e-3, 3e-4], "batch_size": [32, 64], "hidden_dim": [128, 256]},
        "full":  {"lr": [3e-3, 1e-3, 3e-4], "batch_size": [32, 64, 128],
                  "hidden_dim": [128, 256]},
    },
}
# Held fixed across the grid (shared embedding config for both models)
FIXED_HP = {"embed_dim": 128, "dropout": 0.5, "num_filters": 100, "num_layers": 1,
            "weight_decay": 0.01}


def expand_grid(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_arrays(data, split, row_mask=None):
    """Select (X, lengths, y) for one split, optionally further restricted by a mask."""
    mask = (data["meta"]["split"] == split).to_numpy()
    if row_mask is not None:
        mask = mask & np.asarray(row_mask, dtype=bool)
    return data["X"][mask], data["lengths"][mask], data["y"][mask]


def make_loader(X, lengths, y, batch_size, shuffle, seed=SEED):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.long),
                       torch.as_tensor(lengths, dtype=torch.long),
                       torch.as_tensor(y, dtype=torch.long))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def _trim(x, lengths):
    """Sequences are right-padded, so cut each batch to its longest sequence."""
    return x[:, : int(lengths.max())]


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss, n, preds, trues, probs = 0.0, 0, [], [], []
    for x, lengths, y in loader:
        x = _trim(x, lengths).to(device)
        y = y.to(device)
        logits = model(x, lengths)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    return {
        "loss": total_loss / max(n, 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(num_classes)), zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
        "y_true": y_true, "y_pred": y_pred, "probs": np.concatenate(probs),
    }


def train_model(model_type, train_arrays, val_arrays, hp, vocab_size, num_classes,
                epochs=10, patience=3, device=None, verbose=True, seed=SEED):
    """Train one configuration. Returns dict(model, history, best_epoch, best_val, hp)."""
    device = device or get_device()
    set_seed(seed)
    hp = {**FIXED_HP, **hp}
    model = build_model(model_type, vocab_size, num_classes, **hp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                                  weight_decay=hp["weight_decay"])
    train_loader = make_loader(*train_arrays, hp["batch_size"], shuffle=True, seed=seed)
    val_loader = make_loader(*val_arrays, 256, shuffle=False)

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n, correct = 0.0, 0, 0
        for x, lengths, y in train_loader:
            x, y = _trim(x, lengths).to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        val = evaluate(model, val_loader, device, num_classes, criterion)
        row = {"epoch": epoch, "train_loss": tot / n, "train_acc": correct / n,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "val_cm": val["cm"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if verbose:
            print(f"    epoch {epoch:2d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
                  f"| val loss {val['loss']:.4f} acc {val['acc']:.3f} F1 {val['macro_f1']:.3f} "
                  f"| {row['seconds']}s")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stopping (no val F1 gain for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    best_row = history[best_epoch - 1]
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_val": {"acc": best_row["val_acc"], "macro_f1": best_row["val_macro_f1"],
                         "loss": best_row["val_loss"]},
            "hp": hp, "n_params": count_parameters(model)}


def grid_search(model_type, train_arrays, val_arrays, vocab_size, num_classes,
                grid="small", epochs=10, patience=3, device=None, verbose=True):
    """Train every configuration in the grid; return (best_run, results DataFrame)."""
    grid_dict = GRIDS[model_type][grid] if isinstance(grid, str) else grid
    configs = list(expand_grid(grid_dict))
    best, rows = None, []
    for i, hp in enumerate(configs, 1):
        if verbose:
            print(f"  [{model_type.upper()} {i}/{len(configs)}] {hp}")
        run = train_model(model_type, train_arrays, val_arrays, hp, vocab_size,
                          num_classes, epochs, patience, device, verbose)
        rows.append({**{k: str(v) if isinstance(v, tuple) else v for k, v in hp.items()},
                     "best_epoch": run["best_epoch"], "val_acc": run["best_val"]["acc"],
                     "val_macro_f1": run["best_val"]["macro_f1"],
                     "val_loss": run["best_val"]["loss"], "n_params": run["n_params"]})
        if best is None or run["best_val"]["macro_f1"] > best["best_val"]["macro_f1"]:
            if best is not None:
                del best["model"]
            best = run
        else:
            del run["model"]
    results = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    return best, results


def save_checkpoint(run, path, model_type, mode, label_map, vocab_size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hp = {k: list(v) if isinstance(v, tuple) else v for k, v in run["hp"].items()}
    torch.save({"state_dict": run["model"].state_dict(), "model_type": model_type,
                "mode": mode, "hp": hp, "label_map": label_map, "vocab_size": vocab_size,
                "best_epoch": run["best_epoch"]}, path)


def load_checkpoint(path, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    model = build_model(ckpt["model_type"], ckpt["vocab_size"], len(ckpt["label_map"]),
                        **ckpt["hp"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cnn", "rnn"], required=True)
    p.add_argument("--mode", choices=["input", "output", "both"], default="output")
    p.add_argument("--grid", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = p.parse_args()

    data = load_processed(args.mode, args.processed_dir)
    K = len(data["label_map"])
    device = get_device()
    print(f"Device: {device} | classes: {K} | vocab: {data['vocab_size']:,}")
    best, results = grid_search(args.model, split_arrays(data, "train"), split_arrays(data, "val"),
                                data["vocab_size"], K, args.grid, args.epochs, args.patience, device)
    test = evaluate(best["model"], make_loader(*split_arrays(data, "test"), 256, False), device, K)

    out = RESULTS_DIR / "single_runs" / f"{args.model}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "grid_results.csv", index=False)
    save_checkpoint(best, MODELS_DIR / f"{args.model}_{args.mode}.pt", args.model, args.mode,
                    data["label_map"], data["vocab_size"])
    with open(out / "test_metrics.json", "w") as f:
        json.dump({"acc": test["acc"], "macro_f1": test["macro_f1"], "cm": test["cm"],
                   "hp": {k: str(v) for k, v in best["hp"].items()}}, f, indent=2)
    print(f"\nBest config: {best['hp']}\nTest acc {test['acc']:.4f} | macro-F1 {test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
