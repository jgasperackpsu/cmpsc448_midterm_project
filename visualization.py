"""Figures for the report (matplotlib, saved as PNG)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import RESULTS_DIR

# Validated categorical slots (CVD-safe order) + text / grid tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MODEL_COLORS = {"CNN": SERIES[0], "RNN": SERIES[1]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 9.5,
    "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Dataset overview
# --------------------------------------------------------------------------- #
def plot_dataset_overview(meta_csv, out_dir=RESULTS_DIR / "eda"):
    meta = pd.read_csv(meta_csv)
    out_dir = Path(out_dir)
    fams = list(dict.fromkeys(meta.sort_values("label")["LLM_name"]))

    # 1) rows per family and split
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.crosstab(meta["LLM_name"], meta["split"]).reindex(fams)[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(fams))
    w = 0.27
    for i, (split, color) in enumerate(zip(["train", "val", "test"], SERIES)):
        bars = ax.bar(x + (i - 1) * w, counts[split], w - 0.03, color=color, label=split)
        ax.bar_label(bars, fontsize=7, color=TEXT_2, padding=1)
    ax.set_xticks(x, fams)
    ax.set_ylabel("Prompt/response pairs")
    ax.set_title("Observations per LLM family and split", pad=22)
    ax.set_ylim(0, counts.values.max() * 1.12)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    _save(fig, out_dir / "split_counts.png")

    # 2) response length distribution per family
    fig, ax = plt.subplots(figsize=(7, 3.4))
    data = [meta.loc[meta["LLM_name"] == f, "output_tokens"].clip(lower=1) for f in fams]
    ax.boxplot(data, tick_labels=fams, showfliers=False, widths=0.5,
               medianprops={"color": SERIES[1], "linewidth": 2},
               boxprops={"color": SERIES[0]}, whiskerprops={"color": SERIES[0]},
               capprops={"color": SERIES[0]})
    ax.set_yscale("log")
    ax.set_ylabel("Response length (tokens, log scale)")
    ax.set_title("Response length by LLM family")
    _save(fig, out_dir / "response_length_by_family.png")

    # 3) task category x family
    if "task_category" in meta:
        ct = pd.crosstab(meta["task_category"], meta["LLM_name"]).reindex(columns=fams)
        _heatmap(ct.values, ct.index.tolist(), fams, "Task category by LLM family (all splits)",
                 out_dir / "task_category_by_family.png", fmt="d")
        ct.to_csv(out_dir / "task_category_by_family.csv")
    counts.to_csv(out_dir / "split_counts.csv")


def _heatmap(values, row_labels, col_labels, title, path, fmt=".2f", vmax=None,
             xlabel=None, ylabel=None):
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(col_labels), 0.8 + 0.6 * len(row_labels)))
    im = ax.imshow(values, cmap=BLUE_RAMP, vmin=0, vmax=vmax or values.max() or 1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    thresh = (vmax or values.max() or 1) * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                    color="white" if v > thresh else TEXT)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Training diagnostics
# --------------------------------------------------------------------------- #
def plot_learning_curves(history, title, path):
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(h["epoch"], h["train_loss"], color=SERIES[0], marker="o", ms=4, label="train")
    a1.plot(h["epoch"], h["val_loss"], color=SERIES[1], marker="o", ms=4, label="validation")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-entropy loss"); a1.set_title("Loss")
    a1.legend()
    a2.plot(h["epoch"], h["val_acc"], color=SERIES[0], marker="o", ms=4, label="val accuracy")
    a2.plot(h["epoch"], h["val_macro_f1"], color=SERIES[1], marker="o", ms=4, label="val macro-F1")
    best = h["val_macro_f1"].idxmax()
    a2.axvline(h.loc[best, "epoch"], color=TEXT_2, lw=1, ls="--")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.set_title("Validation metrics")
    a2.set_ylim(max(0.0, min(h["val_acc"].min(), h["val_macro_f1"].min()) - 0.1), 1.02)
    a2.legend(loc="lower right")
    for a in (a1, a2):
        a.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=TEXT, y=1.03)
    return _save(fig, path)


def plot_confusion_matrix(cm, labels, title, path, normalize=True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    return _heatmap(cm, labels, labels, title, path, fmt=".2f" if normalize else ".0f",
                    vmax=1.0 if normalize else None, xlabel="Predicted family",
                    ylabel="True family")


# --------------------------------------------------------------------------- #
# Experiment comparisons
# --------------------------------------------------------------------------- #
def plot_grouped_metric(df, group_col, groups, title, path, metrics=("acc", "macro_f1"),
                        chance=None, xlabel=None):
    """Grouped bars: x = `groups` (e.g. modes), one bar per model, one panel per metric.

    df needs columns: model, <group_col>, metric, metric_lo, metric_hi (CI optional).
    """
    names = {"acc": "Test accuracy", "macro_f1": "Test macro-F1"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    models = [m for m in ("CNN", "RNN") if m in set(df["model"])]
    x = np.arange(len(groups))
    w = 0.38
    for ax, metric in zip(axes, metrics):
        for i, m in enumerate(models):
            sub = df[df["model"] == m].set_index(group_col).reindex(groups)
            vals = sub[metric].values
            err = None
            if f"{metric}_lo" in sub:
                err = np.vstack([vals - sub[f"{metric}_lo"].values, sub[f"{metric}_hi"].values - vals])
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w - 0.04,
                          color=MODEL_COLORS[m], label=m, yerr=err,
                          error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})
            ax.bar_label(bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals],
                         fontsize=8, color=TEXT, padding=3)
        if chance is not None:
            ax.axhline(chance, color=TEXT_2, ls="--", lw=1, label=f"chance ({chance:.2f})")
        ax.set_xticks(x, [str(g).replace(" (", "\n(") for g in groups])
        ax.set_ylim(0, 1.1)
        ax.set_title(names.get(metric, metric))
        if xlabel:
            ax.set_xlabel(xlabel)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(leg_labels)), key=lambda i: leg_labels[i].startswith("chance"))
    fig.legend([handles[i] for i in order], [leg_labels[i] for i in order], loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.12 if xlabel else -0.04))
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=TEXT, y=1.03)
    return _save(fig, path)


def plot_feature_distributions(features, families, path, title):
    """Small multiples (one panel per stylometric feature), family on the x-axis."""
    cols = [c for c in features.columns if c not in ("LLM_name",)]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow))
    axes = axes.ravel()
    for ax, c in zip(axes, cols):
        means = features.groupby("LLM_name")[c].mean().reindex(families)
        sems = features.groupby("LLM_name")[c].sem().reindex(families)
        bars = ax.bar(families, means, color=SERIES[0], yerr=1.96 * sems, width=0.6,
                      error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 2})
        ax.bar_label(bars, labels=[f"{v:.2f}" if v < 10 else f"{v:.0f}" for v in means],
                     fontsize=7, color=TEXT, padding=2)
        ax.set_title(c.replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    return _save(fig, path)
