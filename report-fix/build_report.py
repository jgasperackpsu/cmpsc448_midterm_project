"""
build_report.py - regenerate the EDA figures and write report/REPORT.md
=======================================================================
Reads ONLY the files already saved by data_cleaning.py, data_processing.py and
research_questions.py. Nothing is retrained, and no other project file is modified.

What it does
------------
1. Rebuilds the dataset-overview figures in results/eda/ (these used to be produced
   only by data_processing.py's command-line entry point, so they were easy to lose).
2. Runs the report-only analyses that need no training:
   per-class F1, top confusions, error analysis by response length / task category /
   refusal-style answers, a CNN+BiLSTM probability-averaging ensemble, and a
   chi-square test of task mix vs family.
3. Fills every {{placeholder}} in report/REPORT_TEMPLATE.md with numbers and
   data-driven text, and writes report/REPORT.md.
4. Prints a checklist of anything missing: input files, figures the report links to,
   and placeholders it could not fill.

Usage
-----
    python build_report.py --authors "Name 1, Name 2, Name 3" --hardware "Google Colab T4 GPU"

To change the wording permanently, edit report/REPORT_TEMPLATE.md (REPORT.md is
overwritten on every build), or make final hand edits to REPORT.md after the last build.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from visualization import GRID, SERIES, SURFACE, TEXT, TEXT_2, _heatmap, _save, plt

ROOT = Path(__file__).resolve().parent
MODELS = [("cnn", "CNN"), ("rnn", "BiLSTM")]
TOKEN_RE = re.compile(r"```|\*\*|__|#+|\w+|[^\w\s]")
REFUSAL_RE = re.compile(
    r"\b(?:i'?m sorry|i am sorry|i apologi[sz]e|i cannot|i can'?t|i'?m (?:not able|unable)|"
    r"i am (?:not able|unable)|as an ai)\b", re.IGNORECASE)

MISSING_INPUTS: list[str] = []
MISSING_FIGS: list[str] = []
NOTES: list[str] = []


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path):
    path = Path(path)
    if not path.exists():
        MISSING_INPUTS.append(_rel(path))
        return None
    return json.loads(path.read_text())


def read_csv(path, **kw):
    path = Path(path)
    if not path.exists():
        MISSING_INPUTS.append(_rel(path))
        return None
    return pd.read_csv(path, **kw)


def ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def f3(x):
    return f"{x:.3f}" if ok(x) else "n/a"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%" if ok(x) else "n/a"


def pts(x):
    if not ok(x):
        return "n/a"
    return "±0.0 pts" if abs(100 * x) < 0.05 else f"{100 * x:+.1f} pts"


def pval(p):
    if not ok(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def ci(e, k):
    if not e or k not in e:
        return "n/a"
    if f"{k}_lo" in e:
        return f"{e[k]:.3f} [{e[k + '_lo']:.3f}, {e[k + '_hi']:.3f}]"
    return f"{e[k]:.3f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def grouped_bars(categories, series, title, ylabel, path, chance=None, ylim=(0, 1.1)):
    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(categories) + 2), 3.5))
    x = np.arange(len(categories))
    n = len(series)
    w = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        vals = np.asarray(vals, dtype=float)
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w - 0.04, color=SERIES[i % len(SERIES)],
                      label=name)
        ax.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "" for v in vals],
                     fontsize=7.5, color=TEXT, padding=2)
    if chance is not None:
        ax.axhline(chance, color=TEXT_2, ls="--", lw=1, label=f"chance ({chance:.2f})")
    ax.set_xticks(x, categories)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, args):
        self.R = Path(args.results_dir)
        self.P = Path(args.processed_dir)
        self.report_dir = Path(args.report_dir)
        self.fig_dir = self.R / "report_figures"
        self.clean = read_csv(args.clean_csv)
        if self.clean is not None:
            for c in ("LLM_input", "LLM_output"):
                if c in self.clean:
                    self.clean[c] = self.clean[c].fillna("").astype(str)
        self.cfg = read_json(self.P / "processing_config.json") or {}
        label_map = read_json(self.P / "label_map.json")
        self.meta = read_csv(self.P / "meta.csv")
        if label_map:
            self.labels = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        elif self.meta is not None:
            self.labels = sorted(self.meta["LLM_name"].unique())
        elif self.clean is not None:
            self.labels = sorted(self.clean["LLM_name"].unique())
        else:
            raise SystemExit("No label_map.json, meta.csv or cleaned CSV found — nothing to report on.")
        self.K = len(self.labels)
        self.chance = 1 / self.K
        self._prepare_meta()

    def _prepare_meta(self):
        """Make sure meta has the columns the report needs, filling gaps from the cleaned CSV."""
        clean, meta = self.clean, self.meta
        if meta is None:
            if clean is None:
                return
            meta = clean[["LLM_name"]].copy()
            NOTES.append("data/processed/meta.csv not found — dataset figures use the cleaned CSV only.")
        self.aligned = (clean is not None and len(clean) == len(meta)
                        and (clean["LLM_name"].values == meta["LLM_name"].values).all())
        if not self.aligned and clean is not None:
            NOTES.append("meta.csv and the cleaned CSV are not row-aligned; analyses that need the "
                         "response text (refusals, code, examples) were skipped.")
        if "label" not in meta:
            meta["label"] = meta["LLM_name"].map({f: i for i, f in enumerate(self.labels)})
        if "task_category" not in meta and self.aligned and "task_category" in clean:
            meta["task_category"] = clean["task_category"].values
        for col, src in (("output_tokens", "LLM_output"), ("input_tokens", "LLM_input")):
            if col not in meta and self.aligned:
                meta[col] = clean[src].str.lower().str.findall(TOKEN_RE).str.len()
                NOTES.append(f"meta.csv had no '{col}' column; it was approximated with a regex tokenizer.")
        self.meta = meta

    # ---- results files ----
    def tm(self, rq, model, mode):
        return read_json(self.R / rq / f"{model}_{mode}" / "test_metrics.json")

    def hist(self, rq, model, mode):
        return read_json(self.R / rq / f"{model}_{mode}" / "history.json")

    def preds(self, rq, model, mode):
        p = self.R / rq / f"{model}_{mode}" / "test_predictions.npz"
        if not p.exists():
            MISSING_INPUTS.append(_rel(p))
            return None
        return dict(np.load(p))

    def test_eval(self, rq, model, mode):
        t = self.tm(rq, model, mode)
        return (t or {}).get("eval", {}).get("test")


# --------------------------------------------------------------------------- #
# Section 1 - dataset
# --------------------------------------------------------------------------- #
def build_eda(D):
    """Regenerate results/eda/*.png and *.csv."""
    meta, out = D.meta, D.R / "eda"
    if meta is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    fams = D.labels

    if "split" in meta:
        counts = pd.crosstab(meta["LLM_name"], meta["split"]).reindex(fams)
        counts = counts[[c for c in ("train", "val", "test") if c in counts]]
        counts.to_csv(out / "split_counts.csv")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(fams))
        w = 0.8 / counts.shape[1]
        for i, split in enumerate(counts.columns):
            bars = ax.bar(x + (i - (counts.shape[1] - 1) / 2) * w, counts[split], w - 0.03,
                          color=SERIES[i], label=split)
            ax.bar_label(bars, fontsize=7, color=TEXT_2, padding=1)
        ax.set_xticks(x, fams)
        ax.set_ylabel("Prompt/response pairs")
        ax.set_ylim(0, counts.values.max() * 1.12)
        ax.set_title("Observations per LLM family and split", pad=22)
        ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
        _save(fig, out / "split_counts.png")

    if "output_tokens" in meta:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        data = [meta.loc[meta["LLM_name"] == f, "output_tokens"].clip(lower=1) for f in fams]
        ax.boxplot(data, tick_labels=fams, showfliers=False, widths=0.5,
                   medianprops={"color": SERIES[1], "linewidth": 2},
                   boxprops={"color": SERIES[0]}, whiskerprops={"color": SERIES[0]},
                   capprops={"color": SERIES[0]})
        max_out = D.cfg.get("max_output_len")
        if max_out:
            ax.axhline(max_out, color=TEXT_2, ls="--", lw=1)
            ax.text(0.55, max_out * 1.06, f"truncation at {max_out} tokens", fontsize=7.5,
                    color=TEXT_2, va="bottom", ha="left")
        ax.set_yscale("log")
        ax.set_ylabel("Response length (tokens, log scale)")
        ax.set_title("Response length by LLM family")
        _save(fig, out / "response_length_by_family.png")

    if "task_category" in meta:
        ct = pd.crosstab(meta["task_category"], meta["LLM_name"]).reindex(columns=fams)
        ct.to_csv(out / "task_category_by_family.csv")
        _heatmap(ct.values, ct.index.tolist(), fams, "Task category by LLM family (all splits)",
                 out / "task_category_by_family.png", fmt="d")


def sec_dataset(D):
    d = {}
    meta, clean, fams = D.meta, D.clean, D.labels
    src = clean if clean is not None else meta
    rows, sid_rates = [], {}
    for f in fams:
        n = int((src["LLM_name"] == f).sum())
        sid = None
        if clean is not None and "selfid_sentences_removed" in clean:
            sid = clean.loc[clean["LLM_name"] == f, "selfid_sentences_removed"].gt(0).mean()
            sid_rates[f] = sid
        med = (meta.loc[meta["LLM_name"] == f, "output_tokens"].median()
               if meta is not None and "output_tokens" in meta else None)
        rows.append([f, f"{n:,}", pct(sid), f"{med:.0f}" if ok(med) else "n/a"])
    total_sid = (clean["selfid_sentences_removed"].gt(0).mean()
                 if clean is not None and "selfid_sentences_removed" in clean else None)
    total_med = meta["output_tokens"].median() if meta is not None and "output_tokens" in meta else None
    rows.append(["**All**", f"**{len(src):,}**", f"**{pct(total_sid)}**",
                 f"**{total_med:.0f}**" if ok(total_med) else "n/a"])
    d["dataset_table"] = md_table(["Family", "Pairs", "Responses with self-ID removed",
                                   "Median response length (tokens)"], rows)
    d["n_total"] = f"{len(src):,}"
    d["n_per_family"] = f"{int(src['LLM_name'].value_counts().median()):,}"

    if sid_rates:
        lo = min(sid_rates, key=sid_rates.get)
        hi = max(sid_rates, key=sid_rates.get)
        d["selfid_sentence"] = (
            f"Self-identification was found and removed in {pct(total_sid)} of responses overall, "
            f"from {pct(sid_rates[lo])} for {lo} to {pct(sid_rates[hi])} for {hi}. Without this step "
            f"a classifier could score points for {hi} simply by spotting its name.")
    else:
        d["selfid_sentence"] = ""

    if meta is not None and "split" in meta:
        s = meta["split"].value_counts()
        d["split_sizes"] = (f"{s.get('train', 0):,} / {s.get('val', 0):,} / {s.get('test', 0):,} "
                            f"(train / validation / test)")
        D.n_test = int(s.get("test", 0))
    else:
        d["split_sizes"] = "n/a"
        D.n_test = None
    d["n_test"] = f"{D.n_test:,}" if D.n_test else "n/a"

    # task mix + chi-square test of independence
    D.task_chi = None
    if meta is not None and "task_category" in meta:
        ct = pd.crosstab(meta["task_category"], meta["LLM_name"]).reindex(columns=fams).fillna(0)
        props = ct / ct.sum(axis=0)
        rows = [[c] + [f"{int(ct.loc[c, f]):,} ({pct(props.loc[c, f], 0)})" for f in fams]
                for c in ct.index]
        d["task_table"] = md_table(["Category"] + fams, rows)
        chi2, p, dof, _ = stats.chi2_contingency(ct.values)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        D.task_chi = (chi2, p, dof, v)
        top = props.idxmax(axis=1)
        biggest = (props.max(axis=1) - props.min(axis=1)).idxmax()
        d["task_sentence"] = (
            f"The task mix is {'not ' if p >= 0.05 else ''}significantly different across families "
            f"(chi-square test of independence: chi2({dof}) = {chi2:.1f}, {pval(p)}, Cramér's V = {v:.2f}). "
            f"The largest spread is in **{biggest}** prompts, which make up "
            f"{pct(props.loc[biggest].max(), 0)} of {top[biggest]}'s data but only "
            f"{pct(props.loc[biggest].min(), 0)} of {props.loc[biggest].idxmin()}'s. "
            f"Because users chose which model to talk to, the families did not receive the same prompts - "
            f"this matters for RQ2 and RQ3.")
    else:
        d["task_table"], d["task_sentence"] = "_Task categories not available._", ""

    # pre-processing statistics
    cfg = D.cfg

    def pctl(key, p):
        v = cfg.get(key, {})
        return v.get(str(p), v.get(str(float(p)), v.get(p)))

    out_p50, out_p95 = pctl("output_len_percentiles", 50), pctl("output_len_percentiles", 95)
    if out_p50 is None and meta is not None and "output_tokens" in meta:
        out_p50, out_p95 = meta["output_tokens"].median(), meta["output_tokens"].quantile(0.95)
    max_in, max_out = cfg.get("max_input_len", 128), cfg.get("max_output_len", 256)
    D.max_out = max_out
    trunc = cfg.get("pct_outputs_truncated")
    if trunc is None and meta is not None and "output_tokens" in meta:
        trunc = (meta["output_tokens"] > max_out).mean()
    tok = cfg.get("tokenizer", "nltk")
    d.update({
        "vocab_size": f"{cfg['vocab_size']:,}" if "vocab_size" in cfg else "n/a",
        "oov": pct(cfg.get("val_output_oov_rate"), 2),
        "min_freq": str(cfg.get("min_freq", 2)),
        "max_in": str(max_in), "max_out": str(max_out), "max_both": str(max_in + 1 + max_out),
        "out_p50": f"{out_p50:.0f}" if ok(out_p50) else "n/a",
        "out_p95": f"{out_p95:.0f}" if ok(out_p95) else "n/a",
        "out_trunc_pct": pct(trunc),
        "in_trunc_pct": pct(cfg.get("pct_inputs_truncated")),
        "tokenizer_desc": {"nltk": "Lower-cased NLTK `word_tokenize`",
                           "spacy": "Lower-cased spaCy rule-based tokenizer (`spacy.blank(\"en\")`)",
                           "regex": "Lower-cased regular-expression tokenizer"}.get(tok, tok),
    })
    D.vocab_size = cfg.get("vocab_size")
    return d


# --------------------------------------------------------------------------- #
# Section 2 - models and training
# --------------------------------------------------------------------------- #
def _hp_val(hp, key):
    v = hp.get(key)
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(str(x) for x in v) + ")"
    if key == "lr":
        return f"{float(v):g}"
    return str(v)


def sec_models(D, args):
    d = {"hardware": args.hardware, "max_epochs": str(args.epochs), "patience": str(args.patience)}
    for m, name in MODELS:
        t = D.tm("rq1", m, "output")
        d[f"{m}_params"] = f"{t['n_params']:,}" if t and "n_params" in t else "n/a"
        d[f"{m}_hp"] = (_hp_val(t["hp"], "kernel_sizes") if m == "cnn" else _hp_val(t["hp"], "hidden_dim")) if t else "n/a"
    d["embed_params"] = f"{D.vocab_size * 128:,}" if D.vocab_size else "n/a"

    # grid actually searched (from RQ1 grid_results.csv)
    grids = {m: read_csv(D.R / "rq1" / f"{m}_output" / "grid_results.csv") for m, _ in MODELS}

    def uniq(m, col):
        g = grids.get(m)
        if g is None or col not in g:
            return "-"
        vals = list(dict.fromkeys(g[col].tolist()))
        return ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)
    grid_rows = [
        ["Learning rate", uniq("cnn", "lr"), uniq("rnn", "lr")],
        ["Batch size", uniq("cnn", "batch_size"), uniq("rnn", "batch_size")],
        ["Kernel sizes", uniq("cnn", "kernel_sizes"), "-"],
        ["Hidden dimension", "-", uniq("rnn", "hidden_dim")],
        ["Configurations per experiment",
         str(len(grids["cnn"])) if grids["cnn"] is not None else "n/a",
         str(len(grids["rnn"])) if grids["rnn"] is not None else "n/a"],
    ]
    d["grid_table"] = md_table(["", "CNN", "BiLSTM"], grid_rows)

    rows = []
    for m, name in MODELS:
        for rq, mode, seq in (("rq1", "output", "response"), ("rq2", "input", "prompt"),
                              ("rq2", "both", "prompt + response")):
            t = D.tm(rq, m, mode)
            if not t:
                continue
            hp = t.get("hp", {})
            g = read_csv(D.R / rq / f"{m}_{mode}" / "grid_results.csv")
            rows.append([name, seq, len(g) if g is not None else "n/a", _hp_val(hp, "lr"),
                         _hp_val(hp, "batch_size"),
                         _hp_val(hp, "kernel_sizes" if m == "cnn" else "hidden_dim"),
                         t.get("best_epoch", "n/a"), f3(t.get("val_macro_f1")),
                         f"{t['n_params']:,}" if "n_params" in t else "n/a"])
    d["hp_table"] = md_table(["Model", "Input sequence", "Configs tried", "lr", "Batch",
                              "Kernels / hidden", "Best epoch", "Val macro-F1", "Parameters"], rows)
    return d


# --------------------------------------------------------------------------- #
# Section 3.1 - RQ1
# --------------------------------------------------------------------------- #
def top_confusions(cms, labels, k=3):
    """Average row-normalised confusion matrices across models; return top off-diagonal cells."""
    norm = []
    for cm in cms:
        cm = np.asarray(cm, float)
        norm.append(cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None))
    norm = np.stack(norm)
    mean = norm.mean(axis=0)
    off = [(mean[i, j], i, j) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    off.sort(reverse=True)
    return [(labels[i], labels[j], [n[i, j] for n in norm]) for _, i, j in off[:k]]


LINEAGE = {
    frozenset({"GPT", "Vicuna"}): "Vicuna was fine-tuned on user-shared ChatGPT conversations (ShareGPT), "
                                  "so it learned to imitate GPT's style",
    frozenset({"LLaMA-2", "Vicuna"}): "Vicuna is itself a fine-tuned LLaMA model, so the two share a base model",
}


def curve_sentence(name, hist, max_epochs=None):
    if not hist:
        return f"No training history was found for the {name}."
    h = pd.DataFrame(hist)
    n = len(h)
    b = int(h["val_macro_f1"].idxmax())
    be = int(h.loc[b, "epoch"])
    vi = int(h["val_loss"].idxmin())
    last = h.iloc[-1]
    s = (f"The {name} trained for {n} epoch{'s' if n != 1 else ''}; its best validation macro-F1 "
         f"({h.loc[b, 'val_macro_f1']:.3f}) came at epoch {be}")
    if max_epochs and n >= max_epochs:
        s += f" and training ran to the {max_epochs}-epoch limit."
    else:
        s += "." if be == n else ", after which early stopping ended training."
    rose = (vi < n - 1 and last["val_loss"] > h["val_loss"].min() * 1.05
            and last["train_loss"] < h.loc[vi, "train_loss"])
    if rose:
        s += (f" Validation loss bottomed out at epoch {int(h.loc[vi, 'epoch'])} ({h['val_loss'].min():.3f}) "
              f"and climbed to {last['val_loss']:.3f} while training loss kept falling to "
              f"{last['train_loss']:.3f} - the model began to over-fit, which is why the checkpoint from "
              f"epoch {be} is the one evaluated.")
    else:
        s += " Validation loss had not started rising by the end of training, so there is no clear sign of over-fitting."
    s += (f" Training accuracy in the last epoch was {last['train_acc']:.3f} versus "
          f"{h.loc[b, 'val_acc']:.3f} validation accuracy at the selected epoch.")
    if be == n and max_epochs and n >= max_epochs:
        s += " Because the best epoch was the last one, a few more epochs might still help."
    return s


def sec_rq1(D):
    d = {}
    ev = {m: D.test_eval("rq1", m, "output") for m, _ in MODELS}
    rows = [[name, ci(ev[m], "acc"), ci(ev[m], "macro_f1")] for m, name in MODELS]
    rows.append(["Chance (uniform guess)", f"{D.chance:.3f}", f"{D.chance:.3f}"])
    d["rq1_table"] = md_table(["Model", "Test accuracy (95% CI)", "Test macro-F1 (95% CI)"], rows)
    if not all(ev.values()):
        d["rq1_text"] = "_RQ1 results were not found; run `python research_questions.py --rq 1`._"
        d["per_class_table"] = d["curves_text"] = ""
        return d

    # per-class F1 table + figure
    pcf = pd.DataFrame({name: ev[m]["per_class_f1"] for m, name in MODELS}).reindex(D.labels)
    pcf["Mean"] = pcf.mean(axis=1)
    d["per_class_table"] = md_table(
        ["Family", "CNN F1", "BiLSTM F1", "Mean"],
        [[f] + [f"{pcf.loc[f, c]:.3f}" for c in pcf.columns] for f in pcf.index])
    grouped_bars(D.labels, {"CNN": pcf["CNN"].values, "BiLSTM": pcf["BiLSTM"].values},
                 "RQ1 — Test F1 per LLM family (response only)", "F1",
                 D.fig_dir / "rq1_per_class_f1.png")

    acc = {m: ev[m]["acc"] for m, _ in MODELS}
    lo = min(ev[m]["acc_lo"] for m, _ in MODELS) if "acc_lo" in ev["cnn"] else min(acc.values())
    level = ("well above" if lo > 1.5 * D.chance else "above" if lo > D.chance else "not reliably above")
    t = (f"Both models identify the LLM family {level} chance ({D.chance:.3f} for {D.K} balanced classes): "
         f"the CNN reaches {ev['cnn']['acc']:.3f} accuracy and {ev['cnn']['macro_f1']:.3f} macro-F1, the "
         f"BiLSTM {ev['rnn']['acc']:.3f} and {ev['rnn']['macro_f1']:.3f}. ")
    mc = read_json(D.R / "rq1" / "cnn_vs_rnn_mcnemar.json")
    diff = acc["cnn"] - acc["rnn"]
    win, lose = ("CNN", "BiLSTM") if diff > 0 else ("BiLSTM", "CNN")
    if mc:
        detail = (f"{mc['a_only_correct']} test responses were classified correctly only by the CNN and "
                  f"{mc['b_only_correct']} only by the BiLSTM")
        if mc["p_value"] < 0.05:
            t += (f"The {win} is significantly more accurate than the {lose} ({100 * abs(diff):.1f} points; "
                  f"exact McNemar test, {pval(mc['p_value'])}; {detail}). ")
        else:
            t += (f"The {100 * abs(diff):.1f}-point gap between the two models is not statistically significant "
                  f"(exact McNemar test, {pval(mc['p_value'])}; {detail}). ")
    D.rq1_mcnemar = mc

    easy, hard = pcf["Mean"].idxmax(), pcf["Mean"].idxmin()
    if pcf["Mean"].max() - pcf["Mean"].min() < 0.005:
        t += "\n\n**Per family.** All families are recognised equally well. "
    else:
        t += (f"\n\n**Per family.** {easy} is the easiest family to recognise (mean F1 {pcf.loc[easy, 'Mean']:.3f} "
              f"across the two models) and {hard} the hardest ({pcf.loc[hard, 'Mean']:.3f}). ")
        same_order = (pcf["CNN"].rank().values == pcf["BiLSTM"].rank().values).mean()
        t += ("Both models rank the families in the same order, so the difficulty comes from the data rather than the architecture. "
              if same_order == 1 else
              "The two models broadly agree on which families are hard. " if same_order >= 0.5 else
              "Interestingly, the two models disagree about which families are hard, which suggests they rely on different cues. ")

    conf = [c for c in top_confusions([ev[m]["cm"] for m, _ in MODELS], D.labels) if np.mean(c[2]) >= 0.005]
    parts = [f"{tr} → {pr} ({pct(r[0], 0)} of {tr} test responses for the CNN, {pct(r[1], 0)} for the BiLSTM)"
             for tr, pr, r in conf]
    t += (f"\n\n**Confusions.** The most frequent errors in the row-normalised confusion matrices are {join_and(parts)}."
          if parts else "\n\n**Confusions.** The confusion matrices are essentially diagonal; there is no systematic confusion.")
    notes = []
    for tr, pr, _ in conf:
        why = LINEAGE.get(frozenset({tr, pr}))
        if why and why not in notes:
            notes.append(why)
    if notes:
        t += " These pairs make sense: " + "; ".join(notes) + "."
    sym = [(a, b) for a, b, _ in conf if any(a2 == b and b2 == a for a2, b2, _ in conf)]
    if sym:
        t += f" The {sym[0][0]}/{sym[0][1]} confusion runs in both directions, i.e. the two styles genuinely overlap."
    d["rq1_text"] = t
    d["curves_text"] = "\n\n".join(curve_sentence(name, D.hist("rq1", m, "output"), D.max_epochs) for m, name in MODELS)
    D.rq1_ev = ev
    return d


# --------------------------------------------------------------------------- #
# Section 3.2 - RQ2
# --------------------------------------------------------------------------- #
def sec_rq2(D):
    d = {}
    modes = [("rq2", "input", "Prompt only"), ("rq1", "output", "Response only"),
             ("rq2", "both", "Prompt + response")]
    ev = {(m, mode): D.test_eval(rq, m, mode) for m, _ in MODELS for rq, mode, _ in modes}
    rows = []
    for m, name in MODELS:
        for metric, mname in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
            rows.append([f"{name} {mname}"] + [ci(ev[(m, mode)], metric) for _, mode, _ in modes])
    d["rq2_table"] = md_table(["", "Prompt only", "Response only", "Prompt + response"], rows)
    an = read_json(D.R / "rq2" / "analysis.json")
    if not an or not all(ev.values()):
        d["rq2_leak_text"] = d["rq2_help_text"] = "_RQ2 results incomplete; run `python research_questions.py --rq 1 2`._"
        d["rq2_conclusion"] = ""
        return d
    A = {m: an.get(name if name != "BiLSTM" else "RNN", an.get(name)) for m, name in MODELS}

    # ---- leak ----
    sig = {m: A[m]["input_only_vs_chance_p"] < 0.05 for m, _ in MODELS}
    pin = pd.DataFrame({name: ev[(m, "input")]["per_class_f1"] for m, name in MODELS}).reindex(D.labels)
    pin["Mean"] = pin.mean(axis=1)
    top2 = pin["Mean"].sort_values(ascending=False)
    accs = " and ".join(f"the {name} {A[m]['input_only_acc']:.3f} ({pval(A[m]['input_only_vs_chance_p'])})"
                        for m, name in MODELS)
    if all(sig.values()):
        t = (f"**Yes, prompts leak identity.** Trained on the prompt alone, both models beat chance "
             f"({D.chance:.3f}): {accs}, one-sided binomial test against chance. ")
    elif any(sig.values()):
        t = (f"**Weakly.** Trained on the prompt alone, only one model clearly beats chance ({D.chance:.3f}): "
             f"{accs}, one-sided binomial test against chance. ")
    else:
        t = (f"**No clear leak.** Trained on the prompt alone, neither model is distinguishable from chance "
             f"({D.chance:.3f}): {accs}, one-sided binomial test against chance. ")
    if any(sig.values()):
        t += (f"The families most predictable from their prompts are {top2.index[0]} (mean F1 {top2.iloc[0]:.3f}) "
              f"and {top2.index[1]} ({top2.iloc[1]:.3f}). ")
        if D.task_chi and D.task_chi[1] < 0.05:
            t += ("This lines up with the task mix in Section 1.5, which differs significantly across families: "
                  "users sent different kinds of questions to different models. ")
        else:
            t += ("The coarse task categories do not differ much across families, so the leak must come from "
                  "finer-grained differences in what users asked each model (topics, phrasing, or when each "
                  "model was available in the Arena). ")
        best_out = max(ev[(m, "output")]["acc"] for m, _ in MODELS)
        best_in = max(A[m]["input_only_acc"] for m, _ in MODELS)
        t += (f"Still, the leak is small next to the response signal: the best prompt-only accuracy "
              f"({best_in:.3f}) is {100 * (best_out - best_in):.0f} points below the best response-only "
              f"accuracy ({best_out:.3f}).")
    d["rq2_leak_text"] = t

    # ---- does the prompt help? ----
    parts = []
    any_sig_gain = False
    for m, name in MODELS:
        a = A[m]
        mc = a["mcnemar_output_vs_both"]
        da, df1 = a["both_minus_output_acc"], a["both_minus_output_macro_f1"]
        sig_m = mc["p_value"] < 0.05
        if sig_m and da > 0:
            verdict = "a significant improvement"
            any_sig_gain = True
        elif sig_m and da < 0:
            verdict = "a significant drop"
        elif da > 0:
            verdict = "a small gain that is not statistically significant"
        elif da < 0:
            verdict = "a small drop that is not statistically significant"
        else:
            verdict = "no change"
        parts.append(f"for the {name}, adding the prompt changes accuracy by {pts(da)} and macro-F1 by "
                     f"{pts(df1)} - {verdict} (exact McNemar test, {pval(mc['p_value'])}; "
                     f"{mc['b_only_correct']} test responses fixed, {mc['a_only_correct']} broken)")
    t = "Comparing prompt + response against response only on the same test rows: " + "; ".join(parts) + "."
    sig_drop = {m for m, _ in MODELS
                if A[m]["mcnemar_output_vs_both"]["p_value"] < 0.05 and A[m]["both_minus_output_acc"] < 0}
    if "rnn" in sig_drop:
        t += (" The BiLSTM drop has a likely architectural cause: its backward pass reads the sequence from the "
              "end, so its final backward state is formed right after reading the prompt, which dilutes the "
              "response information in the representation passed to the classifier.")
    if "cnn" in sig_drop:
        t += (" For the CNN, prompt n-grams compete with response n-grams in the max-pooled features, adding "
              "noise.")
    if any_sig_gain and any(sig.values()):
        t += (" Because the prompt alone carries some identity signal, at least part of this gain can be "
              "explained by prompt leakage rather than by the model reading the response better.")
    elif not any_sig_gain:
        t += (" Either way, the response already carries essentially all of the usable signal: the extra "
              "prompt tokens add input the classifier has to learn to ignore"
              + (", and can even hurt." if sig_drop else "."))
    d["rq2_help_text"] = t
    d["rq2_conclusion"] = (
        "**Answer to RQ2.** "
        + ("Prompts do leak model identity, " if all(sig.values()) else
           "Prompts leak at most a little identity, " if any(sig.values()) else
           "Prompts do not measurably leak identity, ")
        + ("and including them significantly raises the classification ceiling for at least one model."
           if any_sig_gain else
           "but including them does not significantly raise the classification ceiling: the fingerprint lives in the response."))
    D.rq2 = A
    return d


# --------------------------------------------------------------------------- #
# Section 4.1 / 4.2 - error analysis, ensemble, robustness
# --------------------------------------------------------------------------- #
def sec_errors(D):
    d = {"error_tables": "", "error_text": "_Error analysis needs RQ1 test predictions and meta.csv._",
         "ensemble_text": ""}
    meta = D.meta
    if meta is None or "split" not in meta:
        return d
    test_mask = (meta["split"] == "test").values
    mt = meta[test_mask].reset_index(drop=True)
    P = {m: D.preds("rq1", m, "output") for m, _ in MODELS}
    if not all(P.values()):
        return d
    for m, _ in MODELS:
        if len(P[m]["test_y_true"]) != len(mt) or (P[m]["test_y_true"] != mt["label"].values).any():
            NOTES.append("RQ1 test predictions do not line up with meta.csv test rows — error analysis skipped.")
            return d
    y = mt["label"].values
    right = {m: P[m]["test_y_pred"] == y for m, _ in MODELS}
    tables, text = [], []
    out_dir = D.R / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # by response length
    if "output_tokens" in mt:
        mo = D.max_out
        bins = [-1, 20, 50, 100, mo, np.inf]
        names = ["≤20", "21–50", "51–100", f"101–{mo}", f">{mo} (truncated)"]
        b = pd.cut(mt["output_tokens"], bins=bins, labels=names)
        g = pd.DataFrame({"bin": b, **{name: right[m] for m, name in MODELS}}).groupby("bin", observed=False)
        tab = g.agg(n=("CNN", "size"), CNN=("CNN", "mean"), BiLSTM=("BiLSTM", "mean"))
        tab = tab[tab["n"] > 0]
        tab.to_csv(out_dir / "accuracy_by_length.csv")
        tables.append("**Test accuracy by response length (tokens)**\n\n" + md_table(
            ["Response length", "n", "CNN", "BiLSTM"],
            [[i, f"{int(r.n):,}", f"{r.CNN:.3f}", f"{r.BiLSTM:.3f}"] for i, r in tab.iterrows()]))
        grouped_bars(list(tab.index), {"CNN": tab["CNN"].values, "BiLSTM": tab["BiLSTM"].values},
                     "Test accuracy by response length (RQ1 models)", "Accuracy",
                     D.fig_dir / "accuracy_by_length.png", chance=D.chance)
        mean = tab[["CNN", "BiLSTM"]].mean(axis=1)
        mn, mx = mean.idxmin(), mean.idxmax()
        phrase = lambda b: (f"responses longer than {mo} tokens" if str(b).startswith(">")
                            else f"responses of {b} tokens")
        if mean.max() - mean.min() < 0.01:
            s = "**Length.** Accuracy is about the same for every response length. "
        else:
            s = (f"**Length.** Accuracy varies with how much text there is. Both models do worst on {phrase(mn)} "
                 f"({tab.loc[mn, 'CNN']:.3f} CNN / {tab.loc[mn, 'BiLSTM']:.3f} BiLSTM, "
                 f"n = {int(tab.loc[mn, 'n']):,}) and best on {phrase(mx)} "
                 f"({tab.loc[mx, 'CNN']:.3f} / {tab.loc[mx, 'BiLSTM']:.3f}, n = {int(tab.loc[mx, 'n']):,}).")
        if mn == tab.index[0] and mean.max() - mean.min() >= 0.01:
            s += " Very short answers (\"Sure!\", one-line facts) simply contain too few tokens to carry a fingerprint."
        tr, prev = names[-1], names[-2]
        if tr in tab.index and prev in tab.index:
            dt = mean[tr] - mean[prev]
            s += (f" Responses longer than {mo} tokens, which the models only see the first {mo} tokens of, are "
                  + ("classified at least as well as" if dt >= -0.01 else "classified somewhat worse than")
                  + f" responses of {prev} tokens ({mean[tr]:.3f} vs {mean[prev]:.3f} averaged over both models), "
                  + ("so truncation is not costing much." if dt >= -0.01 else
                     "so a longer sequence limit may help."))
        text.append(s)

    # by task category
    if "task_category" in mt:
        g = pd.DataFrame({"cat": mt["task_category"], **{name: right[m] for m, name in MODELS}}).groupby("cat")
        tab = g.agg(n=("CNN", "size"), CNN=("CNN", "mean"), BiLSTM=("BiLSTM", "mean")).sort_values("n", ascending=False)
        tables.append("**Test accuracy by task category**\n\n" + md_table(
            ["Task category", "n", "CNN", "BiLSTM"],
            [[i, f"{int(r.n):,}", f"{r.CNN:.3f}", f"{r.BiLSTM:.3f}"] for i, r in tab.iterrows()]))
        mean = tab[["CNN", "BiLSTM"]].mean(axis=1)
        if mean.max() - mean.min() >= 0.01:
            text.append(f"**Task category.** {mean.idxmax().capitalize()} prompts give the most recognisable "
                        f"responses (mean accuracy {mean.max():.3f}) and {mean.idxmin()} prompts the least "
                        f"({mean.min():.3f}). RQ3 tests whether this gap persists when the models never see other domains in training.")

    # refusals / code (needs response text)
    if D.aligned:
        ct = D.clean[test_mask].reset_index(drop=True)
        refusal = ct["LLM_output"].str[:300].str.contains(REFUSAL_RE)
        code = ct["LLM_output"].str.contains("```", regex=False)
        rows = []
        for label, mask in (("Refusal / disclaimer opening", refusal), ("Contains a code block", code)):
            if mask.sum() >= 10:
                rows.append([label, f"{int(mask.sum()):,}"] + [f"{right[m][mask.values].mean():.3f}" for m, _ in MODELS]
                            + [f"{right[m][~mask.values].mean():.3f}" for m, _ in MODELS])
        if rows:
            tables.append("**Special response types**\n\n" + md_table(
                ["Subset", "n", "CNN (subset)", "BiLSTM (subset)", "CNN (rest)", "BiLSTM (rest)"], rows))
            if refusal.sum() >= 10:
                rr = np.mean([right[m][refusal.values].mean() for m, _ in MODELS])
                rest = np.mean([right[m][~refusal.values].mean() for m, _ in MODELS])
                fam_share = mt.loc[refusal.values, "LLM_name"].value_counts(normalize=True)
                text.append(
                    f"**Refusals and disclaimers.** {pct(refusal.mean())} of test responses open with an apology, "
                    f"refusal or \"As an AI\" disclaimer (most often from {fam_share.index[0]}, "
                    f"{pct(fam_share.iloc[0], 0)} of them). "
                    + (f"They are classified about as accurately as other responses ({rr:.3f} vs {rest:.3f})."
                       if abs(rr - rest) < 0.02 else
                       f"They are classified more accurately than other responses ({rr:.3f} vs {rest:.3f}), "
                       "suggesting each family has its own way of saying no." if rr > rest else
                       f"They are classified less accurately than other responses ({rr:.3f} vs {rest:.3f}), "
                       "suggesting the refusal boilerplate looks similar across families."))

        # misclassified examples for manual inspection
        both_wrong = ~right["cnn"] & ~right["rnn"]
        ex = pd.DataFrame({
            "true_family": mt["LLM_name"],
            "cnn_pred": [D.labels[i] for i in P["cnn"]["test_y_pred"]],
            "bilstm_pred": [D.labels[i] for i in P["rnn"]["test_y_pred"]],
            "task_category": mt.get("task_category"),
            "output_tokens": mt.get("output_tokens"),
            "response_start": ct["LLM_output"].str[:400],
        })[both_wrong]
        ex = ex.sample(min(50, len(ex)), random_state=42)
        ex.to_csv(out_dir / "misclassified_examples.csv", index=False)
        D.n_examples = len(ex)

    # agreement + ensemble
    both_wrong = ~right["cnn"] & ~right["rnn"]
    agree = (P["cnn"]["test_y_pred"] == P["rnn"]["test_y_pred"]).mean()
    oracle = (right["cnn"] | right["rnn"]).mean()
    best_single = max(right[m].mean() for m, _ in MODELS)
    s = (f"**Agreement.** The two models predict the same family for {pct(agree)} of test responses. Both are wrong on "
         f"{pct(both_wrong.mean())}; at least one is right on {pct(oracle)}.")
    if "test_probs" in P["cnn"] and "test_probs" in P["rnn"]:
        ens_pred = (P["cnn"]["test_probs"] + P["rnn"]["test_probs"]).argmax(1)
        ens_acc = (ens_pred == y).mean()
        ens_f1 = f1_score(y, ens_pred, average="macro")
        s += (f" Simply averaging the two models' predicted probabilities (no extra training or tuning) gives "
              f"{ens_acc:.3f} accuracy and {ens_f1:.3f} macro-F1, "
              + (f"{100 * (ens_acc - best_single):.1f} points better than the best single model - the CNN's "
                 f"local n-gram features and the BiLSTM's sequential features are partly complementary."
                 if ens_acc - best_single > 0.005 else
                 "no better than the best single model, so the two architectures mostly pick up the same cues."))
        json.dump({"ensemble_acc": float(ens_acc), "ensemble_macro_f1": float(ens_f1), "agreement": float(agree),
                   "both_wrong": float(both_wrong.mean()), "oracle": float(oracle)},
                  open(out_dir / "ensemble_and_agreement.json", "w"), indent=2)
    text.append(s)
    if D.aligned and getattr(D, "n_examples", 0):
        text.append(f"{D.n_examples} responses that both models got wrong are saved in "
                    "`results/error_analysis/misclassified_examples.csv` for manual reading.")
    d["error_tables"] = "\n\n".join(tables)
    d["error_text"] = "\n\n".join(text)
    return d


def sec_robustness(D):
    ev = getattr(D, "rq1_ev", None)
    if not ev or "acc_lo" not in ev["cnn"]:
        return {"robustness_text": "_Needs RQ1 results with bootstrap intervals._"}
    hw = np.mean([(ev[m]["acc_hi"] - ev[m]["acc_lo"]) / 2 for m, _ in MODELS])
    items = []
    mc = getattr(D, "rq1_mcnemar", None)
    if mc:
        items.append(f"CNN vs BiLSTM on responses (RQ1): {pval(mc['p_value'])} - "
                     + ("significant." if mc["p_value"] < 0.05 else "not significant."))
    for m, name in MODELS:
        a = getattr(D, "rq2", {}).get(m) if hasattr(D, "rq2") else None
        if a:
            items.append(f"{name}, prompt-only vs chance (RQ2): {pval(a['input_only_vs_chance_p'])} - "
                         + ("significant." if a["input_only_vs_chance_p"] < 0.05 else "not significant."))
            p = a["mcnemar_output_vs_both"]["p_value"]
            items.append(f"{name}, response vs prompt + response (RQ2): {pval(p)} - "
                         + ("significant." if p < 0.05 else "not significant."))
    t = (f"With {D.n_test:,} test responses, the 95% bootstrap intervals on accuracy are about "
         f"±{100 * hw:.1f} points wide, so differences of a point or two between configurations should not be "
         f"over-interpreted on their own. For the comparisons that matter we therefore use paired tests on the "
         f"same test rows (exact McNemar) rather than eyeballing overlapping intervals:\n\n"
         + "\n".join(f"- {i}" for i in items)
         + "\n\nAll results come from a single random seed (42). The bootstrap captures test-set sampling "
           "noise but not training randomness; repeating each run with 3-5 seeds would measure that too.")
    return {"robustness_text": t}


# --------------------------------------------------------------------------- #
# Section 4.3 - RQ3
# --------------------------------------------------------------------------- #
def sec_rq3(D):
    df = read_csv(D.R / "rq3" / "summary.csv")
    if df is None or df.empty:
        return {"rq3_table": "_RQ3 not run (`python research_questions.py --rq 3`)._", "rq3_text": "",
                "source_domain": "general"}
    df["model"] = df["model"].replace({"RNN": "BiLSTM"})
    domains = list(dict.fromkeys(df["test_domain"]))
    src_label = next((x for x in domains if x.startswith("in-domain")), domains[0])
    src = re.sub(r".*\((.*)\).*", r"\1", src_label)
    rows = []
    for dom in domains:
        sub = df[df["test_domain"] == dom].set_index("model")
        row = [dom, f"{int(sub['n_test'].iloc[0]):,}"]
        for _, name in MODELS:
            row.append(ci(sub.loc[name].to_dict(), "macro_f1") if name in sub.index else "n/a")
        for _, name in MODELS:
            row.append("-" if dom == src_label or name not in sub.index else pts(-sub.loc[name, "f1_drop_vs_in_domain"]))
        rows.append(row)
    d = {"source_domain": src, "rq3_table": md_table(
        ["Test domain", "n", "CNN macro-F1 (95% CI)", "BiLSTM macro-F1 (95% CI)",
         "CNN change vs in-domain", "BiLSTM change vs in-domain"], rows)}

    zs = df[df["test_domain"] != src_label]
    drops = zs.groupby("model")["f1_drop_vs_in_domain"].mean()
    parts = []
    for _, name in MODELS:
        z = zs[zs["model"] == name]
        if z.empty:
            continue
        worst = z.loc[z["f1_drop_vs_in_domain"].idxmax()]
        inn = df[(df["model"] == name) & (df["test_domain"] == src_label)].iloc[0]
        overlap = (z["macro_f1_hi"] >= inn["macro_f1_lo"]).sum() if "macro_f1_hi" in z else 0
        parts.append(f"the {name} goes from {inn['macro_f1']:.3f} in-domain to an average of "
                     f"{z['macro_f1'].mean():.3f} zero-shot (largest drop on {re.sub(r'.*[(](.*)[)].*', r'\1', worst['test_domain'])}, "
                     f"{pts(-worst['f1_drop_vs_in_domain'])}; {overlap} of {len(z)} zero-shot intervals overlap "
                     f"the in-domain interval)")
    m = drops.mean()
    if m < 0.03:
        verdict = ("Performance barely degrades, so the classifiers learned **model-wide fingerprints** — "
                   "habits of tone, hedging and phrasing that hold whatever the task.")
    elif m < 0.10:
        verdict = ("Performance degrades moderately: the fingerprints are **mostly model-wide**, with a "
                   "domain-specific component that does not transfer.")
    else:
        verdict = ("Performance degrades substantially, so a large part of what the classifiers learn is "
                   "**domain-specific** (how each family handles a particular kind of task) rather than a "
                   "universal style.")
    t = f"Trained only on **{src}** prompts, " + "; ".join(parts) + ". " + verdict
    t1, t3 = D.tm("rq1", "cnn", "output"), D.tm("rq3", "cnn", "output")
    if t1 and t3 and "n_train" in t1 and "n_train" in t3:
        t += (f"\n\nTwo caveats. First, the RQ3 models see only {t3['n_train']:,} training examples "
              f"({pct(t3['n_train'] / t1['n_train'], 0)} of RQ1's {t1['n_train']:,}), so their in-domain scores "
              f"are lower than RQ1's partly because of less data, not only because of the domain restriction. ")
    else:
        t += "\n\nTwo caveats. First, the RQ3 models are trained on a subset of the data. "
    t += ("Second, task categories come from keyword rules, so some prompts are mislabelled, and the smaller "
          "zero-shot test sets have wide confidence intervals.")
    d["rq3_text"] = t
    return d


# --------------------------------------------------------------------------- #
# Section 4.4 - RQ4
# --------------------------------------------------------------------------- #
def sec_rq4(D):
    d = {}
    fb = read_csv(D.R / "rq4" / "style_features_by_family.csv", index_col=0)
    kw = read_json(D.R / "rq4" / "kruskal_wallis.json")
    if fb is not None:
        fb = fb.reindex(D.labels)
        cols = [("response_words_median", "Median words", "{:.0f}"), ("mattr_50_mean", "MATTR-50", "{:.3f}"),
                ("format_density_per_100w_mean", "Markdown items / 100 words", "{:.2f}"),
                ("bullets_mean", "List items", "{:.2f}"), ("bold_mean", "Bold spans", "{:.2f}"),
                ("headers_mean", "Headers", "{:.2f}"), ("code_blocks_mean", "Code blocks", "{:.2f}")]
        cols = [c for c in cols if c[0] in fb]
        d["rq4_feature_table"] = md_table(["Family"] + [c[1] for c in cols],
                                          [[f] + [c[2].format(fb.loc[f, c[0]]) for c in cols] for f in fb.index])
        t = []
        if "response_words_median" in fb:
            s = fb["response_words_median"]
            t.append(f"{s.idxmax()} writes the longest answers (median {s.max():.0f} words) and "
                     f"{s.idxmin()} the shortest ({s.min():.0f}).")
        if "format_density_per_100w_mean" in fb:
            s = fb["format_density_per_100w_mean"]
            t.append(f"{s.idxmax()} uses the most markdown structure ({s.max():.2f} headers/list items/bold "
                     f"spans/code blocks per 100 words) and {s.idxmin()} the least ({s.min():.2f}).")
        if "mattr_50_mean" in fb:
            s = fb["mattr_50_mean"]
            t.append(f"Lexical diversity (MATTR-50, which unlike the raw type-token ratio does not fall "
                     f"automatically as answers get longer) is highest for {s.idxmax()} ({s.max():.3f}) and "
                     f"lowest for {s.idxmin()} ({s.min():.3f}).")
        if kw:
            sig = [k for k, v in kw.items() if v["p_value"] < 0.001]
            non = [k.replace("_", " ") for k, v in kw.items() if v["p_value"] >= 0.05]
            t.append(f"Kruskal-Wallis tests find a difference between families (p < 0.001) for {len(sig)} of "
                     f"{len(kw)} features" + (f"; no significant difference for {join_and(non)}." if non else "."))
        d["rq4_feature_text"] = " ".join(t)
    else:
        d["rq4_feature_table"] = "_RQ4 not run (`python research_questions.py --rq 4`)._"
        d["rq4_feature_text"] = ""

    lr = read_json(D.R / "rq4" / "feature_only_logreg.json")
    best = max((D.rq1_ev[m]["acc"] for m, _ in MODELS), default=None) if getattr(D, "rq1_ev", None) else None
    if lr:
        d["rq4_logreg_text"] = (
            f"A logistic regression on these eight features alone reaches {ci(lr, 'acc')} accuracy and "
            f"{ci(lr, 'macro_f1')} macro-F1, against {D.chance:.3f} for chance"
            + (f" and {best:.3f} for the best neural model. " if best else ". ")
            + ("Length and formatting habits therefore carry real signal on their own, but most of what the "
               "neural models exploit is in the words themselves."
               if best and lr["acc"] < best - 0.05 else
               "Surface shape alone gets close to the neural models, a warning that much of the signal may be "
               "structural."))
    else:
        d["rq4_logreg_text"] = ""

    df = read_csv(D.R / "rq4" / "summary.csv")
    if df is None or df.empty:
        d["rq4_ablation_table"], d["rq4_ablation_text"] = "", ""
        return d
    df["model"] = df["model"].replace({"RNN": "BiLSTM"})
    rows, drops, overl = [], {}, {}
    for _, name in MODELS:
        o = df[(df["model"] == name) & (df["variant"] == "Original")]
        s = df[(df["model"] == name) & (df["variant"] == "Structure stripped")]
        if o.empty or s.empty:
            continue
        o, s = o.iloc[0], s.iloc[0]
        drops[name] = o["acc"] - s["acc"]
        overl[name] = s["acc_hi"] >= o["acc_lo"] if "acc_hi" in s else None
        rows.append([name, ci(o.to_dict(), "acc"), ci(s.to_dict(), "acc"), pts(-drops[name]),
                     f"{o['macro_f1']:.3f}", f"{s['macro_f1']:.3f}"])
    d["rq4_ablation_table"] = md_table(["Model", "Original accuracy", "Stripped accuracy", "Change",
                                        "Original macro-F1", "Stripped macro-F1"], rows)
    if drops:
        m = np.mean(list(drops.values()))
        t = ("Removing all markdown structure changes accuracy by "
             + join_and([f"{pts(-v)} for the {k}" for k, v in drops.items()]) + ". ")
        if all(v for v in overl.values() if v is not None) and overl:
            t += "The confidence intervals before and after overlap. "
        if m < 0.02:
            t += ("The signal is therefore **primarily linguistic**: with headers, bullets, bold and code fences "
                  "gone, word choice, phrasing and sentence patterns still identify the family almost as well.")
        elif m < 0.07:
            t += ("Formatting contributes, but the signal is **mostly linguistic**: the models lose a few points "
                  "without structure and keep the bulk of their accuracy.")
        else:
            t += ("Formatting carries a **large share** of the signal: families that habitually use lists, "
                  "headers or bold become much harder to tell apart once that structure is removed.")
        t += (" Note what stripping cannot remove: response length, the words inside list items and headers, "
              "and the content of code itself, so this ablation gives a lower bound on the structural contribution.")
        d["rq4_ablation_text"] = t
    return d


# --------------------------------------------------------------------------- #
# Summary + rendering
# --------------------------------------------------------------------------- #
def sec_summary(D, parts):
    ev = getattr(D, "rq1_ev", None)
    if not ev:
        return {"summary_text": "_Run RQ1 to generate the summary._"}
    best_m = max(MODELS, key=lambda mn: ev[mn[0]]["macro_f1"])
    s = (f"We built a balanced, de-identified dataset of {parts.get('n_total', 'n/a')} prompt/response pairs from "
         f"{D.K} LLM families in LMSYS-Chat-1M and trained a multi-kernel CNN and a bidirectional LSTM to name the "
         f"family that wrote a response. From the response alone, the best model ({best_m[1]}) reaches "
         f"{ev[best_m[0]]['acc']:.3f} accuracy and {ev[best_m[0]]['macro_f1']:.3f} macro-F1 against "
         f"{D.chance:.3f} chance.")
    if hasattr(D, "rq2"):
        best_in = max(D.rq2[m]["input_only_acc"] for m, _ in MODELS)
        s += f" Prompts alone reach at most {best_in:.3f}"
        s += (", and adding the prompt to the response changes response-only accuracy by "
              + join_and([f"{pts(D.rq2[m]['both_minus_output_acc'])} ({name})" for m, name in MODELS]) + ".")
    return {"summary_text": s}


PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+?)\}\}")


def render(template, values, D):
    def repl(m):
        key = m.group(1).strip()
        if key.startswith("fig:"):
            rel, _, alt = key[4:].partition("|")
            path = D.R / rel.strip()
            if not path.exists():
                MISSING_FIGS.append(_rel(path))
                return f"_(Figure not generated: `{_rel(path)}`)_"
            link = os.path.relpath(path, D.report_dir).replace(os.sep, "/")
            return f"![{alt.strip() or rel}]({link})"
        v = values.get(key)
        if v is None or v == "":
            return m.group(0) if v is None else ""
        return str(v).replace(" - ", " — ")
    return PLACEHOLDER_RE.sub(repl, template)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--authors", default=None, help='e.g. "Name 1, Name 2, Name 3"')
    p.add_argument("--hardware", default=None, help='e.g. "Google Colab T4 GPU" (default: detected)')
    p.add_argument("--epochs", type=int, default=10, help="max epochs used in research_questions.py")
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience used")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    p.add_argument("--clean-csv", default=str(ROOT / "data" / "cleaned" / "lmsys_clean.csv"))
    p.add_argument("--report-dir", default=str(ROOT / "report"))
    p.add_argument("--template", default=None)
    args = p.parse_args()

    if args.hardware is None:
        try:
            import torch
            args.hardware = (torch.cuda.get_device_name(0) + " GPU" if torch.cuda.is_available() else
                             "Apple-silicon GPU (MPS)" if torch.backends.mps.is_available() else "CPU")
        except Exception:
            args.hardware = "CPU"
        NOTES.append(f"Hardware was auto-detected as '{args.hardware}'. If the experiments ran on a different "
                     f"machine, pass --hardware.")

    D = Data(args)
    D.max_epochs = args.epochs
    D.fig_dir.mkdir(parents=True, exist_ok=True)
    print("Rebuilding dataset figures in results/eda/ ...")
    build_eda(D)
    values = {"authors": args.authors or "_[Team member names — pass --authors]_",
              "K": str(D.K), "chance": f"{D.chance:.3f}", "families_list": join_and(D.labels)}
    for fn in (sec_dataset, lambda d: sec_models(d, args), sec_rq1, sec_rq2, sec_errors,
               sec_robustness, sec_rq3, sec_rq4):
        values.update(fn(D))
    values.update(sec_summary(D, values))

    template_path = Path(args.template) if args.template else Path(args.report_dir) / "REPORT_TEMPLATE.md"
    report = render(template_path.read_text(), values, D)
    out = Path(args.report_dir) / "REPORT.md"
    out.write_text(report)

    unfilled = sorted(set(PLACEHOLDER_RE.findall(report)))
    print(f"\nWrote {_rel(out)}")
    print("\n================ CHECKLIST ================")
    print(f"Missing input files ({len(set(MISSING_INPUTS))}):" + "".join(f"\n  - {x}" for x in sorted(set(MISSING_INPUTS))) if MISSING_INPUTS else "Missing input files: none")
    print(f"Figures linked but not found ({len(MISSING_FIGS)}):" + "".join(f"\n  - {x}" for x in MISSING_FIGS) if MISSING_FIGS else "Figures linked but not found: none")
    print(f"Unfilled placeholders ({len(unfilled)}): {', '.join(unfilled)}" if unfilled else "Unfilled placeholders: none")
    for n in dict.fromkeys(NOTES):
        print(f"Note: {n}")
    if args.authors is None:
        print('Reminder: add your names with --authors "Name 1, Name 2, Name 3"')
    print("Section 5 (Lessons & Experience) is written from the project's history — read it and make it your own.")


if __name__ == "__main__":
    main()
