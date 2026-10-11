"""
Phase 3/4 - Research questions
==============================
RQ1  Baseline detectability: CNN and BiLSTM trained on LLM_output only.
RQ2  Prompt leakage: LLM_input only vs LLM_output only (reused from RQ1) vs both.
RQ3  (extra credit) Domain transfer: train on one task category, test zero-shot on others.
RQ4  (extra credit) Structural vs linguistic signal: stylometric features, then retrain on
     responses with all markdown structure removed.

Every experiment writes to results/<rq>/<model>_<mode>/:
    grid_results.csv, history.json, test_metrics.json, test_predictions.npz,
    learning_curves.png, confusion_matrix.png
and the best checkpoint to models/<rq>_<model>_<mode>.pt.
Finished experiments are re-used on the next run unless --force is given, so a long
run can be resumed.

Usage
-----
    python research_questions.py --rq 1 2              # required parts (grid = small)
    python research_questions.py --rq 3 4              # extra credit (re-uses RQ1 hyper-params)
    python research_questions.py --rq 1 2 3 4 --grid tiny --epochs 3   # quick smoke test
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from config import CLEAN_CSV, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from data_processing import load_clean, load_processed, process
from model_training import (evaluate, get_device, grid_search, make_loader, save_checkpoint,
                            split_arrays)
from visualization import (plot_confusion_matrix, plot_feature_distributions,
                           plot_grouped_metric, plot_learning_curves)

MODEL_LABEL = {"cnn": "CNN", "rnn": "RNN"}
MODE_LABEL = {"input": "Prompt only", "output": "Response only", "both": "Prompt + response"}


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05, seed=SEED):
    """95% percentile bootstrap CIs for accuracy and macro-F1 on the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    labels = np.unique(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        accs.append((y_true[i] == y_pred[i]).mean())
        f1s.append(f1_score(y_true[i], y_pred[i], average="macro", labels=labels, zero_division=0))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"acc_lo": float(np.percentile(accs, q[0])), "acc_hi": float(np.percentile(accs, q[1])),
            "macro_f1_lo": float(np.percentile(f1s, q[0])), "macro_f1_hi": float(np.percentile(f1s, q[1]))}


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on paired predictions (same test rows)."""
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b01 = int(np.sum(a_right & ~b_right))     # A right, B wrong
    b10 = int(np.sum(~a_right & b_right))     # A wrong, B right
    n = b01 + b10
    p = 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": p}


# --------------------------------------------------------------------------- #
# One experiment = grid search + test evaluation + artifacts
# --------------------------------------------------------------------------- #
def run_experiment(tag, model_type, mode, data, grid, epochs, patience, device,
                   train_mask=None, val_mask=None, test_sets=None, force=False,
                   title_suffix=""):
    """Train (or re-load) one model and evaluate it.

    test_sets: dict name -> row mask over the full data; default {"test": all test rows}.
    Returns a summary dict (metrics for the primary 'test' set + any extra test sets).
    """
    out = RESULTS_DIR / tag / f"{model_type}_{mode}"
    metrics_path = out / "test_metrics.json"
    if metrics_path.exists() and not force:
        print(f"  [skip] {tag}/{model_type}_{mode} already done (use --force to retrain)")
        return json.loads(metrics_path.read_text())

    out.mkdir(parents=True, exist_ok=True)
    labels = list(data["label_map"])
    K = len(labels)
    print(f"\n=== {tag.upper()} | {MODEL_LABEL[model_type]} | {MODE_LABEL[mode]}{title_suffix} ===")
    train = split_arrays(data, "train", train_mask)
    val = split_arrays(data, "val", val_mask)
    print(f"  train={len(train[2]):,}  val={len(val[2]):,}  classes={K}  vocab={data['vocab_size']:,}")

    best, grid_df = grid_search(model_type, train, val, data["vocab_size"], K, grid, epochs,
                                patience, device)
    grid_df.to_csv(out / "grid_results.csv", index=False)
    (out / "history.json").write_text(json.dumps(best["history"], indent=1))
    save_checkpoint(best, MODELS_DIR / f"{tag}_{model_type}_{mode}.pt", model_type, mode,
                    data["label_map"], data["vocab_size"])

    title = f"{MODEL_LABEL[model_type]} - {MODE_LABEL[mode]}{title_suffix}"
    plot_learning_curves(best["history"], title, out / "learning_curves.png")

    test_sets = test_sets or {"test": None}
    summary = {"tag": tag, "model": MODEL_LABEL[model_type], "mode": mode,
               "hp": {k: list(v) if isinstance(v, tuple) else v for k, v in best["hp"].items()},
               "best_epoch": best["best_epoch"], "n_params": best["n_params"],
               "val_acc": best["best_val"]["acc"], "val_macro_f1": best["best_val"]["macro_f1"],
               "n_train": int(len(train[2])), "eval": {}}
    preds = {}
    for name, mask in test_sets.items():
        Xt, Lt, yt = split_arrays(data, "test", mask)
        if len(yt) == 0:
            continue
        res = evaluate(best["model"], make_loader(Xt, Lt, yt, 256, False), device, K)
        ci = bootstrap_ci(res["y_true"], res["y_pred"])
        report = classification_report(res["y_true"], res["y_pred"], labels=list(range(K)),
                                       target_names=labels, output_dict=True, zero_division=0)
        summary["eval"][name] = {"n": int(len(yt)), "acc": res["acc"], "macro_f1": res["macro_f1"],
                                 **ci, "cm": res["cm"],
                                 "per_class_f1": {l: report[l]["f1-score"] for l in labels}}
        preds[f"{name}_y_true"], preds[f"{name}_y_pred"] = res["y_true"], res["y_pred"]
        preds[f"{name}_probs"] = res["probs"]
        cm_name = "confusion_matrix.png" if name == "test" else f"confusion_matrix_{name}.png"
        plot_confusion_matrix(res["cm"], labels, f"{title} ({name}, row-normalised)", out / cm_name)
        print(f"  {name:>12}: acc {res['acc']:.4f} [{ci['acc_lo']:.3f}, {ci['acc_hi']:.3f}]  "
              f"macro-F1 {res['macro_f1']:.4f} [{ci['macro_f1_lo']:.3f}, {ci['macro_f1_hi']:.3f}]")

    np.savez_compressed(out / "test_predictions.npz", **preds)
    metrics_path.write_text(json.dumps(summary, indent=2))
    return summary


def load_predictions(tag, model_type, mode, name="test"):
    p = np.load(RESULTS_DIR / tag / f"{model_type}_{mode}" / "test_predictions.npz")
    return p[f"{name}_y_true"], p[f"{name}_y_pred"]


def summary_rows(summaries, set_name="test", extra=None):
    rows = []
    for s in summaries:
        e = s["eval"].get(set_name)
        if e is None:
            continue
        rows.append({"model": s["model"], "mode": s["mode"], **(extra or {}),
                     "acc": e["acc"], "acc_lo": e["acc_lo"], "acc_hi": e["acc_hi"],
                     "macro_f1": e["macro_f1"], "macro_f1_lo": e["macro_f1_lo"],
                     "macro_f1_hi": e["macro_f1_hi"], "val_macro_f1": s["val_macro_f1"],
                     "best_epoch": s["best_epoch"], "n_params": s["n_params"],
                     "hp": json.dumps({k: v for k, v in s["hp"].items()
                                       if k in ("lr", "batch_size", "kernel_sizes", "hidden_dim")})})
    return pd.DataFrame(rows)


def best_hp_grid(model_type):
    """Single-config grid holding RQ1's selected hyper-parameters (used by RQ3/RQ4)."""
    path = RESULTS_DIR / "rq1" / f"{model_type}_output" / "test_metrics.json"
    if not path.exists():
        print(f"  [!] RQ1 results for {model_type} not found - falling back to default hyper-params")
        hp = {"lr": 1e-3, "batch_size": 64}
        hp.update({"kernel_sizes": (3, 4, 5)} if model_type == "cnn" else {"hidden_dim": 128})
    else:
        s = json.loads(path.read_text())["hp"]
        hp = {"lr": s["lr"], "batch_size": s["batch_size"]}
        if model_type == "cnn":
            hp["kernel_sizes"] = tuple(s["kernel_sizes"])
        else:
            hp["hidden_dim"] = s["hidden_dim"]
    return {k: [v] for k, v in hp.items()}


# --------------------------------------------------------------------------- #
# RQ1
# --------------------------------------------------------------------------- #
def rq1(args, device):
    data = load_processed("output", args.processed_dir)
    sums = [run_experiment("rq1", m, "output", data, args.grid, args.epochs, args.patience,
                           device, force=args.force) for m in args.models]
    df = summary_rows(sums)
    out = RESULTS_DIR / "rq1"
    df.to_csv(out / "summary.csv", index=False)
    per_class = pd.DataFrame({s["model"]: s["eval"]["test"]["per_class_f1"] for s in sums})
    per_class.to_csv(out / "per_class_f1.csv")
    K = len(data["label_map"])
    plot_grouped_metric(df.assign(group="Response only"), "group", ["Response only"],
                        "RQ1 - Baseline detectability from responses", out / "rq1_test_metrics.png",
                        chance=1 / K)
    if len(args.models) == 2:
        yt, pa = load_predictions("rq1", "cnn", "output")
        _, pb = load_predictions("rq1", "rnn", "output")
        (out / "cnn_vs_rnn_mcnemar.json").write_text(json.dumps(mcnemar(yt, pa, pb), indent=2))
    print("\nRQ1 summary\n" + df[["model", "acc", "macro_f1"]].to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# RQ2
# --------------------------------------------------------------------------- #
def rq2(args, device):
    sums = []
    for mode in ("input", "both"):
        data = load_processed(mode, args.processed_dir)
        sums += [run_experiment("rq2", m, mode, data, args.grid, args.epochs, args.patience,
                                device, force=args.force) for m in args.models]
    # response-only results come from RQ1 (no need to recompute)
    for m in args.models:
        p = RESULTS_DIR / "rq1" / f"{m}_output" / "test_metrics.json"
        if p.exists():
            sums.append(json.loads(p.read_text()))
        else:
            print(f"  [!] RQ1 {m} results missing - run --rq 1 first for the full comparison")
    K = len(load_processed("output", args.processed_dir)["label_map"])
    df = summary_rows(sums)
    df["mode_label"] = df["mode"].map(MODE_LABEL)
    order = [MODE_LABEL[m] for m in ("input", "output", "both")]
    df = (df.assign(_o=df["mode"].map({"input": 0, "output": 1, "both": 2}))
            .sort_values(["model", "_o"]).drop(columns="_o"))
    out = RESULTS_DIR / "rq2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    plot_grouped_metric(df, "mode_label", order, "RQ2 - What the prompt adds",
                        out / "rq2_mode_comparison.png", chance=1 / K)

    # statistical analysis
    analysis = {"chance_accuracy": 1 / K}
    for m in args.models:
        try:
            yt, p_out = load_predictions("rq1", m, "output")
            _, p_both = load_predictions("rq2", m, "both")
            yt_in, p_in = load_predictions("rq2", m, "input")
        except FileNotFoundError:
            continue
        n_in = len(yt_in)
        k_in = int((p_in == yt_in).sum())
        analysis[MODEL_LABEL[m]] = {
            "input_only_acc": k_in / n_in,
            "input_only_vs_chance_p": float(stats.binomtest(k_in, n_in, 1 / K,
                                                           alternative="greater").pvalue),
            "both_minus_output_acc": float((p_both == yt).mean() - (p_out == yt).mean()),
            "both_minus_output_macro_f1": float(
                f1_score(yt, p_both, average="macro") - f1_score(yt, p_out, average="macro")),
            "mcnemar_output_vs_both": mcnemar(yt, p_out, p_both),
        }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print("\nRQ2 summary\n" + df[["model", "mode_label", "acc", "macro_f1"]].to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# RQ3 - domain transfer (extra credit)
# --------------------------------------------------------------------------- #
def rq3(args, device):
    data = load_processed("output", args.processed_dir)
    meta = data["meta"]
    cats = meta["task_category"]
    src = args.source_domain
    targets = [c for c in cats.value_counts().index if c != src]
    print("\nTask categories (all splits):\n" +
          pd.crosstab(cats, meta["LLM_name"]).to_string())
    if (cats == src).sum() == 0:
        raise ValueError(f"Source domain '{src}' not present. Available: {sorted(cats.unique())}")

    test_sets = {f"in-domain ({src})": (cats == src).values}
    test_sets.update({f"zero-shot ({t})": (cats == t).values for t in targets})
    sums = [run_experiment("rq3", m, "output", data, best_hp_grid(m), args.epochs, args.patience,
                           device, train_mask=(cats == src).values, val_mask=(cats == src).values,
                           test_sets=test_sets, force=args.force,
                           title_suffix=f" | trained on '{src}'") for m in args.models]
    rows = []
    for s in sums:
        base = s["eval"][f"in-domain ({src})"]
        for name, e in s["eval"].items():
            rows.append({"model": s["model"], "test_domain": name, "n_test": e["n"],
                         "acc": e["acc"], "acc_lo": e["acc_lo"], "acc_hi": e["acc_hi"],
                         "macro_f1": e["macro_f1"], "macro_f1_lo": e["macro_f1_lo"],
                         "macro_f1_hi": e["macro_f1_hi"],
                         "acc_drop_vs_in_domain": base["acc"] - e["acc"],
                         "f1_drop_vs_in_domain": base["macro_f1"] - e["macro_f1"]})
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "rq3"
    df.to_csv(out / "summary.csv", index=False)
    K = len(data["label_map"])
    plot_grouped_metric(df, "test_domain", list(test_sets), f"RQ3 - Trained on '{src}' prompts only",
                        out / "rq3_domain_transfer.png", chance=1 / K, xlabel="Test domain")
    print("\nRQ3 summary\n" + df[["model", "test_domain", "n_test", "acc", "macro_f1",
                                  "f1_drop_vs_in_domain"]].to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# RQ4 - structural vs linguistic signal (extra credit)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z0-9']+")


def strip_structure(text: str) -> str:
    """Remove markdown structure while keeping the words."""
    t = re.sub(r"```[\w+#.-]*", " ", text)                  # code fences (+ language tag)
    t = t.replace("`", " ")                                  # inline code
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)              # headers
    t = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", " ", t)           # horizontal rules
    t = re.sub(r"(?m)^\s*[-*•+]\s+", "", t)                  # bullets
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)                 # numbered-list markers
    t = re.sub(r"\*\*|__", "", t)                            # bold
    t = re.sub(r"(?m)^\s*>\s?", "", t)                       # block quotes
    t = t.replace("|", " ")                                  # table pipes
    return re.sub(r"[ \t]{2,}", " ", t)


def style_features(text: str, mattr_window: int = 50) -> dict:
    words = _WORD_RE.findall(text.lower())
    n = len(words)
    ttr = len(set(words)) / n if n else 0.0
    if n >= mattr_window:   # moving-average TTR is length-robust
        mattr = float(np.mean([len(set(words[i:i + mattr_window])) / mattr_window
                               for i in range(0, n - mattr_window + 1, 5)]))
    else:
        mattr = ttr
    headers = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", text))
    bullets = len(re.findall(r"(?m)^\s*[-*•+]\s+", text))
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))
    bold = len(re.findall(r"\*\*[^*\n]+\*\*|__[^_\n]+__", text))
    code_blocks = text.count("```") // 2
    md_total = headers + bullets + numbered + bold + code_blocks
    return {"response_words": n, "type_token_ratio": ttr, "mattr_50": mattr,
            "headers": headers, "bullets": bullets + numbered, "bold": bold,
            "code_blocks": code_blocks,
            "format_density_per_100w": 100 * md_total / max(n, 1)}


def rq4(args, device):
    out = RESULTS_DIR / "rq4"
    out.mkdir(parents=True, exist_ok=True)
    base = load_processed("output", args.processed_dir)
    meta = base["meta"]
    df = load_clean(args.clean_csv)
    if len(df) != len(meta) or not (df["LLM_name"].values == meta["LLM_name"].values).all():
        raise RuntimeError("Cleaned CSV does not line up with data/processed - re-run data_processing.py")
    families = list(base["label_map"])

    # ---- 1. stylometric features per family ----
    feats = pd.DataFrame([style_features(t) for t in df["LLM_output"]])
    feats.insert(0, "LLM_name", df["LLM_name"].values)
    feats.to_csv(out / "style_features.csv", index=False)
    table = feats.groupby("LLM_name").agg(["mean", "median"]).reindex(families)
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table.to_csv(out / "style_features_by_family.csv")
    kw = {c: dict(zip(("H", "p_value"), map(float, stats.kruskal(
        *[feats.loc[feats.LLM_name == f, c] for f in families]))))
          for c in feats.columns if c != "LLM_name"}
    (out / "kruskal_wallis.json").write_text(json.dumps(kw, indent=2))
    plot_feature_distributions(feats, families, out / "rq4_style_features.png",
                               "RQ4 - Stylometric profile of each LLM family (mean ± 95% CI)")

    # ---- 2. how far do hand-crafted features alone get? (logistic regression) ----
    tr, te = (meta["split"] == "train").values, (meta["split"] == "test").values
    Xf = np.log1p(feats.drop(columns="LLM_name").values)
    scaler = StandardScaler().fit(Xf[tr])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xf[tr]), base["y"][tr])
    pf = clf.predict(scaler.transform(Xf[te]))
    feat_only = {"acc": float(accuracy_score(base["y"][te], pf)),
                 "macro_f1": float(f1_score(base["y"][te], pf, average="macro")),
                 **bootstrap_ci(base["y"][te], pf)}
    (out / "feature_only_logreg.json").write_text(json.dumps(feat_only, indent=2))
    print(f"\nStyle-feature logistic regression: acc {feat_only['acc']:.4f}  "
          f"macro-F1 {feat_only['macro_f1']:.4f}")

    # ---- 3. retrain on structure-stripped responses (same split, RQ1 hyper-params) ----
    stripped_dir = Path(args.processed_dir).parent / "processed_rq4_stripped"
    cfg = json.loads((Path(args.processed_dir) / "processing_config.json").read_text())
    if not (stripped_dir / "output.npz").exists() or args.force:
        print("\nRe-tokenizing responses with markdown structure stripped ...")
        process(df, stripped_dir, cfg["tokenizer"], cfg["max_input_len"], cfg["max_output_len"],
                cfg["min_freq"], modes=("output",), split=meta["split"].values,
                output_transform=strip_structure)
    stripped = load_processed("output", stripped_dir)
    sums = [run_experiment("rq4", m, "output", stripped, best_hp_grid(m), args.epochs,
                           args.patience, device, force=args.force,
                           title_suffix=" | structure stripped") for m in args.models]

    rows = []
    for s in sums:
        m = s["model"].lower()
        orig_p = RESULTS_DIR / "rq1" / f"{m}_output" / "test_metrics.json"
        if orig_p.exists():
            o = json.loads(orig_p.read_text())["eval"]["test"]
            rows.append({"model": s["model"], "variant": "Original", **{k: o[k] for k in
                         ("acc", "acc_lo", "acc_hi", "macro_f1", "macro_f1_lo", "macro_f1_hi")}})
        e = s["eval"]["test"]
        rows.append({"model": s["model"], "variant": "Structure stripped", **{k: e[k] for k in
                     ("acc", "acc_lo", "acc_hi", "macro_f1", "macro_f1_lo", "macro_f1_hi")}})
    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(out / "summary.csv", index=False)
    K = len(families)
    plot_grouped_metric(df_cmp, "variant", ["Original", "Structure stripped"],
                        "RQ4 - Accuracy with markdown structure removed",
                        out / "rq4_structure_ablation.png", chance=1 / K)
    print("\nRQ4 summary\n" + df_cmp[["model", "variant", "acc", "macro_f1"]].to_string(index=False))
    return df_cmp


# --------------------------------------------------------------------------- #
# Combined markdown table (pasted into the report)
# --------------------------------------------------------------------------- #
def write_results_markdown():
    lines = ["# Results summary (auto-generated by research_questions.py)\n"]
    fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    for rq, title, group in (("rq1", "RQ1 - response only", None),
                             ("rq2", "RQ2 - sequence configurations", "mode_label"),
                             ("rq3", "RQ3 - domain transfer", "test_domain"),
                             ("rq4", "RQ4 - structure ablation", "variant")):
        p = RESULTS_DIR / rq / "summary.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        lines.append(f"\n## {title}\n")
        nice = {"mode_label": "Input sequence", "test_domain": "Test domain", "variant": "Variant"}
        head = ["Model"] + ([nice[group]] if group else []) + \
               ["Test accuracy (95% CI)", "Test macro-F1 (95% CI)"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for _, r in df.iterrows():
            cells = [r["model"]] + ([str(r[group])] if group else []) + [fmt(r, "acc"), fmt(r, "macro_f1")]
            lines.append("| " + " | ".join(cells) + " |")
    (RESULTS_DIR / "results_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_DIR / 'results_summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rq", nargs="+", type=int, default=[1, 2], choices=[1, 2, 3, 4])
    p.add_argument("--models", nargs="+", default=["cnn", "rnn"], choices=["cnn", "rnn"])
    p.add_argument("--grid", default="small", choices=["tiny", "small", "full"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--source-domain", default="general", help="RQ3 training domain")
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--clean-csv", default=str(CLEAN_CSV))
    p.add_argument("--force", action="store_true", help="retrain even if results exist")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    for rq in sorted(args.rq):
        {1: rq1, 2: rq2, 3: rq3, 4: rq4}[rq](args, device)
    write_results_markdown()


if __name__ == "__main__":
    main()
