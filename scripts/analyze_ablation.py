import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output"

ORDER = ["A0", "A1", "A2", "A3", "A4", "B0"]


def load_runs(seed, results_dir):
    runs = {}
    for path in sorted(results_dir.glob(f"*_seed{seed}/results.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        runs[r["arm"]] = r
        preds = np.load(path.parent / "test_predictions.npz")
        r["_y_true"] = preds["y_true"]
        r["_y_pred"] = preds["y_pred"]
    return {a: runs[a] for a in ORDER if a in runs}


def mcnemar(a, b):
    correct_a = a["_y_pred"] == a["_y_true"]
    correct_b = b["_y_pred"] == b["_y_true"]
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default="output",
                    help="directory holding the per-arm run folders")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    runs = load_runs(args.seed, results_dir)
    if not runs:
        raise SystemExit(f"no results found for seed {args.seed} in {results_dir}")

    baseline = runs.get("A0")
    rows = []
    for arm, r in runs.items():
        t, e, tr = r["test"], r["efficiency"], r["training"]
        row = {
            "arm": arm,
            "description": r["description"],
            "test_accuracy": t["accuracy"],
            "macro_f1": t["macro_f1"],
            "params": e["total_params"],
            "size_mb": e["model_size_mb"],
            "latency_ms": e["latency_ms_median"],
            "sec_per_epoch": tr["seconds_per_epoch_median"],
            "best_epoch": tr["best_epoch"],
            "gen_gap": tr["generalization_gap"],
        }
        if baseline and arm != "A0":
            row["d_macro_f1_vs_A0"] = round(t["macro_f1"] - baseline["test"]["macro_f1"], 5)
            row["param_ratio_vs_A0"] = round(
                e["total_params"] / baseline["efficiency"]["total_params"], 4)
            wins_a0, wins_arm, p = mcnemar(baseline, r)
            row["mcnemar_p_vs_A0"] = f"{p:.2e}"
            row["disagreements"] = f"{wins_a0}/{wins_arm}"
        rows.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"summary_seed{args.seed}.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=["arm", "description"] + [k for k in keys if k not in ("arm", "description")])
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'arm':<4} {'macro-F1':>9} {'acc':>7} {'params':>11} {'MB':>7} {'ms':>7} {'s/ep':>7} {'gap':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<4} {r['macro_f1']:>9.4f} {r['test_accuracy']:>7.4f} "
              f"{r['params']:>11,} {r['size_mb']:>7.2f} {r['latency_ms']:>7.2f} "
              f"{r['sec_per_epoch']:>7.0f} {r['gen_gap']:>7.3f}")

    if baseline:
        print("\nvs A0 (McNemar exact, discordant pairs A0-only/arm-only):")
        for r in rows:
            if "mcnemar_p_vs_A0" in r:
                print(f"  {r['arm']}: macro-F1 {r['d_macro_f1_vs_A0']:+.4f}  "
                      f"params x{r['param_ratio_vs_A0']:.3f}  "
                      f"p={r['mcnemar_p_vs_A0']}  ({r['disagreements']})")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    markers = ["o", "s", "^", "D", "v", "*"]
    for ax, xkey, xlabel, logx in (
        (axes[0], "params", "Parameters", True),
        (axes[1], "latency_ms", "CPU latency per image (ms)", False),
    ):
        for i, r in enumerate(rows):
            face = "black" if r["arm"].startswith("A") else "white"
            ax.scatter(r[xkey], r["macro_f1"], marker=markers[i % len(markers)],
                       s=110, facecolors=face, edgecolors="black", zorder=3,
                       label=f"{r['arm']}")
            ax.annotate(r["arm"], (r[xkey], r["macro_f1"]),
                        textcoords="offset points", xytext=(7, -3), fontsize=9)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Macro F1 (test)")
        ax.grid(alpha=0.3, zorder=0)
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_tradeoff.png", dpi=180)
    plt.close()

    print(f"\nsummary -> {results_dir / f'summary_seed{args.seed}.csv'}")
    print(f"figure  -> {FIG_DIR / 'ablation_tradeoff.png'}")


if __name__ == "__main__":
    main()
