#!/usr/bin/env python3
"""
Regenerate Fig. 1 as a GROUPED BAR chart (revision item A6) from the
multi-seed summary: macro F1 mean +/- std over the five seeds, one group
per evaluation regime, one bar per model family. No lines.
Includes the (a2) pooled CSE-CIC-IDS2018 random-split baseline group
(from baseline_ids2018_summary.csv), the baseline against which the
(b2) deltas are computed.
Outputs: results/degradation_chart_grouped.png and
../paper/figures/degradation_chart.png (same filename the paper includes).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")

REGIMES = [
    ("a_random_ddos2019", "(a) Random\n(CIC-DDoS2019)"),
    ("a2_random_ids2018_pooled", "(a2) Random\n(CSE-CIC-IDS2018)"),
    ("b1_time_ddos2019_day1_to_day2", "(b1) Time-ordered\n(2019 day1>day2)"),
    ("b2_time_ids2018_early_to_late", "(b2) Time-ordered\n(2018 early>late)"),
    ("c_cross_ddos2019_to_ids2018", "(c) Cross-dataset\n(2019>2018)"),
]
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "MLP"]
COLORS = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c",
          "XGBoost": "#d62728", "MLP": "#ff7f0e"}


def main():
    summ = pd.read_csv(os.path.join(RESULTS, "multiseed_summary.csv"))
    base = pd.read_csv(os.path.join(RESULTS, "baseline_ids2018_summary.csv"))
    summ = pd.concat([summ, base], ignore_index=True)
    x = np.arange(len(REGIMES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for i, model in enumerate(MODELS):
        means, stds = [], []
        for reg, _ in REGIMES:
            row = summ[(summ.regime == reg) & (summ.model == model)]
            assert len(row) == 1, (reg, model)
            means.append(float(row["macro_f1_mean"].iloc[0]))
            stds.append(float(row["macro_f1_std"].iloc[0]))
        print(model, [f"{m:.4f}+/-{s:.4f}" for m, s in zip(means, stds)])
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
               label=model, color=COLORS[model], edgecolor="black",
               linewidth=0.4, error_kw={"elinewidth": 0.8})
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 across evaluation regimes "
                 "(mean +/- std over 5 seeds, measured results)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in REGIMES])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=4, framealpha=0.9)
    fig.tight_layout()
    out1 = os.path.join(RESULTS, "degradation_chart_grouped.png")
    out2 = os.path.join(ROOT, "..", "paper", "figures",
                        "degradation_chart.png")
    fig.savefig(out1, dpi=200)
    fig.savefig(out2, dpi=200)
    print("wrote", out1, "and", out2)


if __name__ == "__main__":
    main()
