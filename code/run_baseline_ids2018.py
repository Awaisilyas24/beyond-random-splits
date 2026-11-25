#!/usr/bin/env python3
"""
Review round 2, item #2 (Muhammad, 2026-11-17): regime (b2)'s Delta was
computed against regime (a) -- a CIC-DDoS2019 baseline -- although (b2)
runs on CSE-CIC-IDS2018. This script creates (b2)'s own within-dataset
baseline: a random stratified 70/30 split of the POOLED CSE-CIC-IDS2018
data (early + late pools, deduplicated exactly as run_experiments_v2.py
builds pool18), same pipeline, same 4 models, same 5 seeds, same
feats18 feature set as regime (b2).

Outputs (results/): baseline_summary_seed<seed>.csv (checkpoints),
baseline_ids2018_per_seed.csv, baseline_ids2018_summary.csv,
b2_deltas_corrected.csv (paired per-seed Delta of (b2) vs this baseline).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments_v2 as v2

REGIME = "a2_random_ids2018_pooled"


def main():
    d18_e, f3 = v2.load_pool("ids2018_early")
    d18_l, f4 = v2.load_pool("ids2018_late")
    feats18 = sorted(set(f3) & set(f4))
    pool18, ndup = v2.dedup(pd.concat([d18_e, d18_l], ignore_index=True),
                            feats18)
    print(f"pool18: {len(pool18)} rows ({ndup} cross-pool duplicates "
          f"removed); prevalence={pool18['label'].mean():.4f}; "
          f"features={len(feats18)}", flush=True)

    frames = []
    for seed in v2.SEEDS:
        ckpt = os.path.join(v2.RESULTS, f"baseline_summary_seed{seed}.csv")
        if os.path.exists(ckpt):
            frames.append(pd.read_csv(ckpt))
            print(f"seed {seed}: checkpoint exists, skipped", flush=True)
            continue
        tr, te = train_test_split(pool18, test_size=0.30,
                                  random_state=seed,
                                  stratify=pool18["label"])
        Xtr, ytr = v2.xy(tr, feats18)
        Xte, yte = v2.xy(te, feats18)
        srows, _, _ = v2.fit_all(REGIME, seed, Xtr, ytr, Xte, yte)
        df = pd.DataFrame(srows)
        df.to_csv(ckpt, index=False)
        frames.append(df)
        print(f"SEED {seed} DONE", flush=True)

    allt = pd.concat(frames, ignore_index=True)
    allt.to_csv(os.path.join(v2.RESULTS, "baseline_ids2018_per_seed.csv"),
                index=False)
    mets = ["accuracy", "macro_f1", "benign_f1", "ddos_f1", "ddos_precision",
            "ddos_recall", "benign_recall", "roc_auc", "pr_auc", "benign_fpr"]
    agg = allt.groupby(["regime", "model"])[mets].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(
        os.path.join(v2.RESULTS, "baseline_ids2018_summary.csv"),
        index=False)

    # Paired per-seed deltas: (b2) minus this within-dataset baseline
    b2 = pd.read_csv(os.path.join(v2.RESULTS, "multiseed_per_seed.csv"))
    b2 = b2[b2["regime"] == "b2_time_ids2018_early_to_late"]
    m = b2.merge(allt, on=["seed", "model"], suffixes=("_b2", "_base"))
    rows = []
    for model, g in m.groupby("model"):
        row = {"model": model}
        for met in ["accuracy", "macro_f1", "ddos_f1", "ddos_recall"]:
            d = g[f"{met}_b2"] - g[f"{met}_base"]
            row[f"delta_{met}_mean"] = d.mean()
            row[f"delta_{met}_std"] = d.std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(v2.RESULTS, "b2_deltas_corrected.csv"), index=False)
    print("BASELINE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
