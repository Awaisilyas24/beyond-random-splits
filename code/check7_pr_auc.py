#!/usr/bin/env python3
"""
Hand-check for review item #7 (Muhammad, 2026-11-17): is the PR-AUC in
Table X (regime b2) sklearn's average_precision_score, or a trapezoid
auc(recall, precision)?

Refits regime (b2) at seed 42 with the v2 harness's own functions,
computes BOTH candidate quantities plus ROC-AUC for every model, and
compares against the saved results/ms_summary_seed42.csv values.
Also dumps the PR-curve shape for Random Forest to explain the number.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, auc,
                             precision_recall_curve, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments_v2 as v2

seed = 42
d18_e, f3 = v2.load_pool("ids2018_early")
d18_l, f4 = v2.load_pool("ids2018_late")
feats18 = sorted(set(f3) & set(f4))

te_b2, removed = v2.remove_test_overlap(d18_e, d18_l, feats18)
Xtr, ytr = v2.xy(d18_e, feats18)
Xte, yte = v2.xy(te_b2, feats18)
print(f"b2: train={len(Xtr)} test={len(Xte)} overlap_removed={removed} "
      f"prevalence={yte.mean():.4f}")

saved = pd.read_csv(os.path.join(v2.RESULTS, "ms_summary_seed42.csv"))
saved = saved[saved["regime"] == "b2_time_ids2018_early_to_late"].set_index(
    "model")

for name, model in v2.make_models(seed).items():
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    ap = average_precision_score(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    trap = auc(rec, prec)
    roc = roc_auc_score(yte, proba)
    s = saved.loc[name]
    print(f"\n{name}: saved roc={s['roc_auc']:.6f} pr={s['pr_auc']:.6f}")
    print(f"  recomputed roc_auc_score          = {roc:.6f}")
    print(f"  recomputed average_precision_score= {ap:.6f}  "
          f"(diff vs saved: {ap - s['pr_auc']:+.6f})")
    print(f"  recomputed trapezoid auc(rec,prec)= {trap:.6f}  "
          f"(diff vs saved: {trap - s['pr_auc']:+.6f})")
    if name == "Random Forest":
        # PR-curve shape: precision at recall deciles (step function)
        print("  RF precision at recall thresholds "
              "(share of positives captured at >= that precision):")
        order = np.argsort(-proba, kind="mergesort")
        ys = yte[order]
        tp = np.cumsum(ys)
        prec_at_k = tp / (np.arange(len(ys)) + 1)
        rec_at_k = tp / ys.sum()
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            k = np.searchsorted(rec_at_k, q)
            k = min(k, len(ys) - 1)
            print(f"    recall>={q:.2f}: precision={prec_at_k[k]:.4f} "
                  f"(k={k + 1})")
        u, counts = np.unique(proba, return_counts=True)
        print(f"  RF distinct scores: {len(u)}; largest tie block: "
              f"{counts.max()} points at score {u[counts.argmax()]:.3f}; "
              f"points with score>0: {(proba > 0).sum()} "
              f"(positives with score>0: {((proba > 0) & (yte == 1)).sum()}"
              f" of {int(yte.sum())})")
