#!/usr/bin/env python3
"""
Revision runs (Part B of Muhammad's 2026-11-17 revision checklist) for:
"Beyond Random Splits: A Leakage-Controlled, Time-Aware Evaluation of
Machine Learning DDoS Detectors for Cloud Environments"

Extends run_experiments.py (v1, seed 42) with:
  B1  Dedup ablation: regime (a) WITH vs WITHOUT exact-duplicate removal
      (without = pools from prepare_data_nodedup.py, no pooled dedup either;
      every other step identical).
  B2  Multiple seeds: all regimes x all models for each seed in SEEDS;
      per-seed CSVs are checkpointed, mean +/- std aggregated at the end.
  B3  ROC-AUC, PR-AUC (average precision) and benign false-positive rate
      for every regime/model (all four models expose predict_proba).
  B4  Per-attack recall for ALL models in regimes (a) and (b1)
      (v1 reported only the best model). Subtype spelling 'UDP-lag'
      (day 01-12) is normalised to 'UDPLag' (day 03-11 spelling): same
      attack, two per-day label spellings in the release files.
  B5  Cross-dataset flood control:
      (i)  c_floodtest: regime-(c) models, test restricted to benign +
           LOIC-UDP flows (the UDP-flood attack of CSE-CIC-IDS2018);
      (ii) c2_floodtrain: train on CIC-DDoS2019 benign + flood subtypes
           (Syn, UDP, DrDoS_UDP) only, test on CSE-CIC-IDS2018 benign +
           LOIC-UDP only. Per-subtype recall under full regime (c) is
           also recorded (HOIC vs LOIC-UDP).

Usage: python run_experiments_v2.py [seed ...]   (default: all SEEDS)
Outputs (results/): ms_summary_seed<seed>.csv, ms_perattack_seed<seed>.csv
(checkpoints), multiseed_per_seed.csv, multiseed_summary.csv,
dedup_ablation.csv, perattack_multiseed.csv, run_info_v2.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEEDS = [42, 17, 101, 555, 2024]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PREP = os.path.join(ROOT, "data", "prepared")
PREP_ND = os.path.join(ROOT, "data", "prepared_nodedup")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

FLOOD_2019 = {"Syn", "UDP", "DrDoS_UDP"}          # B5 training subtypes
FLOOD_2018 = {"DDOS attack-LOIC-UDP"}             # B5 test subtype


def load_pool(name, prep=PREP):
    df = pd.read_csv(os.path.join(prep, name + ".csv"))
    feats = [c for c in df.columns if c not in ("label", "subtype")]
    return df, feats


def dedup(df, feats):
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
    return df, before - len(df)


def remove_test_overlap(train, test, feats):
    h = pd.util.hash_pandas_object(train[feats], index=False)
    train_hashes = set(h.to_numpy())
    ht = pd.util.hash_pandas_object(test[feats], index=False)
    mask = ~ht.isin(train_hashes)
    removed = int((~mask).sum())
    return test.loc[mask].reset_index(drop=True), removed


def make_models(seed):
    imp = ("imputer", SimpleImputer(strategy="median"))
    return {
        "Logistic Regression": Pipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       random_state=seed))]),
        "Random Forest": Pipeline([
            imp,
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                           random_state=seed))]),
        "XGBoost": Pipeline([
            imp,
            ("clf", XGBClassifier(n_estimators=300, max_depth=6,
                                  learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.9, tree_method="hist",
                                  n_jobs=-1, eval_metric="logloss",
                                  random_state=seed))]),
        "MLP": Pipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                  max_iter=60, early_stopping=True,
                                  random_state=seed))]),
    }


def norm_subtype(s):
    return "UDPLag" if s == "UDP-lag" else s


def metric_row(regime, model, seed, yte, pred, proba, n_train,
               fit_s=None, latency=None):
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    row = {
        "seed": seed, "regime": regime, "model": model,
        "n_train": n_train, "n_test": int(len(yte)),
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": rep["macro avg"]["f1-score"],
        "benign_f1": rep["0"]["f1-score"],
        "ddos_f1": rep["1"]["f1-score"],
        "ddos_precision": rep["1"]["precision"],
        "ddos_recall": rep["1"]["recall"],
        "benign_recall": rep["0"]["recall"],
        "roc_auc": roc_auc_score(yte, proba),
        "pr_auc": average_precision_score(yte, proba),
        "benign_fpr": float((pred[yte == 0] == 1).mean()),
        "fit_seconds": fit_s,
    }
    if latency is not None:
        row["latency_ms_per_1000"], row["throughput_flows_per_s"] = latency
    return row


def per_attack_rows(regime, model, seed, yte, pred, subtypes):
    sub = pd.DataFrame({"subtype": [norm_subtype(s) for s in subtypes],
                        "y": yte, "p": pred})
    rows = []
    for st, g in sub[sub["y"] == 1].groupby("subtype"):
        rows.append({"seed": seed, "regime": regime, "model": model,
                     "attack_subtype": st, "support": int(len(g)),
                     "recall": float((g["p"] == 1).mean())})
    return rows


def fit_all(regime, seed, Xtr, ytr, Xte, yte, subtypes_te=None,
            latency_too=False, attack_regimes=()):
    """Fit all 4 models for one regime; return (summary_rows, attack_rows,
    fitted) where fitted maps model -> (pred, proba) for reuse (B5)."""
    srows, arows, fitted = [], [], {}
    for name, model in make_models(seed).items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        lat = None
        if latency_too:
            lat_sample = Xte.iloc[:10_000]
            times = []
            for _ in range(3):
                t0 = time.time()
                model.predict(lat_sample)
                times.append(time.time() - t0)
            med = float(np.median(times))
            lat = (med / len(lat_sample) * 1e6, len(lat_sample) / med)
        srows.append(metric_row(regime, name, seed, yte, pred, proba,
                                len(Xtr), round(fit_s, 1), lat))
        fitted[name] = (pred, proba)
        if regime in attack_regimes and subtypes_te is not None:
            arows += per_attack_rows(regime, name, seed, yte, pred,
                                     subtypes_te)
        r = srows[-1]
        print(f"  [{seed}] {regime:34s} {name:20s} acc={r['accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f} ddosF1={r['ddos_f1']:.4f} "
              f"roc={r['roc_auc']:.4f} fit={fit_s:.0f}s", flush=True)
    return srows, arows, fitted


def xy(df, feats):
    return df[feats], df["label"].to_numpy()


def run_seed(seed, pools):
    (d19_1, d19_2, d18_e, d18_l, nd19_1, nd19_2,
     feats19, feats18, feats_common) = pools
    srows, arows = [], []

    pool19, _ = dedup(pd.concat([d19_1, d19_2], ignore_index=True), feats19)
    pool18, _ = dedup(pd.concat([d18_e, d18_l], ignore_index=True), feats18)
    pool19_nd = pd.concat([nd19_1, nd19_2], ignore_index=True)  # NO dedup

    # ---- (a) random split, WITH dedup (v1 protocol) -----------------------
    tr, te = train_test_split(pool19, test_size=0.30, random_state=seed,
                              stratify=pool19["label"])
    Xtr, ytr = xy(tr, feats19)
    Xte, yte = xy(te, feats19)
    s, a, _ = fit_all("a_random_ddos2019", seed, Xtr, ytr, Xte, yte,
                      subtypes_te=te["subtype"].to_numpy(), latency_too=True,
                      attack_regimes=("a_random_ddos2019",))
    srows += s; arows += a

    # ---- B1: (a) random split, WITHOUT dedup ------------------------------
    tr, te = train_test_split(pool19_nd, test_size=0.30, random_state=seed,
                              stratify=pool19_nd["label"])
    Xtr, ytr = xy(tr, feats19)
    Xte, yte = xy(te, feats19)
    s, _, _ = fit_all("a_random_ddos2019_NODEDUP", seed, Xtr, ytr, Xte, yte)
    srows += s

    # ---- (b1) time-ordered CIC-DDoS2019 ------------------------------------
    te_b, _ = remove_test_overlap(d19_1, d19_2, feats19)
    Xtr, ytr = xy(d19_1, feats19)
    Xte, yte = xy(te_b, feats19)
    s, a, _ = fit_all("b1_time_ddos2019_day1_to_day2", seed, Xtr, ytr,
                      Xte, yte, subtypes_te=te_b["subtype"].to_numpy(),
                      attack_regimes=("b1_time_ddos2019_day1_to_day2",))
    srows += s; arows += a

    # ---- (b2) time-ordered CSE-CIC-IDS2018 ---------------------------------
    te_b2, _ = remove_test_overlap(d18_e, d18_l, feats18)
    Xtr, ytr = xy(d18_e, feats18)
    Xte, yte = xy(te_b2, feats18)
    s, _, _ = fit_all("b2_time_ids2018_early_to_late", seed, Xtr, ytr,
                      Xte, yte)
    srows += s

    # ---- (c) cross-dataset --------------------------------------------------
    te_c, _ = remove_test_overlap(pool19, pool18, feats_common)
    Xtr, ytr = xy(pool19, feats_common)
    Xte, yte = xy(te_c, feats_common)
    s, a, fitted_c = fit_all("c_cross_ddos2019_to_ids2018", seed, Xtr, ytr,
                             Xte, yte,
                             subtypes_te=te_c["subtype"].to_numpy(),
                             attack_regimes=("c_cross_ddos2019_to_ids2018",))
    srows += s; arows += a

    # ---- B5(i): regime-(c) models, test restricted to benign + LOIC-UDP ----
    sub_c = te_c["subtype"].to_numpy()
    mask = np.isin(sub_c, ["Benign"] + list(FLOOD_2018))
    for name, (pred, proba) in fitted_c.items():
        srows.append(metric_row("c_floodtest_benign_plus_loicudp", name,
                                seed, yte[mask], pred[mask], proba[mask],
                                len(Xtr)))

    # ---- B5(ii): flood-only training -> LOIC-UDP test -----------------------
    tr_f = pool19[pool19["subtype"].isin(["Benign"] + list(FLOOD_2019))]
    te_f = pool18[pool18["subtype"].isin(["Benign"] + list(FLOOD_2018))]
    te_f, _ = remove_test_overlap(tr_f, te_f, feats_common)
    Xtr, ytr = xy(tr_f, feats_common)
    Xte, yte = xy(te_f, feats_common)
    s, _, _ = fit_all("c2_floodtrain_to_loicudp", seed, Xtr, ytr, Xte, yte)
    srows += s

    pd.DataFrame(srows).to_csv(
        os.path.join(RESULTS, f"ms_summary_seed{seed}.csv"), index=False)
    pd.DataFrame(arows).to_csv(
        os.path.join(RESULTS, f"ms_perattack_seed{seed}.csv"), index=False)
    print(f"SEED {seed} DONE", flush=True)


def aggregate(seeds):
    frames = [pd.read_csv(os.path.join(RESULTS, f"ms_summary_seed{s}.csv"))
              for s in seeds
              if os.path.exists(os.path.join(RESULTS, f"ms_summary_seed{s}.csv"))]
    allt = pd.concat(frames, ignore_index=True)
    allt.to_csv(os.path.join(RESULTS, "multiseed_per_seed.csv"), index=False)
    mets = ["accuracy", "macro_f1", "benign_f1", "ddos_f1", "ddos_precision",
            "ddos_recall", "benign_recall", "roc_auc", "pr_auc", "benign_fpr"]
    agg = allt.groupby(["regime", "model"])[mets].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(RESULTS, "multiseed_summary.csv"), index=False)

    # B1 ablation table: with vs without dedup (regime a)
    sub = allt[allt["regime"].isin(
        ["a_random_ddos2019", "a_random_ddos2019_NODEDUP"])]
    abl = sub.groupby(["regime", "model"])[mets].agg(["mean", "std"])
    abl.columns = [f"{m}_{s}" for m, s in abl.columns]
    abl.reset_index().to_csv(
        os.path.join(RESULTS, "dedup_ablation.csv"), index=False)

    pframes = [pd.read_csv(os.path.join(RESULTS, f"ms_perattack_seed{s}.csv"))
               for s in seeds
               if os.path.exists(os.path.join(RESULTS, f"ms_perattack_seed{s}.csv"))]
    if pframes:
        pa = pd.concat(pframes, ignore_index=True)
        pag = pa.groupby(["regime", "model", "attack_subtype"]).agg(
            recall_mean=("recall", "mean"), recall_std=("recall", "std"),
            support_mean=("support", "mean")).reset_index()
        pag.to_csv(os.path.join(RESULTS, "perattack_multiseed.csv"),
                   index=False)
    print("AGGREGATE DONE", flush=True)


def main():
    seeds = [int(x) for x in sys.argv[1:]] or SEEDS
    d19_1, f1 = load_pool("ddos2019_day1")
    d19_2, f2 = load_pool("ddos2019_day2")
    d18_e, f3 = load_pool("ids2018_early")
    d18_l, f4 = load_pool("ids2018_late")
    nd19_1, _ = load_pool("ddos2019_day1", PREP_ND)
    nd19_2, _ = load_pool("ddos2019_day2", PREP_ND)
    feats19 = sorted(set(f1) & set(f2))
    feats18 = sorted(set(f3) & set(f4))
    feats_common = sorted(set(feats19) & set(feats18))
    pools = (d19_1, d19_2, d18_e, d18_l, nd19_1, nd19_2,
             feats19, feats18, feats_common)
    print(f"features common={len(feats_common)}; nodedup pools: "
          f"day1={len(nd19_1)} day2={len(nd19_2)}", flush=True)
    for seed in seeds:
        run_seed(seed, pools)
    done = [s for s in SEEDS
            if os.path.exists(os.path.join(RESULTS, f"ms_summary_seed{s}.csv"))]
    if done:
        aggregate(done)
    with open(os.path.join(RESULTS, "run_info_v2.json"), "w") as fh:
        json.dump({"seeds_requested": SEEDS, "seeds_completed": done,
                   "b5_flood_train_subtypes": sorted(FLOOD_2019),
                   "b5_test_subtype": sorted(FLOOD_2018),
                   "note": "B1 NODEDUP pools from prepare_data_nodedup.py; "
                           "all other pools = v1 prepared (deduped)."},
                  fh, indent=2)
    print("V2 RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
