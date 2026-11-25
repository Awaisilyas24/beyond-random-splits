#!/usr/bin/env python3
"""
Run the three-regime evaluation from the approved outline (Option 1).

Paper: "Beyond Random Splits: A Leakage-Controlled, Time-Aware Evaluation of
Machine Learning DDoS Detectors for Cloud Environments"

Regimes
-------
(a) random        : pooled CIC-DDoS2019 sample, stratified 70/30 split
                    (the protocol used by most of the reviewed literature).
(b) time-ordered  : train on earlier captures, test on later captures -
                    (b1) CIC-DDoS2019 day 01-12 -> day 03-11 (the release's own
                         training/testing day design);
                    (b2) CSE-CIC-IDS2018 early days (14+20 Feb) -> later days
                         (21 Feb + 02 Mar).
(c) cross-dataset : train on pooled CIC-DDoS2019, test on pooled
                    CSE-CIC-IDS2018 (DDoS + benign), shared-feature
                    intersection only.

Leakage controls
----------------
* Exact duplicate rows (features+label) removed inside every pool beforehand
  (prepare_data.py) and again after pooling.
* For (b) and (c), test rows whose feature vector exactly matches a training
  row are removed from the test set (count reported).
* All preprocessing (median imputation, scaling) lives in an sklearn Pipeline
  fitted on the training split ONLY.

Models: Logistic Regression, Random Forest, XGBoost, MLP (sklearn
MLPClassifier - TensorFlow/Keras was not used; the outline allows either, and
the sklearn MLP keeps the study runnable on this CPU-only VM).

Metrics: per-class precision/recall/F1, macro F1, accuracy, inference latency
(ms per 1,000 flows, median of 3 runs on up to 10k test rows) and throughput.
Explainability: SHAP (TreeExplainer) for the best regime-(a) model, with
permutation importance as documented fallback.

Outputs (results/): regime_summary.csv, metrics_long.csv,
per_attack_recall.csv, shap_top_features.csv, run_info.json
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PREP = os.path.join(ROOT, "data", "prepared")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_pool(name):
    df = pd.read_csv(os.path.join(PREP, name + ".csv"))
    feats = [c for c in df.columns if c not in ("label", "subtype")]
    return df, feats


def dedup(df, feats):
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
    return df, before - len(df)


def remove_test_overlap(train, test, feats):
    """Drop test rows whose full feature vector occurs in the training set."""
    h = pd.util.hash_pandas_object(train[feats], index=False)
    train_hashes = set(h.to_numpy())
    ht = pd.util.hash_pandas_object(test[feats], index=False)
    mask = ~ht.isin(train_hashes)
    removed = int((~mask).sum())
    return test.loc[mask].reset_index(drop=True), removed


def make_models():
    imp = ("imputer", SimpleImputer(strategy="median"))
    return {
        "Logistic Regression": Pipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       random_state=SEED))]),
        "Random Forest": Pipeline([
            imp,
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                           random_state=SEED))]),
        "XGBoost": Pipeline([
            imp,
            ("clf", XGBClassifier(n_estimators=300, max_depth=6,
                                  learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.9, tree_method="hist",
                                  n_jobs=-1, eval_metric="logloss",
                                  random_state=SEED))]),
        "MLP": Pipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                  max_iter=60, early_stopping=True,
                                  random_state=SEED))]),
    }


summary_rows, long_rows, attack_rows = [], [], []
run_info = {"seed": SEED, "regimes": {}}


def evaluate(regime, Xtr, ytr, Xte, yte, subtypes_te=None):
    print(f"\n=== Regime {regime}: train={len(Xtr)} test={len(Xte)} "
          f"features={Xtr.shape[1]} ===", flush=True)
    results = {}
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pred = model.predict(Xte)
        rep = classification_report(yte, pred, output_dict=True,
                                    zero_division=0)
        acc = accuracy_score(yte, pred)
        macro_f1 = rep["macro avg"]["f1-score"]
        # latency: median of 3 runs over up to 10k test rows
        lat_sample = Xte.iloc[:10_000]
        times = []
        for _ in range(3):
            t0 = time.time()
            model.predict(lat_sample)
            times.append(time.time() - t0)
        med = float(np.median(times))
        ms_per_1000 = med / len(lat_sample) * 1e6
        throughput = len(lat_sample) / med
        results[name] = (model, pred, macro_f1)
        summary_rows.append({
            "regime": regime, "model": name, "n_train": len(Xtr),
            "n_test": len(Xte), "n_features": Xtr.shape[1],
            "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "benign_f1": round(rep["0"]["f1-score"], 4),
            "ddos_f1": round(rep["1"]["f1-score"], 4),
            "ddos_recall": round(rep["1"]["recall"], 4),
            "fit_seconds": round(fit_s, 1),
            "latency_ms_per_1000": round(ms_per_1000, 3),
            "throughput_flows_per_s": round(throughput, 1)})
        for cls in ("0", "1"):
            long_rows.append({
                "regime": regime, "model": name,
                "class": "benign" if cls == "0" else "ddos",
                "precision": round(rep[cls]["precision"], 4),
                "recall": round(rep[cls]["recall"], 4),
                "f1": round(rep[cls]["f1-score"], 4),
                "support": int(rep[cls]["support"])})
        print(f"  {name:20s} acc={acc:.4f} macroF1={macro_f1:.4f} "
              f"ddosF1={rep['1']['f1-score']:.4f} "
              f"lat={ms_per_1000:.2f} ms/1k fit={fit_s:.0f}s", flush=True)
    # per-attack recall for the best model (DDoS2019 subtypes)
    if subtypes_te is not None:
        best = max(results, key=lambda k: results[k][2])
        pred = results[best][1]
        sub = pd.DataFrame({"subtype": subtypes_te, "y": yte, "p": pred})
        for st, g in sub[sub["y"] == 1].groupby("subtype"):
            attack_rows.append({
                "regime": regime, "best_model": best, "attack_subtype": st,
                "support": int(len(g)),
                "recall": round(float((g["p"] == 1).mean()), 4)})
    return results


def xy(df, feats):
    return df[feats], df["label"].to_numpy()


def main():
    d19_1, f19_1 = load_pool("ddos2019_day1")
    d19_2, f19_2 = load_pool("ddos2019_day2")
    d18_e, f18_e = load_pool("ids2018_early")
    d18_l, f18_l = load_pool("ids2018_late")

    feats19 = sorted(set(f19_1) & set(f19_2))
    feats18 = sorted(set(f18_e) & set(f18_l))
    feats_common = sorted(set(feats19) & set(feats18))
    run_info["features"] = {"ddos2019": len(feats19), "ids2018": len(feats18),
                            "common_intersection": len(feats_common)}
    print(f"features: 2019={len(feats19)} 2018={len(feats18)} "
          f"common={len(feats_common)}", flush=True)

    pool19, dup = dedup(pd.concat([d19_1, d19_2], ignore_index=True), feats19)
    pool18, dup18 = dedup(pd.concat([d18_e, d18_l], ignore_index=True), feats18)
    run_info["pool_dedup_removed"] = {"ddos2019_pooled": int(dup),
                                      "ids2018_pooled": int(dup18)}

    # ---- Regime (a): random stratified split on pooled CIC-DDoS2019 -------
    tr, te = train_test_split(pool19, test_size=0.30, random_state=SEED,
                              stratify=pool19["label"])
    Xtr, ytr = xy(tr, feats19)
    Xte, yte = xy(te, feats19)
    res_a = evaluate("a_random_ddos2019", Xtr, ytr, Xte, yte,
                     subtypes_te=te["subtype"].to_numpy())

    # ---- Regime (b1): time-ordered within CIC-DDoS2019 (day1 -> day2) -----
    te_b, removed = remove_test_overlap(d19_1, d19_2, feats19)
    run_info["regimes"]["b1_test_overlap_removed"] = removed
    Xtr, ytr = xy(d19_1, feats19)
    Xte, yte = xy(te_b, feats19)
    evaluate("b1_time_ddos2019_day1_to_day2", Xtr, ytr, Xte, yte,
             subtypes_te=te_b["subtype"].to_numpy())

    # ---- Regime (b2): time-ordered within CSE-CIC-IDS2018 (early -> late) -
    te_b2, removed2 = remove_test_overlap(d18_e, d18_l, feats18)
    run_info["regimes"]["b2_test_overlap_removed"] = removed2
    Xtr, ytr = xy(d18_e, feats18)
    Xte, yte = xy(te_b2, feats18)
    evaluate("b2_time_ids2018_early_to_late", Xtr, ytr, Xte, yte)

    # ---- Regime (c): cross-dataset CIC-DDoS2019 -> CSE-CIC-IDS2018 --------
    te_c, removed3 = remove_test_overlap(pool19, pool18, feats_common)
    run_info["regimes"]["c_test_overlap_removed"] = removed3
    Xtr, ytr = xy(pool19, feats_common)
    Xte, yte = xy(te_c, feats_common)
    evaluate("c_cross_ddos2019_to_ids2018", Xtr, ytr, Xte, yte)

    # ---- SHAP for the best regime-(a) model --------------------------------
    best_name = max(res_a, key=lambda k: res_a[k][2])
    best_model = res_a[best_name][0]
    shap_method, top = "none", None
    try:
        import shap
        sample = Xte.iloc[:0]  # placeholder, replaced below
        Xtr_a, _ = xy(tr, feats19)
        Xte_a, _ = xy(te, feats19)
        sample = Xte_a.sample(n=min(2000, len(Xte_a)), random_state=SEED)
        imputer = best_model.named_steps["imputer"]
        Xs = pd.DataFrame(imputer.transform(sample), columns=feats19)
        clf = best_model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 0) == 3:
                sv = sv[:, :, 1]
            imp = np.abs(sv).mean(axis=0)
            shap_method = "shap.TreeExplainer"
        else:
            raise TypeError("best model is not tree-based; use fallback")
        top = pd.DataFrame({"feature": feats19,
                            "mean_abs_shap": np.round(imp, 5)})
    except Exception as exc:  # documented fallback: permutation importance
        from sklearn.inspection import permutation_importance
        Xtr_a, ytr_a = xy(tr, feats19)
        Xte_a, yte_a = xy(te, feats19)
        samp = Xte_a.sample(n=min(2000, len(Xte_a)), random_state=SEED)
        ys = yte_a[Xte_a.index.get_indexer(samp.index)] \
            if False else te.loc[samp.index, "label"].to_numpy()
        r = permutation_importance(best_model, samp, ys, n_repeats=5,
                                   random_state=SEED, n_jobs=-1)
        top = pd.DataFrame({"feature": feats19,
                            "mean_abs_shap": np.round(r.importances_mean, 5)})
        shap_method = f"permutation_importance (fallback: {type(exc).__name__})"
    if top is not None:
        top = top.sort_values("mean_abs_shap", ascending=False).head(15)
        top.to_csv(os.path.join(RESULTS, "shap_top_features.csv"),
                   index=False)
    run_info["best_model_regime_a"] = best_name
    run_info["explainability_method"] = shap_method
    print(f"\nBest regime-(a) model: {best_name}; explainability: "
          f"{shap_method}", flush=True)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS, "regime_summary.csv"), index=False)
    pd.DataFrame(long_rows).to_csv(
        os.path.join(RESULTS, "metrics_long.csv"), index=False)
    if attack_rows:
        pd.DataFrame(attack_rows).to_csv(
            os.path.join(RESULTS, "per_attack_recall.csv"), index=False)
    with open(os.path.join(RESULTS, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=2)
    print("EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()
