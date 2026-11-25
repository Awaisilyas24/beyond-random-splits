# Results Addendum — Revision Runs (Part B), REAL measured numbers

**Paper:** "Beyond Random Splits: A Leakage-Controlled, Time-Aware Evaluation of
Machine Learning DDoS Detectors for Cloud Environments" (Muhammad Awais)
**Run date:** 17 November 2026 · **Code:** `code/run_experiments_v2.py`
(validation + multi-seed), `code/prepare_data_nodedup.py` (B1 pools),
`code/make_grouped_bar_chart.py` (new Fig. 1)
**Logs:** `logs/run_v2_seed42.log`, `logs/run_v2_rest.log`,
`logs/prepare_nodedup.log`
**Tables:** `results/multiseed_per_seed.csv`, `results/multiseed_summary.csv`,
`results/dedup_ablation.csv`, `results/perattack_multiseed.csv`,
`results/ms_summary_seed<seed>.csv`, `results/ms_perattack_seed<seed>.csv`

> Every number below was measured in the runs logged above. Nothing is
> estimated or borrowed. This addendum extends `results.md` (the v1,
> seed-42 run); it does not replace it.

---

## 0. Harness validation (read first)

`run_experiments_v2.py` at seed 42 **reproduces the v1 results exactly**:
all 16 model fits (4 regimes × 4 models) match `results/regime_summary.csv`
to 4 decimal places on accuracy, macro F1, DDoS F1 and DDoS recall. The v2
harness is therefore the same pipeline with additional measurements, not a
new pipeline.

**Seeds used (B2):** $\mathcal{S} = \{42, 17, 101, 555, 2024\}$ — five seeds
(the plan allowed 5–10; five kept the total wall time to ≈65 minutes on
this VM: ≈26.4 minutes of summed fit time plus prediction, probability
and latency measurement). In regime (a) the seed drives the split and the
model initialisation; in (b1)/(b2)/(c) the partitions are fixed by
construction and the seed affects model fitting only — which is why
Logistic Regression (deterministic given the data) shows std 0.0000 in
(b1)/(b2)/(c). That is expected, not an error.

**Pool verification for B1:** the no-dedup pools were rebuilt with
`prepare_data_nodedup.py` (identical sampling/cleaning, dedup disabled).
Check: applying exact-duplicate removal to the no-dedup pools reproduces
the v1 prepared pools **exactly** (row-set equality on all 77 features +
label + subtype, all four pools). No-dedup pool sizes: CIC-DDoS2019 day 1
= 57,487 (22,274 duplicates present), day 2 = 49,469 (28,904 present);
pooled = 106,956 rows vs 55,274 deduplicated.

## 1. B2 — Multi-seed headline results (mean ± std over 5 seeds)

### Regime (a) — random stratified split, pooled CIC-DDoS2019

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall |
|---|---|---|---|---|
| Logistic Regression | 0.9880 ± 0.0002 | 0.9879 ± 0.0002 | 0.9890 ± 0.0002 | 0.9834 ± 0.0007 |
| Random Forest | 0.9988 ± 0.0001 | 0.9988 ± 0.0001 | 0.9989 ± 0.0001 | 0.9982 ± 0.0002 |
| XGBoost | 0.9993 ± 0.0001 | 0.9993 ± 0.0001 | 0.9994 ± 0.0001 | 0.9991 ± 0.0001 |
| MLP | 0.9948 ± 0.0005 | 0.9948 ± 0.0005 | 0.9953 ± 0.0005 | 0.9925 ± 0.0007 |

### Regime (b1) — CIC-DDoS2019 day 01-12 → day 03-11

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall | Δ macro F1 (per-seed, mean ± std) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9862 ± 0.0000 | 0.9860 ± 0.0000 | 0.9880 ± 0.0000 | 0.9937 ± 0.0000 | **−0.0019 ± 0.0002** |
| Random Forest | 0.8722 ± 0.0507 | 0.8720 ± 0.0504 | 0.8756 ± 0.0523 | 0.8056 ± 0.0888 | −0.1268 ± 0.0503 |
| XGBoost | 0.9685 ± 0.0065 | 0.9678 ± 0.0066 | 0.9725 ± 0.0058 | 0.9808 ± 0.0115 | −0.0315 ± 0.0066 |
| MLP | 0.9882 ± 0.0018 | **0.9880 ± 0.0018** | 0.9897 ± 0.0015 | 0.9966 ± 0.0002 | −0.0068 ± 0.0015 |

→ Confirms the A2 facts with multi-seed means: in (b1) the **highest
absolute macro F1 is the MLP's** (0.9880), and the **smallest drop is
Logistic Regression's** (−0.0019). RF's (b1) score varies strongly by seed
(std 0.050) — driven by its SYN-flood recall (§4).

### Regime (b2) — CSE-CIC-IDS2018 early → late

> **Correction (review round 2, §8):** the Δ column below was originally
> computed against regime (a) — a CIC-DDoS2019 baseline. It has been
> recomputed against (b2)'s own within-dataset baseline, the pooled
> CSE-CIC-IDS2018 random split (regime (a2), §8). All other columns are
> unchanged.

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall | Δ macro F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8628 ± 0.0000 | **0.7890 ± 0.0000** | 0.6643 ± 0.0000 | 0.4974 ± 0.0000 | **−0.2082 ± 0.0003** |
| Random Forest | 0.7271 ± 0.0000 | 0.4210 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | −0.5788 ± 0.0000 |
| XGBoost | 0.6066 ± 0.1818 | 0.5619 ± 0.1516 | 0.4370 ± 0.1326 | 0.4968 ± 0.0001 | −0.4381 ± 0.1515 |
| MLP | 0.7274 ± 0.0002 | 0.4222 ± 0.0007 | 0.0022 ± 0.0012 | 0.0011 ± 0.0006 | −0.5775 ± 0.0007 |

→ In (b2), LR has both the highest absolute macro F1 **and** the smallest
drop. XGBoost's (b2) behaviour is bimodal across seeds (accuracy std
0.18): on some seeds it partially detects the new tools, on others it
collapses toward the always-benign corner while flagging much benign
traffic (benign FPR 0.3522 ± 0.2500, §3).

### Regime (c) — cross-dataset CIC-DDoS2019 → CSE-CIC-IDS2018

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall | Δ macro F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.4961 ± 0.0000 | 0.4242 ± 0.0000 | 0.2207 ± 0.0000 | 0.2520 ± 0.0000 | −0.5637 ± 0.0002 |
| Random Forest | 0.7310 ± 0.0012 | 0.5619 ± 0.0009 | 0.2897 ± 0.0009 | 0.1937 ± 0.0000 | −0.4369 ± 0.0009 |
| XGBoost | 0.6681 ± 0.0038 | 0.4005 ± 0.0014 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | −0.5988 ± 0.0014 |
| MLP | 0.6310 ± 0.1029 | 0.3847 ± 0.0424 | 0.0000 ± 0.0001 | 0.0000 ± 0.0001 | −0.6101 ± 0.0424 |

→ The v1 collapse findings are seed-robust: XGBoost and MLP detect zero
DDoS at the 0.5 threshold on **every** seed.

## 2. B1 — Deduplication ablation (regime (a), with vs without dedup)

Paired per-seed comparison (same five seeds, same splits by seed; the
only difference is whether exact duplicates are removed). Macro F1:

| Model | With dedup | Without dedup | Paired Δ (no-dedup − dedup), mean ± std |
|---|---|---|---|
| Logistic Regression | 0.9879 ± 0.0002 | 0.9921 | **+0.0042 ± 0.0007** |
| Random Forest | 0.9988 ± 0.0001 | 0.9992 | +0.0004 ± 0.0002 |
| XGBoost | 0.9993 ± 0.0001 | 0.9995 | +0.0002 ± 0.0002 |
| MLP | 0.9948 ± 0.0005 | 0.9965 | +0.0017 ± 0.0010 |

(Full metrics incl. accuracy/DDoS F1/recall: `results/dedup_ablation.csv`.)

→ Keeping duplicates inflates the random-split score for **every** model,
but only modestly (+0.0002…+0.0042 macro F1; largest for the linear
model). Deduplication is therefore a real but **minor** part of the
literature's inflation: the bulk of it is structural to random splitting
of same-capture data, exactly as the paper's RQ1 discussion argues — now
measured rather than asserted.

## 3. B3 — ROC-AUC, PR-AUC (average precision), benign FPR (mean ± std)

All four models expose `predict_proba`, so scores exist for every model;
no model had to be excluded.

### Regime (b2)

| Model | ROC-AUC | PR-AUC | Benign FPR |
|---|---|---|---|
| Logistic Regression | 0.8830 ± 0.0000 | 0.9094 ± 0.0000 | 0.0001 ± 0.0000 |
| Random Forest | 0.5065 ± 0.0000 | 0.6344 ± 0.0002 | 0.0000 ± 0.0000 |
| XGBoost | 0.5114 ± 0.0013 | 0.6260 ± 0.0013 | 0.3522 ± 0.2500 |
| MLP | 0.8721 ± 0.0205 | 0.8775 ± 0.0512 | 0.0000 ± 0.0000 |

→ Notable: LR and the MLP retain ranking signal on the unseen tools
(ROC-AUC 0.87–0.88) even where their 0.5-threshold recall collapses
(MLP recall 0.0011) — a threshold/calibration failure, not a total
representation failure. RF ranks at chance (0.5065).

### Regime (c)

| Model | ROC-AUC | PR-AUC | Benign FPR |
|---|---|---|---|
| Logistic Regression | 0.5476 ± 0.0000 | 0.2725 ± 0.0000 | 0.4075 ± 0.0000 |
| Random Forest | 0.4112 ± 0.0220 | 0.2876 ± 0.0079 | 0.0568 ± 0.0017 |
| XGBoost | 0.4454 ± 0.0288 | 0.2715 ± 0.0245 | 0.0680 ± 0.0053 |
| MLP | 0.7012 ± 0.1586 | 0.4009 ± 0.0954 | 0.1199 ± 0.1436 |

→ Cross-dataset, RF and XGBoost rank **worse than chance** (ROC-AUC
0.41–0.45): their scores are anti-informative on the other capture. LR's
partial detection is bought with a 40.75% benign false-positive rate.

(Regimes (a)/(b1) AUC/FPR are in `results/multiseed_summary.csv`; (a)
ROC-AUC ≥ 0.9982 for all models.)

## 4. B4 — Per-attack recall, all models (mean over 5 seeds)

Subtype spelling normalised: 'UDP-lag' (day-1 files) and 'UDPLag'
(day-2 files) are one attack (see `perattack_multiseed.csv`).

### Regime (b1) — recall by attack subtype

| Subtype | Support | LR | RF | XGBoost | MLP |
|---|---|---|---|---|---|
| LDAP | 409 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MSSQL | 2,527 | 0.9996 | 0.9986 | 0.9998 | 0.9996 |
| NetBIOS | 237 | 1.0000 | 1.0000 | 0.9814 | 1.0000 |
| Portmap | 222 | 0.9550 | 0.9577 | 0.9739 | 0.9622 |
| **Syn** | 4,211 | 0.9957 | **0.4852** | 0.9558 | 1.0000 |
| UDP | 3,597 | 0.9997 | 0.9997 | 0.9993 | 0.9997 |
| UDPLag | 205 | 0.7951 | 0.8244 | 0.9005 | 0.8615 |

→ RF's (b1) collapse is localised: it misses **half of the day-2 SYN
flood** (recall 0.4852) while matching the other models on the remaining
subtypes. The SYN flood is the highest-volume day-2 subtype, so this one
failure drives RF's aggregate drop — and its seed variance (whether a
given forest catches the shifted SYN signature varies by seed).

### Regime (a) — XGBoost (best model), merged UDPLag

Recall 0.9915–1.0000 across all 11 subtypes (support-weighted; full
table in `results/perattack_multiseed.csv`). Weakest: WebDDoS
0.9756 ± 0.0171 (support ≈ 65), Portmap 0.9915 ± 0.0078.

### Regime (c) — recall by CSE-CIC-IDS2018 subtype (context for B5/A7)

| Subtype | LR | RF | XGBoost | MLP |
|---|---|---|---|---|
| HOIC | 0.4385 | 0.4378 | 0.0000 | 0.0000 |
| LOIC-UDP | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| LOIC-HTTP | 0.0006 | 0.0000 | 0.0000 | 0.0001 |

## 5. B5 — Cross-dataset flood control (DONE, two variants)

**Design.** The label mapping is clean for UDP-family floods:
CIC-DDoS2019 contains SYN floods (`Syn`), UDP floods (`UDP`) and UDP
reflection (`DrDoS_UDP`); CSE-CIC-IDS2018's flood-type DDoS in our pools
is LOIC-UDP (`DDOS attack-LOIC-UDP`, late pool). Two controls:

- **(i) Flood-test:** the regime-(c) models, evaluated on the test subset
  restricted to benign + LOIC-UDP flows (77,716 flows).
- **(ii) Flood-train:** models trained on CIC-DDoS2019 benign + flood
  subtypes only (Syn, UDP, DrDoS_UDP; 38,777 flows), tested on
  CSE-CIC-IDS2018 benign + LOIC-UDP (77,716 flows, after overlap removal).

### Results (mean ± std over 5 seeds)

(i) Flood-test — DDoS recall / DDoS F1 / benign FPR:

| Model | Recall | DDoS F1 | Benign FPR | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 1.0000 | 0.1005 | 0.4075 | 0.8862 |
| Random Forest | 0.0000 | 0.0000 | 0.0568 | 0.3302 |
| XGBoost | 0.0000 | 0.0000 | 0.0680 | 0.9165 |
| MLP | 0.0000 | 0.0000 | 0.1199 | 0.0561 |

(ii) Flood-train — DDoS recall / DDoS F1 / ROC-AUC:

| Model | Recall | DDoS F1 | ROC-AUC | Benign FPR |
|---|---|---|---|---|
| Logistic Regression | 0.0000 | 0.0000 | 0.0045 | 0.3896 |
| Random Forest | 0.0000 | 0.0000 | 0.9004 | 0.0247 |
| XGBoost | 0.0000 | 0.0000 | 0.8981 | 0.0828 |
| MLP | 0.0000 | 0.0000 | 0.0098 | 0.3057 |

→ **The control does not rescue transfer.** Even when training *and*
testing are restricted to flood-type traffic, no model detects a single
LOIC-UDP flow at the 0.5 threshold, on any seed. RF/XGBoost retain
ranking signal (ROC-AUC ≈ 0.90) but their 2019-calibrated scores sit
below threshold on 2018 flows; LR/MLP scores are near-perfectly
anti-ranked (ROC-AUC 0.005–0.010) in variant (ii). Conclusion supported
by the control: the regime-(c) failure is **not** explained by the
attack-type mix alone; the dataset shift itself (capture environment,
background benign profile, feature distributions) breaks threshold
transfer. Caveat: the control removes attack-type shift from the
*attack* side only — benign traffic still comes from different captures,
so the two shifts cannot be fully separated with these benchmarks.

## 6. Latency (regime (a), median over the 5 seeds)

Per-seed latency on the shared VM is noisy (occasional 3–5× outliers),
so the paper reports medians across seeds:

| Model | Latency (ms / 1k flows) | Throughput (flows/s) |
|---|---|---|
| Logistic Regression | 1.02 | 980,894 |
| Random Forest | 12.51 | 79,937 |
| XGBoost | 8.37 | 119,455 |
| MLP | 8.59 | 116,366 |

Ranking is unchanged from v1: LR fastest by an order of magnitude, RF
slowest.

## 7. What changed relative to v1 (results.md)

- All headline numbers are now 5-seed means ± std; seed-42 values in
  `results.md` remain valid as the seed-42 instance.
- Two v1 point values move slightly under averaging: RF (b1) macro F1
  0.8728 → 0.8720 ± 0.0504; XGBoost (b2) macro F1 0.4514 → 0.5619 ± 0.1516
  (seed 42 was one of its collapse seeds); MLP (c) macro F1
  0.4031 → 0.3847 ± 0.0424. No qualitative conclusion changes.
- New measurements: dedup ablation (B1), ROC-AUC/PR-AUC/benign FPR (B3),
  per-attack recall for all models (B4), flood controls (B5).

## 8. Review round 2 (17 November 2026, later) — baseline fix + code checks

### 8.1 New run: (a2) within-dataset baseline for regime (b2)

**Why:** §1's original (b2) Δ values were computed against regime (a),
which is a CIC-DDoS2019 measurement — the wrong reference for a
CSE-CIC-IDS2018 regime. **Code:** `code/run_baseline_ids2018.py`
(reuses the v2 harness verbatim). **Design:** pooled CSE-CIC-IDS2018
(early + late pools concatenated and deduplicated exactly as the v2
harness builds its pool18; 105,997 rows), random stratified 70/30 split
(train 74,197 / test 31,800), same 77-feature space (feats18), same 4
models, same 5 seeds. **Log:** `logs/run_baseline_ids2018.log`.
**Tables:** `results/baseline_ids2018_per_seed.csv`,
`results/baseline_ids2018_summary.csv`, `results/b2_deltas_corrected.csv`.

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall |
|---|---|---|---|---|
| Logistic Regression | 0.9977 ± 0.0003 | 0.9972 ± 0.0003 | 0.9960 ± 0.0004 | 0.9991 ± 0.0003 |
| Random Forest | 0.9998 ± 0.0000 | 0.9998 ± 0.0000 | 0.9997 ± 0.0001 | 0.9998 ± 0.0001 |
| XGBoost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| MLP | 0.9997 ± 0.0001 | 0.9997 ± 0.0001 | 0.9996 ± 0.0002 | 0.9995 ± 0.0004 |

**Corrected (b2) Δ macro F1** (paired per-seed, (b2) − (a2); old
wrong-baseline values in brackets): LR **−0.2082 ± 0.0003** [−0.1989];
RF **−0.5788 ± 0.0000** [−0.5778]; XGBoost **−0.4381 ± 0.1515**
[−0.4374]; MLP **−0.5775 ± 0.0007** [−0.5726]. Conclusions unchanged:
LR still shows the smallest (b2) drop, and the largest time-ordered
drop is still RF's ≈ 0.58, so the abstract's "falls by up to 0.58"
stands — now against a legitimate baseline. (b1) and (c) Δ values are
unchanged: both regimes train on CIC-DDoS2019, so regime (a) is their
correct baseline. Total evaluation is now 140 fits; summed fit time
≈ 31 minutes.

### 8.2 Check #7 — PR-AUC computation: code is CORRECT (no bug)

Suspicion: Table X (b2) RF PR-AUC 0.6344 with ROC-AUC 0.5065 at attack
prevalence 0.2729 resembles the trapezoid value (1 + 0.2729)/2 ≈ 0.636
of `auc(recall, precision)` rather than `average_precision_score`.
**Evidence:** (i) `run_experiments_v2.py` computes
`"pr_auc": average_precision_score(yte, proba)` (metric_row). (ii)
Hand-check (`code/check7_pr_auc.py`): refitting regime (b2) at seed 42
with the harness's own functions reproduces the saved
`ms_summary_seed42.csv` values **exactly** under
`average_precision_score` for all four models (LR 0.909429,
RF 0.634589, XGBoost 0.623596, MLP 0.906168; differences +0.000000),
while the trapezoid `auc(recall, precision)` gives **different, lower**
values (RF 0.606792, XGBoost 0.604640). The saved numbers are therefore
average precision, not the trapezoid area. **Why the value is
legitimately high:** RF's ranking in (b2) places its top 3,753 test
flows — all of them attacks — above every benign flow (precision
1.0000 over the first quartile of recall; precision 1.0000 at recall
0.10 as well) before collapsing (precision 0.158 at recall 0.50).
Average precision credits that perfectly ranked top segment; ROC-AUC,
averaging over all attack–benign pairs, stays at chance. RF emits only
36 distinct scores on this test set (largest tie block: 17,626 flows at
0.140). A clarifying sentence has been added to Section V-H of the
paper. **No numbers changed.**

### 8.3 Check #8 — the duplicated 0.5619: genuine coincidence

Table V's XGBoost (b2) macro F1 and Table VI's Random Forest (c) macro
F1 both print as 0.5619. Per-seed values from
`results/multiseed_per_seed.csv`:

- XGBoost (b2): {42: 0.451412, 17: 0.727747, 101: 0.463118, 555:
  0.439697, 2024: 0.727660} → mean 0.561927, std 0.151559 (bimodal).
- Random Forest (c): {42: 0.561567, 17: 0.561434, 101: 0.563257, 555:
  0.562267, 2024: 0.561001} → mean 0.561905, std 0.000882 (very stable).

The unrounded means differ (0.561927 vs 0.561905) and the per-seed
values and standard deviations are entirely different; the two means
merely round to the same 4-decimal string. **No copy-paste error; no
numbers changed.**

## 9. Review round 3 (17 November 2026, final) — interpretation correction + CCGrid fixes

A second outside review of the compressed CCGrid version challenged the
round-2 V-B framing. The challenge was checked against
`perattack_multiseed.csv` and the reviewer is correct.

### 9.1 Corrected interpretation: RF's day-2 SYN failure is model-specific, not "within-type drift"

Round 2 (§8) rewrote V-B to describe Random Forest's day-2 SYN-flood
collapse as "within-type drift". The per-attack numbers do not support
that reading. Day-2 SYN recalls in regime (b1) (support 4,211 flows):

| Model | SYN recall (mean ± std) |
|---|---|
| Logistic Regression | 0.9957 ± 0.0000 |
| MLP | 1.0000 ± 0.0000 |
| XGBoost | 0.9558 ± 0.0301 |
| Random Forest | 0.4852 ± 0.2404 |

The same day-2 SYN flows are detected by the other three families, so
the traffic has not become undetectable: RF's failure is a
model-dependent sensitivity (RF-specific instability), not drift of the
attack type itself. V-B was rewritten accordingly in both versions.
UDPLag's day-2 support is only 205 flows, so no claim rests on it.

### 9.2 Other round-3 changes (both versions unless noted)

- **Limitations:** an explicit "Confounded shifts" statement was added —
  in (b1) the time shift is confounded with the arrival of new attack
  types; in (b2) the early→late shift also changes the attack tool
  (LOIC-HTTP → HOIC/LOIC-UDP), so (b2) is in part a tool shift.
- **RF (b2) ranking wording:** "ranks at chance" replaced by the precise
  statement that RF's ROC-AUC is near 0.5 (0.5065) as a mixture of a
  well-ranked segment (its top 3,753 ranked test flows are all attacks)
  and an anti-ranked remainder; the PR-AUC explanation was kept
  consistent with this.
- **Style neutralisation:** "uncomfortable" → "stark"; "honest regimes"
  → "stricter regimes"; "credibility will be won" → "credibility is
  determined".
- **VI-A hedge (CCGrid only):** "plausibly" restored on the sentence
  attributing the larger (b2)/(c) falls to the attack-tool change (the
  main version already carried the hedge).
- **Fig. 1 regenerated** with the (a2) pooled CSE-CIC-IDS2018
  random-split baseline group included (5 groups × 4 models), since the
  (b2) deltas are computed against (a2); captions updated (random-split
  cluster is now 0.988–1.000). The CCGrid caption's duplicated range
  "Tables II–II" was replaced by the single merged-table reference.
- **CCGrid cross-reference fix:** the flood-control reference in regime
  (c) is Section V-I; a line break after the hyphen had made it read
  "(Section VI)" — the reference is now set unbreakable, and a full-PDF
  scan for broken ranges/references is clean.
- **2-page summary** corrected to the same RF-specific-instability
  framing (still exactly 2 pages).

### 9.3 Page counts after round 3

- Main paper: **12 pages** (was 11; growth is the added Limitations text
  and the V-B rewrite — no results were cut).
- CCGrid version: **exactly 10 pages**, letter, 0 undefined references,
  anonymity greps (author name / city / university / GitHub / Gmail)
  all zero. Space was recovered by prose tightening, figure sizing and
  table row pitch (arraystretch 0.78) — no results, tables or
  references were removed.

**No measured number was changed in this round.**
