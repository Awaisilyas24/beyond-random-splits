# Experiment Results — REAL measured numbers

**Paper:** "Beyond Random Splits: A Leakage-Controlled, Time-Aware Evaluation of
Machine Learning DDoS Detectors for Cloud Environments" (Muhammad Awais)
**Run date:** 16 November 2026 · **Seed:** 42 everywhere · **Task:** binary
(benign vs DDoS)
**Code:** `code/download_data.py`, `code/prepare_data.py`,
`code/run_experiments.py` · **Logs:** `logs/` · **Raw tables:** `results/`

> Every number below was measured in the run logged in `logs/run.log`.
> Nothing is estimated, borrowed from other papers, or filled in.

---

## 1. Datasets actually obtained

| Dataset | Source (exact) | Files | How obtained |
| --- | --- | --- | --- |
| CSE-CIC-IDS2018 | AWS Open Data Registry bucket `cse-cic-ids2018`, folder "Processed Traffic Data for ML Algorithms" — `https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/<file>&#32;` | 14-02 (358 MB), 20-02 (4,054 MB), 21-02 (328 MB), 02-03 (352 MB) | 3 files whole; the 4 GB 20-02 file via byte-range slices (see §2) |
| CIC-DDoS2019 | Hugging Face mirror `baalajimaestro/CICDDoS2019` — `https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/main/<path>` (UNB's own page requires a request form; the mirror's layout matches the official release and every file passed the format + documented-label verification in `results/dataset_inspection.json`) | Day 01-12: UDPLag, Syn, DrDoS\_NTP, DrDoS\_UDP, DrDoS\_DNS · Day 03-11: Portmap, UDPLag, Syn, UDP, LDAP | Files ≤ \~410 MB whole; larger files via byte-range slices (see §2) |

Full per-file URLs and sizes: `data/download_manifest.json`, `data/SOURCES.md`.

**Verification performed:** every file has the CICFlowMeter header (88 columns
for CIC-DDoS2019 incl. `Flow ID`, `Source IP`, `Timestamp`, `Label`; 80/84 for
CSE-CIC-IDS2018 in its abbreviated-name variant), and every label observed
belongs to the dataset's documented label set (the one initially-unexpected
pair, `DDOS attack-HOIC` / `DDOS attack-LOIC-UDP`, is the 21-02 file's own
capitalisation and is recorded in the inspection file).

## 2. Sampling (documented, over real rows only)

Files larger than \~410 MB were **not** downloaded whole. For each, 10 evenly
spaced 8 MB HTTP byte-range slices were fetched and only complete CSV lines
kept — a position-stratified sample of the file's genuine rows (\~150k–215k
rows per sliced file). No row was generated or altered. During preparation,
collection was capped per pool at 60,000 benign rows and 15,000 rows \*\*per
attack subtype\*\*, then sampled (seed 42) to at most 40,000 per class per pool
before duplicate removal.

## 3. Pools after cleaning (Table 1 of the paper)

Features: **77 CICFlowMeter features common to both datasets** (CSE-CIC-IDS2018's
abbreviated names were renamed to the canonical ones; identifiers — Flow ID,
IPs, ports, Timestamp — excluded, as were CIC-DDoS2019-only extras
`SimillarHTTP`, `Inbound`, the duplicated `Fwd Header Length.1`, `Unnamed: 0`).

| Pool | Rows | Benign | DDoS | Duplicates removed | Attack subtypes (distinct rows after dedup) |
| --- | --- | --- | --- | --- | --- |
| CIC-DDoS2019 day 01-12 | 35,213 | 16,337 | 18,876 | 22,274 | DrDoS\_NTP 7,767 · DrDoS\_UDP 4,417 · UDP-lag 3,447 · DrDoS\_DNS 1,693 · Syn 1,333 · WebDDoS 219 |
| CIC-DDoS2019 day 03-11 | 20,565 | 8,677 | 11,888 | 28,904 | Syn 4,211 · UDP 3,802 · MSSQL 2,626 · LDAP 569 · NetBIOS 238 · Portmap 225 · UDPLag 217 |
| CSE-CIC-IDS2018 early (14+20 Feb) | 50,997 | 35,997 | 15,000 | 4,003 | DDoS attacks-LOIC-HTTP 15,000 |
| CSE-CIC-IDS2018 late (21 Feb+02 Mar) | 55,003 | 39,992 | 15,011 | 1,727 | DDOS attack-HOIC 13,281 · DDOS attack-LOIC-UDP 1,730 |

**Duplication finding (real, and central to the paper):** after identifier
removal, 22k–29k of the sampled CIC-DDoS2019 rows per day were exact duplicate
feature vectors — attack tooling emits near-identical flows. Any evaluation
that skips deduplication leaks these copies across train/test.

Additional leakage control in regimes (b)/(c): test rows whose feature vector
exactly matched a training row were removed — 504 rows in (b1), 0 in (b2)/(c).

## 4. Environment

Linux VM, 4 vCPU visible (1.5 shared-CPU cell), \~6 GB RAM cell.
Python 3.12.3; scikit-learn 1.9.1; XGBoost 3.4.1; SHAP 0.54.1; pandas 3.1.1;
numpy 2.5.5 (venv; full pins in `requirements.txt`). Total experiment runtime
≈ 3.5 minutes for all 16 model fits.

Models: Logistic Regression (median-impute + scale, max\_iter 1000);
Random Forest (200 trees); XGBoost (300 rounds, depth 6, hist);
MLP = **sklearn MLPClassifier** (128, 64), early stopping — see §7.
All preprocessing inside sklearn Pipelines fitted on training data only.

## 5. Results

### Regime (a) — random stratified split, pooled CIC-DDoS2019 (train 38,691 / test 16,583)

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall | Latency (ms/1k flows) | Throughput (flows/s) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9878 | 0.9877 | 0.9888 | 0.9836 | 0.93 | 1,078,366 |
| Random Forest | 0.9987 | 0.9987 | 0.9988 | 0.9981 | 11.74 | 85,170 |
| **XGBoost** | **0.9993** | **0.9993** | **0.9994** | 0.9989 | 8.15 | 122,769 |
| MLP | 0.9943 | 0.9942 | 0.9948 | 0.9932 | 7.87 | 127,053 |

→ Reproduces the literature's 99–100% claims even after deduplication.

### Regime (b1) — time-ordered, CIC-DDoS2019 day 01-12 → day 03-11 (train 35,213 / test 20,061)

| Model | Accuracy | Macro F1 | DDoS F1 | Δ Macro F1 vs (a) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.9862 | 0.9860 | 0.9880 | −0.002 |
| Random Forest | 0.8730 | 0.8728 | 0.8785 | **−0.126** |
| XGBoost | 0.9654 | 0.9647 | 0.9698 | −0.035 |
| MLP | 0.9885 | 0.9882 | 0.9899 | −0.006 |

### Regime (b2) — time-ordered, CSE-CIC-IDS2018 early → late days (train 50,997 / test 55,003)

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.8628 | 0.7890 | 0.6643 | 0.4974 |
| Random Forest | 0.7271 | 0.4210 | **0.0000** | **0.0000** |
| XGBoost | 0.4740 | 0.4514 | 0.3401 | 0.4967 |
| MLP | 0.7275 | 0.4224 | 0.0027 | 0.0013 |

→ Trained on LOIC-HTTP (20 Feb), Random Forest detects **none** of the later
HOIC/LOIC-UDP traffic (21 Feb); MLP detects 0.13% of it. (Accuracy stays
deceptively high because the test pool is \~73% benign — exactly why the paper
reports per-class metrics, not accuracy alone.)

### Regime (c) — cross-dataset, CIC-DDoS2019 → CSE-CIC-IDS2018 (train 55,274 / test 105,997)

| Model | Accuracy | Macro F1 | DDoS F1 | DDoS recall |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.4961 | 0.4242 | 0.2207 | 0.2520 |
| Random Forest | 0.7306 | 0.5616 | 0.2894 | 0.1937 |
| XGBoost | 0.6636 | 0.3989 | **0.0000** | **0.0000** |
| MLP | 0.6754 | 0.4031 | **0.0000** | **0.0000** |

→ Cross-dataset transfer collapses: the two best random-split models detect
**zero** DDoS flows in the other dataset at the default threshold.

### Per-attack recall (best model per regime; full table in `results/per_attack_recall.csv`)

- (a) XGBoost: recall 0.949–1.000 across all 12 attack subtypes present.
- (b1) MLP: Syn 1.000, MSSQL 0.9988, UDP 0.9997, LDAP 1.000, NetBIOS 1.000,
  Portmap 0.9595, UDPLag 0.8537 (weakest).

### Explainability (SHAP, TreeExplainer on the regime-(a) best model, XGBoost)

Top features by mean |SHAP|: **Bwd Packets/s (2.18)**, \*\*Min Packet Length
(1.65)**, \*\*URG Flag Count (1.22)**, ACK Flag Count (1.00),
Init\_Win\_bytes\_forward (0.78), min\_seg\_size\_forward (0.61), Packet Length Std
(0.60), Fwd Packet Length Min (0.52), Average Packet Size (0.41), Flow Bytes/s
(0.39). Full top-15: `results/shap_top_features.csv`; chart:
`results/shap_bar.png`. Degradation chart: `results/degradation_chart.png`.

## 6. Headline story (what the real numbers say)

Random-split evaluation reproduces the field's 99–100% (XGBoost macro F1
0.9993). The same models degrade sharply under honest conditions: Random
Forest loses 12.6 macro-F1 points across capture days within CIC-DDoS2019;
on CSE-CIC-IDS2018's later days, RF and MLP effectively stop detecting DDoS
at all (recall 0.000 / 0.001); and in cross-dataset transfer, XGBoost and MLP
detect zero attacks (DDoS F1 = 0.0000) despite 0.999/0.994 F1 at home.
Logistic Regression — the weakest model under random splits — is paradoxically
the most robust under time-ordered evaluation within each dataset.

## 7. Deviations from the outline plan (each with its reason)

1. **MLP implementation:** sklearn `MLPClassifier` instead of Keras/TensorFlow
   (the outline allowed either). Reason: CPU-only VM; sklearn keeps the study
   reproducible here. Architecture (128, 64) and early stopping documented.
2. **Regime (b) run twice:** (b1) CIC-DDoS2019 day1→day2 (the outline's
   design) **plus** (b2) CSE-CIC-IDS2018 early→late days, which the dataset's
   multi-day structure makes possible and which produced the study's starkest
   result. (b2) is additional evidence, not a substitution.
3. **Large files slice-sampled** instead of fully downloaded (§2). The 4 GB
   CSE-CIC-IDS2018 20-02 file and the 0.6–2.4 GB CIC-DDoS2019 files were
   position-stratified byte-range samples of the genuine files.
4. **CIC-DDoS2019 source:** Hugging Face mirror (verified) instead of UNB's
   form-gated download. CSE-CIC-IDS2018 came from the official AWS bucket.
5. **Binary task as the core** (per plan fallback): multiclass across datasets
   is not alignable (attack taxonomies differ); per-attack recall breakdowns
   are reported instead for regimes (a)/(b1).
6. **One feature set (77) for all regimes** — the cross-dataset intersection —
   so regimes are directly comparable; CIC-DDoS2019-only columns excluded.
7. **Preparation fixes during the run** (all logged): v1 of the prep script
   wrongly rejected CSE-CIC-IDS2018's abbreviated column names, missed the
   `UDP-lag` label spelling, let one attack type monopolise the collection
   cap, and (after the label fix) missed the `DDOS` capitalisation in the
   21-02 file. Each was caught by the inspection step and corrected before
   any model was trained; v1 pool outputs were discarded and regenerated.

## 8. What did NOT happen / limitations of this run

- No live testbed, no drift adaptation, no adversarial testing (future work,
  as the outline states).
- Results rest on sampled subsets of the benchmarks (caps in §2), not the
  full \~50M-flow releases; within-file sampling was position-stratified, not
  uniform-random over the full files for the whole-downloaded ones (those
  were used in full before class caps).
- Latency/throughput are single-machine CPU measurements on this VM.
- CIC-IDS2017 was not needed: CIC-DDoS2019 was obtained, so no dataset
  substitution took place.