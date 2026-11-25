# Beyond Random Splits — experiment code and data provenance

Companion code/data release for the paper:

**M. Awais, "Beyond Random Splits: A Leakage-Controlled, Time-Aware
Evaluation of Machine Learning DDoS Detectors for Cloud Environments."**

Department of Information Technology, Government College University
Faisalabad, Pakistan.

This package contains everything needed to reproduce the paper's results:
the download, preparation and experiment code; the exact dataset sources
and per-file download manifest; the run logs; and all result tables
(single-seed v1 run and the five-seed revision runs). The datasets
themselves are **not** redistributed here — obtain them from the original
sources listed in `data/SOURCES.md` (CSE-CIC-IDS2018: AWS Open Data
Registry; CIC-DDoS2019: UNB CIC, or the public Hugging Face mirror used
here, see below).

## Layout

- `code/download_data.py` — downloads the benchmark CSVs (whole files, or
  10 × 8 MB byte-range slices for files > ~410 MB; see `data/SOURCES.md`).
- `code/prepare_data.py` — builds the four cleaned data pools
  (77 common CICFlowMeter features, identifiers removed, exact duplicates
  removed, counts reported) → `data/prepared/*.csv`, and writes
  `results/dataset_inspection.json`.
- `code/prepare_data_nodedup.py` — identical to `prepare_data.py` except
  duplicate removal is disabled (used for the dedup ablation, B1) →
  `data/prepared_nodedup/*.csv`.
- `code/run_experiments.py` — the original single-seed (42) three-regime
  evaluation + SHAP analysis → `results/regime_summary.csv` etc.
- `code/run_experiments_v2.py` — the revision runs: five seeds
  (42, 17, 101, 555, 2024), dedup ablation, ROC-AUC / PR-AUC / benign FPR,
  per-attack recall for all models, and the cross-dataset flood control →
  `results/multiseed_*.csv`, `results/dedup_ablation.csv`,
  `results/perattack_multiseed.csv`.
- `data/SOURCES.md`, `data/download_manifest.json` — exact URLs, file
  sizes and download modes.
- `logs/` — full logs of every run.
- `results/` — all result tables (CSV/JSON) reported in the paper and in
  `results.md` / `results_addendum.md`.
- `results.md` — the original run's full account (datasets, sampling,
  pools, numbers, deviations, limitations).
- `results_addendum.md` — the revision runs' methods and numbers.
- `requirements.txt` — pinned Python environment.

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python code/download_data.py          # ~2.2 GB into data/
python code/prepare_data.py           # cleaned pools (with dedup)
python code/prepare_data_nodedup.py   # pools without dedup (ablation)
python code/run_experiments.py        # v1, seed 42 (incl. SHAP)
python code/run_experiments_v2.py     # v2, all five seeds + ablation
```

Expected environment of the original runs: Linux, Python 3.12.3,
scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1, pandas 3.1.1, NumPy 2.5.5
(full pins in `requirements.txt`). The v1 seed-42 numbers in
`results/regime_summary.csv` are reproduced exactly by
`run_experiments_v2.py` at seed 42 (verified).

## Data notes (read before citing the numbers)

- Files larger than ~410 MB were **sampled**, not downloaded whole:
  10 evenly spaced 8 MB HTTP byte-range slices, complete CSV lines only —
  a position-stratified sample of the file's genuine rows. Per-pool
  collection was capped (60,000 benign; 15,000 per attack subtype) and
  each pool sampled (seed 42) to at most 40,000 rows per class before
  duplicate removal. Full details in `results.md` §2–3.
- CIC-DDoS2019 was obtained from the public Hugging Face mirror
  `baalajimaestro/CICDDoS2019` because UNB's download page is form-gated.
  Verification performed: every file carries the CICFlowMeter header
  (88 columns incl. `Flow ID`, `Source IP`, `Timestamp`, `Label`); every
  label observed in the mirrored files appears in the attack taxonomy
  published by UNB for CIC-DDoS2019, and each day's label set is
  consistent with the per-day attack lists UNB publishes for the two
  capture days.
- CSE-CIC-IDS2018 came from the official AWS Open Data bucket.

## License / citation

Code: released for reproducibility of the paper's results; if you use it,
please cite the paper. Datasets remain under their original terms
(CIC-DDoS2019 / CSE-CIC-IDS2018: Canadian Institute for Cybersecurity,
University of New Brunswick — citation of the dataset papers required).
