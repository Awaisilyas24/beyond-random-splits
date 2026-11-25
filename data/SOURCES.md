# Dataset sources (exact URLs)

Downloaded 2026-11-16 by `code/download_data.py`. Full per-file details
(URL, mode, local size) are in `download_manifest.json` next to this file.

## CSE-CIC-IDS2018
- Registry: AWS Open Data Registry, `registry.opendata.aws/cse-cic-ids2018`
- Bucket endpoint: `https://cse-cic-ids2018.s3.amazonaws.com/`
- Folder: `Processed Traffic Data for ML Algorithms/` (CICFlowMeter CSVs)
- Files used:
  - `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` (358 MB, whole) — benign pool (day also contains FTP/SSH brute force, excluded from the binary task)
  - `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` (4,054 MB, **byte-range sliced sample**: 10 × 8 MB evenly spaced slices, complete lines only) — DDoS LOIC-HTTP day
  - `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` (328 MB, whole) — DDoS LOIC-UDP + HOIC day
  - `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` (352 MB, whole) — benign pool (day also contains Botnet, excluded)

## CIC-DDoS2019
- UNB CIC's official download requires a request form; this study uses the
  public Hugging Face mirror `baalajimaestro/CICDDoS2019`
  (`https://huggingface.co/datasets/baalajimaestro/CICDDoS2019`), whose layout
  (capture days `01-12` / `03-11`, one CSV per attack) matches the official
  release. Every file was sanity-checked in `prepare_data.py` against the
  known CICFlowMeter format (~85 columns incl. `Flow ID`, `Source IP`,
  `Timestamp`, `Flow Duration`, ..., `Label`; labels BENIGN + the file's
  attack class) — see `results/dataset_inspection.json`.
- Base URL: `https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/main/`
- Files used (whole ≤ ~410 MB, otherwise 10 × 8 MB byte-range slices):
  - Day 01-12 (release "training day"): `UDPLag.csv` (whole), `Syn.csv`,
    `DrDoS_NTP.csv`, `DrDoS_UDP.csv`, `DrDoS_DNS.csv` (sliced)
  - Day 03-11 (release "testing day"): `Portmap.csv`, `UDPLag.csv` (whole),
    `Syn.csv`, `UDP.csv`, `LDAP.csv` (sliced)

## Sampling note (important, also stated in results.md)
Files above ~410 MB were not downloaded in full. For each such file, 10
evenly spaced 8 MB byte-range slices were fetched and only complete CSV lines
kept, giving a position-stratified sample of that file's genuine rows
(~150k–215k rows per sliced file before class caps). No row was generated,
altered, or substituted; sampling is over real rows of the real files only.
