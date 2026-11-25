#!/usr/bin/env python3
"""
NO-DEDUP VARIANT of prepare_data.py (v2) for the B1 ablation:
identical sampling/cleaning, but exact-duplicate removal is DISABLED.

Fixes relative to v1 (see results.md for the full account):
  * CSE-CIC-IDS2018's "Processed Traffic Data for ML Algorithms" CSVs use
    ABBREVIATED CICFlowMeter column names ('Tot Fwd Pkts', 'Flow Byts/s', ...)
    and one file (20-02) additionally carries Flow ID / IP columns. v1's
    format check expected the long names and wrongly rejected all four files.
    v2 renames the abbreviated names to the canonical CIC-DDoS2019 names
    (RENAME_2018, built from the actual file headers) and validates labels
    against each dataset's documented label set.
  * CIC-DDoS2019 spells one class 'UDP-lag' (not 'UDPLag'); v1's name list
    missed it and silently dropped 366k attack rows. v2 uses the dataset's
    own semantics: CIC-DDoS2019 contains ONLY DDoS attacks + benign, so
    binary = benign iff label == BENIGN. For CSE-CIC-IDS2018 (many attack
    families) only labels containing 'DDoS' count as DDoS here.
  * v1 collected attacks into one pool-level cap, so the first file's attack
    type monopolised the cap. v2 caps collection PER ATTACK SUBTYPE.

Feature set: the 77 CICFlowMeter features common to both datasets after
renaming (identifiers - Flow ID, IPs, ports, Timestamp - excluded, as are
CIC-DDoS2019-only extras 'SimillarHTTP', 'Inbound', the duplicated
'Fwd Header Length.1' column and the 'Unnamed: 0' row index). Using one
feature set for every regime keeps the three regimes comparable.

Cleaning per the approved protocol: exact duplicate rows (features+label)
dropped inside each pool, counts reported. NOTE (a real property of these
benchmarks): after identifier removal, large fractions of attack flows
collapse to identical feature vectors - the drop counts are reported, not
hidden.

Outputs: data/prepared/<pool>.csv, results/dataset_inspection.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
BENIGN_COLLECT_CAP = 60_000     # per pool
SUBTYPE_COLLECT_CAP = 15_000    # per pool, per attack subtype
FINAL_CAP_PER_CLASS = 40_000    # per pool, per binary class, before dedup
CHUNK = 200_000

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
PREP = os.path.join(DATA, "prepared_nodedup")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(PREP, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP",
           "Source Port", "Destination Port", "Src Port", "Dst Port",
           "Timestamp", "Unnamed: 0", "SimillarHTTP", "Inbound",
           "Fwd Header Length.1"]

RENAME_2018 = {
    "Dst Port": "Destination Port", "Src IP": "Source IP",
    "Src Port": "Source Port", "Dst IP": "Destination IP",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total", "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length", "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean", "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count", "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

DDOS2019_LABELS = {"BENIGN", "DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL",
                   "DrDoS_NTP", "DrDoS_NetBIOS", "DrDoS_SNMP", "DrDoS_SSDP",
                   "DrDoS_UDP", "Syn", "TFTP", "UDPLag", "UDP-lag", "UDP",
                   "WebDDoS", "Portmap", "LDAP", "MSSQL", "NetBIOS", "SNMP"}
IDS2018_LABELS = {"Benign", "DDoS attacks-LOIC-HTTP", "DDoS attack-LOIC-UDP",
                  "DDoS attack-HOIC", "DDOS attack-LOIC-UDP",
                  "DDOS attack-HOIC", "FTP-BruteForce", "SSH-BruteForce",
                  "Bot", "DoS attacks-Hulk", "DoS attacks-GoldenEye",
                  "DoS attacks-Slowloris", "DoS attacks-SlowHTTPTest",
                  "Brute Force -Web", "Brute Force -XSS", "SQL Injection",
                  "Infilteration"}

# file tag -> (pool, keep: 'ddos+benign' | 'benign-only')
FILE_POOLS = [
    ("ddos2019", "0112_UDPLag", "ddos2019_day1", "ddos+benign"),
    ("ddos2019", "0112_Syn", "ddos2019_day1", "ddos+benign"),
    ("ddos2019", "0112_DrDoS_NTP", "ddos2019_day1", "ddos+benign"),
    ("ddos2019", "0112_DrDoS_UDP", "ddos2019_day1", "ddos+benign"),
    ("ddos2019", "0112_DrDoS_DNS", "ddos2019_day1", "ddos+benign"),
    ("ddos2019", "0311_Portmap", "ddos2019_day2", "ddos+benign"),
    ("ddos2019", "0311_UDPLag", "ddos2019_day2", "ddos+benign"),
    ("ddos2019", "0311_Syn", "ddos2019_day2", "ddos+benign"),
    ("ddos2019", "0311_UDP", "ddos2019_day2", "ddos+benign"),
    ("ddos2019", "0311_LDAP", "ddos2019_day2", "ddos+benign"),
    ("ids2018", "2018-02-14_bruteforce", "ids2018_early", "benign-only"),
    ("ids2018", "2018-02-20_ddos_loic_http", "ids2018_early", "ddos+benign"),
    ("ids2018", "2018-02-21_ddos_loic_udp_hoic", "ids2018_late", "ddos+benign"),
    ("ids2018", "2018-03-02_botnet", "ids2018_late", "benign-only"),
]


def classify(dataset, raw):
    """-> (binary in {'benign','ddos',None}, subtype) for one raw label."""
    s = str(raw).strip()
    if dataset == "ddos2019":
        if s.upper() == "BENIGN":
            return "benign", "Benign"
        return "ddos", s          # dataset contains only DDoS attacks
    # ids2018
    if s == "Benign":
        return "benign", "Benign"
    if "ddos" in s.lower():   # file spells it 'DDOS attack-HOIC' etc.
        return "ddos", s
    return None, s                # other attack families: excluded, counted


def main():
    inspection, pools, collected = {}, {}, {}

    for dataset, tag, pool, keep in FILE_POOLS:
        path = os.path.join(DATA, dataset, tag + ".csv")
        if not os.path.exists(path):
            inspection[tag] = {"error": "file missing"}
            print(f"[missing] {path}", flush=True)
            continue
        header = pd.read_csv(path, nrows=0)
        raw_cols = [c.strip() for c in header.columns]
        info = {"file": path, "bytes": os.path.getsize(path),
                "n_columns": len(raw_cols), "labels": {},
                "unexpected_labels": []}
        info["format_ok"] = ("Label" in raw_cols
                             and "Flow Duration" in raw_cols
                             and 78 <= len(raw_cols) <= 90)
        print(f"[inspect] {tag}: {len(raw_cols)} cols, "
              f"format_ok={info['format_ok']}", flush=True)
        if not info["format_ok"]:
            inspection[tag] = info
            continue
        documented = DDOS2019_LABELS if dataset == "ddos2019" else IDS2018_LABELS
        pools.setdefault(pool, [])
        excluded_other = 0
        for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if dataset == "ids2018":
                chunk = chunk.rename(columns=RENAME_2018)
            for lab, n in chunk["Label"].value_counts().items():
                lab = str(lab).strip()
                info["labels"][lab] = info["labels"].get(lab, 0) + int(n)
                if lab not in documented and lab not in info["unexpected_labels"]:
                    info["unexpected_labels"].append(lab)
            cls = [classify(dataset, raw) for raw in chunk["Label"]]
            chunk["_binary"] = [c[0] for c in cls]
            chunk["_subtype"] = [c[1] for c in cls]
            excluded_other += int((chunk["_binary"].isna()).sum())
            for binary in ("benign", "ddos"):
                if keep == "benign-only" and binary == "ddos":
                    continue
                sub_all = chunk[chunk["_binary"] == binary]
                if not len(sub_all):
                    continue
                if binary == "benign":
                    key = (pool, "benign")
                    room = BENIGN_COLLECT_CAP - collected.get(key, 0)
                    if room <= 0:
                        continue
                    sub = sub_all if len(sub_all) <= room else \
                        sub_all.sample(n=room, random_state=SEED)
                    pools[pool].append(sub)
                    collected[key] = collected.get(key, 0) + len(sub)
                else:
                    for st, sub_all_st in sub_all.groupby("_subtype"):
                        key = (pool, st)
                        room = SUBTYPE_COLLECT_CAP - collected.get(key, 0)
                        if room <= 0:
                            continue
                        sub = sub_all_st if len(sub_all_st) <= room else \
                            sub_all_st.sample(n=room, random_state=SEED)
                        pools[pool].append(sub)
                        collected[key] = collected.get(key, 0) + len(sub)
        info["excluded_non_ddos_attack_rows_seen"] = excluded_other
        inspection[tag] = info
        print(f"    labels={info['labels']} unexpected="
              f"{info['unexpected_labels']}", flush=True)

    # canonical feature set = intersection over pools of non-ID columns
    frames = {}
    for pool, parts in pools.items():
        if parts:
            frames[pool] = pd.concat(parts, ignore_index=True)
    common = None
    for df in frames.values():
        cols = {c for c in df.columns
                if c not in ID_COLS and c not in
                ("Label", "_binary", "_subtype")}
        common = cols if common is None else (common & cols)
    feats = sorted(common)
    print(f"[features] canonical set: {len(feats)}", flush=True)

    summary = {}
    for pool, df in frames.items():
        df = df.copy()
        df["label"] = np.where(df["_binary"] == "benign", 0, 1)
        df["subtype"] = df["_subtype"]
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        df[feats] = X
        df = df[df[feats].isna().mean(axis=1) < 0.5]
        # final per-class sample (seeded) before dedup
        parts = []
        for cls in (0, 1):
            sub = df[df["label"] == cls]
            if len(sub) > FINAL_CAP_PER_CLASS:
                sub = sub.sample(n=FINAL_CAP_PER_CLASS, random_state=SEED)
            parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
        before = len(df)
        n_dup = int(df.duplicated(subset=feats + ["label"]).sum())
        # NO-DEDUP VARIANT: duplicates are KEPT (B1 ablation). Record count.
        df = df.reset_index(drop=True)
        print(f"[nodedup] {pool}: keeping {n_dup} duplicate rows", flush=True)
        out = os.path.join(PREP, pool + ".csv")
        df[feats + ["label", "subtype"]].to_csv(out, index=False)
        summary[pool] = {
            "rows_after_cleaning": int(len(df)),
            "duplicates_removed": 0, "duplicates_present": n_dup,
            "benign": int((df["label"] == 0).sum()),
            "ddos": int((df["label"] == 1).sum()),
            "n_features": len(feats),
            "subtypes": {k: int(v) for k, v in
                         df.loc[df["label"] == 1, "subtype"]
                         .value_counts().items()},
        }
        print(f"[pool] {pool}: {summary[pool]}", flush=True)

    with open(os.path.join(RESULTS, "dataset_inspection_nodedup.json"), "w") as fh:
        json.dump({"files": inspection, "pools": summary,
                   "canonical_features": feats}, fh, indent=2)
    print("PREPARE DONE")


if __name__ == "__main__":
    sys.exit(main())
