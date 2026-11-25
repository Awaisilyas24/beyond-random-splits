#!/usr/bin/env python3
"""
Download the benchmark datasets from their public sources.

Sources (documented, legitimate public access):
  * CSE-CIC-IDS2018 - AWS Open Data Registry bucket `cse-cic-ids2018`
    (registry.opendata.aws/cse-cic-ids2018), folder
    "Processed Traffic Data for ML Algorithms" (CICFlowMeter CSVs).
  * CIC-DDoS2019 - Hugging Face mirror `baalajimaestro/CICDDoS2019`, whose file
    layout (01-12 / 03-11 capture days, one CSV per attack) matches the official
    UNB CIC release. Files are sanity-checked later against the known
    CICFlowMeter format (see prepare_data.py).

Very large CSVs are NOT downloaded whole. Instead we fetch N evenly spaced
byte-range slices (HTTP Range) across each file and keep only complete lines.
This yields a position-stratified sample of the file's rows while keeping the
download tractable. The sampling is documented in data/SOURCES.md and in the
run outputs; every row used is a genuine row from the real dataset files.

Usage: python download_data.py   (idempotent; skips files already present)
"""
import json
import os
import subprocess
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SLICE_MB = 8          # size of one byte-range slice
N_SLICES_BIG = 10     # slices fetched for files above the whole-file threshold
WHOLE_MAX_MB = 410    # files at/below this size are downloaded whole

S3 = ("https://cse-cic-ids2018.s3.amazonaws.com/"
      "Processed%20Traffic%20Data%20for%20ML%20Algorithms/")
HF = "https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/main/"

# (dataset, tag, url, known_size_mb or None -> HEAD)
FILES = [
    # ---- CSE-CIC-IDS2018 -------------------------------------------------
    ("ids2018", "2018-02-14_bruteforce", S3 + "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 358),
    ("ids2018", "2018-02-20_ddos_loic_http", S3 + "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", 4054),
    ("ids2018", "2018-02-21_ddos_loic_udp_hoic", S3 + "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv", 328),
    ("ids2018", "2018-03-02_botnet", S3 + "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", 352),
    # ---- CIC-DDoS2019, capture day 01-12 (training day in the release) ---
    ("ddos2019", "0112_UDPLag", HF + "01-12/UDPLag.csv", 158),
    ("ddos2019", "0112_Syn", HF + "01-12/Syn.csv", 637),
    ("ddos2019", "0112_DrDoS_NTP", HF + "01-12/DrDoS_NTP.csv", 645),
    ("ddos2019", "0112_DrDoS_UDP", HF + "01-12/DrDoS_UDP.csv", 1506),
    ("ddos2019", "0112_DrDoS_DNS", HF + "01-12/DrDoS_DNS.csv", 2133),
    # ---- CIC-DDoS2019, capture day 03-11 (testing day in the release) ----
    ("ddos2019", "0311_Portmap", HF + "03-11/Portmap.csv", 79),
    ("ddos2019", "0311_UDPLag", HF + "03-11/UDPLag.csv", 320),
    ("ddos2019", "0311_Syn", HF + "03-11/Syn.csv", 1877),
    ("ddos2019", "0311_UDP", HF + "03-11/UDP.csv", 1793),
    ("ddos2019", "0311_LDAP", HF + "03-11/LDAP.csv", 871),
]


def curl(args, out=None):
    cmd = ["curl", "-sL", "--fail", "--retry", "3", "--max-time", "1200"] + args
    if out:
        cmd += ["-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed {args}: {r.stderr[:300]}")
    return r.stdout


def head_size(url):
    # Follow redirects and take the LAST Content-Length (the real file's);
    # the first one in the chain can be a tiny redirect page (Hugging Face).
    out = curl(["-I", url])
    sizes = [int(line.split(":")[1].strip()) for line in out.splitlines()
             if line.lower().startswith("content-length:")]
    if not sizes:
        raise RuntimeError(f"no content-length for {url}")
    return sizes[-1]


def fetch_whole(url, dest):
    curl([url], out=dest)


def fetch_sliced(url, dest, size_bytes, n_slices=N_SLICES_BIG, slice_mb=SLICE_MB):
    """Fetch n evenly spaced byte-range slices; keep complete lines only.

    Slice 0 starts at byte 0 so it carries the CSV header. For every later
    slice the first (partial) line is dropped; the header is prepended once.
    """
    slice_bytes = slice_mb * 1024 * 1024
    step = size_bytes // n_slices
    parts = []
    header = None
    for i in range(n_slices):
        start = 0 if i == 0 else i * step
        end = min(start + slice_bytes - 1, size_bytes - 1)
        tmp = dest + f".part{i}"
        curl(["-r", f"{start}-{end}", url], out=tmp)
        with open(tmp, "rb") as fh:
            raw = fh.read()
        os.remove(tmp)
        lines = raw.split(b"\n")
        if i == 0:
            header = lines[0]
            body = lines[1:]
        else:
            body = lines[1:]  # drop partial first line
        # drop trailing partial line of each slice (except it is harmless if empty)
        if body and body[-1] != b"":
            body = body[:-1]
        parts.extend([ln for ln in body if ln.strip()])
        print(f"    slice {i}: bytes {start}-{end}, kept {len(body)} lines", flush=True)
    with open(dest, "wb") as fh:
        fh.write(header + b"\n")
        for ln in parts:
            fh.write(ln + b"\n")


def main():
    manifest = []
    for dataset, tag, url, size_mb in FILES:
        ddir = os.path.join(BASE, dataset)
        os.makedirs(ddir, exist_ok=True)
        dest = os.path.join(ddir, tag + ".csv")
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"[skip] {dataset}/{tag} already present", flush=True)
            mode = "present"
        elif size_mb <= WHOLE_MAX_MB:
            print(f"[whole] {dataset}/{tag} ({size_mb} MB)", flush=True)
            fetch_whole(url, dest)
            mode = "whole"
        else:
            size = head_size(url)
            print(f"[sliced] {dataset}/{tag} ({size/1e6:.0f} MB -> "
                  f"{N_SLICES_BIG}x{SLICE_MB}MB slices)", flush=True)
            fetch_sliced(url, dest, size)
            mode = f"sliced {N_SLICES_BIG}x{SLICE_MB}MB"
        manifest.append({"dataset": dataset, "tag": tag, "url": url,
                         "mode": mode,
                         "local_bytes": os.path.getsize(dest)})
    with open(os.path.join(BASE, "download_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
