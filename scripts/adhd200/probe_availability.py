"""ADHD-200 availability probe (read-only, no bulk download).

Joins phenotyped subjects to the subjects that actually have the filtered+GSR
CC200 output on PCP S3, per site (Peking pooled), so we can sanity-check N BEFORE
downloading. Writes data/adhd200/processed/available_subjects.csv for the
downloader to consume.

No imaging is downloaded here; only S3 directory listings + the small per-site
phenotype CSVs are fetched.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data/adhd200/processed"
OUT.mkdir(parents=True, exist_ok=True)

BUCKET = "https://fcp-indi.s3.amazonaws.com"
PROJ = "data/Projects/ADHD200"
FF_PIPE = f"{PROJ}/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark__freq-filter"
PHENO_DIR = f"{PROJ}/RawDataBIDS"


def s3_list_prefixes(prefix: str) -> list[str]:
    """All CommonPrefixes (subject dirs) under prefix, following pagination."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "delimiter": "/", "prefix": prefix + "/",
             "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = BUCKET + "/?" + urllib.parse.urlencode(q)
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        for chunk in xml.split("<CommonPrefixes>")[1:]:
            p = chunk.split("<Prefix>")[1].split("</Prefix>")[0]
            out.append(p)
        if "<IsTruncated>true</IsTruncated>" in xml:
            token = xml.split("<NextContinuationToken>")[1].split(
                "</NextContinuationToken>")[0]
        else:
            break
    return out


def s3_list_keys(prefix: str, suffix: str) -> list[str]:
    """All object keys under prefix ending with suffix (paginated)."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = BUCKET + "/?" + urllib.parse.urlencode(q)
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        for chunk in xml.split("<Key>")[1:]:
            k = chunk.split("</Key>")[0]
            if k.endswith(suffix):
                out.append(k)
        if "<IsTruncated>true</IsTruncated>" in xml:
            token = xml.split("<NextContinuationToken>")[1].split(
                "</NextContinuationToken>")[0]
        else:
            break
    return out


def site_of(pheno_filename: str) -> str:
    """Map a phenotype filename to a pooled site label."""
    base = pheno_filename.replace("_phenotypic.csv", "")
    base = base.replace("_TestRelease", "")
    if base.startswith("Peking"):
        return "PEKING"          # pool Peking_1/2/3
    return base.upper()


def main() -> None:
    print("[1] Listing filtered-pipeline subject dirs (paginated)...")
    dirs = s3_list_prefixes(FF_PIPE)
    avail_ids = set()
    for d in dirs:
        stem = d.rstrip("/").split("/")[-1]          # e.g. 0010020_session_1
        sid = stem.split("_session_")[0]
        avail_ids.add(sid.zfill(7))
    print(f"    filtered pipeline: {len(dirs)} dirs, "
          f"{len(avail_ids)} unique subject IDs")

    print("[2] Listing phenotype CSVs...")
    pheno_keys = s3_list_keys(PHENO_DIR + "/", "_phenotypic.csv")
    fnames = [k.split("/")[-1] for k in pheno_keys]
    print(f"    found {len(fnames)} phenotype CSVs: {sorted(fnames)}")

    print("[3] Building subject table + joining to availability...")
    rows = []
    peking_samples: dict[str, int] = {}
    for fn in fnames:
        url = f"{BUCKET}/{PHENO_DIR}/{fn}"
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        site = site_of(fn)
        sample_label = fn.replace("_phenotypic.csv", "")
        for _, r in df.iterrows():
            try:
                sid = str(int(float(r["ScanDir ID"]))).zfill(7)
            except (ValueError, TypeError, KeyError):
                continue
            has = sid in avail_ids
            rows.append({"site": site, "sample": sample_label, "id": sid,
                         "age": r.get("Age"), "gender": r.get("Gender"),
                         "dx": r.get("DX"), "available": has})
            if site == "PEKING" and has:
                peking_samples[sample_label] = peking_samples.get(sample_label, 0) + 1

    tab = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    tab.to_csv(OUT / "available_subjects.csv", index=False)

    print("\n" + "=" * 60)
    print("PER-SITE AVAILABILITY (phenotyped vs has filtered CC200)")
    print("=" * 60)
    g = (tab.groupby("site")
            .agg(phenotyped=("id", "size"), available=("available", "sum"))
            .sort_values("available", ascending=False))
    for site, r in g.iterrows():
        flag = "  <-- under 30 available" if r["available"] < 30 else ""
        print(f"  {site:12s} phenotyped={int(r['phenotyped']):4d}  "
              f"available={int(r['available']):4d}{flag}")
    print("-" * 60)
    print(f"  {'TOTAL':12s} phenotyped={int(g['phenotyped'].sum()):4d}  "
          f"available={int(g['available'].sum()):4d}")
    print(f"\n  Peking pooled from: {peking_samples}")
    n_sites_ge30 = int((g["available"] >= 30).sum())
    print(f"\n  Sites with >=30 available (pre-QC): {n_sites_ge30}")
    print(f"  -> site pairs if that holds post-QC: "
          f"{n_sites_ge30*(n_sites_ge30-1)//2}")
    print(f"\n  wrote {OUT/'available_subjects.csv'}")


if __name__ == "__main__":
    main()
