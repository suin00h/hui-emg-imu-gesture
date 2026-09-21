"""Fetch the dataset and checkpoints from Zenodo.

    python scripts/download.py                  # everything
    python scripts/download.py --only S01.h5

Set ZENODO_TOKEN if the record is restricted.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RECORD = None          # filled in once the record is published
ROOT = Path(__file__).resolve().parent.parent


def fetch_manifest(record):
    url = f"https://zenodo.org/api/records/{record}"
    req = urllib.request.Request(url)
    token = os.environ.get("ZENODO_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    import json
    with urllib.request.urlopen(req) as r:
        return json.load(r)["files"]


def download(entry, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = entry["checksum"].split(":")[-1]
    if dest.exists() and md5(dest) == want:
        print(f"  have {dest.name}")
        return
    print(f"  get  {dest.name} ({entry['size'] / 1e6:.0f} MB)", flush=True)
    req = urllib.request.Request(entry["links"]["self"])
    token = os.environ.get("ZENODO_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    if md5(dest) != want:
        raise RuntimeError(f"checksum mismatch for {dest.name}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.record:
        sys.exit("no Zenodo record id: pass --record")
    for e in fetch_manifest(a.record):
        name = e["key"]
        if a.only and name not in a.only:
            continue
        sub = "checkpoints" if name.startswith("checkpoints") else "dataset"
        download(e, ROOT / sub / name)


if __name__ == "__main__":
    main()
