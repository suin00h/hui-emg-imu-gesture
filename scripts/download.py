"""Fetch the dataset and trained weights from the GitHub release.

    python scripts/download.py                    # everything
    python scripts/download.py --only S01.h5      # one subject
    python scripts/download.py --skip-weights
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "suin00h/hui-emg-imu-gesture"
TAG = "v1.0-data"
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = "checkpoints-ours.zip"


def assets(repo, tag):
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(url) as r:
        return json.load(r)["assets"]


def get(asset, dest):
    if dest.exists() and dest.stat().st_size == asset["size"]:
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {dest.name} ({asset['size'] / 1e6:.0f} MB)", flush=True)
    with urllib.request.urlopen(asset["browser_download_url"]) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()

    for asset in assets(a.repo, a.tag):
        name = asset["name"]
        if a.only and name not in a.only:
            continue
        if name == WEIGHTS:
            if a.skip_weights:
                continue
            tmp = ROOT / WEIGHTS
            get(asset, tmp)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(ROOT)
            tmp.unlink()
            print(f"  unpacked {WEIGHTS}")
        else:
            get(asset, ROOT / "dataset" / name)

    print("\nnow run: python scripts/verify.py")


if __name__ == "__main__":
    main()

