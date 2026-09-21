"""Evaluate every adaptation method on cached features.

    python scripts/run_benchmark.py --features runs/<tag> --out runs/<tag>/benchmark.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.adapt.evaluate import METHODS, evaluate
from src.adapt.protocol import make_fold
from src.io import SUBJECTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--draws", type=int, default=12)
    a = ap.parse_args()

    frames = []
    for s in SUBJECTS:
        p = a.features / f"S{s:02d}.npz"
        if not p.exists():
            print(f"[missing] {p}")
            continue
        d = np.load(p)
        fold = make_fold(s, d["f_src"], d["y_src"], d["f_tgt"], d["y_tgt"], d["sess_tgt"])
        frames.append(evaluate(fold, a.methods.split(","), draws=a.draws))
        print(f"[ok] S{s:02d}", flush=True)

    df = pd.concat(frames)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    m = df[df.class_id == -1].groupby(["method", "subject"]).f1.mean().groupby("method").mean()
    print(m.sort_values().mul(100).round(2).to_string())


if __name__ == "__main__":
    main()
