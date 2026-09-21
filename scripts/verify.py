"""Check the downloaded dataset and reproduce one published number.

    python scripts/verify.py                  # data only
    python scripts/verify.py --features runs/<tag>
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from src.io import DATA, SUBJECTS

EXPECT = {"prototype-rectification": 72.57}


def check_data(root):
    ok = True
    sums = {}
    manifest = Path(root) / "MD5SUMS"
    if manifest.exists():
        sums = {l.split()[1]: l.split()[0] for l in manifest.read_text().split("\n") if l.strip()}
    for s in SUBJECTS:
        p = Path(root) / f"S{s:02d}.h5"
        if not p.exists():
            print(f"  MISSING {p.name}")
            ok = False
            continue
        with h5py.File(p) as f:
            l, sess = f["label"][:], f["session"][:]
            good = (f["emg"].shape[1:] == (1500, 9) and f["imu"].shape[1:] == (150, 6)
                    and len(np.unique(l[l >= 0])) == 16 and len(np.unique(sess)) == 5)
        if p.name in sums:
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    h.update(b)
            good = good and h.hexdigest() == sums[p.name]
        print(f"  {p.name}  {'ok' if good else 'FAILED'}")
        ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA)
    ap.add_argument("--features", type=Path)
    a = ap.parse_args()

    print("dataset")
    ok = check_data(a.root)
    if a.features:
        import pandas as pd
        from src.adapt.evaluate import evaluate
        from src.adapt.protocol import make_fold
        print("\nbenchmark")
        frames = []
        for s in SUBJECTS:
            p = a.features / f"S{s:02d}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            frames.append(evaluate(make_fold(s, d["f_src"], d["y_src"], d["f_tgt"], d["y_tgt"],
                                             d["sess_tgt"]), list(EXPECT)))
        df = pd.concat(frames)
        got = df[df.class_id == -1].groupby(["method", "subject"]).f1.mean().groupby(
            "method").mean() * 100
        for m, want in EXPECT.items():
            d = abs(got[m] - want)
            print(f"  {m}: {got[m]:.2f} (expected {want:.2f}, diff {d:.2f})"
                  f"  {'ok' if d < 0.5 else 'FAILED'}")
            ok = ok and d < 0.5
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
