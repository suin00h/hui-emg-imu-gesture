"""Feature cache from a released checkpoint, without retraining.

    python scripts/extract_features.py --checkpoint checkpoints/ours/S01.ckpt --target 1 --out runs/ours
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.folds import load_fold
from src.model.lightning import GestureModule

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/encoder_ours.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    arrays = load_fold(a.target)
    me = (cfg.get("interventions") or {}).get("masked_electrode")
    module = GestureModule(arrays.n_classes, arrays.emg_src.shape[-1], arrays.imu_src.shape[-1],
                           masked_electrode=me is not None, model_kwargs=cfg["model"])
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"S{a.target:02d}.npz",
        f_src=module.fused_features(arrays.emg_src, arrays.imu_src, device=dev),
        y_src=arrays.y_src, subj_src=arrays.subj_src,
        f_tgt=module.fused_features(arrays.emg_tgt, arrays.imu_tgt, device=dev),
        y_tgt=arrays.y_tgt, sess_tgt=arrays.sess_tgt)
    print(f"[ok] S{a.target:02d}")


if __name__ == "__main__":
    main()
