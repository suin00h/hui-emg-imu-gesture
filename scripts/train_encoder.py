"""Train one leave-one-subject-out encoder and cache its features.

    python scripts/train_encoder.py --config configs/encoder_ours.yaml --target 1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import FoldDataModule
from src.folds import load_fold
from src.model.lightning import GestureModule

ROOT = Path(__file__).resolve().parent.parent


def config_tag(cfg):
    """Everything that changes the encoder goes into the cache name. Selecting fields by hand is
    how a baseline and an intervention arm end up silently sharing one cache file."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    tag = config_tag(cfg)
    out = a.out / tag
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    feat = out / f"S{a.target:02d}.npz"
    if feat.exists():
        print(f"[skip] {feat}")
        return

    pl.seed_everything(cfg["train"]["seed"] + a.target, workers=True)
    arrays = load_fold(a.target)
    iv = cfg.get("interventions") or {}
    rot = iv.get("channel_rotation")
    dm = FoldDataModule(arrays, cfg["train"]["batch_size"], cfg["train"]["val_ratio"],
                        a.num_workers, cfg["train"]["seed"],
                        rotation=None if rot is None else
                        {"span": rot["span"], "arm_only": rot["classes"] == "arm"})
    me = iv.get("masked_electrode")
    module = GestureModule(
        arrays.n_classes, arrays.emg_src.shape[-1], arrays.imu_src.shape[-1],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        max_epochs=cfg["train"]["max_epochs"], warmup_epochs=cfg["train"]["warmup_epochs"],
        masked_electrode=me is not None,
        n_mask=(me or {}).get("n_mask", 1), aux_weight=(me or {}).get("weight", 1.0),
        model_kwargs=cfg["model"])

    ck = ModelCheckpoint(dirpath=out / "ckpt", filename=f"S{a.target:02d}",
                         monitor="val/f1", mode="max", save_top_k=1)
    trainer = pl.Trainer(max_epochs=cfg["train"]["max_epochs"], accelerator="auto", devices=1,
                         logger=False, callbacks=[ck], enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(module, dm)
    if ck.best_model_path:
        module.load_state_dict(torch.load(ck.best_model_path, map_location="cpu",
                                          weights_only=False)["state_dict"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.savez_compressed(
        feat,
        f_src=module.fused_features(arrays.emg_src, arrays.imu_src, device=dev),
        y_src=arrays.y_src, subj_src=arrays.subj_src,
        f_tgt=module.fused_features(arrays.emg_tgt, arrays.imu_tgt, device=dev),
        y_tgt=arrays.y_tgt, sess_tgt=arrays.sess_tgt)
    (out / "config.yaml").write_text(a.config.read_text())
    print(f"[ok] S{a.target:02d} val/f1={float(ck.best_model_score):.4f} -> {feat}")


if __name__ == "__main__":
    main()
