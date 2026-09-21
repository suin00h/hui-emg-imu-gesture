"""Read the public per-subject files."""
from pathlib import Path

import h5py
import numpy as np

from src.preprocess import emg_envelope, imu_filter

DATA = Path(__file__).resolve().parent.parent / "dataset"
SUBJECTS = list(range(1, 12))


def load_subject(subject, root=DATA, processed=True, keep_rest=False):
    """-> emg [N, 150, 9], imu [N, 150, 6], label [N], session [N]."""
    with h5py.File(Path(root) / f"S{subject:02d}.h5", "r") as f:
        label, session = f["label"][:].astype(np.int64), f["session"][:].astype(np.int64)
        sel = slice(None) if keep_rest else np.flatnonzero(label >= 0)
        emg, imu = f["emg"][sel], f["imu"][sel]
        label, session = label[sel], session[sel]
    if processed:
        emg, imu = emg_envelope(emg), imu_filter(imu)
    return emg, imu, label, session
