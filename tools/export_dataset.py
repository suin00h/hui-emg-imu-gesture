"""Write the public per-subject dataset from the internal corpus.

Raw 1 kHz EMG windows and 100 Hz IMU windows, one file per subject. Calibration blocks and the
seven recorded gestures outside the 16-class benchmark are dropped; rest windows are kept as -1.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/suin/workspace/omnisense")

import h5py
import numpy as np
from configs.config import Config
from data.labels import TICKLABELS, UNIFIED, UNIFIED_MAP, n_gesture_classes
from data.omnisense_dataset import OmniSenseDatasetBuilder

OUT = Path("/home/suin/workspace/hui-emg-imu-gesture/dataset")
CLASSES = TICKLABELS[UNIFIED][:-1]


def unified_label(exp, label):
    """16-class index, -1 for rest, None for anything not published."""
    if (exp, label) in UNIFIED_MAP:
        return UNIFIED_MAP[(exp, label)]
    return -1 if label == n_gesture_classes(exp) else None


def main():
    import os
    os.chdir("/home/suin/workspace/omnisense")
    cfg = Config.from_yaml(Path("/home/suin/workspace/omnisense/configs/dense-s20.yaml"))
    builder = OmniSenseDatasetBuilder(cfg.fetch, cfg.process)
    raw = builder.fetcher.fetch()
    emg = np.asarray(raw["emg"], np.float32)
    imu = np.asarray(builder.select_imu(raw["imu"], ["acc", "gyro"]), np.float32)
    meta = raw["metadata"]

    mapped = np.array([unified_label(e, l) if unified_label(e, l) is not None else -99
                       for e, l in zip(meta["experiment"], meta["label"])], np.int16)
    keep = mapped != -99
    print(f"{len(meta)} windows -> {keep.sum()} kept, {(~keep).sum()} dropped")

    OUT.mkdir(parents=True, exist_ok=True)
    subj = meta["subject"].to_numpy(int)
    sess = meta["subsession"].to_numpy(int)
    for s in sorted(np.unique(subj)):
        sel = keep & (subj == s)
        path = OUT / f"S{s:02d}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("emg", data=emg[sel], compression="gzip", compression_opts=1)
            f.create_dataset("imu", data=imu[sel], compression="gzip", compression_opts=1)
            f.create_dataset("label", data=mapped[sel])
            f.create_dataset("session", data=sess[sel].astype(np.int8))
            f.attrs.update(
                subject=int(s), emg_fs=1000, imu_fs=100,
                window_seconds=1.5, window_step_seconds=0.2,
                arm="right", redonned_between_sessions=False,
                emg_channel_order="ch1..ch9, clockwise from the dorsal direction, ch9 adjacent to ch1",
                imu_channel_order="acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
                class_names=CLASSES, rest_label=-1,
                note="each window is filtered independently downstream; windows cannot be "
                     "concatenated back into a continuous recording",
            )
        n = int(sel.sum())
        print(f"  S{s:02d}  {n:6d} windows  {path.stat().st_size/1e6:7.1f} MB", flush=True)


if __name__ == "__main__":
    main()
