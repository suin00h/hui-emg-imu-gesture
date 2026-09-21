"""Leave-one-subject-out folds."""
from dataclasses import dataclass

import numpy as np

from src.io import SUBJECTS, load_subject


@dataclass
class FoldArrays:
    emg_src: np.ndarray
    imu_src: np.ndarray
    y_src: np.ndarray
    subj_src: np.ndarray
    emg_tgt: np.ndarray
    imu_tgt: np.ndarray
    y_tgt: np.ndarray
    sess_tgt: np.ndarray
    n_classes: int


def _source_zscore(a, fit):
    """Per-channel standardisation fitted on the source subjects only, clipped to +-5.

    Fitted on the source because the held-out person's statistics are not available at enrolment.
    The masked-electrode target is an RMS of this signal, so changing it changes every number.
    """
    mu = a[fit].mean(axis=(0, 1), keepdims=True)
    sd = a[fit].std(axis=(0, 1), keepdims=True) + 1e-8
    return np.clip((a - mu) / sd, -5, 5).astype(np.float32)


def load_fold(target, root=None, n_classes=16):
    emg, imu, y, sess, subj = [], [], [], [], []
    for s in SUBJECTS:
        kw = {} if root is None else {"root": root}
        e, i, l, ss = load_subject(s, **kw)
        emg.append(e); imu.append(i); y.append(l); sess.append(ss)
        subj.append(np.full(len(l), s))
    emg = np.concatenate(emg); imu = np.concatenate(imu)
    y = np.concatenate(y); sess = np.concatenate(sess); subj = np.concatenate(subj)

    src = subj != target
    emg, imu = _source_zscore(emg, src), _source_zscore(imu, src)
    tgt = ~src
    return FoldArrays(emg[src], imu[src], y[src], subj[src],
                      emg[tgt], imu[tgt], y[tgt], sess[tgt], n_classes)
