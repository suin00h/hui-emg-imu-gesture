"""Raw windows -> encoder input."""
import numpy as np
from scipy.signal import butter, filtfilt

EMG_FS, IMU_FS = 1000, 100


def emg_envelope(x, fs=EMG_FS, band=(20.0, 450.0), lowpass=30.0, order=4, decimate=10):
    """[N, T, 9] raw -> [N, T/decimate, 9] linear envelope.

    Each window is filtered on its own; filtfilt pads at the window edges, so two overlapping
    windows do not agree in their overlap and cannot be stitched back together.
    The 30 Hz low-pass is the anti-alias filter for the decimation that follows.
    """
    bb, ab = butter(order, [band[0] / (0.5 * fs), band[1] / (0.5 * fs)], btype="band")
    bl, al = butter(order, lowpass / (0.5 * fs), btype="low")
    env = np.abs(filtfilt(bb, ab, x, axis=1))
    env = filtfilt(bl, al, env, axis=1)
    return env[:, ::decimate, :].astype(np.float32)


def imu_filter(x, fs=IMU_FS, lowpass=20.0, acc_highpass=0.5, order=4):
    """[N, T, 6] -> same shape. The high-pass removes gravity from the accelerometer only."""
    bl, al = butter(order, lowpass / (0.5 * fs), btype="low")
    out = filtfilt(bl, al, x, axis=1)
    bh, ah = butter(order, acc_highpass / (0.5 * fs), btype="high")
    out[:, :, :3] = filtfilt(bh, ah, out[:, :, :3], axis=1)
    return out.astype(np.float32)
