"""Signal panels for the architecture figure: signal only, no axes.

Sized for an inset a couple of centimetres wide, so line widths are set at that size rather than
shrunk from a large preview.

    python scripts/make_figures.py --channels 7,9,1,3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.io import DATA, SUBJECTS  # noqa: E402
from src.labels import ARM, COUNTING  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs/figures"
EMG_CH = 9


def pick(root, groups, score):
    """The window of any subject whose amplitude contrast against that subject's rest is highest."""
    best = None
    for s in SUBJECTS:
        with h5py.File(Path(root) / f"S{s:02d}.h5", "r") as f:
            label = f["label"][:]
            rest = np.flatnonzero(label == -1)[:400]
            gest = np.flatnonzero(np.isin(label, groups))[:800]
            if len(rest) == 0 or len(gest) == 0:
                continue
            key = "imu" if score == "motion" else "emg"
            base = np.abs(f[key][np.sort(rest)]).mean() + 1e-9
            block = f[key][np.sort(gest)]
        v = (np.abs(np.diff(block, axis=1)).mean(axis=(1, 2)) if score == "motion"
             else np.sqrt((block ** 2).mean(1)).min(1)) / base
        i = int(v.argmax())
        if best is None or v[i] > best[0]:
            best = (v[i], block[i], f"S{s:02d}")
    return best


def lanes(x, lane=0.62):
    """One robust divisor per modality, so relative channel amplitudes stay true."""
    x = x - x.mean(0, keepdims=True)
    s = np.percentile(np.abs(x), 99.5)
    return np.clip(x / (2 * s + 1e-12), -lane, lane) if s > 0 else x


def panel(name, sig, colours, lw, width, lane_scale=1.0):
    fig, ax = plt.subplots(figsize=(width, width * 3 / 4))
    Z = lanes(sig) * lane_scale
    t = np.arange(len(sig)) / len(sig)
    for j in reversed(range(Z.shape[1])):
        ax.plot(t, Z[:, j] + (Z.shape[1] - 1 - j), lw=lw, color=colours[j],
                solid_joinstyle="round", solid_capstyle="round")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.62 * lane_scale - 0.06, Z.shape[1] - 1 + 0.62 * lane_scale + 0.06)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, transparent=(ext == "pdf"), pad_inches=0)
    plt.close(fig)
    print(f"  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA)
    ap.add_argument("--channels", default="7,9,1,3")
    ap.add_argument("--width", type=float, default=1.2)
    ap.add_argument("--decimate", type=int, default=5)
    ap.add_argument("--emg-lw", type=float, default=0.45)
    ap.add_argument("--imu-lw", type=float, default=1.0)
    a = ap.parse_args()
    chans = [int(c) - 1 for c in a.channels.split(",")]
    shrink = (a.width / 1.6) ** 0.75

    _, emg, src_e = pick(a.root, COUNTING, "amplitude")
    _, imu, src_i = pick(a.root, ARM, "motion")
    print(f"EMG {src_e}, IMU {src_i}")

    # Colour is bound to the electrode, not to the row: under a rotation the traces move to another
    # row and carry their colour, which is what makes the shift visible.
    ramp = [matplotlib.colors.to_hex(c) for c in plt.cm.Greens(np.linspace(0.99, 0.30, EMG_CH))]
    imu_c = ["#a01f4c", "#c94a75", "#e57ba0"]
    k = a.decimate
    lw = a.emg_lw * (1 + 0.5 * np.log10(k)) * shrink

    panel("emg", emg[::k, chans], [ramp[c] for c in chans], lw, a.width)
    panel("imu", imu[:, :3], imu_c, a.imu_lw * shrink, a.width, lane_scale=1.15)  # accelerometer
    for s in (1, -1):
        panel(f"emg-shift{s:+d}", np.roll(emg, s, axis=-1)[::k, chans],
              [ramp[(c - s) % EMG_CH] for c in chans], lw, a.width)
    for c in chans:
        m = emg.copy()
        m[:, c] = 0.0
        panel(f"emg-zero-ch{c + 1}", m[::k, chans], [ramp[i] for i in chans], lw, a.width)


if __name__ == "__main__":
    main()
