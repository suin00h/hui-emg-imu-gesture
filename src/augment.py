import numpy as np

from src.labels import ARM


def channel_rotation(emg, label, rng, span=1, arm_only=True):
    """Cyclic shift of the electrode ring by an integer step in [-span, span].

    Applied to arm commands only by default: a movement is recognised by its temporal shape and
    survives a rotation, while a held posture is recognised by which electrode is loaded and does
    not.
    """
    if arm_only and int(label) not in ARM:
        return emg
    k = int(rng.integers(-span, span + 1))
    return np.ascontiguousarray(np.roll(emg, k, axis=-1), dtype=np.float32) if k else emg
