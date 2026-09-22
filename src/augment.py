import numpy as np

from src.labels import ARM


def channel_rotation(emg, label, rng, span=1, arm_only=True):
    """Cyclic shift of the electrode ring by an integer step in [-span, span].

    An arm command is recognised by the trajectory of the arm and survives a rotation of the ring; a
    counting gesture is recognised by which electrodes are loaded, which a rotation destroys.
    """
    if arm_only and int(label) not in ARM:
        return emg
    k = int(rng.integers(-span, span + 1))
    return np.ascontiguousarray(np.roll(emg, k, axis=-1), dtype=np.float32) if k else emg
