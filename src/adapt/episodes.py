from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """One enrolment draw.

    `enroll_session` is the session that ENROLS; its windows form the unlabelled pool and the
    support is drawn from it. Every other session is scored.
    """

    enroll_session: int
    draw: int
    support_idx: dict
    train_mask: np.ndarray
    test_mask: np.ndarray


def support_centroids(Zt, episode, n_classes):
    return np.stack([Zt[episode.support_idx[c]].mean(0) for c in range(n_classes)])


def iter_episodes(fold, shots=1, draws=12, seed=42):
    for sess in sorted(np.unique(fold.sess_t)):
        train = fold.sess_t == sess
        if train.sum() == 0:
            continue
        pools = {c: np.flatnonzero(train & (fold.yt == c)) for c in range(fold.n_classes)}
        if any(len(v) < shots for v in pools.values()):
            continue                       # a session missing a class cannot supply a support set
        for d in range(draws):
            rng = np.random.default_rng(seed + fold.target * 1000 + int(sess) * 97 + d * 7 + shots)
            sup = {c: rng.choice(v, shots, replace=False) for c, v in pools.items()}
            yield Episode(int(sess), d, sup, train, ~train)
