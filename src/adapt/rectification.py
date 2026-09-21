"""Prototype rectification (Liu et al., ECCV 2020) over the enrolment pool."""
import numpy as np

from src.adapt.geometry import pdist2
from src.adapt.heads import HEADS, PrototypeHead, soft_assign


def cross_class_shift(Cshot, Zpool):
    return Zpool.mean(0, keepdims=True) - Cshot.mean(0, keepdims=True)


def rectify(Cshot, Cs, Zpool, alpha=0.7, prior=0.25, n_iter=3, temperature=1.0):
    """alpha shrinks the one-shot prototype toward the source prototype; prior is how much of that
    shrunk prototype every self-training step keeps."""
    P0 = alpha * Cshot + (1.0 - alpha) * Cs + cross_class_shift(Cshot, Zpool)
    P = P0.copy()
    for _ in range(n_iter):
        w = soft_assign(Zpool, P, temperature)
        mass = w.sum(0)[:, None]
        P = prior * P0 + (1.0 - prior) * np.where(
            mass > 1e-6, (w.T @ Zpool) / np.maximum(mass, 1e-9), P)
    return P


def make_rectified_head(alpha=0.7, prior=0.25, n_iter=3, temperature=1.0):
    def build(ctx, amap=None):
        P = rectify(ctx.Cshot, ctx.Cs, ctx.pool(), alpha, prior, n_iter, temperature)
        return PrototypeHead(P if amap is None else amap.apply(P))
    return build


HEADS["prototype-rectification"] = make_rectified_head()
