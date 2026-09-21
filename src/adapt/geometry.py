from dataclasses import dataclass, field

import numpy as np


def pdist2(A, B=None):
    """Squared euclidean distances [n, m] without the (n, m, d) intermediate."""
    B = A if B is None else B
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


@dataclass
class AffineMap:
    """X -> (X - t_center) @ M + s_center, target frame to source frame."""

    M: np.ndarray
    t_center: np.ndarray
    s_center: np.ndarray
    family: str = "unknown"
    meta: dict = field(default_factory=dict)

    def apply(self, X):
        return (X - self.t_center) @ self.M + self.s_center


def procrustes(A, B):
    """Proper orthogonal R minimising ||A R - B||_F, det(R) = +1."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    s = np.ones(U.shape[1])
    s[-1] = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag(s) @ Vt


def fit_rotation(Ct, Cs):
    tc, sc = Ct.mean(0, keepdims=True), Cs.mean(0, keepdims=True)
    return AffineMap(procrustes(Ct - tc, Cs - sc), tc, sc, "rotation-SO")


def identity_map(dim):
    z = np.zeros((1, dim))
    return AffineMap(np.eye(dim), z, z, "identity")
