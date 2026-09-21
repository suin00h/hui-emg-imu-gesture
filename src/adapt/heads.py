import numpy as np

from src.adapt.geometry import pdist2

HEADS = {}


class PrototypeHead:
    def __init__(self, prototypes):
        self.P = prototypes

    def predict(self, X):
        return np.argmin(pdist2(X, self.P), axis=1)


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def centroids(X, y, k):
    return np.stack([X[y == c].mean(0) for c in range(k)])


def soft_assign(X, C, temp=1.0):
    z = -pdist2(X, C) / max(temp, 1e-9)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(1, keepdims=True) + 1e-12)
