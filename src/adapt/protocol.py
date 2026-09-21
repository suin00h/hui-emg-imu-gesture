from dataclasses import dataclass, field

import numpy as np


@dataclass
class Fold:
    """One leave-one-subject-out fold of frozen features, standardised by source statistics."""

    target: int
    n_classes: int
    Zs: np.ndarray
    ys: np.ndarray
    Zt: np.ndarray
    yt: np.ndarray
    sess_t: np.ndarray
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def dim(self):
        return self.Zs.shape[1]

    def source_centroids(self):
        return np.stack([self.Zs[self.ys == c].mean(0) for c in range(self.n_classes)])

    def source_head(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        if "head" not in self.cache:
            sc = StandardScaler().fit(self.Zs)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(self.Zs), self.ys)
            self.cache["head"] = SourceHead(sc, clf)
        return self.cache["head"]


@dataclass
class SourceHead:
    scaler: object
    clf: object

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))


def make_fold(target, F_src, y_src, F_tgt, y_tgt, sess_tgt, n_classes=16):
    mu, sd = F_src.mean(0, keepdims=True), F_src.std(0, keepdims=True) + 1e-6
    return Fold(target, n_classes, (F_src - mu) / sd, y_src,
                (F_tgt - mu) / sd, y_tgt, sess_tgt)
