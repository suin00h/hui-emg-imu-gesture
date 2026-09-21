"""Source-only, the two label-free alignments, and the three one-shot baselines we construct."""
import numpy as np

from src.adapt.geometry import AffineMap, fit_rotation, identity_map
from src.adapt.heads import HEADS, PrototypeHead, centroids, l2

ESTIMATORS = {}


def source_only(ctx):
    return identity_map(ctx.fold.dim)


def statistic_alignment(ctx):
    """Match the pool's per-dimension mean and variance to the source. Diagonal: cannot rotate."""
    Zt, Zs = ctx.pool(), ctx.fold.Zs
    mt, st = Zt.mean(0, keepdims=True), Zt.std(0, keepdims=True) + 1e-6
    ms, ss = Zs.mean(0, keepdims=True), Zs.std(0, keepdims=True) + 1e-6
    return AffineMap(np.diag((ss / st).ravel()), mt, ms, "diag-stat")


def subspace_alignment(ctx, k=32):
    """Fernando et al. (ICCV 2013), lifted back to the full space; rank k by construction."""
    Zt, Zs = ctx.pool(), ctx.fold.Zs
    mt, ms = Zt.mean(0, keepdims=True), Zs.mean(0, keepdims=True)
    k = int(min(k, Zt.shape[0] - 1, Zt.shape[1]))
    if "src_basis" not in ctx.fold.cache:
        ctx.fold.cache["src_basis"] = np.linalg.svd(Zs - ms, full_matrices=False)[2]
    Pt = np.linalg.svd(Zt - mt, full_matrices=False)[2][:k].T
    Ps = ctx.fold.cache["src_basis"][:k].T
    return AffineMap(Pt @ (Pt.T @ Ps) @ Ps.T, mt, ms, f"subspace-{k}")


def procrustes_map(ctx):
    return fit_rotation(ctx.Cshot, ctx.Cs)


def make_prototype_head(alpha=1.0):
    """Nearest prototype; alpha < 1 shrinks the one-shot prototype toward the source prototype."""
    def build(ctx, amap=None):
        P = alpha * ctx.Cshot + (1.0 - alpha) * ctx.Cs
        return PrototypeHead(P if amap is None else amap.apply(P))
    return build


def make_simpleshot(centre=True):
    """Wang et al. (2019): centre by the source mean, L2-normalise, nearest prototype."""
    def build(ctx, amap=None):
        Xs, ys = ctx.support()
        mu = ctx.fold.Zs.mean(0, keepdims=True) if centre else 0.0
        return _CL2NHead(l2(centroids(Xs - mu, ys, ctx.n_classes)), mu)
    return build


class _CL2NHead:
    def __init__(self, P, mu):
        self.P, self.mu = P, mu

    def predict(self, X):
        from src.adapt.geometry import pdist2
        return np.argmin(pdist2(l2(X - self.mu), self.P), axis=1)


def make_logistic_refit(C=1.0):
    """Refit a linear classifier on the support windows alone."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def build(ctx, amap=None):
        X, y = ctx.support()
        X = X if amap is None else amap.apply(X)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X), y)
        return _RefitHead(sc, clf)
    return build


class _RefitHead:
    def __init__(self, scaler, clf):
        self.scaler, self.clf = scaler, clf

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))


ESTIMATORS["source-only"] = source_only
ESTIMATORS["statistic-alignment"] = statistic_alignment
ESTIMATORS["subspace-alignment"] = subspace_alignment
ESTIMATORS["procrustes"] = procrustes_map
HEADS["protonet"] = make_prototype_head()
HEADS["simpleshot"] = make_simpleshot()
HEADS["logistic-refit"] = make_logistic_refit()
