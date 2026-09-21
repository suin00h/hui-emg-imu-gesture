"""Five transductive few-shot methods, re-implemented on the enrolment pool.

Each was formulated to consume the unlabelled query set. Here they receive the enrolment session
instead, so every query is still decided on its own. All five end with class prototypes, so the
read-out is the same nearest-prototype rule used elsewhere.
"""
import numpy as np

from src.adapt.geometry import pdist2
from src.adapt.heads import HEADS, PrototypeHead, centroids, l2, soft_assign


def _sinkhorn(cost, n_iter=30, reg=0.05):
    n, k = cost.shape
    P = np.exp(-cost / max(reg, 1e-9))
    P /= P.sum() + 1e-12
    r, c = np.full(n, 1.0 / n), np.full(k, 1.0 / k)
    for _ in range(n_iter):
        P *= (r / (P.sum(1) + 1e-12))[:, None]
        P *= (c / (P.sum(0) + 1e-12))[None, :]
    return P / (P.sum(1, keepdims=True) + 1e-12)


def make_tim(alpha=None, temp=15.0, steps=300, lr=1e-3, w_ce=0.1, w_marg=1.0, w_cond=0.1):
    """Boudiaf et al. (NeurIPS 2020); alpha switches Shannon entropy for Tsallis (alpha-TIM,
    Veilleux et al., NeurIPS 2021). Torch is used only as a CPU autograd engine."""
    def build(ctx, amap=None):
        import torch
        Xs, ys = ctx.support()
        Xu = ctx.pool()
        if amap is not None:
            Xs, Xu = amap.apply(Xs), amap.apply(Xu)
        Zs = torch.from_numpy(np.ascontiguousarray(Xs, np.float32))
        Zu = torch.from_numpy(np.ascontiguousarray(Xu, np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(ys, np.int64))
        W = torch.nn.Parameter(torch.from_numpy(
            np.ascontiguousarray(centroids(Xs, ys, ctx.n_classes), np.float32)))
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(-0.5 * temp * torch.cdist(Zs, W) ** 2, yt)
            p = (-0.5 * temp * torch.cdist(Zu, W) ** 2).softmax(1)
            pbar = p.mean(0)
            if alpha is None:
                h_cond = -(p * p.clamp_min(1e-12).log()).sum(1).mean()
                h_marg = -(pbar * pbar.clamp_min(1e-12).log()).sum()
            else:
                h_cond = ((1.0 - (p.clamp_min(1e-12) ** alpha).sum(1)) / (alpha - 1.0)).mean()
                h_marg = (1.0 - (pbar.clamp_min(1e-12) ** alpha).sum()) / (alpha - 1.0)
            (w_ce * ce - w_marg * h_marg + w_cond * h_cond).backward()
            opt.step()
        return PrototypeHead(W.detach().numpy().astype(np.float64))
    return build


def make_ptmap(beta=0.5, n_iter=20, reg=0.05):
    """Hu et al. (2021): power transform, then Sinkhorn against prototype re-estimation."""
    def build(ctx, amap=None):
        Xs, ys = ctx.support()
        Xu = ctx.pool()
        if amap is not None:
            Xs, Xu = amap.apply(Xs), amap.apply(Xu)
        both = np.concatenate([Xs, Xu])
        both = l2(np.power(both - both.min() + 1e-6, beta))
        centre = both.mean(0, keepdims=True)
        both = both - centre
        Xs, Xu = both[:len(Xs)], both[len(Xs):]
        P = centroids(Xs, ys, ctx.n_classes)
        for _ in range(n_iter):
            Q = _sinkhorn(pdist2(Xu, P), reg=reg)
            mass = Q.sum(0)[:, None]
            P = 0.5 * (np.where(mass > 1e-6, (Q.T @ Xu) / np.maximum(mass, 1e-9), P)
                       + centroids(Xs, ys, ctx.n_classes))
        return _TransformedHead(P, centre, beta)
    return build


class _TransformedHead:
    def __init__(self, P, centre, beta):
        self.P, self.centre, self.beta = P, centre, beta

    def predict(self, X):
        X = l2(np.power(X - X.min() + 1e-6, self.beta)) - self.centre
        return np.argmin(pdist2(X, self.P), axis=1)


def make_protolp(lam=1.0, alpha=0.2, n_step=5):
    """Zhu & Koniusz (CVPR 2023): the graph is built from prototype assignments, not sample
    distances, and is refined together with the prototypes."""
    def build(ctx, amap=None):
        Xs, ys = ctx.support()
        Xu = ctx.pool()
        if amap is not None:
            Xs, Xu = amap.apply(Xs), amap.apply(Xu)
        Xs, Xu = l2(Xs), l2(Xu)
        k = ctx.n_classes
        X = np.concatenate([Xs, Xu])
        Yl = np.zeros((len(X), k))
        Yl[np.arange(len(Xs)), ys] = 1.0
        C = centroids(Xs, ys, k)
        eye = np.eye(len(X))
        for _ in range(n_step):
            Z = soft_assign(X, C)
            W = (Z / (Z.sum(0) + 1e-12)[None, :]) @ Z.T
            A = Z.T @ Z + lam * (Z.T @ (eye - W) @ Z)
            Yt = np.clip(Z @ np.linalg.solve(A + 1e-6 * np.eye(k), Z.T @ Yl), 0, None)
            Yt /= Yt.sum(1, keepdims=True) + 1e-12
            mass = Yt.sum(0)[:, None]
            C = (1 - alpha) * C + alpha * np.where(
                mass > 1e-6, (Yt.T @ X) / np.maximum(mass, 1e-9), C)
        return _L2Head(C)
    return build


class _L2Head:
    def __init__(self, P):
        self.P = P

    def predict(self, X):
        return np.argmin(pdist2(l2(X), self.P), axis=1)


def _knn_affinity(X, B, gamma):
    A = np.exp(-gamma * pdist2(X))
    np.fill_diagonal(A, 0.0)
    keep = np.argsort(-A, axis=1)[:, :B]
    M = np.zeros_like(A, dtype=bool)
    np.put_along_axis(M, keep, True, axis=1)
    A = A * (M | M.T)
    dg = 1.0 / np.sqrt(A.sum(1) + 1e-9)
    return A * dg[:, None] * dg[None, :]


def make_pslp(B=8, gamma=10.0, T=10, alpha=0.9, beta=0.2, dim=40):
    """PSLP (ACM TOMM 2025): soft-label the nodes from the prototypes, propagate, move the
    prototypes toward the result. Paper defaults for the imbalanced one-shot setting.

    The prototype update divides out the class mass; without it the prototype scales with the
    number of nodes. Joint message passing is omitted: it smooths the features along the graph,
    which a held-out query has no position in.
    """
    def build(ctx, amap=None):
        Xs, ys = ctx.support()
        Xu = ctx.pool()
        if amap is not None:
            Xs, Xu = amap.apply(Xs), amap.apply(Xu)
        X = np.concatenate([Xs, Xu])
        mu = X.mean(0, keepdims=True)
        d = int(min(dim, X.shape[1], len(X) - 1))
        P = np.linalg.svd(X - mu, full_matrices=False)[2][:d].T
        Xp = l2((X - mu) @ P)
        n_s, k = len(Xs), ctx.n_classes
        L = _knn_affinity(Xp, B, gamma)
        C = centroids(Xp[:n_s], ys, k)
        eye = np.eye(len(Xp))
        for _ in range(T):
            Z = soft_assign(Xp, C, temp=1.0 / gamma)
            Z[np.arange(n_s), :] = 0.0
            Z[np.arange(n_s), ys] = 1.0
            Z = np.clip(np.linalg.solve(eye - alpha * L, Z), 0.0, None)
            Z /= Z.sum(1, keepdims=True) + 1e-12
            mass = Z.sum(0)[:, None]
            C = (1.0 - beta) * C + beta * np.where(
                mass > 1e-6, (Z.T @ Xp) / np.maximum(mass, 1e-9), C)
        return _ProjHead(mu, P, C)
    return build


class _ProjHead:
    def __init__(self, mu, P, C):
        self.mu, self.P, self.C = mu, P, C

    def predict(self, X):
        return np.argmin(pdist2(l2((X - self.mu) @ self.P), self.C), axis=1)


HEADS["tim"] = make_tim()
HEADS["alpha-tim"] = make_tim(alpha=7.0)
HEADS["ptmap"] = make_ptmap()
HEADS["protolp"] = make_protolp()
HEADS["pslp"] = make_pslp()
