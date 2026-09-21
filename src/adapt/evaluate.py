import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import src.adapt  # noqa: F401  registers every method
from src.adapt.baselines import ESTIMATORS, source_only
from src.adapt.context import Context
from src.adapt.episodes import iter_episodes
from src.adapt.heads import HEADS

METHODS = ["source-only", "statistic-alignment", "subspace-alignment", "procrustes",
           "simpleshot", "protonet", "logistic-refit",
           "ptmap", "alpha-tim", "tim", "protolp", "pslp", "prototype-rectification"]


def evaluate(fold, methods=METHODS, shots=1, draws=12, seed=42):
    """A method is (feature map, read-out). Methods without a read-out keep the frozen source
    classifier, so a row difference is a difference in adaptation."""
    head = fold.source_head()
    Cs = fold.source_centroids()
    rows = []
    for ep in iter_episodes(fold, shots, draws, seed):
        ctx = Context.build(fold, ep, Cs)
        Xq, yq = fold.Zt[ep.test_mask], fold.yt[ep.test_mask]
        for m in methods:
            amap = ESTIMATORS.get(m, source_only)(ctx)
            read_out = HEADS[m](ctx, amap) if m in HEADS else head
            pred = read_out.predict(amap.apply(Xq))
            per = f1_score(yq, pred, average=None,
                           labels=list(range(fold.n_classes)), zero_division=0)
            rows.append(dict(subject=fold.target, session=ep.enroll_session, draw=ep.draw,
                             method=m, class_id=-1, f1=float(per.mean())))
            rows += [dict(subject=fold.target, session=ep.enroll_session, draw=ep.draw,
                          method=m, class_id=c, f1=float(v)) for c, v in enumerate(per)]
    return pd.DataFrame(rows)


def _check_registered():
    missing = [m for m in METHODS if m not in HEADS and m not in ESTIMATORS]
    if missing:
        raise RuntimeError(f"unregistered methods: {missing}")


_check_registered()
