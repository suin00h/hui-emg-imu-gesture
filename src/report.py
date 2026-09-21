"""CSV of per-episode scores -> the paper's tables."""
import numpy as np
import pandas as pd

from src.labels import COUNTING

NAMES = {
    "source-only": "Source-only", "statistic-alignment": "Statistic Alignment",
    "subspace-alignment": "Subspace Alignment", "procrustes": "Procrustes",
    "simpleshot": "SimpleShot", "protonet": "ProtoNet", "logistic-refit": "Logistic Refit",
    "ptmap": "PT-MAP", "alpha-tim": "$\\alpha$-TIM", "tim": "TIM", "protolp": "ProtoLP",
    "pslp": "PSLP", "prototype-rectification": "Prototype Rectification",
}


def summarise(df, group=None):
    """Mean over episodes within a subject, then over subjects; plus both spreads."""
    if group is None:
        ep = df[df.class_id == -1]
    else:
        sel = df.class_id.isin(COUNTING) if group == "counting" else \
            (df.class_id >= 0) & ~df.class_id.isin(COUNTING)
        ep = df[sel]
    ep = ep.groupby(["method", "subject", "session", "draw"]).f1.mean()
    subj = ep.groupby(["method", "subject"]).mean()
    return pd.DataFrame({
        "macro_f1": subj.groupby("method").mean() * 100,
        "std_episode": ep.groupby("method").std() * 100,
        "std_subject": subj.groupby("method").std() * 100,
    })


def table_one(df, order=None):
    s = summarise(df)
    order = order or s.macro_f1.sort_values().index
    lines = []
    for m in order:
        lines.append(f"{NAMES.get(m, m):32s} & {s.macro_f1[m]:5.2f} \\\\")
    return "\n".join(lines)
