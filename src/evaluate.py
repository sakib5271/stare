"""In-region and cross-region evaluation (§9).

Cross-region drop = in-region test PR-AUC minus the PR-AUC on a held-out biome the
model never trained on (AZ->boreal, AZ->SoCal). This is the H3 robustness axis; we
report it on the DEPLOYED int8 artifact so the number reflects what ships.
"""
import numpy as np

from .dataio import load_split, prepare_inputs
from .metrics import eval_scores


def evaluate_region(score_fn, kind, region, split="test", target_fpr=0.01, size=None):
    """Return metrics dict, or None if this region/split hasn't been ingested yet."""
    try:
        xs, xt, y = load_split(region, split, size=size)
    except FileNotFoundError:
        return None
    X = prepare_inputs(kind, xs, xt)
    scores = score_fn(X)
    m = eval_scores(y, scores, target_fpr)
    m["region"] = region
    m["n"] = int(len(y))
    return m


def evaluate_all(score_fn, kind, train_region, cross_regions, target_fpr=0.01, size=None):
    """Return {'in_region':..., 'cross':{region:...}, 'cross_region_drop':{...}}.
    Cross-regions that aren't on disk yet are skipped (drop = NaN), so a single-biome
    (e.g. FLAME-only) run still produces in-region metrics."""
    in_r = evaluate_region(score_fn, kind, train_region, "test", target_fpr, size)
    cross, drop = {}, {}
    for r in cross_regions:
        m = evaluate_region(score_fn, kind, r, "test", target_fpr, size)
        if m is None:
            drop[r] = float("nan")
        else:
            cross[r] = m
            drop[r] = round(in_r["pr_auc"] - m["pr_auc"], 4)
    return {"in_region": in_r, "cross": cross, "cross_region_drop": drop}
