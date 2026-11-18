"""Rare-event metrics (§9): PR-AUC and recall at a fixed low false-positive rate."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def recall_at_fpr(y_true, y_score, target_fpr=0.01):
    """Highest recall achievable while keeping FPR <= target_fpr (early detection)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    thr = np.quantile(neg, 1.0 - target_fpr)     # threshold that admits target_fpr of negs
    recall = float((pos >= thr).mean())
    return recall, float(thr)


def eval_scores(y_true, y_score, target_fpr=0.01):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    pr_auc = float(average_precision_score(y_true, y_score)) if y_true.sum() else float("nan")
    rec, thr = recall_at_fpr(y_true, y_score, target_fpr)
    pred = (y_score >= thr).astype(int) if not np.isnan(thr) else (y_score >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary",
                                                  zero_division=0)
    return {"pr_auc": round(pr_auc, 4), f"recall@{target_fpr:g}fpr": round(rec, 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f1), 4), "threshold": round(float(thr), 4)}
