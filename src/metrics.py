"""Biometric verification metrics.

Convention: scores are *similarity* scores (higher = more likely genuine);
a sample is **accepted** when ``score >= threshold``.

* FAR (False Acceptance Rate)  = accepted impostors / all impostors
* FRR (False Rejection Rate)   = rejected genuine   / all genuine
* EER (Equal Error Rate)       = the operating point where FAR == FRR
* HTER                         = (FAR + FRR) / 2 at a fixed threshold
* ROC-AUC                      = P(score(genuine) > score(impostor))
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

FPR_GRID = np.linspace(0.0, 1.0, 1001)


def eer_from_scores(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """EER by linear interpolation of the ROC where FPR crosses FNR."""
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    return eer_from_roc(fpr, tpr)


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    diff = fpr - fnr                      # increasing from -1 to +1
    i = np.searchsorted(diff, 0.0)        # first index with diff >= 0
    if i == 0:
        return float(fpr[0])
    if i >= len(diff):
        return float(fnr[-1])
    # Interpolate between points i-1 (diff<0) and i (diff>=0)
    w = -diff[i - 1] / (diff[i] - diff[i - 1])
    return float(fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))


def eer_threshold(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Score threshold at which empirical FAR and FRR are closest.

    Used on *validation* data to fix an operating point a priori.
    """
    cand = np.unique(np.r_[genuine, impostor])
    g = np.sort(genuine)
    imp = np.sort(impostor)
    # FRR(t) = #genuine < t ; FAR(t) = #impostor >= t
    frr = np.searchsorted(g, cand, side="left") / len(g)
    far = 1.0 - np.searchsorted(imp, cand, side="left") / len(imp)
    k = np.argmin(np.maximum(far, frr) + 1e-9 * np.abs(far - frr))
    # Place the threshold half-way to the next lower candidate for stability.
    lower = cand[k - 1] if k > 0 else cand[k] - 1e-9
    return float(0.5 * (cand[k] + lower))


def rates_at_threshold(genuine, impostor, thr: float) -> dict:
    far = float(np.mean(impostor >= thr)) if len(impostor) else np.nan
    frr = float(np.mean(genuine < thr)) if len(genuine) else np.nan
    n_g, n_i = len(genuine), len(impostor)
    correct = (genuine >= thr).sum() + (impostor < thr).sum()
    return {"far": far, "frr": frr, "hter": 0.5 * (far + frr),
            "accuracy": float(correct / (n_g + n_i))}


def auc(genuine, impostor) -> float:
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    return float(roc_auc_score(y, np.r_[genuine, impostor]))


def interpolated_tpr(genuine, impostor) -> np.ndarray:
    """TPR sampled on a fixed FPR grid (for vertically averaged ROC curves)."""
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    fpr, tpr, _ = roc_curve(y, np.r_[genuine, impostor])
    # Step interpolation: best TPR achievable with FPR <= grid value.
    idx = np.searchsorted(fpr, FPR_GRID, side="right") - 1
    return tpr[idx]
