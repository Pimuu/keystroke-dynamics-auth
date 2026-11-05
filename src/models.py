"""Verification models.  Every model exposes the same tiny interface:

    model.fit(X_genuine, X_impostor=None)   -> self
    model.score(X)                          -> np.ndarray, higher = more genuine

Two families are compared:

* **Distance / threshold (one-class) detectors** -- trained on the claimed
  user's samples only.  Their score is a negated distance to the enrolment
  template; accept if score >= threshold.
    - Euclidean       (baseline, Killourhy & Maxion 2009)
    - Scaled Manhattan (per-feature mean absolute deviation scaling; the best
      detector in the CMU benchmark)
    - Mahalanobis     (Ledoit-Wolf shrunk covariance, so it remains defined
      when there are fewer samples than features)
    - One-class SVM   (RBF kernel, standardised features)

* **Discriminative two-class classifier** -- trained on genuine samples *and*
  samples from a disjoint set of "background" users.
    - Random Forest   (probability of the genuine class as score)
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class EuclideanDetector:
    name = "Euclidean"
    needs_impostors = False

    def fit(self, Xg, Xi=None):
        self.mean_ = Xg.mean(axis=0)
        return self

    def score(self, X):
        return -np.linalg.norm(X - self.mean_, axis=1)


class ScaledManhattanDetector:
    name = "Scaled Manhattan"
    needs_impostors = False

    def fit(self, Xg, Xi=None):
        self.mean_ = Xg.mean(axis=0)
        mad = np.abs(Xg - self.mean_).mean(axis=0)
        self.mad_ = np.maximum(mad, 1e-6)  # guard against constant features
        return self

    def score(self, X):
        return -(np.abs(X - self.mean_) / self.mad_).sum(axis=1)


class MahalanobisDetector:
    name = "Mahalanobis"
    needs_impostors = False

    def fit(self, Xg, Xi=None):
        self.scaler_ = StandardScaler().fit(Xg)
        Z = self.scaler_.transform(Xg)
        self.cov_ = LedoitWolf().fit(Z)
        return self

    def score(self, X):
        return -np.sqrt(self.cov_.mahalanobis(self.scaler_.transform(X)))


class OneClassSVMDetector:
    name = "One-class SVM"
    needs_impostors = False

    def __init__(self, nu: float = 0.1):
        self.nu = nu

    def fit(self, Xg, Xi=None):
        self.scaler_ = StandardScaler().fit(Xg)
        self.svm_ = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu)
        self.svm_.fit(self.scaler_.transform(Xg))
        return self

    def score(self, X):
        return self.svm_.decision_function(self.scaler_.transform(X))


class RandomForestVerifier:
    name = "Random Forest"
    needs_impostors = True

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, Xg, Xi=None):
        if Xi is None or len(Xi) == 0:
            raise ValueError("Random Forest needs impostor (background) samples")
        X = np.vstack([Xg, Xi])
        y = np.r_[np.ones(len(Xg)), np.zeros(len(Xi))]
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators, class_weight="balanced",
            min_samples_leaf=1, max_features="sqrt", n_jobs=-1,
            random_state=self.seed)
        self.rf_.fit(X, y)
        return self

    def score(self, X):
        return self.rf_.predict_proba(X)[:, 1]


def make_models(seed: int = 0) -> list:
    """Fresh, unfitted instances of all compared models (fixed order)."""
    return [EuclideanDetector(), ScaledManhattanDetector(), MahalanobisDetector(),
            OneClassSVMDetector(), RandomForestVerifier(seed=seed)]


MODEL_NAMES = [m.name for m in make_models()]
