"""Experiment runners.  Each returns tidy pandas DataFrames; nothing is
written to disk here (see ``run_pipeline.py``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_generation import generate_dataset
from .features import extract_features, feature_names, feature_subsets
from .metrics import (auc, eer_from_roc, eer_from_scores, eer_threshold,
                      interpolated_tpr, rates_at_threshold, FPR_GRID)
from .models import make_models
from .protocol import ProtocolData

IMPOSTOR_TYPES = ["zero-effort", "external", "mimicry"]


def evaluate_user(split, model) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Fit one model for one target user and evaluate it on the test split."""
    model.fit(split.X_fit, split.X_bg_fit if model.needs_impostors else None)

    # 1) Fix the operating point a priori on validation data.
    thr = eer_threshold(model.score(split.X_val_gen), model.score(split.X_bg_val))

    # 2) Score the test data (never seen during fitting or thresholding).
    s_gen = model.score(split.X_test_gen)
    s_imp = model.score(split.X_test_imp)

    row = {"user": split.target, "model": model.name,
           "eer": eer_from_scores(s_gen, s_imp), "auc": auc(s_gen, s_imp),
           "threshold": thr, **rates_at_threshold(s_gen, s_imp, thr)}
    for t in IMPOSTOR_TYPES:
        m = split.test_imp_type == t
        key = t.replace("-", "_")
        row[f"eer_{key}"] = eer_from_scores(s_gen, s_imp[m])
        row[f"far_{key}"] = float(np.mean(s_imp[m] >= thr))

    # Per-user normalisation statistics (validation genuine scores) so that
    # scores of different users can be pooled in one histogram.
    s_val = model.score(split.X_val_gen)
    mu, sd = s_val.mean(), s_val.std() + 1e-12
    scores = pd.DataFrame({
        "user": split.target, "model": model.name,
        "label": np.r_[np.ones(len(s_gen), int), np.zeros(len(s_imp), int)],
        "kind": np.r_[["genuine"] * len(s_gen), split.test_imp_type],
        "session": np.r_[split.test_gen_session, np.zeros(len(s_imp), int)],
        "score": np.r_[s_gen, s_imp],
        "score_norm": (np.r_[s_gen, s_imp] - mu) / sd,
        "accept": np.r_[s_gen, s_imp] >= thr,
    })
    return row, scores, interpolated_tpr(s_gen, s_imp)


def load_features(seed: int, cfg: ExperimentConfig) -> pd.DataFrame:
    return extract_features(generate_dataset(seed, cfg.gen))


def run_main(feats: pd.DataFrame, cfg: ExperimentConfig, cols=None):
    """Main experiment: all models x all users, full feature set."""
    cols = cols or feature_names()
    pdata = ProtocolData(feats, cols, cfg.proto, cfg.seed)
    rows, scores, rocs = [], [], {}
    for u in pdata.users:
        split = pdata.split(u)
        for model in make_models(cfg.seed):
            r, s, tpr = evaluate_user(split, model)
            rows.append(r)
            scores.append(s)
            rocs.setdefault(model.name, []).append(tpr)
    per_user = pd.DataFrame(rows)
    mean_roc = {m: np.mean(v, axis=0) for m, v in rocs.items()}
    return per_user, pd.concat(scores, ignore_index=True), mean_roc


def summarize(per_user: pd.DataFrame, mean_roc: dict | None = None) -> pd.DataFrame:
    """Overall metrics: mean and std over users for each model."""
    metric_cols = [c for c in per_user.columns if c not in ("user", "model", "threshold")]
    agg = per_user.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    if mean_roc is not None:
        agg["eer_avg_roc"] = [eer_from_roc(FPR_GRID, mean_roc[m]) for m in agg.model]
    return agg


def run_session_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    """EER per test session (genuine from session q vs. all impostors)."""
    out = []
    for (model, user), g in scores.groupby(["model", "user"], sort=False):
        imp = g.loc[g.label == 0, "score"].to_numpy()
        for q, gq in g[g.label == 1].groupby("session"):
            out.append({"model": model, "user": user, "session": int(q),
                        "eer": eer_from_scores(gq.score.to_numpy(), imp)})
    df = pd.DataFrame(out)
    return df.groupby(["model", "session"], sort=False).eer.agg(["mean", "std"]).reset_index()


def run_learning_curve(feats, cfg: ExperimentConfig) -> pd.DataFrame:
    """EER as a function of the number of enrolment samples."""
    pdata = ProtocolData(feats, feature_names(), cfg.proto, cfg.seed)
    out = []
    for n in cfg.proto.learning_curve_sizes:
        for u in pdata.users:
            split = pdata.split(u, n_fit=n)
            for model in make_models(cfg.seed):
                model.fit(split.X_fit, split.X_bg_fit if model.needs_impostors else None)
                out.append({"n_enrol": n, "model": model.name, "user": u,
                            "eer": eer_from_scores(model.score(split.X_test_gen),
                                                   model.score(split.X_test_imp))})
    df = pd.DataFrame(out)
    return df.groupby(["model", "n_enrol"], sort=False).eer.agg(["mean", "std"]).reset_index()


def run_feature_ablation(feats, cfg: ExperimentConfig) -> pd.DataFrame:
    """Mean EER per (feature subset, model)."""
    out = []
    for name, cols in feature_subsets().items():
        pdata = ProtocolData(feats, cols, cfg.proto, cfg.seed)
        for u in pdata.users:
            split = pdata.split(u)
            for model in make_models(cfg.seed):
                model.fit(split.X_fit, split.X_bg_fit if model.needs_impostors else None)
                out.append({"features": name, "n_features": len(cols), "model": model.name,
                            "user": u, "eer": eer_from_scores(model.score(split.X_test_gen),
                                                              model.score(split.X_test_imp))})
    df = pd.DataFrame(out)
    return (df.groupby(["features", "n_features", "model"], sort=False)
            .eer.agg(["mean", "std"]).reset_index())


def run_seed_robustness(cfg: ExperimentConfig) -> pd.DataFrame:
    """Re-generate the whole dataset under different seeds and re-run."""
    out = []
    for s in cfg.proto.robustness_seeds:
        c = ExperimentConfig(seed=s, gen=cfg.gen, proto=cfg.proto)
        per_user, _, _ = run_main(load_features(s, c), c)
        for m, g in per_user.groupby("model", sort=False):
            out.append({"seed": s, "model": m, "eer": g.eer.mean(), "auc": g.auc.mean(),
                        "far": g.far.mean(), "frr": g.frr.mean(), "hter": g.hter.mean()})
    return pd.DataFrame(out)


def compare_models(per_user: pd.DataFrame, reference: str = "Random Forest",
                   n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Paired comparison of per-user EER against a reference model.

    For each model: mean EER difference (model - reference) over users, a 95 %
    paired bootstrap CI (resampling users), the number of users on which the
    reference is better, and an exact two-sided sign-test p-value.
    """
    from math import comb

    piv = per_user.pivot(index="user", columns="model", values="eer")
    rng = np.random.default_rng(seed)
    n = len(piv)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = []
    for m in per_user.model.unique():
        if m == reference:
            continue
        d = (piv[m] - piv[reference]).to_numpy()
        boots = d[idx].mean(axis=1)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        k, nn = min(wins, losses), wins + losses
        p = min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / 2 ** nn) if nn else 1.0
        out.append({"model": m, "reference": reference, "mean_eer_diff": d.mean(),
                    "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
                    "reference_better_users": wins, "reference_worse_users": losses,
                    "sign_test_p": p})
    return pd.DataFrame(out)


def user_difficulty(per_user: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between per-user EER and the user's latent
    generator parameters (only possible with synthetic data)."""
    out = []
    for m, g in per_user.groupby("model", sort=False):
        d = latent.merge(g[["user", "eer"]], on="user").drop(columns="user")
        rho = d.corr(method="spearman")["eer"].drop("eer")
        out.append({"model": m, **rho.to_dict()})
    return pd.DataFrame(out)
