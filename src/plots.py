"""Figures for the paper/README (matplotlib only, static PNG).

Colour follows the entity, never its rank: every model keeps the same colour
in every figure.  Palette = validated categorical order (CVD-checked).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_generation import key_tokens  # noqa: E402
from .metrics import FPR_GRID, eer_from_roc  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

MODEL_COLORS = {
    "Scaled Manhattan": BLUE, "Random Forest": ORANGE, "Mahalanobis": AQUA,
    "One-class SVM": YELLOW, "Euclidean": MAGENTA,
}
TYPE_COLORS = {"zero-effort": BLUE, "external": ORANGE, "mimicry": AQUA}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def plot_roc(mean_roc: dict, path):
    """Vertically averaged per-user ROC curves with the EER point marked."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, lim in zip(axes, (1.0, 0.3)):
        ax.plot([0, 1], [1, 0], ls="--", lw=1, color=INK2, label="EER line (FAR = FRR)")
        for m, tpr in mean_roc.items():
            eer = eer_from_roc(FPR_GRID, tpr)
            ax.plot(FPR_GRID, tpr, color=MODEL_COLORS[m], label=f"{m} (EER {eer:.1%})")
            ax.scatter([eer], [1 - eer], s=60, color=MODEL_COLORS[m],
                       edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.set_xlim(0, lim)
        ax.set_ylim(1 - lim if lim < 1 else 0, 1.0)
        ax.set_xlabel("False Acceptance Rate (FAR)")
        ax.set_ylabel("True Acceptance Rate (1 - FRR)")
    axes[0].set_title("ROC, averaged over 40 users")
    axes[1].set_title("Zoom: low-error region (dots = EER)")
    axes[0].legend(loc="lower right", fontsize=9)
    _save(fig, path)


def plot_score_distributions(scores: pd.DataFrame, path,
                             models=("Scaled Manhattan", "Random Forest")):
    """Pooled genuine vs. impostor score histograms, one panel per impostor type."""
    _style()
    types = ["zero-effort", "external", "mimicry"]
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 3.4 * len(models)))
    for r, m in enumerate(models):
        sm = scores[scores.model == m]
        # Distance scores are z-normalised per user (validation genuine stats);
        # the RF output is already a probability in [0, 1].
        col = "score" if m == "Random Forest" else "score_norm"
        gen = sm.loc[sm.kind == "genuine", col]
        lo, hi = np.percentile(sm[col], [0.5, 99.8]) if col == "score_norm" else (0, 1)
        bins = np.linspace(lo, hi, 60)
        for c, t in enumerate(types):
            ax = axes[r, c]
            imp = sm.loc[sm.kind == t, col]
            ax.hist(gen.clip(lo, hi), bins=bins, density=True, alpha=0.55, color=BLUE,
                    label=f"genuine (n={len(gen)})")
            ax.hist(imp.clip(lo, hi), bins=bins, density=True, alpha=0.55, color=ORANGE,
                    label=f"{t} impostor (n={len(imp)})")
            ax.set_title(f"{m}: genuine vs {t}")
            ax.set_xlabel("probability of genuine class" if col == "score"
                          else "score (z-normalised per user)")
            ax.set_ylabel("density")
            ax.legend(fontsize=8, loc="upper left")
    _save(fig, path)


def plot_per_user_eer(per_user: pd.DataFrame, path,
                      models=("Scaled Manhattan", "Random Forest")):
    """Per-user EER bars for the best distance model and the classifier."""
    _style()
    piv = per_user.pivot(index="user", columns="model", values="eer")[list(models)]
    piv = piv.sort_values(models[0])
    x = np.arange(len(piv))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, m in enumerate(models):
        ax.bar(x + (i - 0.5) * w, piv[m] * 100, width=w - 0.04, color=MODEL_COLORS[m],
               label=f"{m} (mean {piv[m].mean():.1%})")
        ax.axhline(piv[m].mean() * 100, color=MODEL_COLORS[m], lw=1.2, ls="--")
    ax.set_xticks(x, piv.index, rotation=90, fontsize=8)
    ax.set_xlim(-0.7, len(piv) - 0.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("user (sorted by Scaled Manhattan EER)")
    ax.set_title("Per-user Equal Error Rate (dashed = mean over users)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_impostor_types(summary: pd.DataFrame, path):
    _style()
    types = ["zero_effort", "external", "mimicry"]
    labels = {"zero_effort": "zero-effort", "external": "external", "mimicry": "mimicry"}
    models = list(summary.model)
    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, t in enumerate(types):
        v = summary[f"eer_{t}_mean"].to_numpy() * 100
        bars = ax.bar(x + (i - 1) * w, v, width=w - 0.03, color=TYPE_COLORS[labels[t]],
                      label=f"{labels[t]} impostors")
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, models)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean per-user EER (%)")
    ax.set_title("EER by impostor type")
    ax.legend()
    _save(fig, path)


def plot_learning_curve(lc: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, g in lc.groupby("model", sort=False):
        ax.plot(g.n_enrol, g["mean"] * 100, marker="o", ms=7, color=MODEL_COLORS[m],
                label=f"{m} ({g['mean'].iloc[-1]:.1%} at n={g.n_enrol.iloc[-1]})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lc.n_enrol.unique(), [str(v) for v in lc.n_enrol.unique()])
    ax.set_xlabel("enrolment samples per user")
    ax.set_ylabel("mean per-user EER (%)")
    ax.set_title("Learning curve: EER vs. enrolment size")
    ax.legend()
    _save(fig, path)


def plot_feature_ablation(ab: pd.DataFrame, path):
    """Heat-map of mean EER (rows = feature subsets, cols = models)."""
    _style()
    piv = ab.pivot(index="features", columns="model", values="mean")
    piv = piv.loc[ab.features.unique(), ab.model.unique()] * 100
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(piv.to_numpy(), cmap="Blues_r", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=15)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    vmid = np.nanmean(piv.to_numpy())
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v < vmid else INK)
    fig.colorbar(im, ax=ax, label="mean EER (%)")
    ax.set_title("Feature ablation: mean per-user EER (%)")
    _save(fig, path)


def plot_session_eer(sess: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for m, g in sess.groupby("model", sort=False):
        ax.plot(g.session, g["mean"] * 100, marker="o", ms=7, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(sorted(sess.session.unique()))
    ax.set_xlabel("test session of the genuine samples (enrolment = sessions 1-4)")
    ax.set_ylabel("mean per-user EER (%)")
    ax.set_title("Template ageing: EER per test session")
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_data_overview(feats: pd.DataFrame, path, users=("u01", "u02", "u03", "u04")):
    """Mean +/- std hold and down-down profiles of a few users."""
    _style()
    tok = key_tokens()
    enr = feats[feats.population == "enrolled"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for prefix, ax, xl in (("H.", axes[0], tok), ("DD.", axes[1],
                                                   [f"{a}>{b}" for a, b in zip(tok[:-1], tok[1:])])):
        cols = [c for c in feats.columns if c.startswith(prefix)]
        x = np.arange(len(cols))
        for k, (u, c) in enumerate(zip(users, colors)):
            g = enr[enr.subject == u][cols]
            ax.errorbar(x + (k - 1.5) * 0.12, g.mean(), yerr=g.std(), fmt="o-", ms=4, lw=1.5,
                        color=c, capsize=2, label=u)
        ax.set_xticks(x, xl, rotation=60, fontsize=8)
        ax.set_ylabel("milliseconds")
    axes[0].set_title("Hold time per key (mean ± sd)")
    axes[1].set_title("Down-down latency per digraph (mean ± sd)")
    axes[0].legend(fontsize=8)
    pop = feats.groupby("population")
    for (p, g), c in zip(pop, [BLUE, ORANGE, AQUA]):
        axes[2].hist(g.speed_cps, bins=50, density=True, alpha=0.55, color=c, label=p)
    axes[2].set_title("Typing speed by population")
    axes[2].set_xlabel("characters per second")
    axes[2].set_ylabel("density")
    axes[2].legend()
    _save(fig, path)
