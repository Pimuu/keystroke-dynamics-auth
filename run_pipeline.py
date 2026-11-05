"""End-to-end reproducible pipeline.

    python run_pipeline.py            # full run (~5-10 min on 4 CPU cores)
    python run_pipeline.py --quick    # skip the multi-seed robustness study

Steps: generate synthetic event log -> extract features -> per-user
verification for 5 models -> metrics CSVs + figures in results/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from src import plots
from src.config import (EVENTS_FILE, FEATURES_FILE, FIG_DIR, RESULTS_DIR,
                        ExperimentConfig)
from src.data_generation import generate_dataset, latent_user_parameters
from src.experiments import (compare_models, run_feature_ablation, run_learning_curve, run_main,
                             run_seed_robustness, run_session_analysis, summarize,
                             user_difficulty)
from src.features import extract_features


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="skip multi-seed robustness study")
    args = ap.parse_args()

    cfg = ExperimentConfig(seed=args.seed)
    t0 = time.time()
    for d in (RESULTS_DIR, FIG_DIR, EVENTS_FILE.parent):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Data --------------------------------------------------------------
    events = generate_dataset(cfg.seed, cfg.gen)
    events.to_csv(EVENTS_FILE, index=False, float_format="%.1f")
    log(f"generated {events.sample_id.nunique()} samples ({len(events)} key events) "
        f"-> {EVENTS_FILE.relative_to(EVENTS_FILE.parents[1])}", t0)

    # Re-read from disk so the pipeline exercises the same path real data would.
    feats = extract_features(pd.read_csv(EVENTS_FILE, keep_default_na=False))
    feats.to_csv(FEATURES_FILE, index=False, float_format="%.4f")
    log(f"extracted {feats.shape[1] - 6} features per sample", t0)

    # 2) Main experiment -----------------------------------------------------
    per_user, scores, mean_roc = run_main(feats, cfg)
    summary = summarize(per_user, mean_roc)
    per_user.to_csv(RESULTS_DIR / "metrics_per_user.csv", index=False, float_format="%.5f")
    summary.to_csv(RESULTS_DIR / "metrics_overall.csv", index=False, float_format="%.5f")
    scores.to_csv(RESULTS_DIR / "scores.csv.gz", index=False, float_format="%.5f")
    pd.DataFrame({"fpr": np.round(np.linspace(0, 1, 1001), 4),
                  **{m: v for m, v in mean_roc.items()}}).to_csv(
        RESULTS_DIR / "roc_mean.csv", index=False, float_format="%.5f")
    log("main experiment done", t0)
    cols = ["model", "eer_mean", "eer_std", "auc_mean", "far_mean", "frr_mean",
            "accuracy_mean", "hter_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name = summary.sort_values("eer_mean").model.iloc[0]
    cmp_ = compare_models(per_user, reference=best_name)
    cmp_.to_csv(RESULTS_DIR / "model_comparison.csv", index=False, float_format="%.5g")
    print(cmp_.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    diff = user_difficulty(per_user, latent_user_parameters(cfg.seed, cfg.gen))
    diff.to_csv(RESULTS_DIR / "user_difficulty_spearman.csv", index=False, float_format="%.4f")

    sess = run_session_analysis(scores)
    sess.to_csv(RESULTS_DIR / "session_eer.csv", index=False, float_format="%.5f")

    # 3) Secondary experiments ----------------------------------------------
    lc = run_learning_curve(feats, cfg)
    lc.to_csv(RESULTS_DIR / "learning_curve.csv", index=False, float_format="%.5f")
    log("learning curve done", t0)
    ab = run_feature_ablation(feats, cfg)
    ab.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False, float_format="%.5f")
    log("feature ablation done", t0)
    rob = None
    if not args.quick:
        rob = run_seed_robustness(cfg)
        rob.to_csv(RESULTS_DIR / "seed_robustness.csv", index=False, float_format="%.5f")
        log("seed robustness done", t0)

    # 4) Figures ------------------------------------------------------------
    plots.plot_roc(mean_roc, FIG_DIR / "roc_curves.png")
    plots.plot_score_distributions(scores, FIG_DIR / "score_distributions.png")
    plots.plot_per_user_eer(per_user, FIG_DIR / "per_user_eer.png")
    plots.plot_impostor_types(summary, FIG_DIR / "eer_by_impostor_type.png")
    plots.plot_learning_curve(lc, FIG_DIR / "learning_curve.png")
    plots.plot_feature_ablation(ab, FIG_DIR / "feature_ablation.png")
    plots.plot_session_eer(sess, FIG_DIR / "session_eer.png")
    plots.plot_data_overview(feats, FIG_DIR / "data_overview.png")
    log("figures written", t0)

    # 5) Machine-readable summary -------------------------------------------
    best = summary.sort_values("eer_mean").iloc[0]
    out = {
        "seed": cfg.seed,
        "n_enrolled_users": int(feats[feats.population == "enrolled"].subject.nunique()),
        "n_samples": int(len(feats)),
        "best_model_by_mean_eer": best.model,
        "overall": summary.set_index("model").round(5).to_dict(orient="index"),
        "paired_comparison_vs_best": cmp_.set_index("model").round(5).to_dict(orient="index"),
        "seed_robustness": (None if rob is None else
                            rob.groupby("model", sort=False).eer.agg(["mean", "std"])
                            .round(5).to_dict(orient="index")),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__},
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    log(f"done. best model by mean EER: {best.model} ({best.eer_mean:.2%})", t0)


if __name__ == "__main__":
    main()
