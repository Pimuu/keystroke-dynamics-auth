"""Synthetic keystroke-dynamics dataset generator.

Why synthetic?  The canonical public benchmark (CMU DSL-StrongPassword,
Killourhy & Maxion 2009) was not downloadable from the build environment, so
the pipeline ships with a documented generative model instead.  The model is
deliberately *not* a trivially separable Gaussian blob per user: it encodes
the main sources of variation reported in the keystroke literature.

Generative model (all times in milliseconds)
--------------------------------------------
For every user ``u`` and key position ``k`` of the passphrase:

* **Mean hold time**  ``mu_H[u,k] = base_hold * key_factor_H[k] * h_u * eps_uk``
  where ``h_u`` is the user's global hold tendency (log-normal, correlated with
  typing speed) and ``eps_uk`` a per-key idiosyncrasy (log-normal).
* **Mean down-down latency** ``mu_DD[u,k] = base_dd * digraph_factor[k] * s_u * eta_uk``
  where ``digraph_factor`` comes from keyboard geometry (hand alternation is
  fast, same-finger transitions and Shift/number-row reaches are slow),
  ``s_u`` is the user's speed and ``eta_uk`` a per-digraph idiosyncrasy.

A single typed sample (repetition ``r`` in session ``q``) then draws

* a whole-sample **tempo** factor (the user is a bit faster/slower today),
* a **learning** slow-down ``1 + g_u * exp(-r / tau_u)`` (users get faster as
  they become familiar with the phrase -- this makes early enrolment samples
  differ from later test samples, i.e. template ageing),
* **session drift** (global and per-feature log-normal offsets per session),
* independent log-normal **jitter** per hold/latency with user-specific CV,
* rare **hesitation pauses** (exponential, added to a latency).

Up-down (flight) times are *derived*: ``UD_k = DD_k - H_k`` and can therefore be
negative when a fast typist presses the next key before releasing the
previous one ("rollover"), exactly as in real key-logger data.

Three impostor populations are generated:

1. *zero-effort*: other enrolled users typing their own rhythm (built in);
2. *external*: ``n_external`` never-enrolled users (unseen at training time);
3. *mimicry*: for each target, attackers whose timing profile is pulled
   towards the target's in log space (weight ``alpha``) with extra variability
   -- a model of an attacker who has observed the victim's rhythm
   (cf. Tey et al., NDSS 2013).

The output is a *raw event log* (one row per key press/release), so feature
extraction in ``features.py`` works exactly as it would on real data.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import EVENTS_FILE, KEY_GEOMETRY, PASSPHRASE, ExperimentConfig, GenerationConfig


# --------------------------------------------------------------------------
# Passphrase helpers
# --------------------------------------------------------------------------
def passphrase_keys(phrase: str = PASSPHRASE) -> list[str]:
    """Physical keys of the passphrase, terminated by Return."""
    return list(phrase) + ["Return"]


def key_tokens(phrase: str = PASSPHRASE) -> list[str]:
    """Unique, column-safe token per key position (``s``, ``s2``, ``period`` ...)."""
    rename = {".": "period", "4": "four", "2": "two"}
    seen: dict[str, int] = {}
    tokens = []
    for k in passphrase_keys(phrase):
        base = rename.get(k, k)
        seen[base] = seen.get(base, 0) + 1
        tokens.append(base if seen[base] == 1 else f"{base}{seen[base]}")
    return tokens


def hold_factor(key: str) -> float:
    """Relative difficulty/duration of holding a key."""
    hand, finger, row = KEY_GEOMETRY[key]
    f = 1.0
    if key == "B":
        f *= 1.30   # Shift chord: the letter is held while Shift is managed
    if key == "Return":
        f *= 1.15
    if row == 0:
        f *= 1.05   # number-row reach
    if finger == 3:
        f *= 1.05   # pinky
    return f


def digraph_factor(k1: str, k2: str) -> float:
    """Relative down-down latency of the transition ``k1 -> k2``."""
    h1, f1, r1 = KEY_GEOMETRY[k1]
    h2, f2, r2 = KEY_GEOMETRY[k2]
    f = 1.0
    if h1 != h2:
        f *= 0.75                      # hand alternation is fast
    elif k1 == k2:
        f *= 1.10                      # key repeat
    elif f1 == f2:
        f *= 1.35                      # same finger, different key: slow
    f *= 1.0 + 0.08 * abs(r1 - r2)     # vertical travel
    if r2 == 0:
        f *= 1.15                      # reaching the number row
    if k2 == "B":
        f *= 1.45                      # preparing the Shift chord
    if k2 == ".":
        f *= 1.20                      # punctuation
    if k1 == ".":
        f *= 1.10                      # cognitive chunk boundary
    if k2 == "Return":
        f *= 1.25                      # end-of-phrase confirmation
    return f


def _lognormal_mean_one(rng: np.random.Generator, cv, size) -> np.ndarray:
    """Multiplicative noise with mean 1 and coefficient of variation ``cv``."""
    sigma = np.sqrt(np.log1p(np.asarray(cv) ** 2))
    return np.exp(rng.normal(-0.5 * sigma ** 2, sigma, size=size))


# --------------------------------------------------------------------------
# User profiles
# --------------------------------------------------------------------------
def draw_profile(rng: np.random.Generator, cfg: GenerationConfig) -> dict:
    """Draw one user's latent typing profile."""
    keys = passphrase_keys()
    n = len(keys)
    z_speed = rng.normal()
    # Fast typists tend to also have short holds (correlation 0.4).
    z_hold = 0.4 * z_speed + np.sqrt(1 - 0.4 ** 2) * rng.normal()
    speed = np.exp(cfg.sigma_user_speed * z_speed)
    hold = np.exp(cfg.sigma_user_hold * z_hold)

    hf = np.array([hold_factor(k) for k in keys])
    df = np.array([digraph_factor(keys[i], keys[i + 1]) for i in range(n - 1)])
    mu_h = cfg.base_hold_ms * hf * hold * np.exp(rng.normal(0, cfg.sigma_key_hold, n))
    mu_dd = cfg.base_dd_ms * df * speed * np.exp(rng.normal(0, cfg.sigma_key_dd, n - 1))

    u = lambda r: rng.uniform(*r)  # noqa: E731 - tiny local helper
    return {
        "mu_h": mu_h,
        "mu_dd": mu_dd,
        "cv_h": u(cfg.cv_hold_range),
        "cv_dd": u(cfg.cv_dd_range),
        "tempo_sigma": u(cfg.tempo_sigma_range),
        "pause_prob": u(cfg.pause_prob_range),
        "learn_gain": u(cfg.learning_gain_range),
        "learn_tau": u(cfg.learning_tau_range),
    }


def session_offsets(rng, cfg: GenerationConfig, n_sessions: int, n_keys: int) -> dict:
    """Per-session log-space drift of one user (global + per feature)."""
    return {
        "g": rng.normal(0, cfg.session_sigma, n_sessions),
        "h": rng.normal(0, cfg.session_key_sigma, (n_sessions, n_keys)),
        "dd": rng.normal(0, cfg.session_key_sigma, (n_sessions, n_keys - 1)),
    }


def mimic_profile(attacker: dict, target: dict, alpha: float, cfg: GenerationConfig) -> dict:
    """Attacker profile pulled towards the target's means (log-space blend)."""
    p = dict(attacker)
    for key in ("mu_h", "mu_dd"):
        p[key] = np.exp((1 - alpha) * np.log(attacker[key]) + alpha * np.log(target[key]))
    # Imitating a foreign rhythm is harder to do consistently.
    p["cv_h"] = attacker["cv_h"] * cfg.mimic_extra_cv
    p["cv_dd"] = attacker["cv_dd"] * cfg.mimic_extra_cv
    p["tempo_sigma"] = attacker["tempo_sigma"] * cfg.mimic_extra_cv
    return p


# --------------------------------------------------------------------------
# Sampling one repetition
# --------------------------------------------------------------------------
def sample_repetition(rng, prof: dict, rep: int, sess: int, offs: dict | None,
                      cfg: GenerationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (press_times, release_times) for one typed passphrase."""
    n = len(prof["mu_h"])
    tempo = np.exp(rng.normal(0, prof["tempo_sigma"]))
    learn = 1.0 + prof["learn_gain"] * np.exp(-rep / prof["learn_tau"])
    if offs is None:
        g, dh, ddd = 0.0, np.zeros(n), np.zeros(n - 1)
    else:
        g, dh, ddd = offs["g"][sess], offs["h"][sess], offs["dd"][sess]

    hold = (prof["mu_h"] * tempo ** 0.3 * learn ** 0.3 * np.exp(0.5 * g + dh)
            * _lognormal_mean_one(rng, prof["cv_h"], n))
    dd = (prof["mu_dd"] * tempo * learn * np.exp(g + ddd)
          * _lognormal_mean_one(rng, prof["cv_dd"], n - 1))
    pauses = rng.random(n - 1) < prof["pause_prob"]
    dd = dd + pauses * rng.exponential(cfg.pause_mean_ms, n - 1)
    hold = np.maximum(hold, 20.0)  # physical lower bounds of a key stroke
    dd = np.maximum(dd, 25.0)

    press = np.concatenate([[0.0], np.cumsum(dd)])
    release = press + hold
    q = cfg.timer_resolution_ms
    return np.round(press / q) * q, np.round(release / q) * q


def _enrolled_profiles(rng_prof, cfg: GenerationConfig, n_keys: int):
    """Latent profiles + session offsets of all enrolled users (one RNG stream)."""
    profiles, offsets = {}, {}
    for i in range(cfg.n_enrolled):
        sid = f"u{i + 1:02d}"
        profiles[sid] = draw_profile(rng_prof, cfg)
        offsets[sid] = session_offsets(rng_prof, cfg, cfg.n_sessions, n_keys)
    return profiles, offsets


def latent_user_parameters(seed: int = 42, cfg: GenerationConfig | None = None) -> pd.DataFrame:
    """Ground-truth generator parameters of each enrolled user (for analysis only).

    Available only because the data is synthetic; lets us ask *why* some users
    are harder to verify than others.
    """
    cfg = cfg or GenerationConfig()
    rng_prof = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[0])
    profiles, _ = _enrolled_profiles(rng_prof, cfg, len(passphrase_keys()))
    rows = []
    for sid, p in profiles.items():
        rows.append({"user": sid, "cv_hold": p["cv_h"], "cv_dd": p["cv_dd"],
                     "tempo_sigma": p["tempo_sigma"], "pause_prob": p["pause_prob"],
                     "learn_gain": p["learn_gain"], "learn_tau": p["learn_tau"],
                     "mean_hold_ms": p["mu_h"].mean(), "mean_dd_ms": p["mu_dd"].mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Whole dataset
# --------------------------------------------------------------------------
def generate_dataset(seed: int = 42, cfg: GenerationConfig | None = None) -> pd.DataFrame:
    """Generate the full raw event log.

    Columns: sample_id, subject, population (enrolled|external|mimic),
    target (victim of a mimic sample, else ''), session, rep, key_index,
    key, press_ms, release_ms.
    """
    cfg = cfg or GenerationConfig()
    keys = passphrase_keys()
    n_keys = len(keys)
    # Independent streams so that e.g. changing the number of mimic attempts
    # does not change the enrolled users' data.
    ss = np.random.SeedSequence(seed)
    rng_prof, rng_samp, rng_ext, rng_mim = (np.random.default_rng(s) for s in ss.spawn(4))

    rows = []  # (sample_id, subject, population, target, session, rep, press, release)

    # 1) Enrolled users -------------------------------------------------------
    enrolled, offsets = _enrolled_profiles(rng_prof, cfg, n_keys)
    reps_per_session = cfg.reps_per_user // cfg.n_sessions
    for sid, prof in enrolled.items():
        offs = offsets[sid]
        for r in range(cfg.reps_per_user):
            sess = min(r // reps_per_session, cfg.n_sessions - 1)
            p, rel = sample_repetition(rng_samp, prof, r, sess, offs, cfg)
            rows.append((sid, "enrolled", "", sess + 1, r, p, rel))

    # 2) External (never enrolled) impostors ----------------------------------
    external = {}
    for j in range(cfg.n_external):
        sid = f"x{j + 1:02d}"
        prof = draw_profile(rng_ext, cfg)
        external[sid] = prof
        for r in range(cfg.reps_external):
            p, rel = sample_repetition(rng_ext, prof, r, 0, None, cfg)
            rows.append((sid, "external", "", 1, r, p, rel))

    # 3) Mimicry attacks: 5 attackers x (n_mimic/5) attempts per target -------
    ext_ids = list(external)
    n_att = 5
    per_att = cfg.n_mimic_per_target // n_att
    for tid, tprof in enrolled.items():
        attackers = rng_mim.choice(ext_ids, size=n_att, replace=False)
        for a in attackers:
            alpha = rng_mim.uniform(*cfg.mimic_alpha_range)
            mprof = mimic_profile(external[a], tprof, alpha, cfg)
            for r in range(per_att):
                # Attackers have practised (rep offset 30 => mostly learned).
                p, rel = sample_repetition(rng_mim, mprof, 30 + r, 0, None, cfg)
                rows.append((f"{a}>{tid}", "mimic", tid, 1, r, p, rel))

    # Flatten into a long event table ---------------------------------------
    n_samples = len(rows)
    df = pd.DataFrame({
        "sample_id": np.repeat(np.arange(n_samples), n_keys),
        "subject": np.repeat([r[0] for r in rows], n_keys),
        "population": np.repeat([r[1] for r in rows], n_keys),
        "target": np.repeat([r[2] for r in rows], n_keys),
        "session": np.repeat([r[3] for r in rows], n_keys),
        "rep": np.repeat([r[4] for r in rows], n_keys),
        "key_index": np.tile(np.arange(n_keys), n_samples),
        "key": np.tile(keys, n_samples),
        "press_ms": np.concatenate([r[5] for r in rows]),
        "release_ms": np.concatenate([r[6] for r in rows]),
    })
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the synthetic keystroke event log.")
    ap.add_argument("--seed", type=int, default=ExperimentConfig().seed)
    ap.add_argument("--out", default=str(EVENTS_FILE))
    args = ap.parse_args()
    df = generate_dataset(args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {df.sample_id.nunique()} samples / {len(df)} events -> {args.out}")


if __name__ == "__main__":
    main()
