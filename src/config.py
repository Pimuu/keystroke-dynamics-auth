"""Central configuration for the keystroke-dynamics verification pipeline.

Every number that influences the dataset or the experimental protocol lives
here so that a single seed + this file fully determine every result.
"""
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

EVENTS_FILE = DATA_DIR / "keystroke_events.csv.gz"   # raw press/release log
FEATURES_FILE = DATA_DIR / "keystroke_features.csv"  # one row per typed sample

# --------------------------------------------------------------------------
# Passphrase and keyboard geometry
# --------------------------------------------------------------------------
# The fixed passphrase every subject types (terminated by Return, as in the
# CMU benchmark).  The capital "B" requires Shift, "." is punctuation and
# "42" sits on the number row -- this mix produces realistic rhythm variety.
PASSPHRASE = "sunset.Bay42"

# Key -> (hand, finger, row) on a QWERTY layout.
# hand: 0 = left, 1 = right; finger: 0 = index .. 3 = pinky;
# row: 0 = number row, 1 = top, 2 = home, 3 = bottom.
KEY_GEOMETRY = {
    "s": (0, 2, 2), "u": (1, 0, 1), "n": (1, 0, 3), "e": (0, 1, 1),
    "t": (0, 0, 1), ".": (1, 2, 3), "B": (0, 0, 3), "a": (0, 3, 2),
    "y": (1, 0, 1), "4": (0, 0, 0), "2": (0, 2, 0), "Return": (1, 3, 2),
}


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters of the synthetic population (see data_generation.py)."""

    n_enrolled: int = 40          # users who enrol and are later verified
    reps_per_user: int = 200      # repetitions of the passphrase per user
    n_sessions: int = 8           # reps are spread evenly over sessions
    n_external: int = 20          # never-enrolled impostor population
    reps_external: int = 20       # repetitions per external impostor
    n_mimic_per_target: int = 50  # imitation attempts generated per target

    # Global timing scale (milliseconds)
    base_hold_ms: float = 95.0
    base_dd_ms: float = 210.0

    # Between-user spread (log-normal sigmas)
    sigma_user_speed: float = 0.30
    sigma_user_hold: float = 0.20
    sigma_key_hold: float = 0.15      # per-user, per-key idiosyncrasy
    sigma_key_dd: float = 0.22        # per-user, per-digraph idiosyncrasy

    # Within-user variability ranges (drawn per user, uniform)
    cv_hold_range: tuple = (0.08, 0.22)
    cv_dd_range: tuple = (0.12, 0.35)
    tempo_sigma_range: tuple = (0.04, 0.12)  # whole-sample speed jitter
    pause_prob_range: tuple = (0.005, 0.03)  # per-digraph hesitation prob.
    pause_mean_ms: float = 250.0

    # Non-stationarity
    session_sigma: float = 0.06        # session-level global drift
    session_key_sigma: float = 0.04    # session-level per-feature drift
    learning_gain_range: tuple = (0.10, 0.40)  # initial slow-down factor
    learning_tau_range: tuple = (15.0, 60.0)   # reps until ~63 % learned

    # Mimicry attacker: log-space interpolation weight towards the target
    mimic_alpha_range: tuple = (0.45, 0.85)
    mimic_extra_cv: float = 1.35       # imitation adds variability

    timer_resolution_ms: float = 0.1   # key-logger timestamp quantisation


@dataclass(frozen=True)
class ProtocolConfig:
    """Per-user verification protocol (see protocol.py)."""

    n_fit: int = 80               # genuine reps used to fit the model
    n_val: int = 20               # genuine reps used to pick the threshold
    test_start: int = 100         # genuine reps >= this index are test data
    n_background_users: int = 20  # other enrolled users used as training negatives
    background_fit_reps: int = 10
    background_val_reps: int = 5
    impostor_test_reps: int = 10  # reps per held-out zero-effort impostor
    external_test_reps: int = 10  # reps per external impostor
    learning_curve_sizes: tuple = (5, 10, 20, 40, 80)
    robustness_seeds: tuple = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    proto: ProtocolConfig = field(default_factory=ProtocolConfig)
