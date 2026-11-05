"""Feature extraction: raw press/release events -> fixed-length timing vector.

For a passphrase of ``n`` keys (including Return) each sample yields

* ``H.<k>``        hold time of key k                     (n features)
* ``DD.<k>.<k+1>`` down-down latency press(k+1)-press(k)  (n-1 features)
* ``UD.<k>.<k+1>`` up-down latency  press(k+1)-release(k) (n-1 features,
                   negative when keys overlap / "rollover")
* ``total_ms``     press of first key -> release of last key
* ``speed_cps``    typing speed in characters per second

This mirrors the feature set of the CMU benchmark (Killourhy & Maxion 2009)
plus two global rhythm descriptors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generation import key_tokens

META_COLS = ["sample_id", "subject", "population", "target", "session", "rep"]


def feature_names(include_global: bool = True) -> list[str]:
    t = key_tokens()
    names = [f"H.{k}" for k in t]
    names += [f"DD.{a}.{b}" for a, b in zip(t[:-1], t[1:])]
    names += [f"UD.{a}.{b}" for a, b in zip(t[:-1], t[1:])]
    if include_global:
        names += ["total_ms", "speed_cps"]
    return names


def extract_features(events: pd.DataFrame) -> pd.DataFrame:
    """Vectorised extraction over a long event table (one row per key)."""
    events = events.sort_values(["sample_id", "key_index"])
    n_keys = events["key_index"].max() + 1
    n_samples = events["sample_id"].nunique()
    if len(events) != n_samples * n_keys:
        raise ValueError("every sample must contain exactly one event per key")

    press = events["press_ms"].to_numpy().reshape(n_samples, n_keys)
    release = events["release_ms"].to_numpy().reshape(n_samples, n_keys)

    hold = release - press
    dd = press[:, 1:] - press[:, :-1]
    ud = press[:, 1:] - release[:, :-1]
    total = release[:, -1] - press[:, 0]
    speed = n_keys / (total / 1000.0)

    X = np.hstack([hold, dd, ud, total[:, None], speed[:, None]])
    meta = events.groupby("sample_id", sort=True)[META_COLS[1:]].first().reset_index()
    feats = pd.DataFrame(X, columns=feature_names())
    return pd.concat([meta, feats], axis=1)


# Named feature subsets used in the ablation study.
def feature_subsets() -> dict[str, list[str]]:
    names = feature_names()
    pick = lambda p: [c for c in names if c.startswith(p)]  # noqa: E731
    return {
        "H": pick("H."),
        "DD": pick("DD."),
        "UD": pick("UD."),
        "H+DD": pick("H.") + pick("DD."),
        "H+UD": pick("H.") + pick("UD."),
        "H+DD+UD": pick("H.") + pick("DD.") + pick("UD."),
        "All (+speed)": names,
    }
