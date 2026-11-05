"""Per-user verification protocol: who is used for what.

For every enrolled *target* user ``u`` (one verification problem per user):

  genuine reps   0 .. 79  (sessions 1-4)  -> fit the model (enrolment)
  genuine reps  80 .. 99  (session 4)     -> validation: choose the threshold
  genuine reps 100 .. 199 (sessions 5-8)  -> TEST genuine (later sessions =>
                                             realistic template ageing)

The remaining 39 enrolled users are randomly partitioned (seeded per target):

  20 background users  -> reps 0-9 are training negatives (only used by the
                          two-class classifier), reps 10-14 validation negatives
  19 held-out users    -> reps 100-109 are TEST zero-effort impostors

Plus, at test time only:
  external impostors   -> 10 reps of each of 20 never-enrolled users
  mimicry attacks      -> the 50 imitation attempts generated for target u

Background and test impostor populations are disjoint, so the classifier is
never evaluated on impostors it has seen during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProtocolConfig


@dataclass
class UserSplit:
    target: str
    X_fit: np.ndarray
    X_val_gen: np.ndarray
    X_bg_fit: np.ndarray
    X_bg_val: np.ndarray
    X_test_gen: np.ndarray
    test_gen_session: np.ndarray
    X_test_imp: np.ndarray
    test_imp_type: np.ndarray      # 'zero-effort' | 'external' | 'mimicry'
    test_imp_subject: np.ndarray


class ProtocolData:
    """Pre-indexes the feature table into dense arrays for fast splitting."""

    def __init__(self, feats: pd.DataFrame, feature_cols: list[str],
                 cfg: ProtocolConfig, seed: int):
        self.cfg, self.seed, self.cols = cfg, seed, feature_cols
        enr = feats[feats.population == "enrolled"].sort_values(["subject", "rep"])
        self.users = sorted(enr.subject.unique())
        n_rep = enr.groupby("subject").size().iloc[0]
        self.E = enr[feature_cols].to_numpy().reshape(len(self.users), n_rep, -1)
        self.E_session = enr["session"].to_numpy().reshape(len(self.users), n_rep)

        ext = feats[feats.population == "external"].sort_values(["subject", "rep"])
        self.ext_users = sorted(ext.subject.unique())
        n_ext_rep = ext.groupby("subject").size().iloc[0]
        self.X_ext = ext[feature_cols].to_numpy().reshape(len(self.ext_users), n_ext_rep, -1)

        mim = feats[feats.population == "mimic"]
        self.mimic = {t: g[feature_cols].to_numpy() for t, g in mim.groupby("target")}
        self.mimic_subj = {t: g["subject"].to_numpy() for t, g in mim.groupby("target")}

    def split(self, target: str, n_fit: int | None = None) -> UserSplit:
        c = self.cfg
        ui = self.users.index(target)
        n_fit = c.n_fit if n_fit is None else n_fit
        G = self.E[ui]

        # Deterministic partition of the other enrolled users for this target.
        others = [i for i in range(len(self.users)) if i != ui]
        rng = np.random.default_rng([self.seed, ui])
        others = list(rng.permutation(others))
        bg, held = others[: c.n_background_users], others[c.n_background_users:]

        bg_fit = self.E[bg, : c.background_fit_reps].reshape(-1, G.shape[1])
        bg_val = self.E[bg, c.background_fit_reps:
                        c.background_fit_reps + c.background_val_reps].reshape(-1, G.shape[1])

        sl = slice(c.test_start, c.test_start + c.impostor_test_reps)
        zero = self.E[held, sl].reshape(-1, G.shape[1])
        zero_subj = np.repeat([self.users[i] for i in held], c.impostor_test_reps)
        ext = self.X_ext[:, : c.external_test_reps].reshape(-1, G.shape[1])
        ext_subj = np.repeat(self.ext_users, c.external_test_reps)
        mim = self.mimic[target]

        X_imp = np.vstack([zero, ext, mim])
        imp_type = np.array(["zero-effort"] * len(zero) + ["external"] * len(ext)
                            + ["mimicry"] * len(mim))
        imp_subj = np.r_[zero_subj, ext_subj, self.mimic_subj[target]]

        return UserSplit(
            target=target,
            # Enrolment uses the first n_fit reps (learning-curve experiments
            # shrink n_fit; validation and test sets stay fixed).
            X_fit=G[:n_fit],
            X_val_gen=G[c.n_fit: c.n_fit + c.n_val],
            X_bg_fit=bg_fit, X_bg_val=bg_val,
            X_test_gen=G[c.test_start:],
            test_gen_session=self.E_session[ui, c.test_start:],
            X_test_imp=X_imp, test_imp_type=imp_type, test_imp_subject=imp_subj,
        )
