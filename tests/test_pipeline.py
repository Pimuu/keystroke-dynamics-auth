"""Sanity tests (stdlib unittest; run with ``python -m unittest discover tests``)."""
import unittest

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, GenerationConfig
from src.data_generation import generate_dataset, key_tokens
from src.features import extract_features, feature_names
from src.metrics import eer_from_scores, eer_threshold, rates_at_threshold
from src.models import make_models
from src.protocol import ProtocolData

SMALL = GenerationConfig(n_enrolled=6, reps_per_user=200, n_external=5,
                         reps_external=20, n_mimic_per_target=10)


class TestFeatures(unittest.TestCase):
    def test_hand_computed_sample(self):
        n = len(key_tokens())
        press = np.arange(n) * 200.0            # DD = 200 ms everywhere
        hold = np.full(n, 80.0)
        hold[0] = 250.0                          # key 0 overlaps key 1
        ev = pd.DataFrame({"sample_id": 0, "subject": "u", "population": "enrolled",
                           "target": "", "session": 1, "rep": 0,
                           "key_index": np.arange(n), "key": key_tokens(),
                           "press_ms": press, "release_ms": press + hold})
        f = extract_features(ev).iloc[0]
        self.assertAlmostEqual(f["H.s"], 250.0)
        self.assertAlmostEqual(f["DD.s.u"], 200.0)
        self.assertAlmostEqual(f["UD.s.u"], -50.0)   # rollover => negative flight
        self.assertAlmostEqual(f["UD.u.n"], 120.0)
        self.assertAlmostEqual(f["total_ms"], press[-1] + 80.0)
        self.assertEqual(len(feature_names()), 3 * n - 2 + 2)


class TestMetrics(unittest.TestCase):
    def test_eer_extremes(self):
        self.assertEqual(eer_from_scores(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 0.0)
        rng = np.random.default_rng(0)
        same = eer_from_scores(rng.normal(size=20000), rng.normal(size=20000))
        self.assertAlmostEqual(same, 0.5, delta=0.02)

    def test_gaussian_eer_matches_theory(self):
        # Two unit-variance Gaussians 2 sigma apart: EER = Phi(-1) = 0.1587
        rng = np.random.default_rng(1)
        e = eer_from_scores(rng.normal(2, 1, 50000), rng.normal(0, 1, 50000))
        self.assertAlmostEqual(e, 0.1587, delta=0.006)

    def test_threshold_balances_errors(self):
        rng = np.random.default_rng(2)
        g, i = rng.normal(2, 1, 5000), rng.normal(0, 1, 5000)
        r = rates_at_threshold(g, i, eer_threshold(g, i))
        self.assertAlmostEqual(r["far"], r["frr"], delta=0.01)


class TestDataAndProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = generate_dataset(7, SMALL)
        cls.feats = extract_features(cls.events)

    def test_deterministic(self):
        again = generate_dataset(7, SMALL)
        pd.testing.assert_frame_equal(self.events, again)

    def test_shapes_and_sanity(self):
        n_samples = 6 * 200 + 5 * 20 + 6 * 10
        self.assertEqual(self.feats.shape[0], n_samples)
        self.assertTrue((self.feats.filter(like="H.") > 0).all().all())
        self.assertTrue((self.feats.filter(like="DD.") > 0).all().all())

    def test_split_has_no_leakage(self):
        proto = ExperimentConfig().proto.__class__(n_background_users=3)
        pdata = ProtocolData(self.feats, feature_names(), proto, seed=0)
        s = pdata.split("u01")
        self.assertEqual(len(s.X_fit), 80)
        self.assertEqual(len(s.X_test_gen), 100)
        self.assertNotIn("u01", set(s.test_imp_subject))
        # Background users (training negatives) never appear as test impostors.
        bg = {tuple(r) for r in s.X_bg_fit.round(4)}
        self.assertFalse(bg & {tuple(r) for r in s.X_test_imp.round(4)})
        self.assertEqual(set(s.test_imp_type), {"zero-effort", "external", "mimicry"})

    def test_models_rank_genuine_higher(self):
        pdata = ProtocolData(self.feats, feature_names(),
                             ExperimentConfig().proto.__class__(n_background_users=3), seed=0)
        s = pdata.split("u02")
        for m in make_models():
            m.fit(s.X_fit, s.X_bg_fit)
            self.assertGreater(np.median(m.score(s.X_test_gen)),
                               np.median(m.score(s.X_test_imp)), m.name)


if __name__ == "__main__":
    unittest.main()
