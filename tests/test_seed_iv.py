import unittest

import numpy as np
import torch

from data.seed_iv import (
    SeedIVArrays,
    get_subject_split,
    standardize_from_training_subjects,
)
from models.seed_iv_baselines import SeedIVBaseline
from models.seed_iv_fuzzy_attention import (
    IntervalType2FuzzyReliability,
    SeedIVFuzzyAttention,
)


class SeedIVDataTests(unittest.TestCase):
    def test_loso_subject_folds_are_disjoint_complete_and_balanced(self):
        test_subjects = []
        validation_subjects = []
        for fold in range(15):
            split = get_subject_split(fold)
            train = set(split["train"])
            validation = set(split["validation"])
            test = set(split["test"])
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, set(range(1, 16)))
            self.assertEqual((len(train), len(validation), len(test)), (13, 1, 1))
            test_subjects.extend(test)
            validation_subjects.extend(validation)

        self.assertEqual(sorted(test_subjects), list(range(1, 16)))
        self.assertEqual(sorted(validation_subjects), list(range(1, 16)))

    def test_standardization_uses_training_subjects_and_imputes_nan(self):
        eeg = np.arange(4 * 62 * 5, dtype=np.float32).reshape(4, 62, 5)
        eye = np.arange(4 * 31, dtype=np.float32).reshape(4, 31)
        eye[0, 5] = np.nan
        arrays = SeedIVArrays(
            eeg=eeg,
            eye=eye,
            labels=np.array([0, 1, 2, 3]),
            subjects=np.array([1, 1, 2, 2]),
            sessions=np.ones(4),
            trials=np.ones(4),
        )
        normalized = standardize_from_training_subjects(arrays, (1,))
        self.assertTrue(np.isfinite(normalized.eeg).all())
        self.assertTrue(np.isfinite(normalized.eye).all())
        self.assertAlmostEqual(float(normalized.eye[0, 5]), 0.0)


class SeedIVModelTests(unittest.TestCase):
    def test_all_baselines_have_four_class_logits_and_gradients(self):
        eeg = torch.randn(3, 62, 5)
        eye = torch.randn(3, 31)
        for mode in SeedIVBaseline.valid_modes:
            with self.subTest(mode=mode):
                model = SeedIVBaseline(mode=mode)
                logits = model(eeg, eye)
                self.assertEqual(logits.shape, (3, 4))
                logits.sum().backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.parameters()
                    )
                )

    def test_attention_ablation_diagnostics(self):
        model = SeedIVFuzzyAttention(use_fuzzy=False)
        logits, diagnostics = model(
            torch.randn(2, 62, 5),
            torch.randn(2, 31),
            return_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(diagnostics["attention"].shape, (1, 2, 4, 10, 10))
        attention_sum = diagnostics["attention"].sum(dim=-1)
        self.assertTrue(torch.allclose(attention_sum, torch.ones_like(attention_sum)))

    def test_it2_reliability_intervals_and_gradients(self):
        fuzzy = IntervalType2FuzzyReliability(num_rules=8)
        weights, diagnostics = fuzzy(torch.rand(3, 5, 3))
        self.assertEqual(weights.shape, (3, 5))
        self.assertTrue(
            torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)
        )
        self.assertTrue(
            torch.all(
                diagnostics["interval_upper"]
                >= diagnostics["interval_lower"]
            )
        )
        weights[:, 0].sum().backward()
        self.assertIsNotNone(fuzzy.center_logits.grad)

    def test_full_fuzzy_attention_diagnostics_and_gradients(self):
        model = SeedIVFuzzyAttention(use_fuzzy=True)
        logits, diagnostics = model(
            torch.randn(2, 62, 5),
            torch.randn(2, 31),
            return_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(diagnostics["eeg_reliability"].shape, (2, 5))
        self.assertEqual(diagnostics["eye_reliability"].shape, (2, 5))
        self.assertEqual(diagnostics["eeg_interval"].shape, (2, 5, 2))
        self.assertEqual(diagnostics["attention"].shape, (1, 2, 4, 10, 10))
        logits.sum().backward()
        self.assertIsNotNone(model.eeg_fuzzy.center_logits.grad)


if __name__ == "__main__":
    unittest.main()
