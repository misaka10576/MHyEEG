import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.seed_iv_eeg_baselines import SeedIVDGCNN, normalize_adjacency
from run_seed_iv_eeg_baselines import (
    build_neural_model,
    classification_metrics,
    save_json_atomic,
    summarize_results,
)


class SeedIVEegBaselineModelTests(unittest.TestCase):
    def test_dgcnn_outputs_four_logits_and_backpropagates(self):
        model = SeedIVDGCNN()
        logits = model(
            eeg=torch.randn(3, 62, 5),
            eye=torch.randn(3, 31),
        )
        self.assertEqual(logits.shape, (3, 4))
        logits.sum().backward()
        self.assertIsNotNone(model.adjacency_logits.grad)
        self.assertTrue(torch.isfinite(model.adjacency_logits.grad).all())

    def test_normalized_adjacency_is_finite_symmetric_and_positive_degree(self):
        adjacency = normalize_adjacency(torch.randn(62, 62))
        self.assertEqual(adjacency.shape, (62, 62))
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertTrue(torch.allclose(adjacency, adjacency.T, atol=1e-6))
        self.assertTrue(torch.all(adjacency.sum(dim=1) > 0))

    def test_neural_model_registry_builds_mlp_and_dgcnn(self):
        args = SimpleNamespace(
            hidden_dim=128,
            embedding_dim=64,
            dropout_rate=0.3,
            graph_channels=32,
            chebyshev_order=2,
            classifier_dim=64,
        )
        for name in ("mlp", "dgcnn"):
            with self.subTest(name=name):
                self.assertIsInstance(build_neural_model(name, args), torch.nn.Module)


class SeedIVEegBaselineResultTests(unittest.TestCase):
    def test_classification_metrics_have_expected_schema(self):
        metrics = classification_metrics(
            targets=[0, 1, 2, 3],
            predictions=[0, 1, 1, 3],
        )
        self.assertAlmostEqual(metrics["accuracy_pct"], 75.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 4)
        self.assertEqual(
            set(metrics["per_class"]),
            {"neutral", "sad", "fear", "happy"},
        )

    def test_summary_marks_only_all_subjects_as_complete(self):
        args = SimpleNamespace(seed=0, smoke_test=False)
        fold_results = [
            {
                "test_subject": subject,
                "accuracy_pct": 50.0 + subject,
                "macro_f1": 0.5,
            }
            for subject in range(1, 16)
        ]
        summary = summarize_results("mlp", args, fold_results)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_folds"], 15)
        self.assertGreater(summary["aggregate"]["accuracy_pct"]["std"], 0)

    def test_json_is_written_atomically_without_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.json"
            save_json_atomic(path, {"complete": True})
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
