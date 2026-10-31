import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluate_seed_iv_paper import _validate_coverage
from main_seed_iv import (
    classification_metrics,
    fold_coverage_complete,
    metrics_from_confusion_matrix,
    save_loso_summary,
    summarize_loso_results,
)
from training import Trainer


class SeedIVTrainingTests(unittest.TestCase):
    @staticmethod
    def _loader(batch_size):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 3, 0])
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, labels),
            batch_size=batch_size,
            shuffle=False,
        )

    @staticmethod
    def _trainer(checkpoint_folder):
        network = torch.nn.Linear(2, 4)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
        return Trainer(
            network,
            optimizer,
            epochs=1,
            use_cuda=False,
            checkpoint_folder=str(checkpoint_folder),
            sample_weights=np.ones(4, dtype=np.float32),
            label_smoothing=0.05,
            max_grad_norm=1.0,
        )

    def test_evaluation_loss_is_invariant_to_batch_partition(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._trainer(temporary_directory)
            with patch("training.wandb.log"):
                batch_two = trainer.evaluate(self._loader(2), split="validation")
                batch_three = trainer.evaluate(self._loader(3), split="validation")

        self.assertAlmostEqual(batch_two["loss"], batch_three["loss"], places=6)
        self.assertAlmostEqual(
            batch_two["accuracy"],
            batch_three["accuracy"],
            places=6,
        )
        self.assertAlmostEqual(batch_two["f1"], batch_three["f1"], places=6)

    def test_smoothed_training_with_gradient_clipping_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._trainer(temporary_directory)
            with (
                patch("training.wandb.run", SimpleNamespace(name="unit-test")),
                patch("training.wandb.log"),
            ):
                result = trainer.train(
                    self._loader(2),
                    self._loader(3),
                    max_lr=1e-3,
                    div_factor=10,
                    final_div_factor=100,
                    pct_start=0.3,
                )

            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            self.assertIn("minimum_val_loss", result)
            self.assertTrue(
                all(
                    parameter.grad is not None
                    for parameter in trainer.net.parameters()
                )
            )

    def test_loso_summary_uses_subject_level_sample_standard_deviation(self):
        results = [
            {
                "test_loss": 1.0,
                "test_accuracy": 40.0,
                "test_f1": 0.4,
            },
            {
                "test_loss": 3.0,
                "test_accuracy": 60.0,
                "test_f1": 0.6,
            },
        ]
        aggregate = summarize_loso_results(results)

        self.assertAlmostEqual(aggregate["test_loss"]["mean"], 2.0)
        self.assertAlmostEqual(
            aggregate["test_accuracy"]["std"],
            np.sqrt(200.0),
        )
        self.assertAlmostEqual(
            aggregate["test_f1"]["std"],
            np.sqrt(0.02),
        )

    def test_partial_loso_summary_is_written_atomically(self):
        args = SimpleNamespace(model="SeedIVIT2FuzzyAttention", seed=0)
        results = [
            {
                "fold": 0,
                "test_loss": 1.0,
                "test_accuracy": 50.0,
                "test_f1": 0.5,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary_path, payload = save_loso_summary(
                temporary_directory,
                args,
                results,
            )

            self.assertTrue(summary_path.is_file())
            self.assertFalse(
                (Path(temporary_directory) / "loso_summary.json.tmp").exists()
            )
            self.assertEqual(payload["completed_folds"], 1)
            self.assertEqual(payload["expected_folds"], 15)

    def test_classification_metrics_include_paper_outputs(self):
        metrics = classification_metrics(
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 1, 1, 2, 3, 3, 0],
        )
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (4, 4))
        self.assertEqual(
            np.asarray(metrics["normalized_confusion_matrix"]).shape,
            (4, 4),
        )
        self.assertEqual(
            set(metrics["per_class"]),
            {"neutral", "sad", "fear", "happy"},
        )
        self.assertAlmostEqual(metrics["accuracy_pct"], 62.5)
        reconstructed = metrics_from_confusion_matrix(
            metrics["confusion_matrix"]
        )
        self.assertAlmostEqual(
            reconstructed["macro_f1"],
            metrics["macro_f1"],
        )

    def test_paper_coverage_accepts_each_subject_exactly_once(self):
        folds = []
        for fold in range(15):
            test_subject = fold + 1
            validation_subject = (fold + 1) % 15 + 1
            train_subjects = [
                subject
                for subject in range(1, 16)
                if subject not in (test_subject, validation_subject)
            ]
            folds.append(
                {
                    "fold": fold,
                    "train_subjects": train_subjects,
                    "validation_subject": validation_subject,
                    "test_subject": test_subject,
                }
            )
        _validate_coverage(folds)
        self.assertTrue(fold_coverage_complete(folds))
        folds[-1]["test_subject"] = 1
        self.assertFalse(fold_coverage_complete(folds))


if __name__ == "__main__":
    unittest.main()
