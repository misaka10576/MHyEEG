from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from BERN.data import RunArrays, SESSION_LABELS
from BERN.summarize import build_summary, load_results
from BERN.training import TrainingConfig, run_experiment
from BERN.training import _attention_diagnostics


class TinyBERN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 4)

    def forward(self, eeg, eye):
        features = torch.stack((eeg.mean(dim=(1, 2)), eye.mean(dim=1)), dim=1)
        return self.classifier(features)


def synthetic_run(subject: int = 1, session: int = 1) -> RunArrays:
    rng = np.random.default_rng(7)
    eeg, eye, labels, trials, windows = [], [], [], [], []
    for trial, label in enumerate(SESSION_LABELS[session], start=1):
        eeg.append(rng.normal(label, 0.1, size=(2, 62, 5)).astype(np.float32))
        eye.append(rng.normal(label, 0.1, size=(2, 31)).astype(np.float32))
        labels.extend([label, label])
        trials.extend([trial, trial])
        windows.extend([0, 1])
    return RunArrays(
        eeg=np.concatenate(eeg),
        eye=np.concatenate(eye),
        labels=np.asarray(labels, dtype=np.int64),
        trials=np.asarray(trials, dtype=np.int8),
        window_in_trial=np.asarray(windows, dtype=np.int16),
        subject=subject,
        session=session,
    )


class TrainingProtocolTests(unittest.TestCase):
    def test_attention_diagnostics_are_finite_and_detect_nonuniformity(self):
        weights = torch.softmax(torch.randn(5, 4, 8, 8), dim=-1)
        diagnostics = _attention_diagnostics([[weights, weights.square() / weights.square().sum(-1, keepdim=True)]])
        self.assertEqual(len(diagnostics), 2)
        for layer in diagnostics:
            self.assertTrue(np.isfinite(layer["normalized_entropy"]))
            self.assertTrue(np.isfinite(layer["relative_uniform_deviation"]))
            self.assertTrue(np.isfinite(layer["attention_variance"]))
            self.assertGreater(layer["relative_uniform_deviation"], 0.0)
            self.assertGreater(layer["attention_variance"], 0.0)

    def test_two_stage_run_saves_recoverable_leakage_safe_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            config = TrainingConfig(
                mode="eye",
                subject=1,
                session=1,
                dataset_root="unused",
                output_dir=str(output),
                cache_dir=str(Path(directory) / "cache"),
                seed=2024,
                batch_size=8,
                max_epochs=2,
                patience=1,
                device="cpu",
                pretrained=False,
                deterministic=True,
            )
            metrics = run_experiment(
                synthetic_run(),
                config,
                model_builder=lambda _: TinyBERN(),
            )
            run_dir = output / "eye" / "subject_01" / "session_1"
            for name in (
                "config.json",
                "selection.json",
                "selection_best.pt",
                "final_checkpoint.pt",
                "predictions.json",
                "test_evaluation.json",
                "metrics.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertFalse((run_dir / "selection_last.pt").exists())
            self.assertFalse((run_dir / "final_last.pt").exists())
            self.assertEqual(metrics["test_evaluations"], 1)
            self.assertEqual(metrics["final_train_trials"], list(range(1, 17)))
            self.assertEqual(metrics["test_trials"], list(range(17, 25)))
            checkpoint = torch.load(
                run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epochs"], metrics["best_epoch"])
            self.assertEqual(checkpoint["training_trials"], list(range(1, 17)))
            self.assertFalse(checkpoint["test_metrics_observed_during_training"])
            predictions = json.loads(
                (run_dir / "predictions.json").read_text(encoding="utf-8")
            )
            self.assertEqual({row["trial"] for row in predictions}, set(range(17, 25)))

            # Simulate interruption after atomic predictions but before metrics:
            # resume must reconstruct metrics without a second test pass.
            (run_dir / "metrics.json").unlink()
            recovered = run_experiment(
                synthetic_run(),
                config,
                resume=True,
                model_builder=lambda _: TinyBERN(),
            )
            self.assertEqual(recovered, metrics)

            # Resume must skip a fully completed run without a second test pass.
            second = run_experiment(
                synthetic_run(),
                config,
                resume=True,
                model_builder=lambda _: TinyBERN(),
            )
            self.assertEqual(second, metrics)

            loaded = load_results(output)
            self.assertEqual(len(loaded), 1)
            summary = build_summary(loaded)
            self.assertEqual(summary["total_completed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
