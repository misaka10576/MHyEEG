import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data.seed_iv import SeedIVArrays, SeedIVDataset, get_subject_split
from models.seed_iv_maet import ReverseLayerF, SeedIVMAET
from run_seed_iv_maet import (
    DomainLabeledDataset,
    MAETTrainer,
    build_domain_mapping,
    classification_metrics,
    save_json_atomic,
    summarize_results,
)


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, count=8, with_domains=False):
        generator = torch.Generator().manual_seed(7)
        self.eeg = torch.randn(count, 62, 5, generator=generator)
        self.eye = torch.randn(count, 31, generator=generator)
        self.labels = torch.arange(count) % 4
        self.domains = torch.arange(count) % 13
        self.with_domains = with_domains

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            "eeg": self.eeg[index],
            "eye": self.eye[index],
        }
        if self.with_domains:
            return item, self.labels[index], self.domains[index]
        return item, self.labels[index]


def small_model(domain_generalization=False, mode="concat"):
    return SeedIVMAET(
        mode=mode,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        mixffn_start_layer_index=1,
        domain_generalization=domain_generalization,
    )


class SeedIVMAETModelTests(unittest.TestCase):
    def test_all_modes_output_four_logits_and_ignore_unused_modality(self):
        eeg = torch.randn(3, 62, 5)
        eye = torch.randn(3, 31)
        cases = {
            "eeg": (eeg, torch.randn(3, 2)),
            "eye": (torch.randn(3, 2), eye),
            "concat": (eeg, eye),
        }
        for mode, inputs in cases.items():
            with self.subTest(mode=mode):
                logits = small_model(mode=mode)(*inputs)
                self.assertEqual(logits.shape, (3, 4))
                self.assertTrue(torch.isfinite(logits).all())

    def test_dg_outputs_thirteen_domains_and_backpropagates(self):
        model = small_model(domain_generalization=True)
        emotion_logits, domain_logits = model(
            torch.randn(4, 62, 5),
            torch.randn(4, 31),
            alpha=0.5,
        )
        self.assertEqual(emotion_logits.shape, (4, 4))
        self.assertEqual(domain_logits.shape, (4, 13))
        (emotion_logits.sum() + domain_logits.sum()).backward()
        self.assertIsNotNone(model.eeg_transform.transform.weight.grad)
        self.assertTrue(
            torch.isfinite(model.eeg_transform.transform.weight.grad).all()
        )

    def test_gradient_reversal_has_expected_direction_and_scale(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        weights = torch.tensor([2.0, -1.0, 4.0])
        (ReverseLayerF.apply(x, 0.25) * weights).sum().backward()
        self.assertTrue(torch.allclose(x.grad, -0.25 * weights))

    def test_input_shapes_are_seed_iv_specific(self):
        model = small_model(mode="concat")
        with self.assertRaisesRegex(ValueError, r"62, 5"):
            model(torch.randn(2, 310), torch.randn(2, 31))
        with self.assertRaisesRegex(ValueError, r"31"):
            model(torch.randn(2, 62, 5), torch.randn(2, 33))


class SeedIVMAETDomainTests(unittest.TestCase):
    @staticmethod
    def arrays():
        subjects = np.arange(1, 16, dtype=np.int16)
        return SeedIVArrays(
            eeg=np.zeros((15, 62, 5), dtype=np.float32),
            eye=np.zeros((15, 31), dtype=np.float32),
            labels=np.arange(15, dtype=np.int64) % 4,
            subjects=subjects,
            sessions=np.ones(15, dtype=np.int8),
            trials=np.ones(15, dtype=np.int8),
        )

    def test_fold_mapping_is_complete_and_excludes_validation_and_test(self):
        split = get_subject_split(0)
        mapping = build_domain_mapping(0)
        self.assertEqual(set(mapping), set(split["train"]))
        self.assertEqual(set(mapping.values()), set(range(13)))
        self.assertFalse(set(split["validation"] + split["test"]) & set(mapping))

        dataset = DomainLabeledDataset(
            SeedIVDataset(self.arrays(), split["train"]),
            mapping,
        )
        observed = {dataset[index][2] for index in range(len(dataset))}
        self.assertEqual(observed, set(range(13)))

    def test_domain_dataset_rejects_held_out_subjects(self):
        split = get_subject_split(0)
        with self.assertRaisesRegex(ValueError, "absent from domain mapping"):
            DomainLabeledDataset(
                SeedIVDataset(self.arrays(), split["validation"]),
                build_domain_mapping(0),
            )


class SeedIVMAETTrainingTests(unittest.TestCase):
    @staticmethod
    def trainer(domain_generalization, epochs=2):
        model = small_model(domain_generalization=domain_generalization)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return MAETTrainer(
            model=model,
            optimizer=optimizer,
            class_weights=np.ones(4, dtype=np.float32),
            epochs=epochs,
            device=torch.device("cpu"),
        )

    def test_vanilla_and_dg_single_step_losses_are_finite(self):
        vanilla_batch = next(
            iter(torch.utils.data.DataLoader(DictDataset(), batch_size=4))
        )
        dg_batch = next(
            iter(
                torch.utils.data.DataLoader(
                    DictDataset(with_domains=True),
                    batch_size=4,
                )
            )
        )
        for trainer, batch in (
            (self.trainer(False), vanilla_batch),
            (self.trainer(True), dg_batch),
        ):
            with self.subTest(domain_generalization=trainer.model.domain_generalization):
                loss, details, _, _ = trainer.compute_training_loss(batch, epoch=0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(np.isfinite(list(details.values())).all())

    def test_fit_saves_checkpoint_and_result_json_schema(self):
        trainer = self.trainer(False)
        train_loader = torch.utils.data.DataLoader(
            DictDataset(),
            batch_size=4,
            shuffle=False,
        )
        validation_loader = torch.utils.data.DataLoader(
            DictDataset(),
            batch_size=4,
            shuffle=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = trainer.fit(
                train_loader,
                validation_loader,
                checkpoint_path=root / "best.pt",
                patience=2,
                max_lr=1e-3,
                div_factor=10,
                final_div_factor=100,
                pct_start=0.5,
            )
            self.assertTrue((root / "best.pt").is_file())

            metrics, _, _ = trainer.evaluate(validation_loader)
            payload = {
                "accuracy_pct": metrics["accuracy_pct"],
                "macro_f1": metrics["macro_f1"],
                **classification_metrics([0, 1, 2, 3], [0, 1, 1, 3]),
            }
            save_json_atomic(root / "metrics.json", payload)
            with (root / "metrics.json").open(encoding="utf-8") as input_file:
                saved = json.load(input_file)
            self.assertIn("accuracy_pct", saved)
            self.assertIn("macro_f1", saved)
            self.assertIn("per_class", saved)
            self.assertIn("checkpoint_path", result)

    def test_summary_requires_each_test_subject_once(self):
        args = SimpleNamespace(
            mode="concat",
            domain_generalization=True,
            seed=0,
            smoke_test=False,
        )
        results = [
            {
                "test_subject": subject,
                "accuracy_pct": 50.0,
                "macro_f1": 0.5,
            }
            for subject in range(1, 16)
        ]
        summary = summarize_results(args, results)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_test_subjects"], list(range(1, 16)))


if __name__ == "__main__":
    unittest.main()
