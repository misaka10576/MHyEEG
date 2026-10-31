from __future__ import annotations

import tempfile
import unittest
import io
from pathlib import Path

import torch

from BERN.model import (
    BERN,
    EEGEncoder,
    EyeEncoder,
    ResNet18ResidualLayers,
    VectorTokenCrossAttention,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    def test_all_modes_return_four_logits(self):
        eeg = torch.randn(2, 62, 5)
        eye = torch.randn(2, 31)
        for mode in ("eye", "eeg", "fusion"):
            with self.subTest(mode=mode):
                model = BERN(mode=mode, pretrained=False).eval()
                with torch.inference_mode():
                    output = model(eeg, eye)
                self.assertEqual(tuple(output.shape), (2, 4))
                self.assertTrue(torch.isfinite(output).all())

    def test_fusion_returns_vector_token_multihead_attention(self):
        model = BERN(mode="fusion", pretrained=False).eval()
        with torch.inference_mode():
            logits, weights = model(
                torch.randn(2, 62, 5),
                torch.randn(2, 31),
                return_attention=True,
            )
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(len(weights), 2)
        self.assertEqual([tuple(value.shape) for value in weights], [(2, 4, 8, 8)] * 2)
        self.assertEqual(model.fusion_variant, "vector_token_v2")

    def test_intermediate_shapes_match_the_paper_constrained_design(self):
        eeg_encoder = EEGEncoder(pretrained=False).eval()
        with torch.inference_mode():
            embedding, shapes = eeg_encoder.forward_features(
                torch.randn(2, 62, 5), return_shapes=True
            )
        self.assertEqual(shapes["grid"], (2, 5, 9, 9))
        self.assertEqual(shapes["resized"], (2, 5, 112, 112))
        self.assertEqual(shapes["conv3d"], (2, 64, 1, 56, 56))
        self.assertEqual(shapes["stem"], (2, 64, 56, 56))
        self.assertEqual(shapes["embedding"], (2, 128))
        self.assertEqual(tuple(embedding.shape), (2, 128))

        eye_encoder = EyeEncoder().eval()
        with torch.inference_mode():
            eye_embedding, eye_shapes = eye_encoder.forward_features(
                torch.randn(2, 31), return_shapes=True
            )
        self.assertEqual([eye_shapes[f"block{i}"] for i in range(1, 5)], [(2, 31)] * 4)
        self.assertEqual(tuple(eye_embedding.shape), (2, 128))
        for block in eye_encoder.blocks:
            self.assertEqual(block.linear1.in_features, 31)
            self.assertEqual(block.linear1.out_features, 15)
            self.assertEqual(block.linear2.in_features, 15)
            self.assertEqual(block.linear2.out_features, 31)

    def test_vector_attention_is_sensitive_nonuniform_and_backpropagates(self):
        layers = torch.nn.ModuleList(
            [VectorTokenCrossAttention(), VectorTokenCrossAttention()]
        )
        eeg = torch.randn(3, 8, 32, requires_grad=True)
        eye = torch.randn(3, 8, 32, requires_grad=True)
        first, attention = layers[0](eeg, eye, return_attention=True)
        output = layers[1](first, eye)
        self.assertEqual(tuple(attention.shape), (3, 4, 8, 8))
        self.assertGreater(float(attention.detach().var()), 0.0)
        changed_query = layers[0](eeg.detach() + torch.randn_like(eeg), eye.detach())
        zero_eye = layers[0](eeg.detach(), torch.zeros_like(eye))
        shuffled_eye = layers[0](eeg.detach(), eye.detach().roll(1, dims=0))
        self.assertFalse(torch.allclose(first.detach(), changed_query))
        self.assertFalse(torch.allclose(first.detach(), zero_eye))
        self.assertFalse(torch.allclose(first.detach(), shuffled_eye))
        uniform = torch.full_like(attention, 1.0 / attention.shape[-1])
        self.assertFalse(torch.allclose(attention, uniform))
        output.square().mean().backward()
        self.assertIsNotNone(eeg.grad)
        self.assertIsNotNone(eye.grad)
        for layer in layers:
            self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))

    def test_fusion_checkpoint_round_trip(self):
        model = BERN(mode="fusion", pretrained=False).eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = BERN(mode="fusion", pretrained=False).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        eeg, eye = torch.randn(1, 62, 5), torch.randn(1, 31)
        with torch.inference_mode():
            expected = model(eeg, eye)
            actual = restored(eeg, eye)
        torch.testing.assert_close(actual, expected)

    def test_resnet_loader_requires_and_covers_every_residual_key(self):
        source = ResNet18ResidualLayers()
        state = source.state_dict()
        target_key = "layer1.0.conv1.weight"
        state[target_key] = torch.full_like(state[target_key], 0.125)
        # Official checkpoints also include stem/head keys; they must be ignored.
        state["conv1.weight"] = torch.randn(64, 3, 7, 7)
        state["fc.weight"] = torch.randn(1000, 512)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resnet18-f37072fd.pth"
            torch.save(state, path)
            target = ResNet18ResidualLayers()
            report = target.load_imagenet_weights(weights_path=path)
        self.assertEqual(report["loaded_keys"], report["expected_keys"])
        torch.testing.assert_close(
            target.state_dict()[target_key],
            torch.full_like(target.state_dict()[target_key], 0.125),
        )

    def test_cached_official_weights_when_present(self):
        path = Path("BERN/cache/resnet18-f37072fd.pth")
        if not path.is_file():
            self.skipTest("Official weight cache has not been populated.")
        model = ResNet18ResidualLayers()
        report = model.load_imagenet_weights(weights_path=path)
        self.assertEqual(report["loaded_keys"], report["expected_keys"])
        self.assertEqual(report["sha256_prefix"], "f37072fd")


if __name__ == "__main__":
    unittest.main()
