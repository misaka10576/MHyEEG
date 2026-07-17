"""Shape, validation, attention, and gradient tests without dataset access."""

import unittest

import torch

from models.cross_attention_fusion import CrossModalAttentionFusion
from models.cross_attention_h2 import CrossAttentionH2


class CrossAttentionFusionTests(unittest.TestCase):
    def test_shape_attention_normalization_and_gradient(self):
        fusion = CrossModalAttentionFusion(
            input_dims=(8, 12, 16, 20),
            d_model=16,
            num_heads=4,
            num_layers=2,
            output_dim=32,
            dropout=0.0,
        )
        features = [
            torch.randn(3, dimension, requires_grad=True)
            for dimension in (8, 12, 16, 20)
        ]

        output, info = fusion(features, return_attention=True)

        self.assertEqual(output.shape, (3, 32))
        self.assertEqual(info["cross_attention"].shape, (2, 3, 4, 4, 3))
        self.assertEqual(info["pooling_weights"].shape, (3, 4))
        torch.testing.assert_close(
            info["pooling_weights"].sum(dim=1),
            torch.ones(3),
            atol=1e-6,
            rtol=1e-6,
        )
        output.mean().backward()
        self.assertTrue(all(feature.grad is not None for feature in features))

    def test_invalid_configuration_and_inputs(self):
        with self.assertRaises(ValueError):
            CrossModalAttentionFusion(input_dims=(8,), d_model=16, num_heads=4)
        with self.assertRaises(ValueError):
            CrossModalAttentionFusion(input_dims=(8, 12), d_model=15, num_heads=4)
        with self.assertRaises(ValueError):
            CrossModalAttentionFusion(
                input_dims=(8, 12), d_model=16, num_heads=4, num_layers=0
            )

        fusion = CrossModalAttentionFusion(
            input_dims=(8, 12), d_model=16, num_heads=4
        )
        with self.assertRaises(ValueError):
            fusion([torch.randn(2, 8)])
        with self.assertRaises(ValueError):
            fusion([torch.randn(2, 7), torch.randn(2, 12)])
        with self.assertRaises(ValueError):
            fusion([torch.randn(2, 8), torch.randn(3, 12)])

    def test_full_model_forward(self):
        model = CrossAttentionH2(
            dropout_rate=0.0,
            attention_dim=64,
            attention_heads=4,
            attention_layers=1,
            attention_dropout=0.0,
        ).eval()
        batch_size = 2
        inputs = (
            torch.randn(batch_size, 4, 600),
            torch.randn(batch_size, 1, 1280),
            torch.randn(batch_size, 10, 1280),
            torch.randn(batch_size, 3, 1280),
        )

        with torch.no_grad():
            logits, info = model(*inputs, return_attention=True)

        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(info["cross_attention"].shape, (1, batch_size, 4, 4, 3))
        self.assertEqual(info["pooling_weights"].shape, (batch_size, 4))


if __name__ == "__main__":
    unittest.main()
