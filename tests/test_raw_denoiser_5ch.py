"""Phase 2B smoke test — 5-channel RawDenoiser forward pass (no training)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from nsa.config import ModelConfig
from nsa.models import RawDenoiserDenoiser, build_model, count_params
from nsa.raw_domain import (
    RawDenoiser,
    fusion_confidence,
    stack_fusion_input,
    to_fusion_tensor,
)
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed


class RawDenoiser5ChSmokeTest(unittest.TestCase):
    def test_stack_fusion_input_shape(self):
        h, w = 64, 80
        fused = np.random.rand(h, w, 4).astype(np.float32)
        weight = np.full((h, w, 1), 8.0, dtype=np.float32)
        x = stack_fusion_input(fused, weight, k_cap=16.0)
        self.assertEqual(x.shape, (h, w, 5))
        np.testing.assert_allclose(x[..., 4], 0.5, rtol=0, atol=1e-6)

    def test_fusion_confidence_clamps(self):
        w = np.array([[[0.0], [8.0], [20.0]]], dtype=np.float32)
        c = fusion_confidence(w, k_cap=16.0)
        np.testing.assert_allclose(c.ravel(), [0.0, 0.5, 1.0], rtol=0, atol=1e-6)

    def test_raw_denoiser_4ch_backward_compat(self):
        model = RawDenoiser(base_channels=16, block_depth=2, in_ch=4)
        x = torch.rand(1, 4, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 32, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_raw_denoiser_5ch_forward(self):
        model = RawDenoiser(base_channels=16, block_depth=2, in_ch=5, out_ch=4)
        x = torch.rand(1, 5, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 32, 32))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((y >= 0.0).all() and (y <= 1.0).all())

    def test_build_model_raw_denoiser_family(self):
        cfg = ModelConfig(model_family="raw_denoiser", base_channels=16, block_depth=2)
        model = build_model(cfg)
        self.assertIsInstance(model, RawDenoiserDenoiser)
        x = torch.rand(2, 5, 48, 48)
        y = model(x)
        self.assertEqual(y.shape, (2, 4, 48, 48))
        self.assertGreater(count_params(model), 0)

    def test_end_to_end_with_synthetic_burst(self):
        rng = np.random.default_rng(662)
        frames = [rng.random((32, 40, 4), dtype=np.float32) for _ in range(8)]
        fused, weight = fuse_burst_packed(frames, FusionConfig(n_frames=8, k_cap=16.0))
        x = to_fusion_tensor(fused, weight, k_cap=16.0)
        self.assertEqual(tuple(x.shape), (1, 5, 32, 40))

        model = RawDenoiser(base_channels=16, block_depth=2, in_ch=5, out_ch=4)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 4, 32, 40))
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
