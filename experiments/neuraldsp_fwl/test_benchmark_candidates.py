"""Correctness checks for the standalone full-frame benchmark architecture."""

from __future__ import annotations

import unittest

import torch

from benchmark_original_neuraldsp import CANDIDATES
from experiments.neuraldsp_fwl.benchmark_candidates import (
    FullFrameNeuralDSP,
    PaddedWindowAttention,
)
from experiments.neuraldsp_fwl.model import WindowAttention


class CandidateTests(unittest.TestCase):
    def test_all_presets_dense_output(self) -> None:
        for name, config in CANDIDATES.items():
            with self.subTest(candidate=name), torch.inference_mode():
                model = FullFrameNeuralDSP(16, 24, 64, **config).eval()
                output = model(torch.randn(2, 16, 24, 64, 1))
                self.assertEqual(tuple(output.shape), (2, 16, 24, 64, 4))
                self.assertTrue(output.is_contiguous())
                self.assertTrue(torch.isfinite(output).all().item())

    def test_full_frame_shapes_on_meta(self) -> None:
        for name, config in CANDIDATES.items():
            with self.subTest(candidate=name), torch.inference_mode():
                model = FullFrameNeuralDSP(**config).to("meta").eval()
                output = model(torch.empty(1, 336, 400, 256, 1, device="meta"))
                self.assertEqual(tuple(output.shape), (1, 336, 400, 256, 4))

    def test_custom_widths_depths_and_backward(self) -> None:
        model = FullFrameNeuralDSP(
            16, 24, 32, patch_size=8, channels=(8, 12, 16, 24), depths=(1, 2, 1, 2, 1)
        )
        output = model(torch.randn(1, 16, 24, 32, 1))
        self.assertEqual(tuple(output.shape), (1, 16, 24, 32, 4))
        output.square().mean().backward()
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all().item())

    def test_no_padding_matches_existing_attention(self) -> None:
        for shift in (False, True):
            with self.subTest(shift=shift):
                reference = WindowAttention(8, (2, 4), 2, shift)
                padded = PaddedWindowAttention(8, (4, 8), (2, 4), 2, shift)
                padded.load_state_dict(reference.state_dict())
                x = torch.randn(2, 4, 8, 8)
                torch.testing.assert_close(padded(x), reference(x))

    def test_padding_keys_do_not_dilute_real_values(self) -> None:
        # Uniform real values should remain uniform under averaging attention,
        # including at padded edges and when windows shift across those edges.
        for shift in (False, True):
            for resolution in ((3, 5), (1, 1), (42, 50)):
                with self.subTest(shift=shift, resolution=resolution), torch.no_grad():
                    attention = PaddedWindowAttention(8, resolution, (2, 4), 2, shift)
                    attention.qkv.weight.zero_()
                    attention.qkv.weight[16:].copy_(torch.eye(8))
                    attention.qkv.bias.zero_()
                    attention.proj.weight.copy_(torch.eye(8))
                    attention.proj.bias.zero_()
                    attention.relative_position_bias_table.zero_()
                    x = torch.ones(2, *resolution, 8)
                    torch.testing.assert_close(attention(x), x)

    def test_zero_depths_and_invalid_configs(self) -> None:
        model = FullFrameNeuralDSP(8, 8, 32, depths=(0, 0, 0, 0, 0))
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.randn(1, 8, 8, 32, 1)).shape), (1, 8, 8, 32, 4))
        for config in (
            {"height": 335},
            {"patch_size": 30},
            {"depths": (1, 1, 1)},
            {"depths": (0, -1, 1, 1, 0)},
            {"channels": (16, 16, 32)},
            {"heads": 3},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                FullFrameNeuralDSP(**config)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
