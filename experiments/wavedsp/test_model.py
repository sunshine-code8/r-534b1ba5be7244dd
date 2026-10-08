"""Correctness checks for the in-place WaveDSP v2 redesign.

Run: python -m experiments.wavedsp.test_model
"""

from __future__ import annotations

import contextlib
import io
import unittest

import torch
from torch.nn import functional as F

from benchmark_wavedsp import NEURALDSP_CONFIGS, parse_args, run_model
from experiments.wavedsp import FastWaveDSP, FineWaveDSP, waveform_auxiliary_loss
from experiments.wavedsp.model import (
    BottleneckTemporalAttention,
    DualProjectionEmbedding,
    GroupedBinHead,
    ResidualPatchEmbedding,
)


class WaveDSPTests(unittest.TestCase):
    def test_shape_layout_and_backward(self) -> None:
        for cls in (FastWaveDSP, FineWaveDSP):
            with self.subTest(model=cls.__name__):
                model = cls(time_bins=32)
                # Batch > 1; odd H/W exercise padding at each merge and exact skip recovery.
                x = torch.randn(2, 9, 11, 32, 1)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 9, 11, 32, 4))
                self.assertTrue(logits.is_contiguous())
                targets = torch.randint(4, (2, 9, 11, 32))
                F.cross_entropy(logits.permute(0, 4, 1, 2, 3), targets).backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all().item(), name)
                first = cls(time_bins=32, layout="channels_first")
                first.load_state_dict(model.state_dict())
                with torch.inference_mode():
                    other = first(x.permute(0, 4, 1, 2, 3).contiguous())
                    torch.testing.assert_close(other, logits.permute(0, 4, 1, 2, 3))
                    for network, inputs in ((model, x), (first, x.permute(0, 4, 1, 2, 3))):
                        labels = network.predict(inputs)
                        self.assertEqual(labels.dtype, torch.uint8)
                        torch.testing.assert_close(labels, logits.argmax(-1).to(torch.uint8))

    def test_backbone_is_4d_and_attention_only_at_bottleneck(self) -> None:
        for cls in (FastWaveDSP, FineWaveDSP):
            with self.subTest(model=cls.__name__), torch.inference_mode():
                model = cls(time_bins=32)
                shapes = {}
                handles = []

                def check_conv(
                    module: torch.nn.Module, inputs: tuple, output: torch.Tensor
                ) -> None:
                    self.assertEqual(inputs[0].ndim, 4)
                    self.assertEqual(output.ndim, 4)

                for module in model.modules():
                    self.assertNotIsInstance(module, (torch.nn.Conv1d, torch.nn.Conv3d))
                    if isinstance(module, torch.nn.Conv2d):
                        handles.append(module.register_forward_hook(check_conv))
                for name in (
                    "patch_embed",
                    "encoder1",
                    "encoder2",
                    "bottleneck",
                    "up2",
                    "up1",
                    "up0",
                ):
                    handles.append(
                        getattr(model, name).register_forward_hook(
                            lambda m, args, out, name=name: shapes.__setitem__(
                                name, tuple(out.shape)
                            )
                        )
                    )
                if cls is FineWaveDSP:
                    handles.append(
                        model.bottleneck[1].norm.register_forward_pre_hook(
                            lambda m, args: shapes.__setitem__(
                                "attention_sequences", tuple(args[0].shape)
                            )
                        )
                    )
                model(torch.randn(1, 17, 25, 32, 1))
                self.assertEqual(shapes["patch_embed"][-2:], (17, 25))
                self.assertEqual(shapes["encoder1"][-2:], (9, 13))
                self.assertEqual(shapes["encoder2"][-2:], (5, 7))
                self.assertEqual(shapes["bottleneck"][-2:], (3, 4))
                self.assertEqual(shapes["up0"][-2:], (17, 25))
                if cls is FineWaveDSP:
                    self.assertEqual(shapes["attention_sequences"], (12, 8, 8))
                for handle in handles:
                    handle.remove()

    def test_grouped_embedding_does_not_mix_raw_patches(self) -> None:
        model = FastWaveDSP(time_bins=32, patch_size=8, patch_dim=2)
        with torch.no_grad():
            model.patch_embed.amplitude.weight.fill_(1)
            model.patch_embed.amplitude.bias.zero_()
            model.patch_embed.derivative.weight.zero_()
            model.patch_embed.derivative.bias.zero_()
            x = torch.zeros(1, 2, 3, 32, 1)
            x[..., 10, 0] = 7
            embedded = model.patch_embed(model._unpack(x))
            # Four patches, each with amplitude and derivative channels.
            expected = torch.zeros(1, 8, 2, 3)
            expected[:, 2] = F.silu(torch.tensor(7.0))
            torch.testing.assert_close(embedded, expected)

    def test_grouped_head_bin_and_class_order(self) -> None:
        head = GroupedBinHead(time_bins=8, patch_size=2, patch_dim=1)
        with torch.no_grad():
            head.project.weight.zero_()
            head.project.bias.copy_(torch.arange(32))
            native = head(torch.zeros(2, 4, 3, 5))
            expected = torch.arange(32).float().reshape(1, 1, 1, 8, 4).expand(2, 3, 5, 8, 4)
            torch.testing.assert_close(head.format_logits(native, "channels_last"), expected)
            torch.testing.assert_close(
                head.format_logits(native, "channels_first"), expected.permute(0, 4, 1, 2, 3)
            )
            head.project.bias.zero_()
            for t in range(8):
                head.project.bias[t * 4 + t % 4] = 1
            native = head(torch.zeros(2, 4, 3, 5))
            expected_labels = (torch.arange(8) % 4).to(torch.uint8).expand(2, 3, 5, 8)
            torch.testing.assert_close(head.labels(native), expected_labels)

    def test_global_attention_covers_all_tokens(self) -> None:
        att = BottleneckTemporalAttention(16, tokens=4, token_dim=4, heads=1, relative_bias=False)
        att.norm = torch.nn.Identity()
        with torch.no_grad():
            for p in att.parameters():
                p.zero_()
            att.qkv.weight[8:].copy_(torch.eye(4))
            att.proj.weight.copy_(torch.eye(4))
        x = torch.randn(2, 4, 4)
        torch.testing.assert_close(att.attend(x), x + x.mean(dim=1, keepdim=True))

    def test_bottleneck_chunking_preserves_logits_and_gradients(self) -> None:
        whole = FineWaveDSP(time_bins=32, attention_chunk_size=0)
        split = FineWaveDSP(time_bins=32, attention_chunk_size=3)
        split.load_state_dict(whole.state_dict())
        x = torch.randn(1, 17, 25, 32, 1)
        a, b = whole(x), split(x)
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(whole.parameters(), split.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)

    def test_ablation_and_patch_overrides(self) -> None:
        with torch.inference_mode():
            for model in (
                FastWaveDSP(time_bins=32, patch_size=8, bottleneck_mixer=False),
                FineWaveDSP(time_bins=32, patch_size=8, temporal_attention=False),
                FineWaveDSP(time_bins=32, relative_bias=False),
            ):
                logits = model(torch.zeros(1, 1, 1, 32, 1))
                self.assertEqual(tuple(logits.shape), (1, 1, 1, 32, 4))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_dual_embedding_difference_and_patch_order(self) -> None:
        embed = DualProjectionEmbedding(time_bins=8, patch_size=4, patch_dim=2)
        with torch.no_grad():
            embed.amplitude.weight.fill_(1)
            embed.amplitude.bias.zero_()
            embed.derivative.weight.fill_(1)
            embed.derivative.bias.zero_()
        # An inter-patch jump of 97 must not leak into the derivative branch.
        raw = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0, 100.0, 100.0, 100.0]).reshape(1, 8, 1, 1)
        expected = F.silu(torch.tensor([6.0, 3.0, 400.0, 0.0]).reshape(1, 4, 1, 1))
        torch.testing.assert_close(embed(raw), expected)
        shifted = raw.clone()
        shifted[:, :4] += 10
        torch.testing.assert_close(embed(shifted)[:, 1::2], embed(raw)[:, 1::2])

    def test_fine_embedding_identity_residual(self) -> None:
        embed = ResidualPatchEmbedding(16, 4, 4)
        with torch.no_grad():
            embed.project.weight.zero_()
            embed.project.bias.zero_()
        raw = torch.randn(2, 16, 3, 5)
        torch.testing.assert_close(embed(raw), raw)

    def test_compressed_embedding_configuration(self) -> None:
        for cls, count, patch_dim, stem, decoder in (
            (FastWaveDSP, 316800, 8, 96, (96, 64, 64)),
            (FineWaveDSP, 958678, 4, 128, (160, 128, 128)),
        ):
            model = cls()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(model.patch_dim, patch_dim)
            self.assertEqual(model.config["stem_channels"], stem)
            self.assertEqual(model.config["decoder_channels"], decoder)
            with torch.inference_mode():
                output = model(torch.zeros(1, 1, 1, 256, 1))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_auxiliary_losses_and_inference_exclusion(self) -> None:
        for cls in (FastWaveDSP, FineWaveDSP):
            for layout in ("channels_last", "channels_first"):
                model = cls(time_bins=32, layout=layout, auxiliary_reconstruction=True)
                x = torch.randn(1, 3, 5, 32, 1)
                if layout == "channels_first":
                    x = x.permute(0, 4, 1, 2, 3).contiguous()
                result = model.forward_with_aux(x)
                self.assertEqual(result["reconstruction"].shape, x.shape)
                torch.testing.assert_close(result["logits"], model(x))
                losses = waveform_auxiliary_loss(result["reconstruction"], x, layout)
                (result["logits"].square().mean() + losses["total"]).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

                def reject_aux(module: torch.nn.Module, args: tuple) -> None:
                    raise AssertionError(
                        "Auxiliary head must not execute during ordinary inference"
                    )

                handle = model.auxiliary_head.register_forward_pre_hook(reject_aux)
                with torch.inference_mode():
                    model(x)
                    model.predict(x)
                handle.remove()
        target = torch.zeros(1, 1, 1, 4, 1)
        reconstruction = torch.arange(4.0).reshape_as(target)
        losses = waveform_auxiliary_loss(reconstruction, target)
        self.assertAlmostEqual(losses["waveform"].item(), 1.5)
        self.assertAlmostEqual(losses["derivative"].item(), 1.0)
        self.assertAlmostEqual(losses["total"].item(), 0.2)
        with self.assertRaises(RuntimeError):
            FastWaveDSP(time_bins=32).forward_with_aux(torch.zeros(1, 1, 1, 32, 1))

    def test_benchmark_complexity_rows(self) -> None:
        args = parse_args([
            "--model", "all", "--device", "cpu", "--height", "9", "--width", "11",
            "--time-bins", "32", "--output", "both", "--metrics-only",
        ])
        for name in ("fast", "fine"):
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()):
                rows = run_model(args, name)
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["parameters"], 0)
            self.assertEqual(rows[0]["parameters"], rows[0]["trainable_parameters"])
            self.assertGreater(rows[0]["parameter_storage_mib"], 0)
            self.assertGreater(rows[0]["flops_per_batch"], 0)
            self.assertEqual(rows[0]["flops_per_batch"], rows[1]["flops_per_batch"])
            self.assertIsNone(rows[0]["latency_ms"])
            self.assertIsNone(rows[0]["fps"])
        args.skip_flops = True
        with contextlib.redirect_stdout(io.StringIO()):
            rows = run_model(args, "fast")
        self.assertIsNone(rows[0]["flops_per_batch"])

    def test_cli_and_validation(self) -> None:
        args = parse_args(["--model", "fine", "--time-bins", "12", "--no-temporal-attention"])
        self.assertTrue(args.no_temporal_attention)
        self.assertTrue(parse_args(["--model", "fast", "--full-head"]).full_head)
        self.assertFalse(parse_args(["--model", "fast"]).full_head)
        self.assertEqual(NEURALDSP_CONFIGS["user_patch32"]["depths"], (0, 1, 2, 1, 0))
        self.assertEqual(NEURALDSP_CONFIGS["user_patch32"]["patch_size"], 32)
        for argv in (
            ["--model", "fine", "--full-head"],
            ["--time-bins", "17"],
            ["--patch-dim", "0"],
            ["--patch-size", "0"],
            ["--patch-dim", "3"],
            ["--stem-channels", "0"],
            ["--attention-heads", "3"],
            ["--attention-chunk-size", "-1"],
            ["--wave-chunk-size", "8192"],
            ["--no-physical-bias"],
            ["--variant", "base"],
        ):
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(argv)
        for cls in (FastWaveDSP, FineWaveDSP):
            for config in ({"time_bins": 17}, {"patch_size": 0}, {"patch_dim": 0}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    cls(**config)
            with self.assertRaises(ValueError):
                cls(time_bins=32)(torch.zeros(1, 3, 4, 16, 1))
            with self.assertRaises(ValueError):
                cls(time_bins=32)(torch.zeros(1, 3, 4, 32, 2))
        with self.assertRaises(ValueError):
            FineWaveDSP(heads=3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(42)
    unittest.main()
