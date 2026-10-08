"""CPU tests for opt-in MAE transfer, freezing and supervised compatibility."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.wavedsp.evaluate import load_model
from experiments.wavedsp.mae_data import preprocessing_settings
from experiments.wavedsp.mae_model import ReconstructionWaveDSP
from experiments.wavedsp.model import ARCHITECTURE_VERSION
from experiments.wavedsp.train import MaskedFocalLoss, SupervisedWaveDSP
from experiments.wavedsp.transfer import check_transfer_resume, load_mae_backbone


def fixture(root: Path, kind: str) -> tuple[dict, dict, dict]:
    size = [9, 11, 32]
    data = {"downsample_z": 32}
    source = ReconstructionWaveDSP(kind, time_bins=32)
    path = root / f"{kind}_mae.pt"
    checkpoint = {
        "stage": "wavedsp_mae_reconstruction",
        "architecture_version": ARCHITECTURE_VERSION,
        "epoch": 100,
        "best_masked_mse": 1.0,
        "config": {"model": kind, "model_kwargs": {}, "target_size": size},
        "backbone": source.backbone_state_dict(),
        "data_identity": {"preprocessing": preprocessing_settings(data, tuple(size))},
    }
    torch.save(checkpoint, path)
    config = {
        "model": kind,
        "model_kwargs": {},
        "target_size": size,
        "supervised": {"num_classes": 4},
        "transfer": {"pretrained_checkpoint": str(path), "freeze_backbone": True},
    }
    return config, data, checkpoint


class TransferTests(unittest.TestCase):
    def test_both_models_load_and_update_only_head_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, full_head in (("fast", False), ("fast", True), ("fine", False)):
                with self.subTest(kind=kind, full_head=full_head):
                    config, data, checkpoint = fixture(root, kind)
                    if kind == "fast":
                        config["model_kwargs"] = {
                            "patch_size": 16,
                            "patch_dim": 8,
                            "stem_channels": 96,
                            "bottleneck_mixer": True,
                            "full_head": full_head,
                        }
                    model = SupervisedWaveDSP(kind, 32, **config["model_kwargs"])
                    head_before = copy.deepcopy(model.network.head.state_dict())
                    load_mae_backbone(model.network, config, data)
                    for name, value in checkpoint["backbone"].items():
                        torch.testing.assert_close(model.network.state_dict()[name], value)
                    for name, value in head_before.items():
                        torch.testing.assert_close(model.network.head.state_dict()[name], value)
                    ordinary = SupervisedWaveDSP(kind, 32, **config["model_kwargs"])
                    ordinary.load_state_dict(model.state_dict(), strict=True)
                    model.set_backbone_frozen(True)
                    model.train()
                    self.assertFalse(model.network.training)
                    self.assertTrue(model.network.head.training)
                    trainable = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ]
                    self.assertEqual(
                        trainable, ["network.head.project.weight", "network.head.project.bias"]
                    )
                    x = torch.randn(2, 1, 32, 11, 9, requires_grad=True)
                    torch.testing.assert_close(model(x), ordinary(x))
                    before = copy.deepcopy(model.state_dict())
                    feature_grads = []
                    hook = model.network.head.register_forward_pre_hook(
                        lambda module, inputs: feature_grads.append(inputs[0].requires_grad)
                    )
                    optimizer = torch.optim.AdamW(
                        (p for p in model.parameters() if p.requires_grad), lr=1e-3
                    )
                    loss = MaskedFocalLoss([0.0001, 0.05, 0.25, 0.7], 2.0)(
                        model(x), torch.randint(0, 4, (2, 32, 11, 9))
                    )
                    loss.backward()
                    optimizer.step()
                    hook.remove()
                    self.assertEqual(feature_grads, [False])
                    self.assertIsNone(x.grad)
                    for name, p in model.named_parameters():
                        if "head." not in name:
                            self.assertIsNone(p.grad)
                            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                    self.assertFalse(
                        torch.equal(
                            model.network.head.project.weight, before["network.head.project.weight"]
                        )
                    )
                    # Stage-two checkpoints retain the existing evaluator's state format.
                    path = root / f"{kind}_stage2.pt"
                    torch.save({"model": model.state_dict(), "config": config}, path)
                    reloaded, _ = load_model(path, kind, torch.device("cpu"))
                    model.eval()
                    torch.testing.assert_close(reloaded(x), model(x))
                    model.set_backbone_frozen(False)
                    model.train()
                    self.assertTrue(model.network.training)
                    self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_reject_incompatible_or_incomplete_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, data, checkpoint = fixture(Path(temporary), "fast")
            for kwargs in ({"patch_size": 8}, {"stem_channels": 128}):
                with self.assertRaisesRegex(ValueError, "architecture differs"):
                    load_mae_backbone(SupervisedWaveDSP("fast", 32, **kwargs).network, config, data)
            with self.assertRaisesRegex(ValueError, "crop/time sampling"):
                load_mae_backbone(
                    SupervisedWaveDSP("fast", 32).network, config, dict(data, z_crop_front=1)
                )
            with self.assertRaisesRegex(ValueError, "kind differs"):
                load_mae_backbone(
                    SupervisedWaveDSP("fine", 32).network, dict(config, model="fine"), data
                )
            checkpoint["backbone"].pop("embedding_compression.weight")
            torch.save(checkpoint, config["transfer"]["pretrained_checkpoint"])
            with self.assertRaisesRegex(ValueError, "Backbone keys differ"):
                load_mae_backbone(SupervisedWaveDSP("fast", 32).network, config, data)

    def test_resume_cannot_change_training_strategy(self) -> None:
        original = {"model": "fast"}
        transfer = {
            "model": "fast",
            "transfer": {"pretrained_checkpoint": "best.pt", "freeze_backbone": True},
        }
        check_transfer_resume({"config": original}, original)
        check_transfer_resume({"config": transfer}, transfer)
        for previous, current in ((original, transfer), (transfer, original)):
            with self.assertRaisesRegex(ValueError, "strategy differs"):
                check_transfer_resume({"config": previous}, current)
        changed = copy.deepcopy(transfer)
        changed["transfer"]["freeze_backbone"] = False
        with self.assertRaisesRegex(ValueError, "freeze_backbone differs"):
            check_transfer_resume({"config": transfer}, changed)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
