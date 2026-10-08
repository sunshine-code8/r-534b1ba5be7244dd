"""Small integration checks for Ghost preprocessing and WaveDSP supervised tensors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np
import torch
import torch.nn.functional as F

from experiments.wavedsp.data import GhostSupervisedDataset
from experiments.wavedsp.train import MaskedFocalLoss, SupervisedWaveDSP


class TrainingIntegrationTests(unittest.TestCase):
    def test_masked_focal_matches_legacy_value_and_gradient(self) -> None:
        alpha = [0.0001, 0.05, 0.25, 0.7]
        for seed in range(3):
            torch.manual_seed(seed)
            logits = torch.randn(2, 4, 5, 6, 7, requires_grad=True)
            reference_logits = logits.detach().clone().requires_grad_(True)
            targets = torch.randint(0, 4, (2, 5, 6, 7))
            targets.flatten()[::9] = -1
            valid = targets != -1
            ce = F.cross_entropy(reference_logits, targets, reduction="none", ignore_index=-1)
            selected = ce[valid]
            weights = torch.tensor(alpha)[targets[valid]]
            reference = (weights * (1 - torch.exp(-selected)).pow(2.0) * selected).mean()
            optimized = MaskedFocalLoss(alpha, 2.0)(logits, targets)
            torch.testing.assert_close(optimized, reference)
            optimized.backward()
            reference.backward()
            torch.testing.assert_close(logits.grad, reference_logits.grad)

    def test_ghost_pairing_preprocessing_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voxel_dir = root / "scene001" / "data" / "hist001"
            annotation_dir = root / "scene001" / "annotation" / "hist001"
            voxel_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            raw = np.arange(8 * 14 * 40, dtype=np.float32).reshape(8, 14, 40)
            labels = np.zeros_like(raw, dtype=np.uint8)
            labels[:, 2:12, 3:] = (raw[:, 2:12, 3:] % 4).astype(np.uint8)
            blosc2.save_array(raw, voxel_dir / "frame_voxel.b2")
            blosc2.save_array(labels, annotation_dir / "frame_annotation_voxel.b2")
            config = {
                "train_voxel_dirs": [str(voxel_dir)],
                "train_annotation_dirs": [str(annotation_dir)],
                "y_crop_top": 2,
                "y_crop_bottom": 2,
                "z_crop_front": 3,
                "downsample_z": 16,
            }
            dataset = GhostSupervisedDataset(config, "train", (8, 10, 16))
            sample = dataset[0]
            self.assertEqual(tuple(sample["voxel_grids"].shape), (1, 16, 10, 8))
            self.assertEqual(tuple(sample["annotations"].shape), (16, 10, 8))
            indices = np.linspace(0, 36, 16, dtype=int) + 3
            self.assertEqual(float(sample["voxel_grids"][0, 0, 0, 0]), float(raw[0, 2, indices[0]]))
            self.assertEqual(int(sample["annotations"][0, 0, 0]), int(labels[0, 2, indices[0]]))
            voxels = sample["voxel_grids"].unsqueeze(0)
            targets = sample["annotations"].unsqueeze(0).long()
            loss_fn = MaskedFocalLoss([0.0001, 0.05, 0.25, 0.7], 2.0)
            for kind in ("fast", "fine"):
                with self.subTest(kind=kind):
                    model = SupervisedWaveDSP(kind, 16)
                    logits = model(voxels)
                    self.assertEqual(tuple(logits.shape), (1, 4, 16, 10, 8))
                    loss = loss_fn(logits, targets)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
