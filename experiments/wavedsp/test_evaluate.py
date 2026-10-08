"""Small end-to-end checks for the independent WaveDSP evaluation entrypoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import blosc2
import numpy as np
import torch
import yaml

from experiments.wavedsp.evaluate import estimate, generate_pcd, load_model, recall
from experiments.wavedsp.train import SupervisedWaveDSP


class WaveDSPEvaluationTests(unittest.TestCase):
    def test_fast_checkpoint_estimate_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voxel_dir = root / "ghost_dataset/scene001/data/hist001"
            annotation_dir = root / "ghost_dataset/scene001/annotation_v1_expand/hist001"
            voxel_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            stem = "frame001"
            waveform = np.zeros((8, 8, 16), dtype=np.uint16)
            waveform[:, :, 1:10] = np.array([1, 2, 4, 6, 8, 6, 4, 2, 1])
            labels = np.zeros_like(waveform, dtype=np.int16)
            labels[:, :, 5] = 3
            labels[0, 0, 5] = -1
            blosc2.save_array(waveform, str(voxel_dir / f"{stem}_voxel.b2"))
            blosc2.save_array(labels, str(annotation_dir / f"{stem}_annotation_voxel.b2"))
            data_config = root / "data.yaml"
            data_config.write_text(yaml.safe_dump({"downsample_z": 16}))
            settings = {
                "model": "fast", "model_kwargs": {}, "target_size": [8, 8, 16],
                "supervised": {"num_classes": 4}, "data_config": str(data_config),
            }
            checkpoint = root / "best.pt"
            torch.save({"model": SupervisedWaveDSP("fast", 16).state_dict(),
                        "config": settings}, checkpoint)
            model, loaded = load_model(checkpoint, "fast", torch.device("cpu"))
            test_config = {
                "test_voxel_dirs": [str(voxel_dir)],
                "test_annotation_dirs": [str(annotation_dir)],
                "downsample_z": 16,
                "use_threshold_prediction": False,
            }
            count = estimate(model, loaded, test_config, root / "estimate", torch.device("cpu"), "fp32")
            self.assertEqual(count, 1)
            prediction = blosc2.load_array(root / "estimate/scene001_hist001_frame001_prediction_voxel.b2")
            self.assertEqual(prediction.shape, waveform.shape)
            self.assertTrue(np.isin(prediction, [0, 1, 2, 3]).all())
            result = recall(model, loaded, test_config, root / "recall.json",
                            torch.device("cpu"), "fp32", checkpoint)
            self.assertEqual(result["frames"], 1)
            self.assertEqual(result["voxel"]["count"], waveform.size - 1)
            self.assertEqual(result["peak"]["count"], 63)
            self.assertEqual(json.loads((root / "recall.json").read_text())["model"], "fast")
            exported = json.loads((root / "confusion_matrix/matrices.json").read_text())
            self.assertEqual(exported["reports"]["peak"]["confusion_matrix"],
                             result["peak"]["confusion_matrix"])
            self.assertTrue((root / "confusion_matrix/peak/row_normalized.png").is_file())
            # Reporting errors must preserve both saved metrics and the return value.
            original_json = (root / "recall.json").read_bytes()
            with patch("experiments.wavedsp.confusion_matrix.export_confusion_matrices",
                       side_effect=RuntimeError("simulated plotting failure")):
                repeated = recall(model, loaded, test_config, root / "recall.json",
                                  torch.device("cpu"), "fp32", checkpoint)
            self.assertEqual(repeated, result)
            self.assertEqual((root / "recall.json").read_bytes(), original_json)

    def test_full_head_supervised_backward_and_checkpoint_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for patch_size in (8, 16):
                with self.subTest(patch_size=patch_size):
                    kwargs = {"patch_size": patch_size, "full_head": True}
                    model = SupervisedWaveDSP("fast", 32, **kwargs)
                    self.assertEqual(model.network.head.project.groups, 1)
                    x = torch.randn(1, 1, 32, 3, 5)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (1, 4, 32, 3, 5))
                    logits.square().mean().backward()
                    self.assertTrue(all(p.grad is not None for p in model.parameters()))
                    path = Path(directory) / "full_head.pt"
                    settings = {
                        "model": "fast", "model_kwargs": kwargs, "target_size": [5, 3, 32],
                        "supervised": {"num_classes": 4},
                    }
                    torch.save({"model": model.state_dict(), "config": settings}, path)
                    loaded, _ = load_model(path, "fast", torch.device("cpu"))
                    with torch.inference_mode():
                        torch.testing.assert_close(loaded(x), logits)
                        labels = loaded.network.predict(x.permute(0, 3, 4, 2, 1))
                        expected = logits.argmax(1).permute(0, 2, 3, 1).to(torch.uint8)
                        torch.testing.assert_close(labels, expected)

    def test_pcd_stage_resumes_only_missing_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred_dir, pcd_dir = root / "estimate", root / "pcd"
            pred_dir.mkdir()
            pcd_dir.mkdir()
            files = [pred_dir / f"frame{i}_prediction_voxel.b2" for i in range(3)]
            for path in files:
                path.touch()
            for suffix in ("", "_gt"):
                (pcd_dir / f"{files[0].stem}{suffix}.pcd").touch()
            config_path = root / "pcd.yaml"
            config_path.write_text(yaml.safe_dump({"ghost_dataset": "data/ghost_dataset"}))

            def complete_pending(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                selected = Path(command[command.index("--pred_dir") + 1])
                pending = sorted(selected.glob("*_prediction_voxel.b2"))
                self.assertEqual([path.name for path in pending], [path.name for path in files[1:]])
                for path in pending:
                    self.assertTrue(path.is_symlink())
                    for suffix in ("", "_gt"):
                        (pcd_dir / f"{path.stem}{suffix}.pcd").touch()

            with patch("experiments.wavedsp.evaluate.subprocess.run", side_effect=complete_pending):
                generate_pcd(pred_dir, pcd_dir, config_path)
            with patch("experiments.wavedsp.evaluate.subprocess.run") as run:
                generate_pcd(pred_dir, pcd_dir, config_path)
                run.assert_not_called()

    def test_checkpoint_kind_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"config": {"model": "fine", "target_size": [8, 8, 16],
                                   "supervised": {"num_classes": 4}}}, path)
            with self.assertRaisesRegex(ValueError, "expected 'fast'"):
                load_model(path, "fast", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
