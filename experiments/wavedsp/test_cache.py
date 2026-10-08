"""Small, CPU-only checks for the versioned Ghost preprocessing cache."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np
import torch

from experiments.wavedsp.cache import GhostCachedDataset, NativeCachedView, build_cache
from experiments.wavedsp.data import GhostSupervisedDataset
from experiments.wavedsp.train import SupervisedWaveDSP, make_datasets


class CacheTests(unittest.TestCase):
    def test_cache_matches_raw_and_resumes_through_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ghost_dataset"
            cache = root / "physical_cache"
            link = root / "data" / "ghost_dataset_cache_v1"
            link.parent.mkdir()
            link.symlink_to(cache, target_is_directory=True)
            config: dict[str, object] = {
                "y_crop_top": 2, "y_crop_bottom": 2,
                "z_crop_front": 3, "z_crop_back": 1, "downsample_z": 16,
            }
            for split, scene in (("train", "scene001"), ("valid", "scene002")):
                voxel_dir = source / scene / "data" / "hist001"
                annotation_dir = source / scene / "annotation_v1_expand" / "hist001"
                voxel_dir.mkdir(parents=True)
                annotation_dir.mkdir(parents=True)
                rng = np.random.default_rng(1 if split == "train" else 2)
                raw = rng.integers(0, 65536, size=(8, 14, 40), dtype=np.uint16)
                labels = rng.integers(0, 4, size=raw.shape, dtype=np.uint8)
                blosc2.save_array(raw, voxel_dir / "frame_voxel.b2")
                blosc2.save_array(labels, annotation_dir / "frame_annotation_voxel.b2")
                config[f"{split}_voxel_dirs"] = [str(voxel_dir)]
                config[f"{split}_annotation_dirs"] = [str(annotation_dir)]
            size = (8, 10, 16)
            meta = build_cache(config, size, source, link)
            self.assertEqual(meta["split_counts"], {"train": 1, "valid": 1})
            self.assertTrue((link / "meta.json").is_file())
            self.assertEqual(meta, build_cache(config, size, source, link))
            train_config = {
                "data_config": str(root / "unused.yaml"),
                "target_size": list(size),
                "cache_root": str(link),
                "supervised": {"num_classes": 4, "ignore_index": -1},
            }
            import yaml
            (root / "unused.yaml").write_text(yaml.safe_dump(config))
            train, valid = make_datasets(train_config, None)
            for split, cached in (("train", train), ("valid", valid)):
                self.assertIsInstance(cached, GhostCachedDataset)
                raw_dataset = GhostSupervisedDataset(config, split, size, source)
                expected, actual = raw_dataset[0], cached[0]
                self.assertEqual(actual["voxel_grids"].dtype, torch.uint16)
                self.assertTrue(torch.equal(expected["voxel_grids"], actual["voxel_grids"].float()))
                self.assertTrue(torch.equal(expected["annotations"], actual["annotations"]))
                self.assertEqual(expected["frame_id"], actual["frame_id"])
                native = NativeCachedView(cached)[0]
                self.assertTrue(native["voxel_grids"].is_contiguous())
                self.assertTrue(native["annotations"].is_contiguous())
                restored_voxel = native["voxel_grids"].unsqueeze(0).permute(0, 4, 3, 2, 1)
                restored_label = native["annotations"].unsqueeze(0).permute(0, 3, 2, 1)
                self.assertTrue(torch.equal(restored_voxel[0], actual["voxel_grids"]))
                self.assertTrue(torch.equal(restored_label[0], actual["annotations"]))
                if split == "train":
                    model = SupervisedWaveDSP("fast", size[2])
                    with torch.no_grad():
                        old_logits = model(actual["voxel_grids"].unsqueeze(0).float())
                        new_logits = model(restored_voxel.float())
                    torch.testing.assert_close(old_logits, new_logits)
            bad = dict(config, y_crop_top=1)
            with self.assertRaisesRegex(ValueError, "preprocessing settings"):
                GhostCachedDataset(link, "train", size, data_config=bad)


if __name__ == "__main__":
    unittest.main()
