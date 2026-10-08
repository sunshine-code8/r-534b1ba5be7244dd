"""Stage-two FWLDataset view over the completed fixed-preprocessing Ghost cache."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_fwl import FWLDataset


class FWLCachedDataset(FWLDataset):
    """Keep FWLDataset's random spatial crop while reading cached full frames."""

    def __init__(self, *, cache_root: str, cache_split: str, **dataset_kwargs: Any) -> None:
        if cache_split not in ("train", "valid"):
            raise ValueError("cache_split must be train or valid")
        requested_divide = int(dataset_kwargs.get("divide", 1))
        super().__init__(**{**dataset_kwargs, "divide": 1})
        self.cache_root = Path(cache_root)
        meta = json.loads((self.cache_root / "meta.json").read_text())
        state = json.loads((self.cache_root / "build_state.json").read_text())
        crop = {
            "y_crop_top": self.y_crop_top,
            "y_crop_bottom": self.y_crop_bottom,
            "z_crop_front": self.z_crop_front,
            "z_crop_back": self.z_crop_back,
        }
        size = meta.get("target_size")
        if (
            meta.get("version") != 1
            or meta.get("complete") is not True
            or state.get("fingerprint") != meta.get("fingerprint")
            or meta.get("crop") != crop
            or meta.get("downsample_z") != self.downsample_z
            or meta.get("num_classes") != self.num_classes
            or meta.get("ignore_index") != -1
            or self.target_size is None
            or not isinstance(size, list)
            or len(size) != 3
            or size[2] != self.target_size[2]
            or any(full < part for full, part in zip(size, self.target_size))
        ):
            raise ValueError("Cache metadata does not match stage-two preprocessing")
        self.cache_size = tuple(size)
        entries = json.loads((self.cache_root / f"{cache_split}_index.json").read_text())
        if len(entries) != meta["split_counts"][cache_split]:
            raise ValueError(f"Cache {cache_split} index count differs from metadata")
        source_root = Path(meta["source_root"]).resolve(strict=True)
        original_pairs = set(
            zip(
                (path.resolve() for path in self.voxel_files),
                (path.resolve() for path in self.annotation_files),
            )
        )
        indexed_pairs = {
            (source_root / entry["voxel"], source_root / entry["annotation"]) for entry in entries
        }
        if len(original_pairs) != len(self.voxel_files) or indexed_pairs != original_pairs:
            raise ValueError(
                f"Cache {cache_split} frame pairs differ from configured source directories"
            )
        entries.sort(key=lambda entry: str(source_root / entry["voxel"]))
        self.voxel_files = [self.cache_root / entry["voxel"] for entry in entries]
        self.annotation_files = [self.cache_root / entry["annotation"] for entry in entries]
        if requested_divide > 1:
            indices = sorted(random.sample(range(len(entries)), len(entries) // requested_divide))
            self.voxel_files = [self.voxel_files[i] for i in indices]
            self.annotation_files = [self.annotation_files[i] for i in indices]
        self.divide = requested_divide

        # The cache already contains these fixed operations. FWLDataset still
        # chooses one shared random spatial crop for voxel and annotation.
        self.y_crop_top = 0
        self.y_crop_bottom = 0
        self.z_crop_front = 0
        self.z_crop_back = 0
        self.downsample_z = None

    def _load_voxel_grid(self, file_path: str) -> np.ndarray:
        array = super()._load_voxel_grid(file_path)
        if array.shape != self.cache_size or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Invalid cached voxel: {file_path}")
        return array

    def _load_annotation_voxel(self, file_path: str) -> np.ndarray:
        array = super()._load_annotation_voxel(file_path)
        if array.shape != self.cache_size or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Invalid cached annotation: {file_path}")
        return array
