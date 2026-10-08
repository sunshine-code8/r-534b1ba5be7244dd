"""Paired Ghost voxel/annotation loading for full-frame WaveDSP training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


def _frame_key(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


class GhostSupervisedDataset(Dataset):
    """Return aligned [1,T,H,W] waveforms and [T,H,W] class labels."""

    def __init__(
        self,
        data_config: dict[str, Any],
        split: str,
        target_size: tuple[int, int, int],
        data_root: Path | None = None,
        num_classes: int = 4,
        ignore_index: int = -1,
    ) -> None:
        if split not in ("train", "valid"):
            raise ValueError(f"Unknown split: {split}")
        self.target_size = target_size  # X,Y,T
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.y_crop_top = int(data_config.get("y_crop_top", 0))
        self.y_crop_bottom = int(data_config.get("y_crop_bottom", 0))
        self.z_crop_front = int(data_config.get("z_crop_front", 0))
        self.z_crop_back = int(data_config.get("z_crop_back", 0))
        self.downsample_z = int(data_config.get("downsample_z", target_size[2]))
        if self.downsample_z != target_size[2]:
            raise ValueError("target_size T must match downsample_z")

        voxel_dirs = data_config[f"{split}_voxel_dirs"]
        annotation_dirs = data_config[f"{split}_annotation_dirs"]
        if len(voxel_dirs) != len(annotation_dirs):
            raise ValueError(f"{split}: voxel and annotation directory counts differ")
        self.pairs: list[tuple[Path, Path]] = []
        for voxel_dir, annotation_dir in zip(voxel_dirs, annotation_dirs):
            voxel_path = self._resolve(voxel_dir, data_root)
            annotation_path = self._resolve(annotation_dir, data_root)
            if not voxel_path.is_dir() or not annotation_path.is_dir():
                raise FileNotFoundError(
                    f"{split}: missing Ghost directory: {voxel_path} or {annotation_path}. "
                    "Set --data-root to the ghost_dataset directory if needed."
                )
            voxels = {_frame_key(path, "_voxel.b2"): path for path in voxel_path.glob("*_voxel.b2")}
            annotations = {
                _frame_key(path, "_annotation_voxel.b2"): path
                for path in annotation_path.glob("*_annotation_voxel.b2")
            }
            if voxels.keys() != annotations.keys():
                missing_labels = sorted(voxels.keys() - annotations.keys())[:3]
                missing_voxels = sorted(annotations.keys() - voxels.keys())[:3]
                raise ValueError(
                    f"{split}: unpaired files in {voxel_path} / {annotation_path}; "
                    f"missing labels={missing_labels}, missing voxels={missing_voxels}"
                )
            self.pairs.extend((voxels[key], annotations[key]) for key in sorted(voxels))
        if not self.pairs:
            raise RuntimeError(f"{split}: no paired Ghost files found")

    @staticmethod
    def _resolve(path: str, data_root: Path | None) -> Path:
        if data_root is None:
            return Path(path)
        source = Path(path)
        parts = source.parts
        if "ghost_dataset" not in parts:
            raise ValueError(f"Expected ghost_dataset in configured path: {path}")
        return data_root.joinpath(*parts[parts.index("ghost_dataset") + 1 :])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        voxel, labels = self.load_preprocessed_arrays(index)
        voxel_path, _ = self.pairs[index]
        waveform = torch.from_numpy(np.ascontiguousarray(voxel, dtype=np.float32))
        valid_labels = (labels >= 0) & (labels < self.num_classes)
        label_dtype = np.uint8 if self.ignore_index >= 0 or np.all(valid_labels) else np.int16
        annotation = torch.from_numpy(np.ascontiguousarray(labels, dtype=label_dtype))
        return {
            "voxel_grids": waveform.permute(2, 1, 0).unsqueeze(0),
            "annotations": annotation.permute(2, 1, 0),
            "frame_id": voxel_path.stem.removesuffix("_voxel"),
        }

    def load_preprocessed_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Apply the training crop/bin selection while retaining source dtypes for caching."""
        voxel_path, annotation_path = self.pairs[index]
        voxel = blosc2.load_array(voxel_path)
        labels = blosc2.load_array(annotation_path)
        if voxel.shape != labels.shape or voxel.ndim != 3:
            raise ValueError(
                f"Mismatched 3D voxel/annotation shapes: {voxel_path} {voxel.shape}, "
                f"{annotation_path} {labels.shape}"
            )
        y_end = voxel.shape[1] - self.y_crop_top
        z_end = voxel.shape[2] - self.z_crop_back
        if self.y_crop_bottom >= y_end or self.z_crop_front >= z_end:
            raise ValueError(f"Crop exceeds source shape {voxel.shape}: {voxel_path}")
        crop = (slice(None), slice(self.y_crop_bottom, y_end), slice(self.z_crop_front, z_end))
        voxel, labels = voxel[crop], labels[crop]
        if self.downsample_z > voxel.shape[2]:
            raise ValueError(f"Cannot downsample {voxel.shape[2]} bins to {self.downsample_z}")
        # Same integer sampling rule as src.utils.downsample_histogram_direction.
        indices = np.linspace(0, voxel.shape[2] - 1, self.downsample_z, dtype=int)
        voxel, labels = voxel[:, :, indices], labels[:, :, indices]
        if voxel.shape != self.target_size:
            raise ValueError(
                f"After preprocessing {voxel_path}: expected (X,Y,T)={self.target_size}, "
                f"got {voxel.shape}. Check source shape and crop settings."
            )
        valid_labels = (labels >= 0) & (labels < self.num_classes)
        if not np.issubdtype(labels.dtype, np.integer) or not np.all(
            valid_labels | (labels == self.ignore_index)
        ):
            raise ValueError(
                f"{annotation_path}: labels must be integers in [0,{self.num_classes - 1}] "
                f"or ignore_index={self.ignore_index}; got dtype={labels.dtype}, "
                f"min={labels.min()}, max={labels.max()}"
            )
        return np.ascontiguousarray(voxel), np.ascontiguousarray(labels)
