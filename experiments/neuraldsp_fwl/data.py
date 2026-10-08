"""Datasets for the isolated NeuralDSP/FWL experiments."""

from __future__ import annotations

import pathlib
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.dataset_fwl import FWLDataset, voxel_collate_fn
from src.utils import downsample_histogram_direction, load_blosc2


def _target_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    size = list(config["target_size"])
    if config.get("downsample_z") is not None:
        size[2] = int(config["downsample_z"])
    return tuple(int(value) for value in size)


class VoxelOnlyDataset(Dataset):
    """FWL-MAE preprocessing without peak files or peak tensors."""

    def __init__(
        self,
        voxel_dirs: list[str],
        data_config: dict[str, Any],
        training: bool,
        seed: int,
    ) -> None:
        self.training = training
        self.seed = seed
        self.target_shape = _target_shape(data_config)
        self.downsample_z = data_config.get("downsample_z")
        self.y_crop_top = int(data_config.get("y_crop_top", 0))
        self.y_crop_bottom = int(data_config.get("y_crop_bottom", 0))
        self.z_crop_front = int(data_config.get("z_crop_front", 0))
        self.z_crop_back = int(data_config.get("z_crop_back", 0))

        files: list[pathlib.Path] = []
        for entry in voxel_dirs:
            path = pathlib.Path(entry)
            if path.is_file() and path.match("*_voxel.b2"):
                files.append(path)
            elif path.is_dir():
                files.extend(path.rglob("*_voxel.b2"))
        self.files = sorted(set(files))

        divide = int(data_config.get("divide", 1))
        if divide > 1:
            rng = random.Random(seed)
            count = len(self.files) // divide
            indices = sorted(rng.sample(range(len(self.files)), count))
            self.files = [self.files[index] for index in indices]
        if not self.files:
            raise RuntimeError(f"no *_voxel.b2 files found in {voxel_dirs}")

    def __len__(self) -> int:
        return len(self.files)

    def _crop_start(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        maximum = [max(0, shape[axis] - self.target_shape[axis]) for axis in range(3)]
        if self.training:
            return tuple(np.random.randint(0, value + 1) if value > 0 else 0 for value in maximum)
        return tuple(value // 2 for value in maximum)

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str]:
        voxel = load_blosc2(self.files[index]).copy()
        y_end = voxel.shape[1] - self.y_crop_top if self.y_crop_top else voxel.shape[1]
        z_end = voxel.shape[2] - self.z_crop_back if self.z_crop_back else voxel.shape[2]
        voxel = voxel[:, self.y_crop_bottom : y_end, self.z_crop_front : z_end]
        if self.downsample_z is not None:
            voxel = downsample_histogram_direction(voxel, int(self.downsample_z))

        if any(voxel.shape[axis] < self.target_shape[axis] for axis in range(3)):
            raise ValueError(
                f"{self.files[index]} shape {voxel.shape} is smaller than {self.target_shape}"
            )
        start = self._crop_start(voxel.shape)
        slices = tuple(
            slice(start[axis], start[axis] + self.target_shape[axis]) for axis in range(3)
        )
        voxel = np.ascontiguousarray(voxel[slices], dtype=np.float32)
        # (X,Y,T) -> (1,T,Y,X)
        tensor = torch.from_numpy(voxel).permute(2, 1, 0).unsqueeze(0)
        return {"voxels": tensor, "indices": index, "frame_id": self.files[index].stem}


class ExperimentFWLDataset(FWLDataset):
    """Original supervised dataset with deterministic validation crops."""

    def __init__(self, *args: Any, training: bool, **kwargs: Any) -> None:
        self.experiment_training = training
        super().__init__(*args, **kwargs)

    def _determine_crop_coordinates(
        self, voxel_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if self.experiment_training:
            return super()._determine_crop_coordinates(voxel_shape)
        if self.target_size is None:
            return (0, 0, 0)
        target = list(self.target_size)
        if self.downsample_z is not None:
            target[2] = self.downsample_z
        return tuple(max(0, voxel_shape[axis] - int(target[axis])) // 2 for axis in range(3))


def build_supervised_dataset(data_config: dict[str, Any], training: bool) -> ExperimentFWLDataset:
    prefix = "train" if training else "valid"
    return ExperimentFWLDataset(
        voxel_dirs=data_config[f"{prefix}_voxel_dirs"],
        annotation_dirs=data_config[f"{prefix}_annotation_dirs"],
        target_size=data_config.get("target_size"),
        downsample_z=data_config.get("downsample_z"),
        divide=int(data_config.get("divide", 1)),
        y_crop_top=int(data_config.get("y_crop_top", 0)),
        y_crop_bottom=int(data_config.get("y_crop_bottom", 0)),
        z_crop_front=int(data_config.get("z_crop_front", 0)),
        z_crop_back=int(data_config.get("z_crop_back", 0)),
        training=training,
    )


def supervised_collate(batch: list[dict]) -> dict:
    return voxel_collate_fn(batch)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
