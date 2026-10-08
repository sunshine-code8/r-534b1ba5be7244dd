"""Versioned, lossless cache for the Ghost crop and temporal-bin selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from experiments.wavedsp.data import GhostSupervisedDataset

CACHE_VERSION = 1


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".tmp_", delete=False) as out:
        temporary = Path(out.name)
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
    os.replace(temporary, path)


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp_", suffix=".b2", delete=False) as out:
        temporary = Path(out.name)
    temporary.unlink()  # Blosc2 requires a new, nonexistent urlpath.
    try:
        blosc2.save_array(array, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry(source_root: Path, voxel: Path, annotation: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    voxel = voxel.resolve()
    annotation = annotation.resolve()
    voxel_relative = voxel.relative_to(source_root)
    annotation_relative = annotation.relative_to(source_root)
    return {
        "voxel": str(voxel_relative),
        "annotation": str(annotation_relative),
        "source_voxel": [voxel.stat().st_size, voxel.stat().st_mtime_ns],
        "source_annotation": [annotation.stat().st_size, annotation.stat().st_mtime_ns],
        "frame_id": voxel.stem.removesuffix("_voxel"),
    }


def _paths(cache_root: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    return cache_root / entry["voxel"], cache_root / entry["annotation"]


def _process_one(item: tuple[GhostSupervisedDataset, int, dict[str, Any], Path]) -> bool:
    dataset, index, entry, cache_root = item
    voxel_path, annotation_path = _paths(cache_root, entry)
    if voxel_path.is_file() and annotation_path.is_file():
        return False
    voxel, annotation = dataset.load_preprocessed_arrays(index)
    _save_array_atomic(voxel_path, voxel)
    _save_array_atomic(annotation_path, annotation)
    return True


def build_cache(
    data_config: dict[str, Any],
    target_size: tuple[int, int, int],
    data_root: Path,
    cache_root: Path,
    num_classes: int = 4,
    ignore_index: int = -1,
    workers: int = 1,
) -> dict[str, Any]:
    """Build or resume a cache. Publish meta.json only after all paired arrays exist."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    source_root = data_root.resolve(strict=True)
    cache_root = cache_root.resolve(strict=False)  # Supports a dangling project symlink.
    if source_root == cache_root or source_root in cache_root.parents:
        raise ValueError("Cache must be outside the raw Ghost dataset")
    datasets = {
        split: GhostSupervisedDataset(
            data_config, split, target_size, source_root, num_classes, ignore_index
        )
        for split in ("train", "valid")
    }
    entries = {
        split: [_entry(source_root, voxel, annotation) for voxel, annotation in dataset.pairs]
        for split, dataset in datasets.items()
    }
    train_paths = {entry["voxel"] for entry in entries["train"]}
    valid_paths = {entry["voxel"] for entry in entries["valid"]}
    if train_paths & valid_paths:
        raise ValueError("Train and validation splits overlap")
    settings = {
        "version": CACHE_VERSION,
        "source_root": str(source_root),
        "target_size": list(target_size),
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "crop": {key: int(data_config.get(key, 0)) for key in (
            "y_crop_top", "y_crop_bottom", "z_crop_front", "z_crop_back"
        )},
        "downsample_z": int(data_config.get("downsample_z", target_size[2])),
        "entries": entries,
    }
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state = {"version": CACHE_VERSION, "fingerprint": fingerprint}
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = cache_root / "build_state.json"
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError(f"{cache_root} contains a cache for different source/configuration; use a new cache version")
    if not state_path.exists() and any(cache_root.iterdir()):
        raise ValueError(f"{cache_root} is nonempty without build_state.json; refusing to mix datasets")
    _write_json_atomic(state_path, state)
    meta_path = cache_root / "meta.json"
    if meta_path.exists():
        current = json.loads(meta_path.read_text())
        if current.get("fingerprint") != fingerprint:
            raise ValueError("Completed cache fingerprint differs; use a new cache version")
    for split, dataset in datasets.items():
        tasks = ((dataset, i, entry, cache_root) for i, entry in enumerate(entries[split]))
        if workers == 1:
            results = map(_process_one, tasks)
            with tqdm(total=len(dataset), desc=f"Cache {split}", unit="frame") as bar:
                for _ in results:
                    bar.update()
        else:
            # Ordered map bounds outstanding work; threads keep memory use below process workers.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in tqdm(pool.map(_process_one, tasks), total=len(dataset), desc=f"Cache {split}", unit="frame"):
                    pass
        for entry in entries[split]:
            if not all(path.is_file() for path in _paths(cache_root, entry)):
                raise RuntimeError(f"Incomplete cache pair: {entry['voxel']}")
        _write_json_atomic(cache_root / f"{split}_index.json", entries[split])
    meta = {
        "version": CACHE_VERSION,
        "complete": True,
        "fingerprint": fingerprint,
        "source_root": str(source_root),
        "target_size": list(target_size),
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "crop": settings["crop"],
        "downsample_z": settings["downsample_z"],
        "split_counts": {split: len(entries[split]) for split in entries},
    }
    _write_json_atomic(meta_path, meta)
    return meta


class GhostCachedDataset(Dataset):
    """Load only cropped and temporally sampled arrays from a completed cache."""

    def __init__(
        self,
        cache_root: Path,
        split: str,
        target_size: tuple[int, int, int],
        num_classes: int = 4,
        ignore_index: int = -1,
        data_config: dict[str, Any] | None = None,
    ) -> None:
        if split not in ("train", "valid"):
            raise ValueError(f"Unknown split: {split}")
        self.cache_root = Path(cache_root)
        meta_path = self.cache_root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Completed cache missing at {meta_path}; run preprocess_cache.py first"
            )
        meta = json.loads(meta_path.read_text())
        if (meta.get("version") != CACHE_VERSION or not meta.get("complete")
                or meta.get("target_size") != list(target_size)
                or meta.get("num_classes") != num_classes
                or meta.get("ignore_index") != ignore_index):
            raise ValueError(f"Cache metadata does not match training settings: {meta_path}")
        if data_config is not None:
            expected_crop = {key: int(data_config.get(key, 0)) for key in (
                "y_crop_top", "y_crop_bottom", "z_crop_front", "z_crop_back"
            )}
            if meta.get("crop") != expected_crop or meta.get("downsample_z") != int(
                data_config.get("downsample_z", target_size[2])
            ):
                raise ValueError("Cache preprocessing settings differ from the training data config")
        self.entries = json.loads((self.cache_root / f"{split}_index.json").read_text())
        if len(self.entries) != meta["split_counts"][split]:
            raise ValueError(f"Cache index length mismatch: {split}")
        self.target_size = target_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        voxel_path, annotation_path = _paths(self.cache_root, entry)
        voxel = blosc2.load_array(voxel_path)
        labels = blosc2.load_array(annotation_path)
        if voxel.shape != self.target_size or labels.shape != self.target_size:
            raise ValueError(f"Invalid cached shape: {voxel_path} {voxel.shape}, {annotation_path} {labels.shape}")
        if not np.issubdtype(voxel.dtype, np.number) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Invalid cached dtypes: {voxel.dtype}, {labels.dtype}")
        return {
            "voxel_grids": torch.from_numpy(np.ascontiguousarray(voxel)).permute(2, 1, 0).unsqueeze(0),
            "annotations": torch.from_numpy(np.ascontiguousarray(labels)).permute(2, 1, 0),
            "frame_id": entry["frame_id"],
        }


class NativeCachedView(Dataset):
    """Expose contiguous cache arrays so DataLoader collates without CPU transposes."""

    def __init__(self, dataset: GhostCachedDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = dict(self.dataset[index])
        # Underlying arrays are contiguous [X,Y,T]. Restore that view before stack.
        sample["voxel_grids"] = sample["voxel_grids"][0].permute(2, 1, 0).unsqueeze(-1)
        sample["annotations"] = sample["annotations"].permute(2, 1, 0)
        return sample
