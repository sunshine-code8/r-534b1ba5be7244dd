"""Read-only, elementwise verification of the stage-two Ghost preprocessing cache."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import blosc2
import numpy as np
import yaml

from experiments.wavedsp.data import GhostSupervisedDataset


def verify(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text())
    root = args.cache_root
    meta = json.loads((root / "meta.json").read_text())
    state = json.loads((root / "build_state.json").read_text())
    source_root = (args.data_root or Path(meta["source_root"])).resolve(strict=True)
    target_size = tuple(meta["target_size"])
    crop = {
        key: int(config.get(key, 0))
        for key in ("y_crop_top", "y_crop_bottom", "z_crop_front", "z_crop_back")
    }
    if (
        meta.get("version") != 1
        or meta.get("complete") is not True
        or state.get("fingerprint") != meta.get("fingerprint")
        or meta.get("crop") != crop
        or meta.get("downsample_z") != int(config["downsample_z"])
        or meta.get("num_classes") != int(config["num_classes"])
        or meta.get("ignore_index") != -1
    ):
        raise ValueError("Cache metadata does not match the stage-two config or build state")
    if source_root != Path(meta["source_root"]).resolve(strict=True):
        raise ValueError("Selected raw data root differs from cache source_root")

    checked = 0
    mismatches = []
    summaries = {}
    split_paths = {}
    for split in ("train", "valid"):
        dataset = GhostSupervisedDataset(
            config, split, target_size, source_root, meta["num_classes"], meta["ignore_index"]
        )
        entries = json.loads((root / f"{split}_index.json").read_text())
        if len(entries) != len(dataset) or len(entries) != meta["split_counts"][split]:
            raise ValueError(f"{split}: index, raw pair and metadata counts differ")
        by_pair = {(v.resolve(), a.resolve()): i for i, (v, a) in enumerate(dataset.pairs)}
        if len(by_pair) != len(dataset):
            raise ValueError(f"{split}: duplicate raw pairs")
        indexed = [(source_root / e["voxel"], source_root / e["annotation"]) for e in entries]
        if len(set(indexed)) != len(entries) or set(indexed) != set(by_pair):
            raise ValueError(f"{split}: cache index and configured raw pairs differ")
        split_paths[split] = {e["voxel"] for e in entries}

        def check(item: tuple[dict[str, Any], tuple[Path, Path]]) -> str | None:
            entry, pair = item
            index = by_pair[pair]
            voxel_path, annotation_path = pair
            if (
                entry["frame_id"] != voxel_path.stem.removesuffix("_voxel")
                or [voxel_path.stat().st_size, voxel_path.stat().st_mtime_ns]
                != entry["source_voxel"]
                or [annotation_path.stat().st_size, annotation_path.stat().st_mtime_ns]
                != entry["source_annotation"]
            ):
                return f"{split}/{entry['voxel']}: source identity or timestamp differs"
            expected_voxel, expected_label = dataset.load_preprocessed_arrays(index)
            cache_voxel = blosc2.load_array(root / entry["voxel"])
            cache_label = blosc2.load_array(root / entry["annotation"])
            for name, expected, actual in (
                ("voxel", expected_voxel, cache_voxel),
                ("annotation", expected_label, cache_label),
            ):
                if expected.shape != actual.shape or expected.dtype != actual.dtype:
                    return (
                        f"{split}/{entry['voxel']}: {name} shape/dtype differs: "
                        f"raw={expected.shape}/{expected.dtype}, cache={actual.shape}/{actual.dtype}"
                    )
                if not np.array_equal(expected, actual):
                    first = tuple(np.argwhere(expected != actual)[0])
                    return (
                        f"{split}/{entry['voxel']}: {name} differs at {first}: "
                        f"raw={expected[first]}, cache={actual[first]}"
                    )
            return None

        tasks = zip(entries, indexed)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, error in enumerate(pool.map(check, tasks), start=1):
                checked += 1
                if error is not None:
                    mismatches.append(error)
                    print(error, flush=True)
                if i % 100 == 0 or i == len(entries):
                    print(
                        f"{split}: {i}/{len(entries)} checked, {len(mismatches)} mismatches",
                        flush=True,
                    )
        summaries[split] = {"checked": len(entries)}
    if split_paths["train"] & split_paths["valid"]:
        raise ValueError("Train/valid cache index overlap")
    return {
        "cache_root": str(root.resolve()),
        "source_root": str(source_root),
        "checked": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "splits": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_train.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/ghost_dataset_cache_v1"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--blosc-threads", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1 or (args.blosc_threads is not None and args.blosc_threads < 1):
        parser.error("--workers and --blosc-threads must be at least 1")
    if args.blosc_threads is not None:
        blosc2.set_nthreads(args.blosc_threads)
    result = verify(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
