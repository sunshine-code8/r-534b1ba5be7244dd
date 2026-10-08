"""Read-only timings inside and outside cached DataLoader workers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, default_collate, get_worker_info

from experiments.wavedsp.train import load_yaml, make_datasets


class TimedDataset(Dataset):
    def __init__(self, dataset: Dataset, native_layout: bool = False) -> None:
        self.dataset = dataset
        self.native_layout = native_layout

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = time.perf_counter()
        sample = dict(self.dataset[index])
        if self.native_layout:
            # Restore the contiguous [X,Y,T] cache order before CPU collation.
            sample["voxel_grids"] = sample["voxel_grids"][0].permute(2, 1, 0).unsqueeze(-1)
            sample["annotations"] = sample["annotations"].permute(2, 1, 0)
        sample["_load_ms"] = (time.perf_counter() - start) * 1000
        return sample


def timed_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    load_ms = [sample.pop("_load_ms") for sample in samples]
    start = time.perf_counter()
    batch = default_collate(samples)
    batch["_collate_ms"] = (time.perf_counter() - start) * 1000
    batch["_load_sum_ms"] = sum(load_ms)
    worker = get_worker_info()
    batch["_worker_id"] = worker.id if worker else -1
    batch["_worker_ready"] = time.perf_counter()
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--pace", type=float, default=0.6, help="Seconds between requests, approximating GPU work")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--native-layout", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    dataset, _ = make_datasets(config, None)
    loader = DataLoader(
        TimedDataset(dataset, native_layout=args.native_layout), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=not args.no_pin_memory,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers else None,
        collate_fn=timed_collate,
    )
    iterator = iter(loader)
    results = []
    try:
        for step in range(args.steps):
            started = time.perf_counter()
            batch = next(iterator)
            received = time.perf_counter()
            row = {
                "step": step + 1,
                "worker": batch["_worker_id"],
                "load_sum_ms": batch["_load_sum_ms"],
                "collate_ms": batch["_collate_ms"],
                "queue_and_pin_ms": (received - batch["_worker_ready"]) * 1000,
                "wait_ms": (received - started) * 1000,
            }
            print(json.dumps(row), flush=True)
            results.append(row)
            del batch
            time.sleep(args.pace)
    finally:
        del iterator, loader
    summary = {
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": not args.no_pin_memory,
        "native_layout": args.native_layout,
        "pace": args.pace,
        "steps": args.steps,
        "mean_ms": {key: statistics.mean(row[key] for row in results) for key in (
            "load_sum_ms", "collate_ms", "queue_and_pin_ms", "wait_ms"
        )},
    }
    print(json.dumps({"summary": summary}), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "steps": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
