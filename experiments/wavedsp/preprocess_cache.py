"""One-time Ghost preprocessing for the shared Fast/Fine WaveDSP cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.wavedsp.cache import build_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/wavedsp/configs/fast_ghost.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/ghost_dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/ghost_dataset_cache_v1"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel frame workers; start with 1 while training is active")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_config = yaml.safe_load(Path(config["data_config"]).read_text())
    size = tuple(int(value) for value in config["target_size"])
    meta = build_cache(
        data_config, size, args.data_root, args.cache_root,
        int(config["supervised"]["num_classes"]),
        int(config["supervised"].get("ignore_index", -1)), args.workers,
    )
    print(json.dumps({"cache_root": str(args.cache_root), **meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
