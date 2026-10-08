"""Build/resume the shared FAST/FINE waveform-only MAE cache (no CUDA needed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.wavedsp.mae_data import build_mae_cache
from experiments.wavedsp.train import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/wavedsp/configs/fast_mae.yaml")
    )
    parser.add_argument("--data-root", type=Path, help="Override mae_dataset root")
    parser.add_argument("--cache-root", type=Path, help="Override output cache directory")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = load_yaml(args.config)
    meta = build_mae_cache(
        load_yaml(Path(config["data_config"])),
        tuple(config["target_size"]),
        args.data_root or Path(config["data_root"]),
        args.cache_root or Path(config["cache_root"]),
        args.workers,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
