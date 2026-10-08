"""Count peaks in a configured dataset split without training preprocessing.

Object/glass/ghost counts come from unexpanded annotation peak positions.
Noise is an estimate: waveform local maxima above the height threshold whose
annotation is zero and which are not near a nonzero annotation. Zero background
bins are never counted. Unannotated regions cannot be identified without a mask.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import blosc2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CLASSES = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
CORRUPT = "20250929162519_t01759130735367000000_000043"


def local_maxima(hist: np.ndarray, height: float, relative: float) -> np.ndarray:
    """Vectorized find_peaks(height=...) equivalent, including flat peak centers.

    Endpoints are excluded; even plateaus use the lower of the two center bins.
    No width/prominence/distance filtering or four-peak truncation is applied.
    """
    rows, bins = hist.shape
    result = np.zeros((rows, bins), dtype=bool)
    if bins < 3:
        return result
    # For each position, find the final bin of its constant-valued run.
    indices = np.arange(bins)[None, :]
    changes = np.ones((rows, bins), dtype=bool)
    changes[:, :-1] = hist[:, :-1] != hist[:, 1:]
    ends = np.minimum.accumulate(np.where(changes, indices, bins)[:, ::-1], axis=1)
    ends = ends[:, ::-1]
    r, start0 = np.nonzero(hist[:, 1:] > hist[:, :-1])
    start = start0 + 1
    end = ends[r, start]
    valid = end < bins - 1
    r, start, end = r[valid], start[valid], end[valid]
    peak_height = hist[r, start]
    thresholds = np.maximum(height, relative * hist.max(axis=1))
    valid = (peak_height > hist[r, end + 1]) & (peak_height >= thresholds[r])
    result[r[valid], (start[valid] + end[valid]) // 2] = True
    return result


def summarize_arrays(
    annotation: np.ndarray,
    waveform: np.ndarray | None,
    block_beams: int,
    noise_height: float,
    noise_relative_height: float,
    match_bins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if annotation.ndim != 3 or not np.isin(annotation, [-1, 0, 1, 2, 3]).all():
        raise ValueError("Expected a 3D annotation containing only -1, 0, 1, 2, 3")
    if waveform is not None and waveform.shape != annotation.shape:
        raise ValueError("Waveform and annotation shapes differ")
    flat = annotation.reshape(-1, annotation.shape[-1])
    wave = None if waveform is None else waveform.reshape(flat.shape)
    totals = np.zeros(4, dtype=np.int64)
    occurrences = np.zeros((3, 4), dtype=np.int64)
    valid_beams = 0
    for start in range(0, len(flat), block_beams):
        ann = flat[start : start + block_beams]
        valid = (ann >= 0).any(axis=1)
        valid_beams += int(valid.sum())
        # Adjacent labels would contradict isolated peak-top annotations.
        if np.any((ann[:, 1:] > 0) & (ann[:, 1:] == ann[:, :-1])):
            raise ValueError("Adjacent same-class labels found; verify unexpanded annotations")
        for label in (1, 2, 3):
            counts = np.count_nonzero(ann == label, axis=1)
            totals[label] += counts.sum()
            occurrences[label - 1] += np.bincount(np.minimum(counts[valid], 3), minlength=4)
        if wave is not None:
            hist = wave[start : start + block_beams]
            if not np.isfinite(hist).all():
                raise ValueError("Waveform contains nonfinite values")
            peaks = local_maxima(hist, noise_height, noise_relative_height)
            occupied = ann > 0
            near = occupied.copy()
            for offset in range(1, min(match_bins + 1, ann.shape[1])):
                near[:, offset:] |= occupied[:, :-offset]
                near[:, :-offset] |= occupied[:, offset:]
            totals[0] += np.count_nonzero(peaks & (ann == 0) & ~near)
    return totals, occurrences, valid_beams


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def discover_frames(
    config: dict, split: str = "train"
) -> tuple[list[tuple[Path, Path]], list[dict]]:
    voxels = config.get(f"{split}_voxel_dirs", [])
    annotations = config.get(f"{split}_annotation_dirs", [])
    if not voxels or len(voxels) != len(annotations):
        raise ValueError(f"Need equally sized nonempty {split}_voxel_dirs/{split}_annotation_dirs")
    frames = []
    skipped = []
    seen = set()
    for voxel_value, annotation_value in zip(voxels, annotations):
        voxel_dir = resolve_path(voxel_value)
        supplied = resolve_path(annotation_value)
        parts = list(supplied.parts)
        versions = [i for i, part in enumerate(parts) if part.startswith("annotation_v")]
        if len(versions) != 1:
            raise ValueError(f"Cannot identify annotation version: {supplied}")
        i = versions[0]
        parts[i] = parts[i].removesuffix("_expand")
        annotation_dir = Path(*parts)
        if (voxel_dir.parent.parent.name, voxel_dir.name) != (
            annotation_dir.parent.parent.name,
            annotation_dir.name,
        ):
            raise ValueError(f"Scene/hist mismatch: {voxel_dir}, {annotation_dir}")
        if not voxel_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f"Missing data or raw annotation directory: {annotation_dir}")
        voxel_map = {p.name.removesuffix("_voxel.b2"): p for p in voxel_dir.glob("*_voxel.b2")}
        ann_map = {
            p.name.removesuffix("_annotation_voxel.b2"): p
            for p in annotation_dir.glob("*_annotation_voxel.b2")
        }
        if not voxel_map or not ann_map:
            raise ValueError(f"Empty waveform or annotation directory: {annotation_dir}")
        for frame_id in sorted(voxel_map.keys() | ann_map.keys()):
            key = (voxel_dir.parent.parent.name, voxel_dir.name, frame_id)
            if key in seen:
                raise ValueError(f"Duplicate {split} frame: {key}")
            seen.add(key)
            if key[0] == "scene003" and key[1] == "hist022" and frame_id == CORRUPT:
                skipped.append({"frame": list(key), "reason": "known corrupted frame"})
                continue
            if frame_id not in voxel_map or frame_id not in ann_map:
                raise FileNotFoundError(f"Unpaired {split} frame: {key}")
            frames.append((voxel_map[frame_id], ann_map[frame_id]))
    return frames, skipped


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    output: Path,
    totals: np.ndarray,
    occurrences: np.ndarray,
    beams: int,
    include_noise: bool,
    split: str = "train",
) -> None:
    peak_sum = int(totals.sum())
    rows = [
        {
            "class": name,
            "peak_count": int(totals[label]),
            "fraction": int(totals[label]) / peak_sum if peak_sum else 0,
            "method": "detected_unmatched_estimate" if label == 0 else "raw_annotation",
        }
        for label, name in CLASSES.items()
        if include_noise or label != 0
    ]
    write_csv(output / "peak_class_counts.csv", ["class", "peak_count", "fraction", "method"], rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        [r["class"] for r in rows],
        [r["peak_count"] for r in rows],
        color=["gray", "#2ca02c", "#1f77b4", "#d62728"]
        if include_noise
        else ["#2ca02c", "#1f77b4", "#d62728"],
    )
    axis.bar_label(bars, fmt="%.0f", padding=3)
    axis.set(ylabel="Number of peaks", title=f"{split.capitalize()} set peak counts")
    axis.margins(y=0.15)
    figure.text(
        0.5,
        0.01,
        "Noise: thresholded unmatched local maxima estimate; others: raw annotations",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output / "peak_class_counts.png", dpi=180)
    plt.close(figure)
    buckets = ["0", "1", "2", "3+"]
    rows = [
        {
            "class": CLASSES[label],
            "peaks_per_beam": bucket,
            "beam_count": int(count),
            "fraction_of_valid_beams": int(count) / beams if beams else 0,
        }
        for label in (1, 2, 3)
        for bucket, count in zip(buckets, occurrences[label - 1])
    ]
    write_csv(
        output / "beam_occurrences.csv",
        ["class", "peaks_per_beam", "beam_count", "fraction_of_valid_beams"],
        rows,
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for label, axis in zip((1, 2, 3), axes):
        bars = axis.bar(buckets, occurrences[label - 1])
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        axis.set(title=CLASSES[label], xlabel="Peaks per beam", ylabel="Number of beams")
        axis.margins(y=0.15)
    figure.tight_layout()
    figure.savefig(output / "beam_occurrences.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--config", default=None, help="Default: configs/config_<split>.yaml")
    parser.add_argument("--output-dir", default=None, help="Default: output/<split>_peak_stats")
    parser.add_argument(
        "--noise-height",
        type=float,
        default=3.0,
        help="Minimum waveform intensity for estimated noise peaks (default: 3)",
    )
    parser.add_argument(
        "--noise-relative-height",
        type=float,
        default=0.0,
        help="Also require this fraction of the beam maximum (default: 0)",
    )
    parser.add_argument(
        "--match-bins",
        type=int,
        default=2,
        help="Exclude estimated noise within this many bins of nonzero labels",
    )
    parser.add_argument("--block-beams", type=int, default=512)
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional partial-run check")
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Only count three semantic classes; do not load waveform arrays",
    )
    args = parser.parse_args()
    args.config = args.config or str(ROOT / f"configs/config_{args.split}.yaml")
    args.output_dir = args.output_dir or str(ROOT / f"output/{args.split}_peak_stats")
    if (
        args.block_beams < 1
        or args.match_bins < 0
        or args.noise_height < 0
        or not np.isfinite(args.noise_height)
        or not 0 <= args.noise_relative_height <= 1
        or (args.limit_frames is not None and args.limit_frames < 1)
    ):
        parser.error("Invalid block size, threshold, matching radius, or frame limit")
    config_path = resolve_path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    frames, skipped = discover_frames(config, args.split)
    selected = frames if args.limit_frames is None else frames[: args.limit_frames]
    if not selected:
        raise ValueError(f"No {args.split} frames to process")
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    totals = np.zeros(4, dtype=np.int64)
    occurrences = np.zeros((3, 4), dtype=np.int64)
    beams = 0
    per_frame = []
    scenes = Counter()
    # A manifest records the exact split and raw versions used, independent of config edits.
    write_csv(
        output / "manifest.csv",
        ["waveform", "raw_annotation"],
        [{"waveform": str(v), "raw_annotation": str(a)} for v, a in selected],
    )
    print(
        f"{args.split.capitalize()} frames: {len(selected)}/{len(frames)}; raw annotations; no crop/downsampling",
        flush=True,
    )
    for index, (voxel_file, annotation_file) in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {annotation_file}", flush=True)
        try:
            annotation = blosc2.load_array(str(annotation_file))
            waveform = None if args.skip_noise else blosc2.load_array(str(voxel_file))
            counts, distribution, valid = summarize_arrays(
                annotation,
                waveform,
                args.block_beams,
                args.noise_height,
                args.noise_relative_height,
                args.match_bins,
            )
        except Exception as error:
            raise RuntimeError(f"Failed on {annotation_file}: {error}") from error
        totals += counts
        occurrences += distribution
        beams += valid
        scenes[annotation_file.parent.parent.parent.name] += 1
        per_frame.append(
            {
                "annotation": str(annotation_file),
                "valid_beams": valid,
                **{name: int(counts[label]) for label, name in CLASSES.items()},
            }
        )
        del annotation, waveform
    write_csv(output / "per_frame.csv", ["annotation", "valid_beams", *CLASSES.values()], per_frame)
    export_results(output, totals, occurrences, beams, not args.skip_noise, args.split)
    metadata = {
        "config": str(config_path),
        "split": args.split,
        "arguments": vars(args),
        "available_frames": len(frames),
        "processed_frames": len(selected),
        "partial_run": len(selected) != len(frames),
        "scene_frame_counts": dict(scenes),
        "valid_beams": beams,
        "skipped": skipped,
        "peak_counts": {
            name: int(totals[label]) if label != 0 or not args.skip_noise else None
            for label, name in CLASSES.items()
        },
        "scope": f"Configured {args.split} frames; full original spatial/temporal extent; no preprocessing",
        "beam_unit": "One (frame, x, y); repeated directions across frames count separately",
        "noise_method": "Local maxima, including plateau centers, above configured height; label 0;"
        " outside matching radius of positive annotations. Estimate, not verified noise GT."
        " No region mask available; zero-labeled unannotated regions may be included.",
        "denominator": "Beams with at least one nonnegative annotation bin; all-negative beams excluded",
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata["peak_counts"], ensure_ascii=False), flush=True)
    print(f"Saved tables and figures to {output}", flush=True)


if __name__ == "__main__":
    main()
