"""Independent full-frame Ghost evaluation for trained Fast/Fine WaveDSP models.

Run from the repository root. The original FWLMAE entrypoints are untouched.
"""

from __future__ import annotations

import argparse
import codecs
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from experiments.wavedsp.train import SupervisedWaveDSP
from scripts.run_estimate import SimpleFWLDataset, upsampling_prediction
from src.config.constants import LABEL_MAP
from src.data.dataset_fwl import FWLDataset
from src.training.fwl_mae_finetune_test import detect_peaks_in_voxel, evaluate_peaks
from src.utils import save_blosc2


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_model(checkpoint_path: Path, kind: str, device: torch.device) -> tuple[SupervisedWaveDSP, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError(f"Not a WaveDSP training checkpoint: {checkpoint_path}")
    settings = checkpoint["config"]
    if settings.get("model") != kind:
        raise ValueError(f"Checkpoint model is {settings.get('model')!r}, expected {kind!r}")
    size = settings.get("target_size")
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError("Checkpoint has no valid target_size [X,Y,T]")
    if int(settings.get("supervised", {}).get("num_classes", -1)) != 4:
        raise ValueError("WaveDSP evaluation requires the trained four-class head")
    model = SupervisedWaveDSP(kind, int(size[2]), **settings.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, settings


def check_preprocessing(settings: dict[str, Any], test_config: dict[str, Any]) -> tuple[int, int, int]:
    size = tuple(int(value) for value in settings["target_size"])
    training_data = read_yaml(Path(settings["data_config"]))
    for key in ("y_crop_top", "y_crop_bottom", "z_crop_front", "z_crop_back"):
        if int(training_data.get(key, 0)) != int(test_config.get(key, 0)):
            raise ValueError(f"Test {key} differs from the WaveDSP training preprocessing")
    if int(training_data.get("downsample_z", size[2])) != int(test_config.get("downsample_z", 0)):
        raise ValueError("Test temporal downsampling differs from WaveDSP training")
    if size[2] != int(test_config["downsample_z"]):
        raise ValueError("Checkpoint time bins differ from test downsample_z")
    return size


def predict_labels(
    model: SupervisedWaveDSP,
    voxels: torch.Tensor,
    threshold: float | None,
    low_confidence_label: int,
    precision: str,
) -> torch.Tensor:
    enabled = voxels.device.type == "cuda" and precision == "bf16"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        logits = model(voxels)
        if threshold is None:
            return logits.argmax(dim=1)
        probabilities = logits.float().softmax(dim=1)
        confidence, labels = probabilities.max(dim=1)
        return torch.where(
            confidence >= threshold, labels, torch.full_like(labels, low_confidence_label)
        )


def frame_name(voxel_file: Path) -> str:
    # Same scene_hist_frame_prediction_voxel.b2 convention as scripts/run_estimate.py.
    return f"{voxel_file.parents[2].name}_{voxel_file.parent.name}_{voxel_file.stem.removesuffix('_voxel')}_prediction_voxel.b2"


def estimate(
    model: SupervisedWaveDSP,
    settings: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    precision: str,
) -> int:
    size = check_preprocessing(settings, config)
    dataset = SimpleFWLDataset(
        voxel_dirs=config["test_voxel_dirs"],
        downsample_z=config["downsample_z"],
        y_crop_top=config.get("y_crop_top", 0),
        y_crop_bottom=config.get("y_crop_bottom", 0),
        z_crop_front=config.get("z_crop_front", 0),
        z_crop_back=config.get("z_crop_back", 0),
        voxel_pattern="*_voxel.b2",
    )
    if not len(dataset):
        raise ValueError("No test voxel files found for estimate")
    expected = {frame_name(path) for path in dataset.voxel_files}
    existing = {path.name for path in output_dir.glob("*_prediction_voxel.b2")}
    if existing - expected:
        raise ValueError(f"Estimate directory contains {len(existing - expected)} unrelated predictions: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = (
        float(config.get("prediction_threshold", 0.5))
        if config.get("use_threshold_prediction", False)
        else None
    )
    for index in tqdm(range(len(dataset)), desc="WaveDSP estimate", unit="frame"):
        sample = dataset[index]
        voxel = sample["voxel_grid"]
        if tuple(voxel.shape) != size:
            raise ValueError(f"{sample['voxel_file']}: got {voxel.shape}, expected {size}")
        input_tensor = torch.from_numpy(np.ascontiguousarray(voxel)).permute(2, 1, 0)
        input_tensor = input_tensor[None, None].to(device=device, dtype=torch.float32)
        labels = predict_labels(model, input_tensor, threshold, -1, precision)
        prediction = labels[0].permute(2, 1, 0).cpu().numpy().astype(np.int16, copy=False)
        restored = upsampling_prediction(prediction, tuple(sample["original_shape"]))
        save_blosc2(str(output_dir / frame_name(Path(sample["voxel_file"]))), np.ascontiguousarray(restored))
    return len(dataset)


def confusion_scores(matrix: np.ndarray) -> dict[str, Any]:
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(4, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(4, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4, dtype=float), where=(precision + recall) != 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros(4, dtype=float), where=(tp + fp + fn) != 0)
    return {
        "count": int(matrix.sum()),
        "accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "miou": float(iou.mean()),
        "per_class": {
            LABEL_MAP[class_id]: {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "iou": float(iou[class_id]),
                "support": int(matrix[class_id].sum()),
            }
            for class_id in range(4)
        },
        "confusion_matrix": matrix.tolist(),
    }


def check_pairs(dataset: FWLDataset) -> None:
    if len(dataset.voxel_files) != len(dataset.annotation_files):
        raise ValueError("Test voxel and annotation counts differ")
    for voxel, annotation in zip(dataset.voxel_files, dataset.annotation_files):
        voxel_path, ann_path = Path(voxel), Path(annotation)
        voxel_key = (voxel_path.parents[2].name, voxel_path.parent.name,
                     voxel_path.name.removesuffix("_voxel.b2"))
        ann_key = (ann_path.parents[2].name, ann_path.parent.name,
                   ann_path.name.removesuffix("_annotation_voxel.b2"))
        if voxel_key != ann_key:
            raise ValueError(f"Test voxel and annotation do not match: {voxel} / {annotation}")


def recall(
    model: SupervisedWaveDSP,
    settings: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    device: torch.device,
    precision: str,
    checkpoint_path: Path,
    *,
    export_confusion: bool = True,
) -> dict[str, Any]:
    size = check_preprocessing(settings, config)
    dataset = FWLDataset(
        voxel_dirs=config["test_voxel_dirs"],
        annotation_dirs=config["test_annotation_dirs"],
        target_size=list(size),
        downsample_z=int(config["downsample_z"]),
        divide=1,
        y_crop_top=int(config.get("y_crop_top", 0)),
        y_crop_bottom=int(config.get("y_crop_bottom", 0)),
        z_crop_front=int(config.get("z_crop_front", 0)),
        z_crop_back=int(config.get("z_crop_back", 0)),
    )
    check_pairs(dataset)
    if not len(dataset):
        raise ValueError("No paired test files found for recall")
    threshold = (
        float(config.get("prediction_threshold", 0.5))
        if config.get("use_threshold_prediction", False)
        else None
    )
    voxel_cm = np.zeros((4, 4), dtype=np.int64)
    peak_cm = np.zeros((4, 4), dtype=np.int64)
    scene_peaks: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    frames: dict[str, int] = defaultdict(int)
    for index in tqdm(range(len(dataset)), desc="WaveDSP recall", unit="frame"):
        sample = dataset[index]
        voxel = np.asarray(sample["voxel_grid"])
        target = np.asarray(sample["annotation"])
        if tuple(voxel.shape) != size or target.shape != voxel.shape:
            raise ValueError(f"{sample['frame_id']}: unexpected shape {voxel.shape}/{target.shape}")
        input_tensor = torch.from_numpy(np.ascontiguousarray(voxel)).permute(2, 1, 0)
        input_tensor = input_tensor[None, None].to(device=device, dtype=torch.float32)
        labels = predict_labels(model, input_tensor, threshold, 0, precision)[0].cpu().numpy()
        target_tyx = target.transpose(2, 1, 0)
        valid = (target_tyx >= 0) & (target_tyx < 4)
        voxel_cm += np.bincount(target_tyx[valid] * 4 + labels[valid], minlength=16).reshape(4, 4)
        raw_tyx = voxel.transpose(2, 1, 0)
        peak = evaluate_peaks(labels, target_tyx, detect_peaks_in_voxel(raw_tyx), [-1], 4)
        peak_cm += peak["peak_confusion_matrix"]
        scene = str(sample["scene_id"])
        scene_peaks[scene] += peak["peak_confusion_matrix"]
        frames[scene] += 1
    scene_scores = {scene: {"frames": frames[scene], **confusion_scores(cm)}
                    for scene, cm in sorted(scene_peaks.items())}
    # The legacy test prints an equal-scene summary in addition to pooled peak metrics.
    scene_values = [item for item in scene_scores.values() if item["count"] > 0]
    mean_p = float(np.mean([item["macro_precision"] for item in scene_values])) if scene_values else 0.0
    mean_r = float(np.mean([item["macro_recall"] for item in scene_values])) if scene_values else 0.0
    result = {
        "model": settings["model"],
        "checkpoint": str(checkpoint_path),
        "frames": len(dataset),
        "target_size_XYT": list(size),
        "threshold": threshold,
        "voxel": confusion_scores(voxel_cm),
        "peak": confusion_scores(peak_cm),
        "peak_by_scene": scene_scores,
        "peak_scene_average": {
            "accuracy": float(np.mean([item["accuracy"] for item in scene_values])) if scene_values else 0.0,
            "macro_precision": mean_p,
            "macro_recall": mean_r,
            "macro_f1": 2 * mean_p * mean_r / (mean_p + mean_r) if mean_p + mean_r else 0.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"frames": result["frames"], "voxel_macro_f1": result["voxel"]["macro_f1"],
                      "peak_recall": result["peak"]["macro_recall"],
                      "peak_scene_average_recall": result["peak_scene_average"]["macro_recall"],
                      "report": str(output_path)}, ensure_ascii=False))
    if not export_confusion:
        return result
    # Optional reporting runs after the original metrics have been saved.
    # A plotting failure must not discard recall results or stop the all pipeline.
    try:
        from experiments.wavedsp.confusion_matrix import export_confusion_matrices

        report_dir = export_confusion_matrices(output_path)
        print(f"Saved confusion matrices: {report_dir}")
    except Exception as error:
        print(f"Warning: confusion matrix export failed: {error}. "
              f"Recall is saved at {output_path}; retry with "
              "python -m experiments.wavedsp.confusion_matrix --recall-json "
              f"{output_path}", file=sys.stderr)
    return result


def generate_pcd(pred_dir: Path, pcd_dir: Path, config_path: Path) -> None:
    files = sorted(pred_dir.glob("*_prediction_voxel.b2"))
    if not files:
        raise ValueError(f"No WaveDSP estimate files in {pred_dir}")
    expected_pcd = {f"{path.stem}{suffix}.pcd" for path in files for suffix in ("", "_gt")}
    existing_pcd = {path.name for path in pcd_dir.glob("*.pcd")}
    if existing_pcd - expected_pcd:
        raise ValueError(f"PCD directory contains {len(existing_pcd - expected_pcd)} unrelated files: {pcd_dir}")
    pending = [path for path in files if not (pcd_dir / f"{path.stem}.pcd").is_file()
               or not (pcd_dir / f"{path.stem}_gt.pcd").is_file()]
    print(f"PCD pairs complete: {len(files) - len(pending)}/{len(files)}; remaining: {len(pending)}")
    if pending:
        settings = read_yaml(config_path)
        # The original batch script has no resume option. A temporary directory of
        # links lets it process only frames without a complete predicted/GT pair.
        with tempfile.TemporaryDirectory(prefix="wavedsp_pcd_") as temporary:
            pending_dir = Path(temporary)
            for path in pending:
                (pending_dir / path.name).symlink_to(path.resolve())
            command = [sys.executable, "src/visualize/vis_pcd_batch.py", "--config", str(config_path),
                       "--pred_dir", str(pending_dir), "--output_dir", str(pcd_dir)]
            if settings.get("ghost_dataset"):
                command += ["--ghost_dataset", str(settings["ghost_dataset"])]
            subprocess.run(command, check=True)
    missing = [path.name for path in files if not (pcd_dir / f"{path.stem}.pcd").is_file()
               or not (pcd_dir / f"{path.stem}_gt.pcd").is_file()]
    if missing:
        raise RuntimeError(f"Missing predicted/GT PCD pairs for {len(missing)} frames: {missing[:3]}")


def stream_command_to_report(command: list[str], report_path: Path) -> None:
    """Tee child stdout/stderr live, including carriage-return progress updates."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            try:
                assert process.stdout is not None
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    chunk = process.stdout.read1(65536)
                    text = decoder.decode(chunk, final=not chunk)
                    if text:
                        report.write(text)
                        report.flush()
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    if not chunk:
                        break
                returncode = process.wait()
            except BaseException:
                # Do not leave a background evaluator running after Ctrl+C or an I/O error.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def ghost_removal(pcd_dir: Path, config_path: Path, report_path: Path) -> None:
    gt_files = list(pcd_dir.glob("*_gt.pcd"))
    if not gt_files:
        raise ValueError(f"No GT PCD files in {pcd_dir}")
    missing = [path.name for path in gt_files if not path.with_name(path.name.replace("_gt.pcd", ".pcd")).is_file()]
    if missing:
        raise ValueError(f"Missing predicted PCD for {len(missing)} GT files: {missing[:3]}")
    command = [sys.executable, "-u", "src/visualize/evaluate_pcd_batch.py", "--config", str(config_path),
               "--root", str(pcd_dir)]
    print(f"Ghost removal: evaluating {len(gt_files)} PCD pairs in {pcd_dir}", flush=True)
    print(f"Live log and results: {report_path}", flush=True)
    stream_command_to_report(command, report_path)
    print(f"Saved report: {report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("estimate", "recall", "pcd", "ghost-removal", "all"))
    parser.add_argument("--model", choices=("fast", "fine"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="WaveDSP best.pt or last.pt")
    parser.add_argument("--output-root", type=Path, help="Default: <checkpoint directory>/evaluation")
    parser.add_argument("--test-config", type=Path, default=Path("configs/config_test.yaml"))
    parser.add_argument("--estimate-config", type=Path, default=Path("configs/config_estimate.yaml"))
    parser.add_argument("--pcd-config", type=Path, default=Path("src/visualize/configs/vis_pcd_batch.yaml"))
    parser.add_argument("--pcd-eval-config", type=Path,
                        default=Path("src/visualize/configs/evaluate_pcd_batch.yaml"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--skip-confusion-matrix", action="store_true",
                        help="Save recall.json without exporting confusion matrix images or CSV reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("estimate", "recall", "all") and args.checkpoint is None:
        raise ValueError("--checkpoint is required for estimate, recall, and all")
    if args.output_root is None and args.checkpoint is None:
        raise ValueError("--output-root is required without --checkpoint")
    output_root = args.output_root or args.checkpoint.parent / "evaluation"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = settings = None
    if args.stage in ("estimate", "recall", "all"):
        model, settings = load_model(args.checkpoint, args.model, device)
    if args.stage in ("estimate", "all"):
        count = estimate(model, settings, read_yaml(args.estimate_config),
                         output_root / "estimate", device, args.precision)
        print(f"Saved {count} estimated frames to {output_root / 'estimate'}")
    if args.stage in ("recall", "all"):
        recall(model, settings, read_yaml(args.test_config), output_root / "recall.json",
               device, args.precision, args.checkpoint,
               export_confusion=not args.skip_confusion_matrix)
    if args.stage in ("pcd", "all"):
        generate_pcd(output_root / "estimate", output_root / "pcd", args.pcd_config)
    if args.stage in ("ghost-removal", "all"):
        ghost_removal(output_root / "pcd", args.pcd_eval_config,
                      output_root / "ghost_removal.txt")


if __name__ == "__main__":
    main()
