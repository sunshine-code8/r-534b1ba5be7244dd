"""Export confusion matrix reports from recall.json without model inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config.constants import LABEL_MAP

LABELS = [LABEL_MAP[index] for index in range(4)]


def matrix_report(scores: dict[str, Any]) -> dict[str, Any]:
    """Rows are true labels; columns are predictions. Preserve int64 counts."""
    matrix = np.asarray(scores["confusion_matrix"])
    if matrix.shape != (4, 4) or matrix.dtype.kind not in "iu" or (matrix < 0).any():
        raise ValueError("Expected a nonnegative integer 4x4 confusion matrix")
    matrix = matrix.astype(np.int64)
    total = int(matrix.sum())
    if "count" in scores and total != scores["count"]:
        raise ValueError("Confusion matrix sum differs from saved count")
    support = matrix.sum(axis=1)
    for index, label in enumerate(LABELS):
        saved_support = scores.get("per_class", {}).get(label, {}).get("support")
        if saved_support is not None and saved_support != int(support[index]):
            raise ValueError(f"Confusion matrix support differs for {label}")
    normalized = np.divide(matrix, support[:, None], out=np.zeros((4, 4)),
                           where=support[:, None] != 0)
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fn = int(support[index]) - tp
        fp = int(matrix[:, index].sum()) - tp
        tn = total - tp - fn - fp
        per_class[label] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                            "confusion_matrix": [[tn, fp], [fn, tp]]}
    return {"count": total, "confusion_matrix": matrix.tolist(),
            "row_normalized": normalized.tolist(), "one_vs_rest": per_class}


def write_matrix_csv(path: Path, matrix: list[list[Any]], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def draw_matrix(ax: Any, matrix: np.ndarray, labels: list[str], title: str,
                normalized: bool = False) -> Any:
    picture = ax.imshow(matrix, cmap="Blues", vmin=0,
                        vmax=1 if normalized else max(1, int(matrix.max())))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = 0.5 if normalized else matrix.max() / 2
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            label = f"{value:.2%}" if normalized else f"{int(value):,}"
            ax.text(col, row, label, ha="center", va="center", fontsize=10,
                    color="white" if value > threshold else "black")
    return picture


def save_figures(directory: Path, name: str, report: dict[str, Any]) -> None:
    # Use the noninteractive canvas directly: no GUI or global backend changes.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for suffix, values, normalized in (
        ("counts", report["confusion_matrix"], False),
        ("row_normalized", report["row_normalized"], True),
    ):
        figure = Figure(figsize=(9, 7), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        picture = draw_matrix(axis, np.asarray(values), LABELS,
                              f"{name}: {suffix.replace('_', ' ')}", normalized)
        figure.colorbar(picture, ax=axis, label="Fraction of true class" if normalized else "Count")
        for extension in ("png", "svg"):
            figure.savefig(directory / f"{suffix}.{extension}", dpi=160)
        figure.clear()

    figure = Figure(figsize=(12, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for axis, label in zip(axes.flat, LABELS):
        draw_matrix(axis, np.asarray(report["one_vs_rest"][label]["confusion_matrix"]),
                    [f"non-{label}", label], f"{label} vs rest (TN, FP / FN, TP)")
    figure.suptitle(f"{name}: one-vs-rest counts")
    for extension in ("png", "svg"):
        figure.savefig(directory / f"one_vs_rest.{extension}", dpi=160)
    figure.clear()


def export_confusion_matrices(recall_path: Path, output_dir: Path | None = None) -> Path:
    """Read saved counts; never modify recall.json or load checkpoints/data."""
    recall_path = Path(recall_path)
    source = json.loads(recall_path.read_text(encoding="utf-8"))
    destination = Path(output_dir) if output_dir is not None else recall_path.parent / "confusion_matrix"
    reports = {name: matrix_report(source[name]) for name in ("peak", "voxel")}
    for scene, scores in sorted(source.get("peak_by_scene", {}).items()):
        if not scene or scene in (".", "..") or "/" in scene or "\\" in scene:
            raise ValueError(f"Invalid scene name: {scene!r}")
        reports[f"peak_by_scene/{scene}"] = matrix_report(scores)
    # Validate all counts before writing any output.
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_recall": str(recall_path.resolve()),
        "metadata": {key: source.get(key) for key in
                     ("model", "checkpoint", "frames", "target_size_XYT", "threshold")},
        "labels": LABELS,
        "orientation": "rows=true, columns=predicted",
        "one_vs_rest_order": ["negative (other classes)", "positive (named class)"],
        "zero_support_policy": "Row-normalized values are zero when true support is zero.",
        "reports": reports,
    }
    (destination / "matrices.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, report in reports.items():
        directory = destination / name
        directory.mkdir(parents=True, exist_ok=True)
        write_matrix_csv(directory / "counts.csv", report["confusion_matrix"], LABELS)
        write_matrix_csv(directory / "row_normalized.csv", report["row_normalized"], LABELS)
        for label, binary in report["one_vs_rest"].items():
            write_matrix_csv(directory / f"{label}_one_vs_rest.csv",
                             binary["confusion_matrix"], [f"non-{label}", label])
    (destination / "README.md").write_text(
        "# Confusion matrices\n\n"
        "Source: " + str(recall_path.resolve()) + "\n\n"
        "Rows = true labels; columns = predicted labels.\n"
        "Class order: noise, object, glass, ghost.\n\n"
        "- peak/: pooled peak positions (main recall evaluation).\n"
        "- voxel/: all valid voxels.\n"
        "- peak_by_scene/: peak positions for each scene.\n"
        "- counts.csv/png/svg: raw four-class counts.\n"
        "- row_normalized.csv/png/svg: each row divided by its true support; "
        "diagonal = recall. CSV values are fractions; figures show percentages. "
        "Zero-support rows are shown as zero.\n"
        "- *_one_vs_rest.csv: binary counts for each named class. "
        "Rows/columns: non-class, class; matrix = [[TN, FP], [FN, TP]].\n"
        "- one_vs_rest.png/svg: four binary matrices in one figure.\n"
        "- matrices.json: counts, normalized matrices, binary counts and source metadata.\n\n"
        "Counts are positions, not frames or objects. Peak and voxel counts must not be mixed. "
        "These reports preserve the saved prediction threshold and are not ghost removal rates.\n",
        encoding="utf-8",
    )
    for name, report in reports.items():
        save_figures(destination / name, name, report)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: <recall directory>/confusion_matrix")
    args = parser.parse_args()
    destination = export_confusion_matrices(args.recall_json, args.output_dir)
    print(f"Saved confusion matrices: {destination}")


if __name__ == "__main__":
    main()
