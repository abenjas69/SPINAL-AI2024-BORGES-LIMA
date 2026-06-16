"""Evaluate Fusion V3 locked on AASCE 2019 real radiographs.

This is a zero-shot locked evaluation: it uses the Spinal-AI2024 Phase 5/MLP
pipeline, the colleague centerline model, and the Fusion V3 holdout lock. It
does not train, tune, sweep thresholds, or select parameters on AASCE.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative o ambiente .venv antes de correr este script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "external_datasets"
    / "ascee_aasce2019"
    / "processed"
    / "ascee_aasce2019_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ascee_aasce2019_fusion_v3_locked"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "sanity_check"
    / "reports"
    / "ascee_aasce2019_fusion_v3_locked.md"
)
DEFAULT_PHASE5_MODEL = PROJECT_ROOT / "models" / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_PHASE9_MODEL = PROJECT_ROOT / "models" / "phase9_cobb_residual_mlp_v2.keras"
DEFAULT_PHASE9_SCALER = PROJECT_ROOT / "models" / "phase9_cobb_residual_mlp_v2_scaler.npz"
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_COLLEAGUE_MODEL = PROJECT_ROOT / "models" / "centerline_daniel_unet_baseline_2000_padding_512.keras"
DEFAULT_FUSION_LOCK = PROJECT_ROOT / "experiments" / "fusion_centerline_mlp_v3_holdout3192" / "fusion_centerline_mlp_v3_lock.json"

ANGLE_NAMES = ("PT", "MT", "TLL")
POINT_ORDER = ("upperLeft", "upperRight", "lowerLeft", "lowerRight")
ROI_SOURCE = "phase5_predicted_points"
EPS = 1.0e-8

PREDICTION_FIELDNAMES = [
    "index",
    "file_name",
    "image_path",
    "width",
    "height",
    "angle_1",
    "angle_2",
    "angle_3",
    "gt_cobb_max",
    "landmark_geometric_cobb",
    "base_status",
    "base_error",
    "mlp_locked_cobb",
    "mlp_locked_abs_error",
    "geom_cobb",
    "geom_abs_error",
    "applied_correction_deg",
    "predicted_residual_deg",
    "vertebra_count",
    "confidence",
    "quality_flags",
    "upper_index",
    "lower_index",
    "span",
    "timing_ms",
    "centerline_status",
    "centerline_error",
    "centerline_points",
    "centerline_mask_pixels",
    "centerline_pred_max_cobb",
    "centerline_abs_error",
    "centerline_bias_corrected",
    "centerline_bias_corrected_abs_error",
    "fusion_v3_locked",
    "fusion_v3_abs_error",
    "fusion_error_delta_vs_mlp",
    "fusion_improved_vs_mlp",
    "fusion_rescued_gt5",
    "fusion_broken_gt5",
    "roi_source",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "crop_width",
    "crop_height",
    "normalized_width",
    "normalized_height",
    "scale",
    "pad_x",
    "pad_y",
    "phase5_predicted_point_count",
    "phase5_selected_vertebrae",
    "phase5_mean_score",
    "prediction_min",
    "prediction_max",
    "prediction_mean",
]


def import_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {path}")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    require_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False)


def append_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def maybe_error(value: Any) -> str:
    if isinstance(value, Mapping):
        code = str(value.get("code", "")).strip()
        message = str(value.get("message", "")).strip()
        if code and message:
            return f"{code}: {message}"
        return code or message
    return str(value or "")


def quality_flags_text(result: Mapping[str, Any]) -> str:
    flags = result.get("qualityFlags", [])
    if not isinstance(flags, list):
        return ""
    values: list[str] = []
    for flag in flags:
        if isinstance(flag, Mapping):
            name = str(flag.get("name", "")).strip()
            if name:
                values.append(name)
        else:
            values.append(str(flag))
    return ";".join(values)


def load_fusion_lock(path: Path) -> dict[str, float]:
    require_file(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    bias = float(payload["centerline_bias_correction_deg"])
    centerline_weight = float(payload["centerline_weight"])
    if not 0.0 <= centerline_weight <= 1.0:
        raise ValueError(f"Peso centerline invalido no lock: {centerline_weight}")
    return {
        "centerline_bias_correction_deg": bias,
        "centerline_weight": centerline_weight,
        "mlp_weight": 1.0 - centerline_weight,
    }


def metric_bundle(pred_values: Sequence[float], gt_values: Sequence[float], total_count: int) -> dict[str, Any]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred_ok = pred[mask]
    gt_ok = gt[mask]
    ok_count = int(pred_ok.size)
    total_count = int(total_count)
    result: dict[str, Any] = {
        "total_images": total_count,
        "num_images": ok_count,
        "missing_predictions": max(total_count - ok_count, 0),
        "coverage_rate": float(ok_count / max(total_count, 1)),
    }
    if ok_count == 0:
        result.update(
            {
                "mae_deg": None,
                "rmse_deg": None,
                "bias_deg": None,
                "median_abs_error_deg": None,
                "p90_abs_error_deg": None,
                "within_3deg_rate": None,
                "within_5deg_rate": None,
                "within_10deg_rate": None,
                "failures_gt5": 0,
                "failures_gt10": 0,
                "paper_smape_pct": None,
                "within_3deg_rate_overall": 0.0,
                "within_5deg_rate_overall": 0.0,
                "within_10deg_rate_overall": 0.0,
                "failures_gt5_overall": total_count,
                "failures_gt10_overall": total_count,
            }
        )
        return result

    errors = pred_ok - gt_ok
    abs_errors = np.abs(errors)
    denominator = np.maximum(np.abs(pred_ok) + np.abs(gt_ok), EPS)
    smape_terms = abs_errors / denominator
    within3 = int(np.sum(abs_errors <= 3.0))
    within5 = int(np.sum(abs_errors <= 5.0))
    within10 = int(np.sum(abs_errors <= 10.0))
    result.update(
        {
            "mae_deg": float(np.mean(abs_errors)),
            "rmse_deg": float(np.sqrt(np.mean(errors**2))),
            "bias_deg": float(np.mean(errors)),
            "median_abs_error_deg": float(np.median(abs_errors)),
            "p90_abs_error_deg": float(np.percentile(abs_errors, 90)),
            "p95_abs_error_deg": float(np.percentile(abs_errors, 95)),
            "within_3deg_rate": float(within3 / ok_count),
            "within_5deg_rate": float(within5 / ok_count),
            "within_10deg_rate": float(within10 / ok_count),
            "failures_gt5": int(np.sum(abs_errors > 5.0)),
            "failures_gt10": int(np.sum(abs_errors > 10.0)),
            "paper_smape_pct": float(100.0 * np.mean(smape_terms)),
            "standard_2x_smape_pct_not_for_curvnet": float(200.0 * np.mean(smape_terms)),
            "within_3deg_rate_overall": float(within3 / max(total_count, 1)),
            "within_5deg_rate_overall": float(within5 / max(total_count, 1)),
            "within_10deg_rate_overall": float(within10 / max(total_count, 1)),
            "failures_gt5_overall": int(total_count - within5),
            "failures_gt10_overall": int(total_count - within10),
        }
    )
    return result


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    numeric = finite_float(value)
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def metric_row(metrics: Mapping[str, Any], label: str) -> str:
    return (
        "| "
        f"{label} | "
        f"{int(metrics.get('num_images') or 0)} | "
        f"{int(metrics.get('total_images') or 0)} | "
        f"{format_metric(metrics.get('coverage_rate'))} | "
        f"{format_metric(metrics.get('mae_deg'))} | "
        f"{format_metric(metrics.get('paper_smape_pct'), 4)}% | "
        f"{format_metric(metrics.get('within_5deg_rate'))} | "
        f"{format_metric(metrics.get('within_5deg_rate_overall'))} | "
        f"{int(metrics.get('failures_gt5') or 0)} | "
        f"{int(metrics.get('failures_gt5_overall') or 0)} | "
        f"{format_metric(metrics.get('rmse_deg'))} | "
        f"{format_metric(metrics.get('p90_abs_error_deg'))} | "
        f"{format_metric(metrics.get('bias_deg'))} |"
    )


def points_from_vertebrae(vertebrae: Any) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    scores: list[float] = []
    if not isinstance(vertebrae, list):
        return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)
    for vertebra in vertebrae:
        if not isinstance(vertebra, Mapping):
            continue
        score = finite_float(vertebra.get("score"), 0.0)
        point_map = vertebra.get("points", {})
        if not isinstance(point_map, Mapping):
            continue
        for point_name in POINT_ORDER:
            point = point_map.get(point_name)
            if isinstance(point, Mapping):
                x_value = finite_float(point.get("x"))
                y_value = finite_float(point.get("y"))
            elif isinstance(point, Sequence) and len(point) >= 2 and not isinstance(point, str):
                x_value = finite_float(point[0])
                y_value = finite_float(point[1])
            else:
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                points.append((x_value, y_value))
        scores.append(score)
    return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def decode_image(path: Path) -> tf.Tensor:
    encoded = tf.io.read_file(str(path))
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return tf.image.decode_jpeg(encoded, channels=1)
    if suffix == ".png":
        return tf.image.decode_png(encoded, channels=1)
    return tf.image.decode_image(encoded, channels=1, expand_animations=False)


def preprocess_centerline_from_points(
    *,
    record: Mapping[str, Any],
    points_abs: np.ndarray,
    scores: np.ndarray,
    roi_padding: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if points_abs.size == 0:
        raise ValueError("sem pontos Phase 5 previstos para construir ROI centerline")
    width = int(record["width"])
    height = int(record["height"])
    points_abs = np.asarray(points_abs, dtype=np.float32).reshape(-1, 2)
    points_abs[:, 0] = np.clip(points_abs[:, 0], 0.0, max(width - 1.0, 0.0))
    points_abs[:, 1] = np.clip(points_abs[:, 1], 0.0, max(height - 1.0, 0.0))
    padding = int(roi_padding)
    xmin = max(0, int(np.floor(float(np.min(points_abs[:, 0])) - padding)))
    ymin = max(0, int(np.floor(float(np.min(points_abs[:, 1])) - padding)))
    xmax = min(width, int(np.ceil(float(np.max(points_abs[:, 0])) + padding)))
    ymax = min(height, int(np.ceil(float(np.max(points_abs[:, 1])) + padding)))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"ROI Phase 5 invalida: {(xmin, ymin, xmax, ymax)}")

    image_path = resolve_project_path(str(record["image_path"]))
    image = decode_image(image_path)
    crop = image[ymin:ymax, xmin:xmax, :]
    crop_height = int(ymax - ymin)
    crop_width = int(xmax - xmin)
    scale = min(image_size / crop_width, image_size / crop_height)
    normalized_width = max(1, int(crop_width * scale))
    normalized_height = max(1, int(crop_height * scale))
    resized = tf.image.resize(
        tf.cast(crop, tf.float32),
        [normalized_height, normalized_width],
        method="bilinear",
    )
    pad_x = (image_size - normalized_width) // 2
    pad_y = (image_size - normalized_height) // 2
    padded = tf.pad(
        resized,
        [
            [pad_y, image_size - normalized_height - pad_y],
            [pad_x, image_size - normalized_width - pad_x],
            [0, 0],
        ],
    )
    normalized = tf.cast(padded, tf.float32) / 255.0
    display = tf.cast(tf.clip_by_value(tf.round(padded), 0.0, 255.0), tf.uint8)
    meta = {
        "image_path": str(image_path),
        "roi_source": ROI_SOURCE,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "normalized_width": normalized_width,
        "normalized_height": normalized_height,
        "scale": float(scale),
        "pad_x": pad_x,
        "pad_y": pad_y,
        "phase5_predicted_point_count": int(points_abs.shape[0]),
        "phase5_selected_vertebrae": int(points_abs.shape[0] // 4),
        "phase5_mean_score": float(np.mean(scores)) if scores.size else 0.0,
    }
    return normalized.numpy().astype(np.float32), display.numpy()[:, :, 0], meta


def ascee_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    angles = list(record["angles_deg"])
    return {
        "file_name": str(record["file_name"]),
        "cobb_angles": {
            "PT": float(angles[0]),
            "MT": float(angles[1]),
            "TLL": float(angles[2]),
        },
    }


def process_centerline_one(
    *,
    centerline_model: tf.keras.Model,
    centerline_v1: ModuleType,
    record: Mapping[str, Any],
    points_abs: np.ndarray,
    scores: np.ndarray,
    roi_padding: int,
    threshold: float,
    output_dir: Path,
    save_overlays: bool,
    overlay_limit: int,
    overlay_count: int,
) -> tuple[dict[str, Any], int]:
    normalized, display_image, meta = preprocess_centerline_from_points(
        record=record,
        points_abs=points_abs,
        scores=scores,
        roi_padding=roi_padding,
        image_size=int(centerline_v1.IMAGE_SIZE),
    )
    rows, overlay_count = centerline_v1.process_batch(
        model=centerline_model,
        batch_items=[(int(record["index"]), ascee_sample(record), normalized, display_image, meta)],
        threshold=float(threshold),
        output_dir=output_dir,
        save_overlays_enabled=save_overlays,
        overlay_limit=int(overlay_limit),
        overlay_count=overlay_count,
    )
    if not rows:
        raise ValueError("centerline nao devolveu linha de predicao")
    return rows[0], overlay_count


def base_prediction_row(record: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    result = response.get("result", {})
    if not isinstance(result, Mapping):
        result = {}
    measurement = result.get("cobbMeasurement", {})
    if not isinstance(measurement, Mapping):
        measurement = {}
    status = str(response.get("status", "UNKNOWN"))
    gt_cobb = finite_float(record.get("gt_cobb_max"))
    mlp_pred = finite_float(result.get("cobbAngleDeg"))
    geom_pred = finite_float(result.get("rawGeometricCobbAngleDeg"))
    return {
        "index": int(record["index"]),
        "file_name": str(record["file_name"]),
        "image_path": str(record["image_path"]),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "angle_1": float(record["angles_deg"][0]),
        "angle_2": float(record["angles_deg"][1]),
        "angle_3": float(record["angles_deg"][2]),
        "gt_cobb_max": gt_cobb,
        "landmark_geometric_cobb": finite_float(record.get("landmark_geometric_cobb")),
        "base_status": status,
        "base_error": "" if status == "OK" else maybe_error(response.get("error", "")),
        "mlp_locked_cobb": mlp_pred,
        "mlp_locked_abs_error": abs(mlp_pred - gt_cobb) if np.isfinite(mlp_pred) and np.isfinite(gt_cobb) else np.nan,
        "geom_cobb": geom_pred,
        "geom_abs_error": abs(geom_pred - gt_cobb) if np.isfinite(geom_pred) and np.isfinite(gt_cobb) else np.nan,
        "applied_correction_deg": finite_float(result.get("appliedCorrectionDeg")),
        "predicted_residual_deg": finite_float(result.get("predictedResidualDeg")),
        "vertebra_count": int(result.get("vertebraCount", 0) or 0),
        "confidence": finite_float(result.get("confidence")),
        "quality_flags": quality_flags_text(result),
        "upper_index": int(measurement.get("upperVertebraIndex", -1) or -1),
        "lower_index": int(measurement.get("lowerVertebraIndex", -1) or -1),
        "span": int(measurement.get("span", 0) or 0),
        "timing_ms": int(response.get("timingMs", 0) or 0),
    }


def fuse_row(base_row: Mapping[str, Any], centerline_row: Mapping[str, Any], lock: Mapping[str, float]) -> dict[str, Any]:
    gt = finite_float(base_row.get("gt_cobb_max"))
    mlp_pred = finite_float(base_row.get("mlp_locked_cobb"))
    centerline_pred = finite_float(centerline_row.get("pred_max_cobb"))
    centerline_corrected = (
        centerline_pred + float(lock["centerline_bias_correction_deg"])
        if np.isfinite(centerline_pred)
        else np.nan
    )
    fusion_pred = (
        float(lock["mlp_weight"]) * mlp_pred + float(lock["centerline_weight"]) * centerline_corrected
        if np.isfinite(mlp_pred) and np.isfinite(centerline_corrected)
        else (mlp_pred if np.isfinite(mlp_pred) else np.nan)
    )
    mlp_error = abs(mlp_pred - gt) if np.isfinite(mlp_pred) and np.isfinite(gt) else np.nan
    fusion_error = abs(fusion_pred - gt) if np.isfinite(fusion_pred) and np.isfinite(gt) else np.nan
    row = dict(base_row)
    row.update(
        {
            "centerline_status": centerline_row.get("status", ""),
            "centerline_error": "",
            "centerline_points": centerline_row.get("centerline_points", ""),
            "centerline_mask_pixels": centerline_row.get("mask_pixels_gt_threshold", ""),
            "centerline_pred_max_cobb": centerline_pred,
            "centerline_abs_error": abs(centerline_pred - gt)
            if np.isfinite(centerline_pred) and np.isfinite(gt)
            else np.nan,
            "centerline_bias_corrected": centerline_corrected,
            "centerline_bias_corrected_abs_error": abs(centerline_corrected - gt)
            if np.isfinite(centerline_corrected) and np.isfinite(gt)
            else np.nan,
            "fusion_v3_locked": fusion_pred,
            "fusion_v3_abs_error": fusion_error,
            "fusion_error_delta_vs_mlp": fusion_error - mlp_error
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else np.nan,
            "fusion_improved_vs_mlp": int(fusion_error < mlp_error)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
            "fusion_rescued_gt5": int(mlp_error > 5.0 and fusion_error <= 5.0)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
            "fusion_broken_gt5": int(mlp_error <= 5.0 and fusion_error > 5.0)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
        }
    )
    for key in (
        "roi_source",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "crop_width",
        "crop_height",
        "normalized_width",
        "normalized_height",
        "scale",
        "pad_x",
        "pad_y",
        "phase5_predicted_point_count",
        "phase5_selected_vertebrae",
        "phase5_mean_score",
        "prediction_min",
        "prediction_max",
        "prediction_mean",
    ):
        row[key] = centerline_row.get(key, "")
    return row


def failed_centerline_row(base_row: Mapping[str, Any], error: str, lock: Mapping[str, float]) -> dict[str, Any]:
    empty_centerline = {
        "status": "ERROR",
        "pred_max_cobb": np.nan,
        "centerline_points": "",
        "mask_pixels_gt_threshold": "",
    }
    row = fuse_row(base_row, empty_centerline, lock)
    row["centerline_status"] = "ERROR"
    row["centerline_error"] = error
    row["roi_source"] = ROI_SOURCE
    return row


def build_metrics(rows: Sequence[Mapping[str, Any]], total_count: int) -> dict[str, dict[str, Any]]:
    gt_values = [finite_float(row.get("gt_cobb_max")) for row in rows]
    return {
        "mlp_locked": metric_bundle(
            [finite_float(row.get("mlp_locked_cobb")) for row in rows],
            gt_values,
            total_count,
        ),
        "geometry_max_angle": metric_bundle(
            [finite_float(row.get("geom_cobb")) for row in rows],
            gt_values,
            total_count,
        ),
        "centerline_only": metric_bundle(
            [finite_float(row.get("centerline_pred_max_cobb")) for row in rows],
            gt_values,
            total_count,
        ),
        "centerline_bias_corrected": metric_bundle(
            [finite_float(row.get("centerline_bias_corrected")) for row in rows],
            gt_values,
            total_count,
        ),
        "fusion_v3_locked": metric_bundle(
            [finite_float(row.get("fusion_v3_locked")) for row in rows],
            gt_values,
            total_count,
        ),
        "gt_landmark_geometry_audit": metric_bundle(
            [finite_float(row.get("landmark_geometric_cobb")) for row in rows],
            gt_values,
            total_count,
        ),
    }


def count_int(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        total += int(value)
    return total


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        "# AASCE 2019 - Fusion V3 Locked",
        "",
        "## Scope",
        "",
        "Zero-shot locked evaluation on real AASCE radiographs.",
        "No AASCE training, threshold tuning, bias fitting, weight sweep, or model selection is performed.",
        "",
        "## Artefacts",
        "",
        f"- manifest: `{payload['manifest_path']}`",
        f"- Phase 5 model: `{payload['phase5_model_path']}`",
        f"- Phase 5 profile: `{payload['profile']}`",
        f"- Phase 9 model: `{payload['phase9_model_path']}`",
        f"- centerline model: `{payload['colleague_model_path']}`",
        f"- fusion lock: `{payload['fusion_lock_path']}`",
        f"- centerline bias correction: `{payload['fusion_lock']['centerline_bias_correction_deg']:.6f}`",
        f"- centerline weight: `{payload['fusion_lock']['centerline_weight']:.4f}`",
        f"- mlp weight: `{payload['fusion_lock']['mlp_weight']:.4f}`",
        "",
        "## Dataset",
        "",
        f"- selected images: `{payload['selected_count']}`",
        f"- processed rows: `{payload['processed_count']}`",
        f"- base OK rows: `{payload['base_ok_count']}`",
        f"- centerline OK rows: `{payload['centerline_ok_count']}`",
        f"- tuning on AASCE: `{payload['tuning_on_ascee']}`",
        f"- roi source: `{payload['roi_source']}`",
        "",
        "## Results",
        "",
        "| method | finite N | total | coverage | MAE OK | SMAPE OK | within5 OK | within5 overall | fail>5 OK | fail>5 overall | RMSE OK | p90 OK | bias OK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("geometry_max_angle", "geometry max-angle"),
        ("mlp_locked", "MLP locked"),
        ("centerline_only", "centerline only"),
        ("centerline_bias_corrected", "centerline bias-corrected"),
        ("fusion_v3_locked", "Fusion V3 locked"),
        ("gt_landmark_geometry_audit", "GT landmark geometry audit"),
    ):
        lines.append(metric_row(metrics[key], label))
    lines.extend(
        [
            "",
            "## Fusion Effect vs MLP",
            "",
            f"- improved rows: `{payload['fusion_effect']['improved_count']}`",
            f"- rescued >5: `{payload['fusion_effect']['rescued_gt5']}`",
            f"- broken >5: `{payload['fusion_effect']['broken_gt5']}`",
            "",
            "## Outputs",
            "",
            f"- predictions CSV: `{payload['predictions_csv']}`",
            f"- metrics JSON: `{payload['metrics_json']}`",
            "",
            "## Notes",
            "",
            "- AASCE ground-truth landmarks are used only for metrics/audit, never for ROI construction.",
            "- Failed base predictions are kept in the denominator for coverage-aware rates.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Fusion V3 locked on AASCE 2019.")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--phase5-model-path", default=str(DEFAULT_PHASE5_MODEL))
    parser.add_argument("--phase9-model-path", default=str(DEFAULT_PHASE9_MODEL))
    parser.add_argument("--phase9-scaler-path", default=str(DEFAULT_PHASE9_SCALER))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-correction-deg", type=float, default=10.0)
    parser.add_argument("--colleague-model", default=str(DEFAULT_COLLEAGUE_MODEL))
    parser.add_argument("--fusion-lock", default=str(DEFAULT_FUSION_LOCK))
    parser.add_argument("--centerline-threshold", type=float, default=0.5)
    parser.add_argument("--roi-padding", type=int, default=20)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest_path)
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)
    prediction_path = output_dir / "ascee_aasce2019_fusion_v3_locked_predictions.csv"
    metrics_path = output_dir / "ascee_aasce2019_fusion_v3_locked_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase5_model_path = resolve_project_path(args.phase5_model_path)
    phase9_model_path = resolve_project_path(args.phase9_model_path)
    phase9_scaler_path = resolve_project_path(args.phase9_scaler_path)
    colleague_model_path = resolve_project_path(args.colleague_model)
    fusion_lock_path = resolve_project_path(args.fusion_lock)
    for path in (phase5_model_path, phase9_model_path, phase9_scaler_path, colleague_model_path, fusion_lock_path):
        require_file(path)
    lock = load_fusion_lock(fusion_lock_path)

    all_records = read_manifest(manifest_path)
    start_index = max(int(args.start_index), 0)
    selected_records = all_records[start_index:]
    if int(args.num_images) > 0:
        selected_records = selected_records[: int(args.num_images)]
    if not selected_records:
        raise ValueError("Nenhuma imagem AASCE selecionada.")

    existing_rows = read_csv(prediction_path) if args.resume else []
    if not args.resume and prediction_path.exists():
        raise ValueError("Predictions CSV ja existe. Use --resume ou escolha outro --output-dir.")
    processed_names = {str(row.get("file_name", "")).strip() for row in existing_rows if row.get("file_name")}
    rows: list[dict[str, Any]] = [dict(row) for row in existing_rows]

    inference = import_script(PROJECT_ROOT / "deployment" / "spinal_ai_inference.py", "ascee_fusion_v3_inference")
    centerline_v1 = import_script(SCRIPTS_DIR / "107_eval_colleague_centerline_model_v1.py", "ascee_fusion_v3_centerline")

    analyzer = inference.SpinalAIAnalyzer(
        phase5_model_path=phase5_model_path,
        phase9_model_path=phase9_model_path,
        phase9_scaler_path=phase9_scaler_path,
        profile=str(args.profile),
        alpha=float(args.alpha),
        max_correction_deg=float(args.max_correction_deg),
    )
    print(f"Loading colleague centerline model: {colleague_model_path}")
    centerline_model = tf.keras.models.load_model(colleague_model_path, compile=False)

    overlay_count = 0
    total = len(selected_records)
    last_reported = len(processed_names)
    print("\nAASCE Fusion V3 locked")
    print(f"selected images: {total}")
    print(f"resume skipped: {len(processed_names)}")
    print(f"output: {output_dir}")
    print(f"roi source: {ROI_SOURCE}")

    for local_index, record in enumerate(selected_records):
        file_name = str(record["file_name"])
        if file_name in processed_names:
            continue

        try:
            response = analyzer.analyze_image_path(
                estudo_id=file_name,
                image_path=str(record["image_path"]),
                include_overlay=False,
            )
            base_row = base_prediction_row(record, response)
            result = response.get("result", {})
            vertebrae = result.get("vertebrae", []) if isinstance(result, Mapping) else []
            points_abs, scores = points_from_vertebrae(vertebrae)
            centerline_row, overlay_count = process_centerline_one(
                centerline_model=centerline_model,
                centerline_v1=centerline_v1,
                record=record,
                points_abs=points_abs,
                scores=scores,
                roi_padding=int(args.roi_padding),
                threshold=float(args.centerline_threshold),
                output_dir=output_dir,
                save_overlays=bool(args.save_overlays),
                overlay_limit=int(args.overlay_limit),
                overlay_count=overlay_count,
            )
            output_row = fuse_row(base_row, centerline_row, lock)
        except Exception as exc:
            if "base_row" not in locals() or str(base_row.get("file_name", "")) != file_name:
                base_row = {
                    "index": int(record["index"]),
                    "file_name": file_name,
                    "image_path": str(record["image_path"]),
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "angle_1": float(record["angles_deg"][0]),
                    "angle_2": float(record["angles_deg"][1]),
                    "angle_3": float(record["angles_deg"][2]),
                    "gt_cobb_max": finite_float(record.get("gt_cobb_max")),
                    "landmark_geometric_cobb": finite_float(record.get("landmark_geometric_cobb")),
                    "base_status": "ERROR",
                    "base_error": str(exc),
                }
            output_row = failed_centerline_row(base_row, str(exc), lock)

        rows.append(output_row)
        append_csv_rows(prediction_path, [output_row], PREDICTION_FIELDNAMES)
        processed_count = len(rows)
        if args.progress_every > 0 and (
            processed_count - last_reported >= int(args.progress_every) or local_index + 1 >= total
        ):
            print(f"processed {processed_count}/{total} AASCE images")
            last_reported = processed_count

    metrics = build_metrics(rows, total)
    base_ok_count = sum(str(row.get("base_status")) == "OK" for row in rows)
    centerline_ok_count = sum(str(row.get("centerline_status")) == "ok" for row in rows)
    payload = {
        "phase": "ascee_aasce2019_fusion_v3_locked",
        "mode": "locked_zero_shot_ascee",
        "tuning_on_ascee": False,
        "roi_source": ROI_SOURCE,
        "manifest_path": manifest_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "selected_count": total,
        "processed_count": len(rows),
        "base_ok_count": int(base_ok_count),
        "centerline_ok_count": int(centerline_ok_count),
        "phase5_model_path": str(args.phase5_model_path),
        "phase9_model_path": str(args.phase9_model_path),
        "phase9_scaler_path": str(args.phase9_scaler_path),
        "profile": str(args.profile),
        "alpha": float(args.alpha),
        "max_correction_deg": float(args.max_correction_deg),
        "colleague_model_path": str(args.colleague_model),
        "fusion_lock_path": fusion_lock_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "fusion_lock": lock,
        "metrics": metrics,
        "fusion_effect": {
            "improved_count": count_int(rows, "fusion_improved_vs_mlp"),
            "rescued_gt5": count_int(rows, "fusion_rescued_gt5"),
            "broken_gt5": count_int(rows, "fusion_broken_gt5"),
        },
        "predictions_csv": prediction_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "metrics_json": metrics_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "report_path": report_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
    }
    write_json(metrics_path, payload)
    write_report(report_path, payload)

    print("\nAASCE Fusion V3 locked complete")
    print(f"processed: {len(rows)}/{total}")
    print(f"base OK: {base_ok_count}/{total}")
    print(f"centerline OK: {centerline_ok_count}/{total}")
    for key, label in (
        ("mlp_locked", "MLP locked"),
        ("centerline_only", "centerline only"),
        ("fusion_v3_locked", "Fusion V3 locked"),
    ):
        row = metrics[key]
        print(
            f"{label}: N={row['num_images']}/{row['total_images']}, "
            f"MAE_OK={format_metric(row.get('mae_deg'))}, "
            f"within5_overall={format_metric(row.get('within_5deg_rate_overall'))}, "
            f"fail>5_overall={row.get('failures_gt5_overall')}"
        )
    print(f"predictions: {prediction_path}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
