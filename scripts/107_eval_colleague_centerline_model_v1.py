"""Evaluate the colleague centerline U-Net on cleaned Spinal-AI2024 annotations.

This script is intentionally standalone. It uses the cleaned annotation JSON from
this project as the source of truth, prepares the 512x512 padded ROI expected by
the colleague model, runs the U-Net mask prediction, converts the mask to a
centerline, and computes regional PT/MT/TLL Cobb estimates from that centerline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
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
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "processed" / "cleaned" / "test_ready_annotations_clean.json"
DEFAULT_COLLEAGUE_MODEL = PROJECT_ROOT / "models" / "centerline_daniel_unet_baseline_2000_padding_512.keras"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "fusion_centerline_model_v1"

IMAGE_SIZE = 512
ANGLE_NAMES = ("PT", "MT", "TLL")
EPS = 1.0e-8

DEFAULT_THRESHOLD = 0.5
DEFAULT_ROI_PADDING = 20
BLOCK_HEIGHT = 10
MIN_PIXELS_PER_BLOCK = 20
CENTERLINE_SMOOTH_WINDOW = 31
CENTERLINE_SMOOTH_POLYORDER = 3

COBB_SMOOTH_WINDOW = 1
COBB_SMOOTH_POLYORDER = 3
ANGLE_SMOOTH_WINDOW = 21
ANGLE_SMOOTH_POLYORDER = 3
TRIM_RATIO = 0.07
MIN_POINT_DISTANCE_GLOBAL = 80
MIN_POINT_DISTANCE_REGION = 50
MIN_REGION_POINTS = 12
MAX_ABS_ANGLE = 80
LOW_PERCENTILE = 3
HIGH_PERCENTILE = 97
CANDIDATE_LIMIT = 25

PT_CURVATURE_SUPPORT_START = 0.07
PT_CURVATURE_SUPPORT_END = 0.40
PT_CURVATURE_PROMINENCE = 2.0
PT_UNSUPPORTED_TARGET_SCALE = 0.55

TLL_CURVATURE_SUPPORT_START = 0.60
TLL_CURVATURE_SUPPORT_END = 0.98
TLL_LOW_AMPLITUDE_START = 0.70
TLL_LOW_AMPLITUDE_LIMIT = 10.0
TLL_SECONDARY_MIN_COBB = 15.0
TLL_UNSUPPORTED_TARGET_SCALE = 0.75

LOCAL_ANGLE_WINDOW = 11
LOCAL_STD_LIMIT = 8.0
LOCAL_STD_PENALTY = 0.08
CLOSE_DISTANCE_LIMIT = 90
CLOSE_DISTANCE_PENALTY = 0.05

REGION_EXPANSION_STEP = 0.05
MAX_REGION_EXPANSION = 0.15

REGIONS = (
    {
        "name": "PT",
        "label": "PT",
        "start": 0.07,
        "end": 0.42,
    },
    {
        "name": "MT",
        "label": "MT",
        "start": 0.05,
        "end": 0.95,
    },
    {
        "name": "TLL",
        "label": "TLL",
        "start": 0.35,
        "end": 1.00,
    },
)


def savgol_filter(values: Sequence[float], window_length: int, polyorder: int) -> np.ndarray:
    """Small local Savitzky-Golay fallback to avoid a SciPy runtime dependency."""
    array = np.asarray(values, dtype=np.float32)
    if len(array) == 0:
        return array.copy()
    window_length = make_odd(min(int(window_length), len(array)))
    if window_length <= polyorder or len(array) < window_length:
        return array.copy()

    half = window_length // 2
    x_window = np.arange(window_length, dtype=np.float32) - float(half)
    padded = np.pad(array, (half, half), mode="edge")
    result = np.empty_like(array, dtype=np.float32)
    degree = min(int(polyorder), window_length - 1)
    for index in range(len(array)):
        y_window = padded[index : index + window_length]
        coeffs = np.polyfit(x_window, y_window, degree)
        result[index] = float(np.polyval(coeffs, 0.0))
    return result


def find_peaks(
    values: Sequence[float],
    *,
    prominence: float,
    distance: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Approximate peak finder matching the limited behavior needed here."""
    array = np.asarray(values, dtype=np.float32)
    if len(array) < 3:
        return np.array([], dtype=np.int32), {"prominences": np.array([], dtype=np.float32)}

    distance = max(1, int(distance))
    candidates: list[tuple[int, float, float]] = []
    for index in range(1, len(array) - 1):
        if array[index] <= array[index - 1] or array[index] <= array[index + 1]:
            continue
        left = array[max(0, index - distance) : index + 1]
        right = array[index : min(len(array), index + distance + 1)]
        local_prominence = float(array[index] - max(float(np.min(left)), float(np.min(right))))
        if local_prominence >= prominence:
            candidates.append((index, float(array[index]), local_prominence))

    selected: list[tuple[int, float]] = []
    for index, height, local_prominence in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(index - chosen_index) >= distance for chosen_index, _ in selected):
            selected.append((index, local_prominence))
    selected.sort(key=lambda item: item[0])
    peaks = np.asarray([index for index, _ in selected], dtype=np.int32)
    prominences = np.asarray([item_prominence for _, item_prominence in selected], dtype=np.float32)
    return peaks, {"prominences": prominences}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_annotations(path: Path) -> list[dict[str, Any]]:
    require_file(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return list(payload["samples"])
    raise ValueError(f"Formato de annotations nao suportado: {path}")


def select_window(items: Sequence[Any], start_index: int, num_images: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_images <= 0:
        return items[start_index:]
    return items[start_index : start_index + num_images]


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def jsonable_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metric_bundle(pred_values: Sequence[float], gt_values: Sequence[float]) -> dict[str, Any]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[mask]
    gt = gt[mask]
    if pred.size == 0:
        return {"num_images": 0}

    errors = pred - gt
    abs_errors = np.abs(errors)
    denominator = np.abs(pred) + np.abs(gt)
    smape_terms = np.divide(
        abs_errors,
        denominator,
        out=np.zeros_like(abs_errors, dtype=np.float32),
        where=denominator > EPS,
    )

    result: dict[str, Any] = {
        "num_images": int(pred.size),
        "mae_deg": float(np.mean(abs_errors)),
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "bias_deg": float(np.mean(errors)),
        "median_abs_error_deg": float(np.median(abs_errors)),
        "p90_abs_error_deg": float(np.percentile(abs_errors, 90)),
        "p95_abs_error_deg": float(np.percentile(abs_errors, 95)),
        "p99_abs_error_deg": float(np.percentile(abs_errors, 99)),
        "max_abs_error_deg": float(np.max(abs_errors)),
        "within_3deg_rate": float(np.mean(abs_errors <= 3.0)),
        "within_5deg_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10deg_rate": float(np.mean(abs_errors <= 10.0)),
        "failures_gt3": int(np.sum(abs_errors > 3.0)),
        "failures_gt5": int(np.sum(abs_errors > 5.0)),
        "failures_gt10": int(np.sum(abs_errors > 10.0)),
        "paper_smape_pct": float(100.0 * np.mean(smape_terms)),
        "standard_2x_smape_pct_not_for_curvnet": float(200.0 * np.mean(smape_terms)),
        "smape_term_median_pct": float(100.0 * np.median(smape_terms)),
        "smape_term_p90_pct": float(100.0 * np.percentile(smape_terms, 90)),
        "mean_pred_deg": float(np.mean(pred)),
        "mean_gt_deg": float(np.mean(gt)),
    }
    if pred.size >= 2 and np.ptp(pred) > 1.0e-6 and np.ptp(gt) > 1.0e-6:
        result["pearson"] = float(np.corrcoef(gt, pred)[0, 1])
    else:
        result["pearson"] = None
    total = float(np.sum((gt - np.mean(gt)) ** 2))
    result["r2"] = float(1.0 - float(np.sum((gt - pred) ** 2)) / total) if total > EPS else None
    return result


def metric_row(metrics: Mapping[str, Any], label: str) -> str:
    if int(metrics.get("num_images", 0)) == 0:
        return f"| {label} | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |"
    return (
        "| "
        f"{label} | "
        f"{int(metrics['num_images'])} | "
        f"{float(metrics['mae_deg']):.3f} | "
        f"{float(metrics['paper_smape_pct']):.4f}% | "
        f"{float(metrics['within_3deg_rate']):.3f} | "
        f"{float(metrics['within_5deg_rate']):.3f} | "
        f"{float(metrics['within_10deg_rate']):.3f} | "
        f"{int(metrics['failures_gt5'])} | "
        f"{int(metrics['failures_gt10'])} | "
        f"{float(metrics['rmse_deg']):.3f} | "
        f"{float(metrics['p90_abs_error_deg']):.3f} | "
        f"{float(metrics['bias_deg']):.3f} |"
    )


def resolve_image_path(sample: Mapping[str, Any]) -> Path:
    raw_path = Path(str(sample.get("image_path", "")))
    if raw_path.is_file():
        return raw_path
    if raw_path and not raw_path.is_absolute():
        project_path = PROJECT_ROOT / raw_path
        if project_path.is_file():
            return project_path

    file_name = str(sample["file_name"])
    split = str(sample.get("split", "")).lower()
    candidates: list[Path] = []
    if split == "test":
        candidates.append(PROJECT_ROOT / "raw" / "images" / "test" / "Spinal-AI2024-subset5" / file_name)
    for subset_index in range(1, 5):
        candidates.append(PROJECT_ROOT / "raw" / "images" / "train" / f"Spinal-AI2024-subset{subset_index}" / file_name)
    candidates.append(PROJECT_ROOT / "raw" / "images" / "test" / "Spinal-AI2024-subset5" / file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Imagem nao encontrada para {file_name}")


def extract_annotation_points(sample: Mapping[str, Any]) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for vertebra in sample.get("vertebrae", []):
        vertebra_points = vertebra.get("points", {})
        if isinstance(vertebra_points, Mapping):
            iterable = vertebra_points.values()
        else:
            iterable = vertebra_points
        for point in iterable:
            if isinstance(point, Sequence) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    if not points:
        raise ValueError(f"Sample sem pontos vertebrais: {sample.get('file_name')}")
    return np.asarray(points, dtype=np.float32)


def calculate_roi(sample: Mapping[str, Any], roi_padding: int) -> dict[str, int]:
    points = extract_annotation_points(sample)
    width = int(sample["width"])
    height = int(sample["height"])
    xmin = max(0, int(float(np.min(points[:, 0])) - roi_padding))
    ymin = max(0, int(float(np.min(points[:, 1])) - roi_padding))
    xmax = min(width, int(float(np.max(points[:, 0])) + roi_padding))
    ymax = min(height, int(float(np.max(points[:, 1])) + roi_padding))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"ROI invalida para {sample.get('file_name')}: {(xmin, ymin, xmax, ymax)}")
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def decode_image(path: Path) -> tf.Tensor:
    encoded = tf.io.read_file(str(path))
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return tf.image.decode_jpeg(encoded, channels=1)
    if suffix == ".png":
        return tf.image.decode_png(encoded, channels=1)
    return tf.image.decode_image(encoded, channels=1, expand_animations=False)


def preprocess_sample(
    sample: Mapping[str, Any],
    *,
    roi_padding: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path = resolve_image_path(sample)
    roi = calculate_roi(sample, roi_padding)
    image = decode_image(image_path)
    crop = image[roi["ymin"] : roi["ymax"], roi["xmin"] : roi["xmax"], :]
    crop_height = int(roi["ymax"] - roi["ymin"])
    crop_width = int(roi["xmax"] - roi["xmin"])

    scale = min(IMAGE_SIZE / crop_width, IMAGE_SIZE / crop_height)
    normalized_width = max(1, int(crop_width * scale))
    normalized_height = max(1, int(crop_height * scale))
    resized = tf.image.resize(
        tf.cast(crop, tf.float32),
        [normalized_height, normalized_width],
        method="bilinear",
    )
    pad_x = (IMAGE_SIZE - normalized_width) // 2
    pad_y = (IMAGE_SIZE - normalized_height) // 2
    padded = tf.pad(
        resized,
        [
            [pad_y, IMAGE_SIZE - normalized_height - pad_y],
            [pad_x, IMAGE_SIZE - normalized_width - pad_x],
            [0, 0],
        ],
    )
    normalized = tf.cast(padded, tf.float32) / 255.0
    display = tf.cast(tf.clip_by_value(tf.round(padded), 0.0, 255.0), tf.uint8)
    meta = {
        "image_path": str(image_path),
        **roi,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "normalized_width": normalized_width,
        "normalized_height": normalized_height,
        "scale": float(scale),
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return normalized.numpy().astype(np.float32), display.numpy()[:, :, 0], meta


def make_odd(value: int) -> int:
    value = int(value)
    if value % 2 == 0:
        value -= 1
    return max(value, 3)


def extract_centerline_by_blocks(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if binary_mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Shape invalido na mascara prevista: {binary_mask.shape}")

    x_points: list[float] = []
    y_points: list[float] = []
    for y_start in range(0, IMAGE_SIZE, BLOCK_HEIGHT):
        y_end = min(y_start + BLOCK_HEIGHT, IMAGE_SIZE)
        block = binary_mask[y_start:y_end, :]
        ys, xs = np.where(block > 0)
        if len(xs) >= MIN_PIXELS_PER_BLOCK:
            x_points.append(float(np.mean(xs)))
            y_points.append(float(y_start + np.mean(ys)))
    return np.asarray(x_points, dtype=np.float32), np.asarray(y_points, dtype=np.float32)


def smooth_centerline(x_points: np.ndarray, y_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x_points) < 5:
        return x_points, y_points
    window = min(CENTERLINE_SMOOTH_WINDOW, len(x_points))
    if window % 2 == 0:
        window -= 1
    if window <= CENTERLINE_SMOOTH_POLYORDER:
        return x_points, y_points
    return (
        savgol_filter(x_points, window_length=window, polyorder=CENTERLINE_SMOOTH_POLYORDER).astype(np.float32),
        y_points,
    )


def smooth_centerline_for_cobb(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) < 7:
        return xs, ys
    window = make_odd(min(COBB_SMOOTH_WINDOW, len(xs)))
    if window <= COBB_SMOOTH_POLYORDER:
        return xs, ys
    return savgol_filter(xs, window_length=window, polyorder=COBB_SMOOTH_POLYORDER).astype(np.float32), ys


def trim_centerline(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(xs)
    start = int(count * TRIM_RATIO)
    end = int(count * (1.0 - TRIM_RATIO))
    if end <= start:
        return xs, ys
    return xs[start:end], ys[start:end]


def calculate_angles_by_arc_length(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    ds = np.sqrt(dx**2 + dy**2)
    ds[ds == 0] = 1.0e-6
    dx_ds = dx / ds
    dy_ds = dy / ds
    angles = np.degrees(np.arctan2(dx_ds, dy_ds))
    angles = ((angles + 90.0) % 180.0) - 90.0

    if len(angles) >= 7:
        window = make_odd(min(ANGLE_SMOOTH_WINDOW, len(angles)))
        if window > ANGLE_SMOOTH_POLYORDER:
            angles = savgol_filter(angles, window_length=window, polyorder=ANGLE_SMOOTH_POLYORDER)
    return np.asarray(angles, dtype=np.float32)


def angular_difference(angle1: float, angle2: float) -> float:
    diff = abs(float(angle2) - float(angle1))
    if diff > 90.0:
        diff = 180.0 - diff
    return abs(diff)


def get_local_angle_stats(angles: np.ndarray, idx: int, window: int = LOCAL_ANGLE_WINDOW) -> tuple[float, float]:
    half = window // 2
    start = max(0, idx - half)
    end = min(len(angles), idx + half + 1)
    local_values = angles[start:end]
    if len(local_values) == 0:
        return float(angles[idx]), 0.0
    return float(np.mean(local_values)), float(np.std(local_values))


def get_effective_angle(angles: np.ndarray, idx: int) -> tuple[float, float, float]:
    raw_angle = float(angles[idx])
    local_mean, local_std = get_local_angle_stats(angles, idx)
    if local_std > LOCAL_STD_LIMIT:
        return 0.75 * raw_angle + 0.25 * local_mean, local_mean, local_std
    return raw_angle, local_mean, local_std


def get_region_indices(ys: np.ndarray, region: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    if len(ys) == 0:
        return np.array([], dtype=np.int32), float(region["start"]), float(region["end"])
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    y_span = y_max - y_min
    if y_span <= 0:
        return np.array([], dtype=np.int32), float(region["start"]), float(region["end"])

    normalized_y = (ys - y_min) / y_span
    expansion = 0.0
    last_indices = np.array([], dtype=np.int32)
    start = float(region["start"])
    end = float(region["end"])
    while expansion <= MAX_REGION_EXPANSION + 1.0e-9:
        start = max(0.0, float(region["start"]) - expansion)
        end = min(1.0, float(region["end"]) + expansion)
        indices = np.where((normalized_y >= start) & (normalized_y <= end))[0]
        last_indices = indices.astype(np.int32)
        if len(indices) >= MIN_REGION_POINTS:
            return last_indices, start, end
        expansion += REGION_EXPANSION_STEP
    return last_indices, start, end


def reorder_extreme_pair(
    i: int,
    j: int,
    best_data: Mapping[str, float],
    ys: np.ndarray,
) -> tuple[int, int, dict[str, float]]:
    if ys[i] <= ys[j]:
        return i, j, dict(best_data)
    return j, i, {
        "angle_i_eff": best_data["angle_j_eff"],
        "angle_j_eff": best_data["angle_i_eff"],
        "angle_i_raw": best_data["angle_j_raw"],
        "angle_j_raw": best_data["angle_i_raw"],
        "angle_i_local": best_data["angle_j_local"],
        "angle_j_local": best_data["angle_i_local"],
        "std_i": best_data["std_j"],
        "std_j": best_data["std_i"],
        "cobb_eff": best_data["cobb_eff"],
        "cobb_raw": best_data["cobb_raw"],
        "score": best_data["score"],
        "y_distance": best_data["y_distance"],
    }


def build_extremes_result(i: int, j: int, best_data: Mapping[str, float], ys: np.ndarray) -> dict[str, Any]:
    top_idx, bottom_idx, data = reorder_extreme_pair(i, j, best_data, ys)
    return {
        "idx_top": int(top_idx),
        "idx_bottom": int(bottom_idx),
        "angle_top": float(data["angle_i_eff"]),
        "angle_bottom": float(data["angle_j_eff"]),
        "angle_top_raw": float(data["angle_i_raw"]),
        "angle_bottom_raw": float(data["angle_j_raw"]),
        "angle_top_local": float(data["angle_i_local"]),
        "angle_bottom_local": float(data["angle_j_local"]),
        "angle_std_top": float(data["std_i"]),
        "angle_std_bottom": float(data["std_j"]),
        "cobb": float(data["cobb_eff"]),
        "cobb_raw": float(data["cobb_raw"]),
        "selection_score": float(data["score"]),
        "y_distance": float(data["y_distance"]),
    }


def calculate_pair_data(angles: np.ndarray, ys: np.ndarray, i: int, j: int) -> dict[str, float]:
    y_distance = abs(float(ys[j]) - float(ys[i]))
    angle_i_eff, angle_i_local, std_i = get_effective_angle(angles, i)
    angle_j_eff, angle_j_local, std_j = get_effective_angle(angles, j)
    cobb_eff = angular_difference(angle_i_eff, angle_j_eff)
    cobb_raw = angular_difference(float(angles[i]), float(angles[j]))
    std_penalty = LOCAL_STD_PENALTY * (std_i + std_j)
    distance_penalty = 0.0
    if y_distance < CLOSE_DISTANCE_LIMIT:
        distance_penalty = CLOSE_DISTANCE_PENALTY * (CLOSE_DISTANCE_LIMIT - y_distance)
    return {
        "angle_i_eff": angle_i_eff,
        "angle_j_eff": angle_j_eff,
        "angle_i_raw": float(angles[i]),
        "angle_j_raw": float(angles[j]),
        "angle_i_local": angle_i_local,
        "angle_j_local": angle_j_local,
        "std_i": std_i,
        "std_j": std_j,
        "cobb_eff": cobb_eff,
        "cobb_raw": cobb_raw,
        "score": cobb_eff - std_penalty - distance_penalty,
        "y_distance": y_distance,
    }


def find_cobb_extremes(
    ys: np.ndarray,
    angles: np.ndarray,
    *,
    candidate_indices: np.ndarray | None = None,
    min_point_distance: int = MIN_POINT_DISTANCE_GLOBAL,
) -> dict[str, Any] | None:
    if candidate_indices is None:
        candidate_indices = np.arange(len(angles))
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    if len(candidate_indices) < MIN_REGION_POINTS:
        return None

    angle_mask = np.abs(angles[candidate_indices]) <= MAX_ABS_ANGLE
    valid_indices = candidate_indices[angle_mask]
    if len(valid_indices) < MIN_REGION_POINTS:
        valid_indices = candidate_indices

    valid_angles = angles[valid_indices]
    min_target = np.percentile(valid_angles, LOW_PERCENTILE)
    max_target = np.percentile(valid_angles, HIGH_PERCENTILE)
    min_candidates = valid_indices[np.argsort(np.abs(angles[valid_indices] - min_target))]
    max_candidates = valid_indices[np.argsort(np.abs(angles[valid_indices] - max_target))]

    best_pair: tuple[int, int] | None = None
    best_score = -1.0e9
    best_data: dict[str, float] | None = None
    for i in min_candidates[:CANDIDATE_LIMIT]:
        for j in max_candidates[:CANDIDATE_LIMIT]:
            if int(i) == int(j):
                continue
            if abs(float(ys[j]) - float(ys[i])) < min_point_distance:
                continue
            pair_data = calculate_pair_data(angles, ys, int(i), int(j))
            if pair_data["score"] > best_score:
                best_score = pair_data["score"]
                best_pair = (int(i), int(j))
                best_data = pair_data

    if best_pair is None or best_data is None:
        return None
    return build_extremes_result(best_pair[0], best_pair[1], best_data, ys)


def find_cobb_extremes_near_relative_cobb(
    ys: np.ndarray,
    angles: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    min_point_distance: int,
    target_scale: float,
) -> dict[str, Any] | None:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    if len(candidate_indices) < MIN_REGION_POINTS:
        return None

    pairs: list[tuple[int, int, dict[str, float]]] = []
    for pos_i, i in enumerate(candidate_indices):
        for j in candidate_indices[pos_i + 1 :]:
            if abs(float(ys[j]) - float(ys[i])) < min_point_distance:
                continue
            pairs.append((int(i), int(j), calculate_pair_data(angles, ys, int(i), int(j))))
    if not pairs:
        return None

    max_cobb = max(pair_data["cobb_eff"] for _, _, pair_data in pairs)
    target_cobb = max_cobb * target_scale
    i, j, best_data = min(
        pairs,
        key=lambda item: (abs(item[2]["cobb_eff"] - target_cobb), -item[2]["score"]),
    )
    best_data = dict(best_data)
    best_data["score"] -= abs(best_data["cobb_eff"] - target_cobb)
    return build_extremes_result(i, j, best_data, ys)


def has_curvature_support(
    ys: np.ndarray,
    angles: np.ndarray,
    *,
    start: float,
    end: float,
) -> bool:
    if len(ys) == 0:
        return False
    y_span = float(np.max(ys) - np.min(ys))
    if y_span <= 0:
        return False
    normalized_y = (ys - np.min(ys)) / y_span
    peak_indices, _ = find_peaks(angles, prominence=PT_CURVATURE_PROMINENCE, distance=8)
    trough_indices, _ = find_peaks(-angles, prominence=PT_CURVATURE_PROMINENCE, distance=8)
    landmark_indices = np.concatenate([peak_indices, trough_indices])
    if len(landmark_indices) == 0:
        return False
    landmark_positions = normalized_y[landmark_indices]
    return bool(np.any((landmark_positions >= start) & (landmark_positions <= end)))


def calculate_lower_lateral_amplitude(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3 or len(ys) < 3:
        return 0.0
    y_span = float(np.max(ys) - np.min(ys))
    if y_span <= 0:
        return 0.0
    normalized_y = (ys - np.min(ys)) / y_span
    lower_indices = np.where(normalized_y >= TLL_LOW_AMPLITUDE_START)[0]
    if len(lower_indices) == 0:
        return 0.0
    trend = np.polyval(np.polyfit(ys, xs, 1), ys)
    residuals = xs - trend
    lower_residuals = residuals[lower_indices]
    return float(np.max(lower_residuals) - np.min(lower_residuals))


def calculate_regional_cobbs(xs: np.ndarray, ys: np.ndarray, angles: np.ndarray) -> dict[str, dict[str, Any]]:
    regional: dict[str, dict[str, Any]] = {}
    pt_has_support = has_curvature_support(
        ys,
        angles,
        start=PT_CURVATURE_SUPPORT_START,
        end=PT_CURVATURE_SUPPORT_END,
    )
    tll_has_support = has_curvature_support(
        ys,
        angles,
        start=TLL_CURVATURE_SUPPORT_START,
        end=TLL_CURVATURE_SUPPORT_END,
    )
    tll_lower_lateral_amplitude = calculate_lower_lateral_amplitude(xs, ys)

    for region in REGIONS:
        indices, start, end = get_region_indices(ys, region)
        extremes = find_cobb_extremes(
            ys,
            angles,
            candidate_indices=indices,
            min_point_distance=MIN_POINT_DISTANCE_REGION,
        )
        status = "ok" if extremes is not None else "no_valid_pair"

        if region["name"] == "PT" and extremes is not None and not pt_has_support:
            secondary = find_cobb_extremes_near_relative_cobb(
                ys,
                angles,
                indices,
                min_point_distance=MIN_POINT_DISTANCE_REGION,
                target_scale=PT_UNSUPPORTED_TARGET_SCALE,
            )
            if secondary is not None:
                extremes = secondary
                status = "ok_secondary_no_upper_curvature"

        if (
            region["name"] == "TLL"
            and extremes is not None
            and extremes["cobb"] >= TLL_SECONDARY_MIN_COBB
            and not tll_has_support
            and tll_lower_lateral_amplitude <= TLL_LOW_AMPLITUDE_LIMIT
        ):
            secondary = find_cobb_extremes_near_relative_cobb(
                ys,
                angles,
                indices,
                min_point_distance=MIN_POINT_DISTANCE_REGION,
                target_scale=TLL_UNSUPPORTED_TARGET_SCALE,
            )
            if secondary is not None:
                extremes = secondary
                status = "ok_secondary_low_lower_curvature"

        regional[str(region["name"])] = {
            "indices": indices,
            "start": start,
            "end": end,
            "extremes": extremes,
            "status": status,
        }
    return regional


def centerline_to_cobb(x_points: np.ndarray, y_points: np.ndarray) -> dict[str, Any]:
    if len(x_points) < MIN_REGION_POINTS:
        return {
            "status": "insufficient_centerline_points",
            "predictions": {name: np.nan for name in ANGLE_NAMES},
            "regional": {},
            "xs_trim": np.asarray([], dtype=np.float32),
            "ys_trim": np.asarray([], dtype=np.float32),
        }

    xs_smooth, ys_smooth = smooth_centerline_for_cobb(x_points, y_points)
    xs_trim, ys_trim = trim_centerline(xs_smooth, ys_smooth)
    if len(xs_trim) < MIN_REGION_POINTS:
        return {
            "status": "insufficient_trimmed_points",
            "predictions": {name: np.nan for name in ANGLE_NAMES},
            "regional": {},
            "xs_trim": xs_trim,
            "ys_trim": ys_trim,
        }

    angles = calculate_angles_by_arc_length(xs_trim, ys_trim)
    if len(angles) != len(xs_trim):
        raise ValueError("Shapes inconsistentes entre centerline e angulos.")
    regional = calculate_regional_cobbs(xs_trim, ys_trim, angles)
    predictions = {}
    valid_region_count = 0
    for name in ANGLE_NAMES:
        region = regional.get(name, {})
        extremes = region.get("extremes") if isinstance(region, Mapping) else None
        cobb = finite_float(extremes.get("cobb")) if isinstance(extremes, Mapping) else np.nan
        predictions[name] = cobb
        if np.isfinite(cobb):
            valid_region_count += 1
    return {
        "status": "ok" if valid_region_count > 0 else "no_valid_regional_cobb",
        "predictions": predictions,
        "regional": regional,
        "xs_trim": xs_trim,
        "ys_trim": ys_trim,
        "angles": angles,
    }


def gt_angles(sample: Mapping[str, Any]) -> dict[str, float]:
    cobb = sample.get("cobb_angles", {})
    return {
        "PT": finite_float(cobb.get("PT")),
        "MT": finite_float(cobb.get("MT")),
        "TLL": finite_float(cobb.get("TLL", cobb.get("TL_L"))),
    }


def finite_max(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else float("nan")


def mark_point(rgb: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int = 4) -> None:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    for yy in range(max(0, yi - radius), min(IMAGE_SIZE, yi + radius + 1)):
        for xx in range(max(0, xi - radius), min(IMAGE_SIZE, xi + radius + 1)):
            if (yy - yi) ** 2 + (xx - xi) ** 2 <= radius**2:
                rgb[yy, xx, :] = color


def save_overlay(
    path: Path,
    display_image: np.ndarray,
    binary_mask: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    cobb_result: Mapping[str, Any],
) -> None:
    rgb = np.repeat(display_image[:, :, None], 3, axis=2).astype(np.uint8)
    mask = binary_mask > 0
    rgb[mask, 0] = np.maximum(rgb[mask, 0], 180)
    rgb[mask, 1] = (0.65 * rgb[mask, 1]).astype(np.uint8)
    rgb[mask, 2] = (0.65 * rgb[mask, 2]).astype(np.uint8)

    for x, y in zip(x_points, y_points):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < IMAGE_SIZE and 0 <= yi < IMAGE_SIZE:
            rgb[yi, xi, :] = (0, 255, 255)

    colors = {
        "PT": ((255, 80, 80), (255, 170, 40)),
        "MT": ((80, 140, 255), (40, 220, 255)),
        "TLL": ((180, 110, 255), (80, 255, 120)),
    }
    xs_trim = np.asarray(cobb_result.get("xs_trim", []), dtype=np.float32)
    ys_trim = np.asarray(cobb_result.get("ys_trim", []), dtype=np.float32)
    for name, region in cobb_result.get("regional", {}).items():
        extremes = region.get("extremes")
        if extremes is None or len(xs_trim) == 0:
            continue
        top_color, bottom_color = colors.get(name, ((255, 255, 0), (255, 255, 0)))
        top_idx = int(extremes["idx_top"])
        bottom_idx = int(extremes["idx_bottom"])
        if 0 <= top_idx < len(xs_trim):
            mark_point(rgb, xs_trim[top_idx], ys_trim[top_idx], top_color)
        if 0 <= bottom_idx < len(xs_trim):
            mark_point(rgb, xs_trim[bottom_idx], ys_trim[bottom_idx], bottom_color)

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = tf.io.encode_png(tf.convert_to_tensor(rgb, dtype=tf.uint8))
    tf.io.write_file(str(path), encoded)


def build_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name in ANGLE_NAMES:
        metrics[name] = metric_bundle(
            [finite_float(row[f"pred_{name.lower()}"]) for row in rows],
            [finite_float(row[f"gt_{name.lower()}"]) for row in rows],
        )
    metrics["max_cobb"] = metric_bundle(
        [finite_float(row["pred_max_cobb"]) for row in rows],
        [finite_float(row["gt_max_cobb"]) for row in rows],
    )
    agg_pred: list[float] = []
    agg_gt: list[float] = []
    for row in rows:
        for name in ANGLE_NAMES:
            agg_pred.append(finite_float(row[f"pred_{name.lower()}"]))
            agg_gt.append(finite_float(row[f"gt_{name.lower()}"]))
    metrics["agg3"] = metric_bundle(agg_pred, agg_gt)
    return metrics


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    selected_count: int,
    processed_count: int,
    metrics: Mapping[str, Mapping[str, Any]],
    error_count: int,
) -> None:
    lines = [
        "# Colleague Centerline Model Evaluation v1",
        "",
        "## Setup",
        "",
        f"- annotations: `{resolve_project_path(args.annotations)}`",
        f"- colleague model: `{resolve_project_path(args.colleague_model)}`",
        f"- selected images: `{selected_count}`",
        f"- processed rows: `{processed_count}`",
        f"- errors: `{error_count}`",
        f"- threshold: `{args.threshold}`",
        f"- roi padding: `{args.roi_padding}`",
        "",
        "## Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("max_cobb", "PT", "MT", "TLL", "agg3"):
        lines.append(metric_row(metrics[label], label))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- predictions CSV: `{path.parent / 'colleague_centerline_predictions.csv'}`",
            f"- metrics JSON: `{path.parent / 'colleague_centerline_metrics.json'}`",
            f"- overlays: `{path.parent / 'overlays'}`",
            "",
            "## Notes",
            "",
            "- This evaluates the colleague U-Net/centerline path in isolation.",
            "- It does not tune fusion weights and does not alter the locked Phase 9 baseline.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_batch(
    *,
    model: tf.keras.Model,
    batch_items: Sequence[tuple[int, Mapping[str, Any], np.ndarray, np.ndarray, dict[str, Any]]],
    threshold: float,
    output_dir: Path,
    save_overlays_enabled: bool,
    overlay_limit: int,
    overlay_count: int,
) -> tuple[list[dict[str, Any]], int]:
    inputs = np.stack([item[2] for item in batch_items], axis=0)
    predictions = model.predict(inputs, verbose=0)[:, :, :, 0]
    rows: list[dict[str, Any]] = []

    for pred_index, (global_index, sample, _, display_image, meta) in enumerate(batch_items):
        prediction = np.asarray(predictions[pred_index], dtype=np.float32)
        binary_mask = (prediction > threshold).astype(np.uint8)
        x_points, y_points = extract_centerline_by_blocks(binary_mask)
        x_smooth, y_smooth = smooth_centerline(x_points, y_points)
        cobb_result = centerline_to_cobb(x_smooth, y_smooth)
        pred_angles = cobb_result["predictions"]
        gt = gt_angles(sample)
        pred_max = finite_max([pred_angles[name] for name in ANGLE_NAMES])
        gt_max = finite_max([gt[name] for name in ANGLE_NAMES])

        regional = cobb_result.get("regional", {})
        row: dict[str, Any] = {
            "index": global_index,
            "file_name": sample.get("file_name", ""),
            "image_path": meta["image_path"],
            "status": cobb_result["status"],
            "centerline_points": int(len(x_smooth)),
            "mask_pixels_gt_threshold": int(np.sum(binary_mask > 0)),
            "prediction_min": float(np.min(prediction)),
            "prediction_max": float(np.max(prediction)),
            "prediction_mean": float(np.mean(prediction)),
            "gt_pt": gt["PT"],
            "gt_mt": gt["MT"],
            "gt_tll": gt["TLL"],
            "pred_pt": pred_angles["PT"],
            "pred_mt": pred_angles["MT"],
            "pred_tll": pred_angles["TLL"],
            "gt_max_cobb": gt_max,
            "pred_max_cobb": pred_max,
            "abs_error_max_cobb": abs(pred_max - gt_max) if np.isfinite(pred_max) and np.isfinite(gt_max) else np.nan,
            **meta,
        }
        for name in ANGLE_NAMES:
            region = regional.get(name, {})
            extremes = region.get("extremes") or {}
            prefix = name.lower()
            row[f"{prefix}_status"] = region.get("status", "")
            row[f"{prefix}_idx_top"] = extremes.get("idx_top", "")
            row[f"{prefix}_idx_bottom"] = extremes.get("idx_bottom", "")
            row[f"{prefix}_selection_score"] = extremes.get("selection_score", "")
        rows.append(row)

        if save_overlays_enabled and overlay_count < overlay_limit:
            overlay_name = f"{global_index:04d}_{Path(str(sample.get('file_name', 'image'))).stem}_centerline.png"
            save_overlay(
                output_dir / "overlays" / overlay_name,
                display_image,
                binary_mask,
                x_smooth,
                y_smooth,
                cobb_result,
            )
            overlay_count += 1
    return rows, overlay_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o U-Net centerline do colega no dataset limpo local."
    )
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--colleague-model", default=str(DEFAULT_COLLEAGUE_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Numero de imagens a avaliar. Use 0 para avaliar todas a partir de start-index.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--roi-padding", type=int, default=DEFAULT_ROI_PADDING)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_path = resolve_project_path(args.annotations)
    model_path = resolve_project_path(args.colleague_model)
    output_dir = resolve_project_path(args.output_dir)
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser > 0")
    if args.overlay_limit < 0:
        raise ValueError("--overlay-limit deve ser >= 0")
    require_file(model_path)

    samples = load_annotations(annotations_path)
    selected_samples = list(select_window(samples, args.start_index, args.num_images))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== COLLEAGUE CENTERLINE MODEL EVAL V1 =====")
    print(f"Annotations: {annotations_path}")
    print(f"Model: {model_path}")
    print(f"Selected images: {len(selected_samples)}")
    print(f"Output: {output_dir}")

    model = tf.keras.models.load_model(model_path, compile=False)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[tuple[int, Mapping[str, Any], np.ndarray, np.ndarray, dict[str, Any]]] = []
    overlay_count = 0

    for offset, sample in enumerate(selected_samples):
        global_index = args.start_index + offset
        try:
            normalized, display_image, meta = preprocess_sample(sample, roi_padding=args.roi_padding)
            pending.append((global_index, sample, normalized, display_image, meta))
        except Exception as exc:
            errors.append({
                "index": global_index,
                "file_name": sample.get("file_name", ""),
                "error": str(exc),
            })
            continue

        if len(pending) >= args.batch_size:
            batch_rows, overlay_count = process_batch(
                model=model,
                batch_items=pending,
                threshold=args.threshold,
                output_dir=output_dir,
                save_overlays_enabled=args.save_overlays,
                overlay_limit=args.overlay_limit,
                overlay_count=overlay_count,
            )
            rows.extend(batch_rows)
            pending = []
            print(f"Processed {len(rows) + len(errors)}/{len(selected_samples)}")

    if pending:
        batch_rows, overlay_count = process_batch(
            model=model,
            batch_items=pending,
            threshold=args.threshold,
            output_dir=output_dir,
            save_overlays_enabled=args.save_overlays,
            overlay_limit=args.overlay_limit,
            overlay_count=overlay_count,
        )
        rows.extend(batch_rows)
        print(f"Processed {len(rows) + len(errors)}/{len(selected_samples)}")

    metrics = build_metrics(rows)
    metrics_payload = {
        "phase": "colleague_centerline_model_v1",
        "annotations": str(annotations_path),
        "colleague_model": str(model_path),
        "selected_images": len(selected_samples),
        "processed_rows": len(rows),
        "errors": errors,
        "threshold": float(args.threshold),
        "roi_padding": int(args.roi_padding),
        "metrics": {
            key: {metric_key: jsonable_metric_value(metric_value) for metric_key, metric_value in value.items()}
            for key, value in metrics.items()
        },
    }

    prediction_path = output_dir / "colleague_centerline_predictions.csv"
    metrics_path = output_dir / "colleague_centerline_metrics.json"
    report_path = output_dir / "colleague_centerline_report.md"
    errors_path = output_dir / "colleague_centerline_errors.csv"

    fieldnames = [
        "index",
        "file_name",
        "image_path",
        "status",
        "centerline_points",
        "mask_pixels_gt_threshold",
        "prediction_min",
        "prediction_max",
        "prediction_mean",
        "gt_pt",
        "gt_mt",
        "gt_tll",
        "pred_pt",
        "pred_mt",
        "pred_tll",
        "gt_max_cobb",
        "pred_max_cobb",
        "abs_error_max_cobb",
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
        "pt_status",
        "pt_idx_top",
        "pt_idx_bottom",
        "pt_selection_score",
        "mt_status",
        "mt_idx_top",
        "mt_idx_bottom",
        "mt_selection_score",
        "tll_status",
        "tll_idx_top",
        "tll_idx_bottom",
        "tll_selection_score",
    ]
    write_csv(prediction_path, rows, fieldnames)
    if errors:
        write_csv(errors_path, errors, ["index", "file_name", "error"])
    write_json(metrics_path, metrics_payload)
    write_report(
        report_path,
        args=args,
        selected_count=len(selected_samples),
        processed_count=len(rows),
        metrics=metrics,
        error_count=len(errors),
    )

    print("\n===== METRICS =====")
    for label in ("max_cobb", "PT", "MT", "TLL", "agg3"):
        row = metrics[label]
        if int(row.get("num_images", 0)) == 0:
            print(f"{label}: sem predicoes validas")
            continue
        print(
            f"{label}: MAE={row['mae_deg']:.3f}, "
            f"SMAPE={row['paper_smape_pct']:.4f}%, "
            f"within5={row['within_5deg_rate']:.4f}, "
            f"failures_gt5={row['failures_gt5']}"
        )
    print(f"\nPredictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    if errors:
        print(f"Errors: {errors_path}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
