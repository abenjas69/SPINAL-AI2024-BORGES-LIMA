"""Inference entrypoint for the Spinal-AI scoliosis model.

This module exposes the current best production-oriented pipeline:

1. download/read a radiograph;
2. run Phase 5 vertebra/quadrilateral detection;
3. apply the anatomical sequence post-processing profile;
4. compute the geometric max Cobb angle;
5. apply the Phase 9 residual MLP calibrator;
6. return an integration-friendly JSON payload.

The labels V01..Vn are positional labels ordered top-to-bottom. They are not
certified anatomical labels such as T1, T2 or L1.
"""

from __future__ import annotations

import argparse
import base64
import io
import importlib.util
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed in the active Python environment. "
        "Activate the project .venv or install the inference requirements."
    ) from exc

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_PHASE5_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_PHASE9_MODEL = MODELS_DIR / "phase9_cobb_residual_mlp_v2.keras"
DEFAULT_PHASE9_SCALER = MODELS_DIR / "phase9_cobb_residual_mlp_v2_scaler.npz"
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_ALPHA = 0.55
DEFAULT_MAX_CORRECTION_DEG = 10.0
MODEL_VERSION = "phase5_radius_hardmining_v1+phase9_cobb_residual_mlp_v2"
POINT_NAMES = ("upperLeft", "upperRight", "lowerLeft", "lowerRight")


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def finite_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def rounded(value: float | np.floating | None, digits: int = 4) -> float | None:
    value = finite_float(value)
    if value is None:
        return None
    return round(value, digits)


def severity_from_cobb(angle_deg: float) -> str:
    angle = float(angle_deg)
    if angle < 10.0:
        return "NAO_SIGNIFICATIVA"
    if angle < 25.0:
        return "LEVE"
    if angle < 40.0:
        return "MODERADA"
    return "GRAVE"


def image_bytes_from_url(url: str, timeout_seconds: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Spinal-AI2024-Inference/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read()


def load_image_tensor(image_bytes: bytes) -> tuple[tf.Tensor, Any, int, int]:
    decoded = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    decoded = tf.ensure_shape(decoded, [None, None, 3])
    shape = tf.shape(decoded)
    height = int(shape[0].numpy())
    width = int(shape[1].numpy())
    image_float = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize(image_float, (512, 512), method="bilinear")

    pil_image = None
    if Image is not None:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return tf.expand_dims(tf.cast(resized, tf.float32), axis=0), pil_image, int(width), int(height)


def point_to_pixel(point: np.ndarray, width: int, height: int) -> dict[str, int]:
    return {
        "x": int(round(float(np.clip(point[0], 0.0, 1.0)) * width)),
        "y": int(round(float(np.clip(point[1], 0.0, 1.0)) * height)),
    }


def normalized_points_to_pixel_dict(row: np.ndarray, width: int, height: int) -> dict[str, dict[str, int]]:
    points = np.asarray(row, dtype=np.float32).reshape(4, 2)
    return {
        name: point_to_pixel(points[index], width, height)
        for index, name in enumerate(POINT_NAMES)
    }


def bbox_from_points(row: np.ndarray, width: int, height: int) -> dict[str, int]:
    points = np.asarray(row, dtype=np.float32).reshape(4, 2)
    x_values = np.clip(points[:, 0], 0.0, 1.0) * width
    y_values = np.clip(points[:, 1], 0.0, 1.0) * height
    x_min = int(round(float(np.min(x_values))))
    y_min = int(round(float(np.min(y_values))))
    x_max = int(round(float(np.max(x_values))))
    y_max = int(round(float(np.max(y_values))))
    return {
        "x": x_min,
        "y": y_min,
        "width": max(0, x_max - x_min),
        "height": max(0, y_max - y_min),
    }


def line_box_segment(point_a: tuple[float, float], point_b: tuple[float, float], width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1 = point_a
    x2, y2 = point_b
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1.0e-6 and abs(dy) < 1.0e-6:
        return point_a, point_b

    candidates: list[tuple[float, float]] = []
    for x in (0.0, float(width - 1)):
        if abs(dx) > 1.0e-6:
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0.0 <= y <= float(height - 1):
                candidates.append((x, y))
    for y in (0.0, float(height - 1)):
        if abs(dy) > 1.0e-6:
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0.0 <= x <= float(width - 1):
                candidates.append((x, y))

    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        if all(math.hypot(candidate[0] - item[0], candidate[1] - item[1]) > 1.0 for item in unique):
            unique.append(candidate)
    if len(unique) < 2:
        scale = max(width, height)
        return (x1 - dx * scale, y1 - dy * scale), (x1 + dx * scale, y1 + dy * scale)

    best_pair = (unique[0], unique[1])
    best_distance = -1.0
    for first_index in range(len(unique) - 1):
        for second_index in range(first_index + 1, len(unique)):
            distance = math.hypot(
                unique[first_index][0] - unique[second_index][0],
                unique[first_index][1] - unique[second_index][1],
            )
            if distance > best_distance:
                best_distance = distance
                best_pair = (unique[first_index], unique[second_index])
    return best_pair


@dataclass
class Phase9Prediction:
    geom_cobb: float
    calibrated_cobb: float
    predicted_residual: float
    applied_correction: float
    geom_pair: Mapping[str, Any]


class SpinalAIAnalyzer:
    """Loads the current Spinal-AI pipeline once and reuses it per image."""

    def __init__(
        self,
        *,
        phase5_model_path: str | Path = DEFAULT_PHASE5_MODEL,
        phase9_model_path: str | Path = DEFAULT_PHASE9_MODEL,
        phase9_scaler_path: str | Path = DEFAULT_PHASE9_SCALER,
        profile: str = DEFAULT_PROFILE,
        alpha: float = DEFAULT_ALPHA,
        max_correction_deg: float = DEFAULT_MAX_CORRECTION_DEG,
    ) -> None:
        self.phase5_model_path = resolve_project_path(phase5_model_path)
        self.phase9_model_path = resolve_project_path(phase9_model_path)
        self.phase9_scaler_path = resolve_project_path(phase9_scaler_path)
        self.profile = str(profile)
        self.alpha = float(alpha)
        self.max_correction_deg = float(max_correction_deg)

        self.phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "deployment_phase5_eval")
        self.phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "deployment_phase5_sequence")
        self.postprocess = import_script("19_postprocess_sequence_v1.py", "deployment_phase5_postprocess")
        self.phase9_v1 = import_script("32_eval_phase9_final_cobb_v1.py", "deployment_phase9_v1")
        self.pair_reranker = import_script("57_train_phase9_endpoint_pair_reranker_v1.py", "deployment_pair_reranker")
        self.residual_calibrator = import_script("59_train_phase9_cobb_residual_calibrator_v1.py", "deployment_residual_calibrator")

        if self.profile not in self.phase5_sequence.PROFILE_CONFIGS:
            raise ValueError(f"Unknown Phase 5 profile: {self.profile}")
        if not self.phase5_model_path.is_file():
            raise FileNotFoundError(f"Phase 5 model not found: {self.phase5_model_path}")
        if not self.phase9_model_path.is_file():
            raise FileNotFoundError(f"Phase 9 MLP model not found: {self.phase9_model_path}")
        if not self.phase9_scaler_path.is_file():
            raise FileNotFoundError(f"Phase 9 scaler not found: {self.phase9_scaler_path}")

        print(f"Loading Phase 5 model: {self.phase5_model_path}")
        self.phase5_model = self.phase5_eval.load_spatial_offset_model_for_eval(self.phase5_model_path)
        print(f"Loading Phase 9 residual MLP: {self.phase9_model_path}")
        self.phase9_model = tf.keras.models.load_model(self.phase9_model_path, compile=False)

        scaler = np.load(self.phase9_scaler_path, allow_pickle=True)
        self.scaler_mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
        self.scaler_std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, -1)
        self.scaler_std = np.where(self.scaler_std < 1.0e-6, 1.0, self.scaler_std).astype(np.float32)
        expected_feature_names = self.residual_calibrator.build_feature_names(self.pair_reranker)
        saved_feature_names = [str(item) for item in np.asarray(scaler["feature_names"]).tolist()]
        if saved_feature_names != expected_feature_names:
            raise ValueError(
                "Phase 9 scaler feature order does not match the current code. "
                "Do not run inference with mismatched model/code artifacts."
            )

    def analyze_url(
        self,
        *,
        estudo_id: str,
        image_url: str,
        include_overlay: bool = False,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        image_bytes = image_bytes_from_url(image_url, timeout_seconds=timeout_seconds)
        return self.analyze_image_bytes(
            estudo_id=estudo_id,
            image_bytes=image_bytes,
            include_overlay=include_overlay,
            source={"type": "url", "imageUrl": image_url},
        )

    def analyze_image_path(
        self,
        *,
        estudo_id: str,
        image_path: str | Path,
        include_overlay: bool = False,
    ) -> dict[str, Any]:
        resolved_path = resolve_project_path(image_path)
        image_bytes = resolved_path.read_bytes()
        return self.analyze_image_bytes(
            estudo_id=estudo_id,
            image_bytes=image_bytes,
            include_overlay=include_overlay,
            source={"type": "file", "path": str(resolved_path)},
        )

    def analyze_image_bytes(
        self,
        *,
        estudo_id: str,
        image_bytes: bytes,
        include_overlay: bool = False,
        source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        image_tensor, pil_image, width, height = load_image_tensor(image_bytes)
        predictions = self.phase5_model(image_tensor, training=False)
        presence = predictions["presence"].numpy()[0]
        bbox = predictions["bbox"].numpy()[0]
        points = predictions["points"].numpy()[0]

        sequence_result = self.postprocess.postprocess_candidates_sequence(
            presence=presence,
            bbox=bbox,
            points=points,
            **self.phase5_sequence.PROFILE_CONFIGS[self.profile],
        )
        selected_points = np.asarray(sequence_result["selected_points"], dtype=np.float32).reshape(-1, 8)
        selected_scores = np.asarray(sequence_result["selected_scores"], dtype=np.float32).reshape(-1)

        if selected_points.shape[0] < 2:
            response = self._insufficient_vertebrae_response(
                estudo_id=estudo_id,
                width=width,
                height=height,
                sequence_result=sequence_result,
                selected_points=selected_points,
                selected_scores=selected_scores,
                source=source,
                start=start,
            )
            if include_overlay:
                response["overlay"] = self._build_overlay_or_unavailable(
                    pil_image=pil_image,
                    selected_points=selected_points,
                    selected_scores=selected_scores,
                    prediction=None,
                )
            return response

        prediction = self._predict_phase9(
            file_name=str(estudo_id),
            selected_points=selected_points,
            selected_scores=selected_scores,
            width=width,
            height=height,
        )
        vertebrae = self._vertebrae_payload(selected_points, selected_scores, width, height)
        quality_flags, confidence = self._quality_flags_and_confidence(
            sequence_result=sequence_result,
            selected_scores=selected_scores,
            prediction=prediction,
        )
        upper_index = int(prediction.geom_pair["upper_index"])
        lower_index = int(prediction.geom_pair["lower_index"])
        endpoint_mean_score = float(
            (selected_scores[upper_index] + selected_scores[lower_index]) / 2.0
        )

        response = {
            "estudoId": str(estudo_id),
            "status": "OK",
            "modelVersion": MODEL_VERSION,
            "source": dict(source or {}),
            "image": {
                "width": int(width),
                "height": int(height),
            },
            "result": {
                "cobbAngleDeg": rounded(prediction.calibrated_cobb),
                "rawGeometricCobbAngleDeg": rounded(prediction.geom_cobb),
                "appliedCorrectionDeg": rounded(prediction.applied_correction),
                "predictedResidualDeg": rounded(prediction.predicted_residual),
                "severity": severity_from_cobb(prediction.calibrated_cobb),
                "confidence": rounded(confidence, 3),
                "vertebraCount": int(selected_points.shape[0]),
                "cobbMeasurement": {
                    "upperVertebraIndex": upper_index,
                    "lowerVertebraIndex": lower_index,
                    "upperVertebraLabel": f"V{upper_index + 1:02d}",
                    "lowerVertebraLabel": f"V{lower_index + 1:02d}",
                    "upperPlateAngleDeg": rounded(prediction.geom_pair.get("upper_angle_deg")),
                    "lowerPlateAngleDeg": rounded(prediction.geom_pair.get("lower_angle_deg")),
                    "span": int(prediction.geom_pair.get("span", lower_index - upper_index)),
                    "endpointMeanScore": rounded(endpoint_mean_score, 4),
                },
                "vertebrae": vertebrae,
                "qualityFlags": quality_flags,
                "modelNotes": [
                    "Vxx labels are positional top-to-bottom labels, not anatomical vertebra labels.",
                    "Phase 9 API inference currently runs without the older auxiliary Cobb feature.",
                ],
            },
            "timingMs": int(round((time.perf_counter() - start) * 1000.0)),
            "overlay": None,
        }
        if include_overlay:
            response["overlay"] = self._build_overlay_or_unavailable(
                pil_image=pil_image,
                selected_points=selected_points,
                selected_scores=selected_scores,
                prediction=prediction,
            )
        return response

    def _predict_phase9(
        self,
        *,
        file_name: str,
        selected_points: np.ndarray,
        selected_scores: np.ndarray,
        width: int,
        height: int,
    ) -> Phase9Prediction:
        candidates = self._build_pair_candidates_for_inference(
            file_name=file_name,
            points=selected_points,
            scores=selected_scores,
            width=float(width),
            height=float(height),
        )
        group = {
            "file_name": file_name,
            "aux_cobb": np.nan,
            "aux_available": False,
            "candidates": candidates,
        }
        geom = self.residual_calibrator.select_pair_by_flag(candidates, "is_geom_pair")
        features = self.residual_calibrator.build_image_features(group, self.pair_reranker).reshape(1, -1)
        if features.shape[1] != self.scaler_mean.shape[1]:
            raise ValueError(
                f"Unexpected feature count: {features.shape[1]} != {self.scaler_mean.shape[1]}"
            )
        scaled = (features.astype(np.float32) - self.scaler_mean) / self.scaler_std
        predicted_residual = float(self.phase9_model.predict(scaled, batch_size=1, verbose=0).reshape(-1)[0])
        applied_correction = float(
            np.clip(
                self.alpha * predicted_residual,
                -self.max_correction_deg,
                self.max_correction_deg,
            )
        )
        geom_cobb = float(geom["angle_deg"])
        return Phase9Prediction(
            geom_cobb=geom_cobb,
            calibrated_cobb=float(geom_cobb + applied_correction),
            predicted_residual=predicted_residual,
            applied_correction=applied_correction,
            geom_pair=geom,
        )

    def _build_pair_candidates_for_inference(
        self,
        *,
        file_name: str,
        points: np.ndarray,
        scores: np.ndarray,
        width: float,
        height: float,
    ) -> list[dict[str, Any]]:
        valid_points = np.asarray(points, dtype=np.float32).reshape(-1, 8)
        valid_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        count = int(valid_points.shape[0])
        if count < 2:
            return []

        top_angles: list[float] = []
        bottom_angles: list[float] = []
        for row in valid_points:
            top_angles.append(self.phase9_v1.line_angle_deg(row[0:2], row[2:4], width, height))
            bottom_angles.append(self.phase9_v1.line_angle_deg(row[4:6], row[6:8], width, height))

        raw_pairs: list[dict[str, Any]] = []
        max_angle = -1.0
        for upper_index in range(count - 1):
            for lower_index in range(upper_index + 1, count):
                angle = float(self.phase9_v1.angle_diff_deg(top_angles[upper_index], bottom_angles[lower_index]))
                max_angle = max(max_angle, angle)
                raw_pairs.append(
                    {
                        "upper_index": int(upper_index),
                        "lower_index": int(lower_index),
                        "angle_deg": angle,
                    }
                )

        mean_score = float(np.mean(valid_scores)) if valid_scores.size else 0.0
        min_score = float(np.min(valid_scores)) if valid_scores.size else 0.0
        score_std = float(np.std(valid_scores)) if valid_scores.size else 0.0
        aux_available = 0.0
        aux_value = 0.0
        candidates: list[dict[str, Any]] = []
        for pair in raw_pairs:
            upper_index = int(pair["upper_index"])
            lower_index = int(pair["lower_index"])
            angle = float(pair["angle_deg"])
            span = int(lower_index - upper_index)
            span_norm = span / max(count - 1, 1)
            center_norm = (float(upper_index) + float(lower_index)) / 2.0 / float(max(count - 1, 1))
            upper_score = float(valid_scores[upper_index])
            lower_score = float(valid_scores[lower_index])
            endpoint_mean = float((upper_score + lower_score) / 2.0)
            endpoint_min = float(min(upper_score, lower_score))
            region_pt, region_mt, region_tll = self.pair_reranker.coarse_region(upper_index, lower_index, count)
            top_sin, top_cos = self.pair_reranker.axial_sin_cos(float(top_angles[upper_index]))
            bottom_sin, bottom_cos = self.pair_reranker.axial_sin_cos(float(bottom_angles[lower_index]))
            aux_signed_delta = 0.0
            feature_values = np.asarray(
                [
                    angle / 60.0,
                    (max_angle - angle) / 60.0,
                    upper_index / max(count - 1, 1),
                    lower_index / max(count - 1, 1),
                    span_norm,
                    center_norm,
                    count / 21.0,
                    upper_score,
                    lower_score,
                    endpoint_mean,
                    endpoint_min,
                    mean_score,
                    min_score,
                    score_std,
                    float(upper_index <= 1),
                    float(lower_index >= count - 2),
                    float(upper_index <= 1 or lower_index >= count - 2),
                    region_pt,
                    region_mt,
                    region_tll,
                    top_sin,
                    top_cos,
                    bottom_sin,
                    bottom_cos,
                    aux_available,
                    aux_value / 60.0,
                    abs(aux_signed_delta) / 60.0,
                    aux_signed_delta / 60.0,
                ],
                dtype=np.float32,
            )
            candidates.append(
                {
                    "file_name": file_name,
                    "upper_index": upper_index,
                    "lower_index": lower_index,
                    "span": span,
                    "angle_deg": angle,
                    "upper_angle_deg": float(top_angles[upper_index]),
                    "lower_angle_deg": float(bottom_angles[lower_index]),
                    "upper_score": upper_score,
                    "lower_score": lower_score,
                    "endpoint_score_mean": endpoint_mean,
                    "endpoint_score_min": endpoint_min,
                    "target_score": np.nan,
                    "is_oracle_pair": 0,
                    "is_geom_pair": 0,
                    "is_aux_pair": 0,
                    "features": feature_values,
                }
            )

        best_geom_index = int(np.argmax([float(item["angle_deg"]) for item in candidates]))
        candidates[best_geom_index]["is_geom_pair"] = 1
        return candidates

    def _vertebrae_payload(
        self,
        selected_points: np.ndarray,
        selected_scores: np.ndarray,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(selected_points):
            rows.append(
                {
                    "index": int(index),
                    "label": f"V{index + 1:02d}",
                    "score": rounded(float(selected_scores[index]), 4),
                    "bbox": bbox_from_points(row, width, height),
                    "points": normalized_points_to_pixel_dict(row, width, height),
                }
            )
        return rows

    def _quality_flags_and_confidence(
        self,
        *,
        sequence_result: Mapping[str, Any],
        selected_scores: np.ndarray,
        prediction: Phase9Prediction,
    ) -> tuple[list[str], float]:
        count = int(sequence_result.get("final_count", selected_scores.size))
        mean_score = float(np.mean(selected_scores)) if selected_scores.size else 0.0
        min_score = float(np.min(selected_scores)) if selected_scores.size else 0.0
        upper_index = int(prediction.geom_pair["upper_index"])
        lower_index = int(prediction.geom_pair["lower_index"])
        endpoint_min = float(min(selected_scores[upper_index], selected_scores[lower_index]))
        correction_abs = abs(float(prediction.applied_correction))

        flags: list[str] = []
        if count < 14:
            flags.append("LOW_VERTEBRA_COUNT")
        if count > 21:
            flags.append("HIGH_VERTEBRA_COUNT")
        if mean_score < 0.75:
            flags.append("LOW_MEAN_VERTEBRA_SCORE")
        if min_score < 0.50:
            flags.append("LOW_MIN_VERTEBRA_SCORE")
        if endpoint_min < 0.70:
            flags.append("LOW_COBB_ENDPOINT_SCORE")
        if correction_abs >= 8.0:
            flags.append("HIGH_PHASE9_CORRECTION")

        count_score = 1.0 if 14 <= count <= 21 else max(0.0, 1.0 - abs(count - 17.0) / 8.0)
        correction_penalty = 0.0
        if correction_abs >= 6.0:
            correction_penalty = 0.08
        if correction_abs >= 8.0:
            correction_penalty = 0.16
        confidence = 0.45 * mean_score + 0.35 * endpoint_min + 0.20 * count_score - correction_penalty
        return flags, float(np.clip(confidence, 0.0, 1.0))

    def _insufficient_vertebrae_response(
        self,
        *,
        estudo_id: str,
        width: int,
        height: int,
        sequence_result: Mapping[str, Any],
        selected_points: np.ndarray,
        selected_scores: np.ndarray,
        source: Mapping[str, Any] | None,
        start: float,
    ) -> dict[str, Any]:
        return {
            "estudoId": str(estudo_id),
            "status": "ERROR",
            "error": {
                "code": "INSUFFICIENT_VERTEBRAE_DETECTED",
                "message": "Fewer than two vertebrae were detected, so Cobb angle cannot be computed.",
            },
            "modelVersion": MODEL_VERSION,
            "source": dict(source or {}),
            "image": {"width": int(width), "height": int(height)},
            "result": {
                "cobbAngleDeg": None,
                "rawGeometricCobbAngleDeg": None,
                "appliedCorrectionDeg": None,
                "severity": None,
                "confidence": 0.0,
                "vertebraCount": int(sequence_result.get("final_count", selected_points.shape[0])),
                "cobbMeasurement": None,
                "vertebrae": self._vertebrae_payload(selected_points, selected_scores, width, height),
                "qualityFlags": ["INSUFFICIENT_VERTEBRAE_DETECTED"],
            },
            "timingMs": int(round((time.perf_counter() - start) * 1000.0)),
            "overlay": None,
        }

    def _build_overlay_or_unavailable(
        self,
        *,
        pil_image: Any,
        selected_points: np.ndarray,
        selected_scores: np.ndarray,
        prediction: Phase9Prediction | None,
    ) -> dict[str, Any]:
        if pil_image is None or ImageDraw is None:
            return {
                "format": "png",
                "encoding": "base64",
                "data": None,
                "error": "Pillow is not installed; install Pillow to generate overlays.",
            }
        image = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for index, row in enumerate(selected_points):
            points = np.asarray(row, dtype=np.float32).reshape(4, 2)
            polygon = [
                (
                    float(np.clip(point[0], 0.0, 1.0)) * width,
                    float(np.clip(point[1], 0.0, 1.0)) * height,
                )
                for point in points
            ]
            color = (245, 158, 11)
            line_width = 2
            if prediction is not None and index in (
                int(prediction.geom_pair["upper_index"]),
                int(prediction.geom_pair["lower_index"]),
            ):
                color = (220, 38, 38)
                line_width = 4
            draw.line(polygon + [polygon[0]], fill=color, width=line_width)
            draw.text((polygon[0][0] + 3, polygon[0][1] + 3), f"V{index + 1:02d}", fill=(255, 255, 255))

        if prediction is not None:
            upper_index = int(prediction.geom_pair["upper_index"])
            lower_index = int(prediction.geom_pair["lower_index"])
            upper_points = np.asarray(selected_points[upper_index], dtype=np.float32).reshape(4, 2)
            lower_points = np.asarray(selected_points[lower_index], dtype=np.float32).reshape(4, 2)

            upper_a = (float(upper_points[0, 0]) * width, float(upper_points[0, 1]) * height)
            upper_b = (float(upper_points[1, 0]) * width, float(upper_points[1, 1]) * height)
            lower_a = (float(lower_points[2, 0]) * width, float(lower_points[2, 1]) * height)
            lower_b = (float(lower_points[3, 0]) * width, float(lower_points[3, 1]) * height)
            upper_line = line_box_segment(upper_a, upper_b, width, height)
            lower_line = line_box_segment(lower_a, lower_b, width, height)
            draw.line(upper_line, fill=(34, 197, 94), width=3)
            draw.line(lower_line, fill=(59, 130, 246), width=3)
            draw.text(
                (12, 12),
                f"Cobb {prediction.calibrated_cobb:.1f} deg",
                fill=(255, 255, 255),
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "format": "png",
            "encoding": "base64",
            "contains": ["vertebra_quadrilaterals", "cobb_endpoint_plate_lines"],
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spinal-AI inference for one image.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-url", default="")
    input_group.add_argument("--image-path", default="")
    parser.add_argument("--estudo-id", default="manual-test")
    parser.add_argument("--include-overlay", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--overlay-path", default="")
    parser.add_argument("--phase5-model-path", default=str(DEFAULT_PHASE5_MODEL))
    parser.add_argument("--phase9-model-path", default=str(DEFAULT_PHASE9_MODEL))
    parser.add_argument("--phase9-scaler-path", default=str(DEFAULT_PHASE9_SCALER))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--url-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = SpinalAIAnalyzer(
        phase5_model_path=args.phase5_model_path,
        phase9_model_path=args.phase9_model_path,
        phase9_scaler_path=args.phase9_scaler_path,
        profile=args.profile,
    )
    if args.image_url:
        response = analyzer.analyze_url(
            estudo_id=args.estudo_id,
            image_url=args.image_url,
            include_overlay=args.include_overlay,
            timeout_seconds=float(args.url_timeout_seconds),
        )
    else:
        response = analyzer.analyze_image_path(
            estudo_id=args.estudo_id,
            image_path=args.image_path,
            include_overlay=args.include_overlay,
        )

    overlay_payload = response.get("overlay")
    if args.overlay_path and isinstance(overlay_payload, dict) and overlay_payload.get("data"):
        overlay_path = resolve_project_path(args.overlay_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_bytes(base64.b64decode(str(overlay_payload["data"])))
        response["overlay"] = {
            key: value
            for key, value in overlay_payload.items()
            if key != "data"
        } | {"path": str(overlay_path)}

    json_text = json.dumps(response, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = resolve_project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text, encoding="utf-8")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
