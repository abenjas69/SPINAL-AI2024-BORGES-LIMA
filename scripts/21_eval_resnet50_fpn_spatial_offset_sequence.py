"""Avaliacao visual da Fase 5 v3 com pos-processamento sequencial.

A avaliacao continua prediction-driven: nao usa o numero real de vertebras para
cortar as previsoes. Alem de contagem, calcula matching entre previsoes finais
e ground truth para detetar casos em que a contagem esta certa mas ha vertebras
saltadas ou deteccoes falsas.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGE_SIZE = 512.0


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_spatial_offset_model_for_eval(model_path: Path) -> tf.keras.Model:
    """Carrega modelos Fase 5 v3/v4/v5/v6/v7 para avaliacao."""
    offset_model = import_script("18_resnet50_fpn_spatial_offset_model_v1.py", "phase5_offset_model")
    try:
        return offset_model.load_resnet50_fpn_spatial_offset_model_v1(model_path)
    except Exception as base_error:
        hard_model = import_script(
            "22_resnet50_fpn_spatial_offset_hard_negative_model_v1.py",
            "phase5_hard_negative_model",
        )
        try:
            return hard_model.load_resnet50_fpn_spatial_offset_hard_negative_model_v1(model_path)
        except Exception as hard_error:
            radius_model = import_script(
                "53_resnet50_fpn_spatial_offset_radius_model_v1.py",
                "phase5_radius_model",
            )
            try:
                return radius_model.load_resnet50_fpn_spatial_offset_radius_model_v1(model_path)
            except Exception as radius_error:
                endpoint_model = import_script(
                    "81_resnet50_fpn_spatial_offset_radius_cobb_endpoint_model_v1.py",
                    "phase5_cobb_endpoint_model",
                )
                try:
                    return endpoint_model.load_resnet50_fpn_spatial_offset_radius_cobb_endpoint_model_v1(model_path)
                except Exception as endpoint_error:
                    anatomical_endpoint_model = import_script(
                        "91_resnet50_fpn_spatial_offset_radius_anatomical_endpoint_model_v1.py",
                        "phase5_anatomical_endpoint_model",
                    )
                    try:
                        return anatomical_endpoint_model.load_resnet50_fpn_spatial_offset_radius_anatomical_endpoint_model_v1(
                            model_path
                        )
                    except Exception as anatomical_endpoint_error:
                        raise RuntimeError(
                            "Nao foi possivel carregar o modelo como Fase 5 v3, "
                            "Fase 5 v4 hard-negative, Fase 5 v5 PositiveRadius, "
                            "Fase 5 v6 CobbEndpoint nem Fase 5 v7 AnatomicalEndpoint."
                        ) from anatomical_endpoint_error


def select_window(items: Sequence[Any], start_index: int, num_items: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_items <= 0:
        return []
    return items[start_index : start_index + num_items]


def points_centroids_px(points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)
    return np.mean(points_array, axis=1) * IMAGE_SIZE


def match_predictions_to_ground_truth(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    max_center_distance_px: float,
) -> dict[str, Any]:
    """Faz matching greedy por distancia de centroide em pixels."""
    gt = np.asarray(gt_points, dtype=np.float32).reshape(-1, 8)
    pred = np.asarray(pred_points, dtype=np.float32).reshape(-1, 8)
    gt_count = int(gt.shape[0])
    pred_count = int(pred.shape[0])

    if gt_count == 0 or pred_count == 0:
        return {
            "matched_count": 0,
            "missed_gt": gt_count,
            "false_pred": pred_count,
            "mean_center_error_px": None,
            "mean_points_mae_px": None,
            "matches": [],
        }

    gt_centers = points_centroids_px(gt)
    pred_centers = points_centroids_px(pred)
    distances = np.linalg.norm(gt_centers[:, None, :] - pred_centers[None, :, :], axis=2)
    candidate_pairs = [
        (float(distances[gt_index, pred_index]), int(gt_index), int(pred_index))
        for gt_index in range(gt_count)
        for pred_index in range(pred_count)
    ]
    candidate_pairs.sort(key=lambda item: item[0])

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    point_errors: list[float] = []
    center_errors: list[float] = []

    for distance, gt_index, pred_index in candidate_pairs:
        if distance > max_center_distance_px:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        point_mae_px = float(np.mean(np.abs(gt[gt_index] - pred[pred_index])) * IMAGE_SIZE)
        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "center_error_px": distance,
                "points_mae_px": point_mae_px,
            }
        )
        center_errors.append(distance)
        point_errors.append(point_mae_px)

    matched_count = len(matches)
    return {
        "matched_count": matched_count,
        "missed_gt": gt_count - matched_count,
        "false_pred": pred_count - matched_count,
        "mean_center_error_px": float(np.mean(center_errors)) if center_errors else None,
        "mean_points_mae_px": float(np.mean(point_errors)) if point_errors else None,
        "matches": matches,
    }


def error_zone_counts(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    matches: Sequence[Mapping[str, Any]],
    endpoint_margin_px: float = 14.0,
) -> dict[str, int]:
    """Separa falsos positivos e misses por topo, fundo e interior."""
    gt = np.asarray(gt_points, dtype=np.float32).reshape(-1, 8)
    pred = np.asarray(pred_points, dtype=np.float32).reshape(-1, 8)
    matched_gt = {int(match["gt_index"]) for match in matches}
    matched_pred = {int(match["pred_index"]) for match in matches}

    counts = {
        "false_top": 0,
        "false_bottom": 0,
        "false_inside": 0,
        "missed_top": 0,
        "missed_bottom": 0,
        "missed_inside": 0,
    }
    if gt.size == 0:
        counts["false_inside"] = int(pred.shape[0])
        return counts

    gt_y = points_centroids_px(gt)[:, 1]
    pred_y = points_centroids_px(pred)[:, 1] if pred.size else np.asarray([], dtype=np.float32)
    top_bound = float(np.min(gt_y))
    bottom_bound = float(np.max(gt_y))
    gt_count = int(gt.shape[0])

    for pred_index, y_value in enumerate(pred_y):
        if pred_index in matched_pred:
            continue
        if float(y_value) < top_bound - endpoint_margin_px:
            counts["false_top"] += 1
        elif float(y_value) > bottom_bound + endpoint_margin_px:
            counts["false_bottom"] += 1
        else:
            counts["false_inside"] += 1

    for gt_index in range(gt_count):
        if gt_index in matched_gt:
            continue
        if gt_index <= 1:
            counts["missed_top"] += 1
        elif gt_index >= gt_count - 2:
            counts["missed_bottom"] += 1
        else:
            counts["missed_inside"] += 1

    return counts


def draw_candidates(
    image: tf.Tensor,
    targets: Mapping[str, Any],
    candidate_points: np.ndarray,
    drawing: ModuleType,
    color: np.ndarray,
) -> np.ndarray:
    canvas = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
    gt_count = int(targets["vertebra_count"])
    gt_points = np.asarray(targets["points"][:gt_count], dtype=np.float32).reshape(gt_count, 4, 2) * IMAGE_SIZE
    drawing.draw_ground_truth_points(canvas, gt_points)

    points = np.asarray(candidate_points, dtype=np.float32).reshape(-1, 4, 2) * IMAGE_SIZE
    for vertebra_points in points:
        drawing.draw_quadrilateral(canvas, vertebra_points, color)

    return canvas


def make_sequence_comparison_overlay(
    image: tf.Tensor,
    targets: Mapping[str, Any],
    postprocessed: Mapping[str, Any],
    drawing: ModuleType,
) -> np.ndarray:
    raw_color = np.array([255, 160, 0], dtype=np.uint8)
    final_color = np.array([255, 0, 0], dtype=np.uint8)
    before = draw_candidates(
        image=image,
        targets=targets,
        candidate_points=postprocessed["raw_points"],
        drawing=drawing,
        color=raw_color,
    )
    after = draw_candidates(
        image=image,
        targets=targets,
        candidate_points=postprocessed["selected_points"],
        drawing=drawing,
        color=final_color,
    )
    separator = np.full((before.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([before, separator, after], axis=1)


def write_details_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "file_name",
        "gt_count",
        "raw_count",
        "nms_count",
        "final_count",
        "count_error",
        "abs_count_error",
        "plausible_count",
        "matched_count",
        "missed_gt",
        "false_pred",
        "mean_center_error_px",
        "mean_points_mae_px",
        "mean_final_score",
        "mean_endpoint_decision_score",
        "endpoint_score_used",
        "endpoint_score_blend",
        "sequence_used_y_gap",
        "selection_method",
        "estimated_y_gap",
        "path_score",
        "endpoint_pruned_top",
        "endpoint_pruned_bottom",
        "gap_filled_count",
        "gap_fill_candidate_count",
        "gap_fill_threshold",
        "endpoint_filled_top",
        "endpoint_filled_bottom",
        "endpoint_fill_candidate_count",
        "endpoint_fill_threshold",
        "false_top",
        "false_bottom",
        "false_inside",
        "missed_top",
        "missed_bottom",
        "missed_inside",
        "selected_indices",
        "overlay_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["selected_indices"] = " ".join(str(i) for i in row["selected_indices"])
            writer.writerow(row)


def mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def summarize_records(
    records: list[dict[str, Any]],
    split: str,
    start_index: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    min_y_gap: float,
    max_match_distance_px: float,
    selection_method: str,
) -> dict[str, Any]:
    if not records:
        return {
            "num_images": 0,
            "split": split,
            "start_index": start_index,
            "confidence_threshold": confidence_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "min_y_gap": min_y_gap,
            "max_match_distance_px": max_match_distance_px,
            "selection_method": selection_method,
            "mean_abs_count_error": None,
            "exact_count_accuracy": None,
            "plausible_count_rate": None,
            "total_false_top": 0,
            "total_false_bottom": 0,
            "total_false_inside": 0,
            "total_missed_top": 0,
            "total_missed_bottom": 0,
            "total_missed_inside": 0,
            "total_endpoint_pruned_top": 0,
            "total_endpoint_pruned_bottom": 0,
            "total_gap_filled": 0,
            "mean_gap_filled": 0.0,
            "total_endpoint_filled_top": 0,
            "total_endpoint_filled_bottom": 0,
        }

    abs_errors = [record["abs_count_error"] for record in records]
    exact = [record["count_error"] == 0 for record in records]
    plausible = [bool(record["plausible_count"]) for record in records]
    return {
        "num_images": len(records),
        "split": split,
        "start_index": start_index,
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "min_y_gap": min_y_gap,
        "max_match_distance_px": max_match_distance_px,
        "selection_method": selection_method,
        "mean_abs_count_error": float(np.mean(abs_errors)),
        "max_abs_count_error": int(np.max(abs_errors)),
        "exact_count_accuracy": float(np.mean(exact)),
        "plausible_count_rate": float(np.mean(plausible)),
        "mean_matched_count": float(np.mean([record["matched_count"] for record in records])),
        "mean_missed_gt": float(np.mean([record["missed_gt"] for record in records])),
        "mean_false_pred": float(np.mean([record["false_pred"] for record in records])),
        "mean_center_error_px": mean_optional([record["mean_center_error_px"] for record in records]),
        "mean_points_mae_px": mean_optional([record["mean_points_mae_px"] for record in records]),
        "total_false_top": int(np.sum([record["false_top"] for record in records])),
        "total_false_bottom": int(np.sum([record["false_bottom"] for record in records])),
        "total_false_inside": int(np.sum([record["false_inside"] for record in records])),
        "total_missed_top": int(np.sum([record["missed_top"] for record in records])),
        "total_missed_bottom": int(np.sum([record["missed_bottom"] for record in records])),
        "total_missed_inside": int(np.sum([record["missed_inside"] for record in records])),
        "mean_false_top": float(np.mean([record["false_top"] for record in records])),
        "mean_false_bottom": float(np.mean([record["false_bottom"] for record in records])),
        "mean_false_inside": float(np.mean([record["false_inside"] for record in records])),
        "mean_missed_top": float(np.mean([record["missed_top"] for record in records])),
        "mean_missed_bottom": float(np.mean([record["missed_bottom"] for record in records])),
        "mean_missed_inside": float(np.mean([record["missed_inside"] for record in records])),
        "total_endpoint_pruned_top": int(np.sum([record["endpoint_pruned_top"] for record in records])),
        "total_endpoint_pruned_bottom": int(np.sum([record["endpoint_pruned_bottom"] for record in records])),
        "total_gap_filled": int(np.sum([record["gap_filled_count"] for record in records])),
        "mean_gap_filled": float(np.mean([record["gap_filled_count"] for record in records])),
        "total_endpoint_filled_top": int(np.sum([record.get("endpoint_filled_top", 0) for record in records])),
        "total_endpoint_filled_bottom": int(np.sum([record.get("endpoint_filled_bottom", 0) for record in records])),
    }


def build_evaluation_record(
    sample: Mapping[str, Any],
    targets: Mapping[str, Any],
    result: Mapping[str, Any],
    max_match_distance_px: float,
    overlay_name: str,
) -> dict[str, Any]:
    gt_count = int(targets["vertebra_count"])
    gt_points = np.asarray(targets["points"][:gt_count], dtype=np.float32).reshape(gt_count, 8)
    match = match_predictions_to_ground_truth(
        gt_points=gt_points,
        pred_points=result["selected_points"],
        max_center_distance_px=max_match_distance_px,
    )
    zones = error_zone_counts(
        gt_points=gt_points,
        pred_points=result["selected_points"],
        matches=match["matches"],
    )
    final_count = int(result["final_count"])
    mean_final_score = (
        float(np.mean(result["selected_scores"]))
        if len(result["selected_scores"]) > 0
        else 0.0
    )
    selected_endpoint_scores = result.get("selected_endpoint_scores", [])
    mean_endpoint_decision_score = (
        float(np.mean(selected_endpoint_scores))
        if len(selected_endpoint_scores) > 0
        else 0.0
    )
    return {
        "file_name": str(sample["file_name"]),
        "gt_count": gt_count,
        "raw_count": int(result["raw_count"]),
        "nms_count": int(result["nms_count"]),
        "final_count": final_count,
        "count_error": final_count - gt_count,
        "abs_count_error": abs(final_count - gt_count),
        "plausible_count": bool(result["plausible_count"]),
        "matched_count": int(match["matched_count"]),
        "missed_gt": int(match["missed_gt"]),
        "false_pred": int(match["false_pred"]),
        "mean_center_error_px": match["mean_center_error_px"],
        "mean_points_mae_px": match["mean_points_mae_px"],
        "mean_final_score": mean_final_score,
        "mean_endpoint_decision_score": mean_endpoint_decision_score,
        "endpoint_score_used": bool(result.get("endpoint_score_used", False)),
        "endpoint_score_blend": float(result.get("endpoint_score_blend", 0.0)),
        "sequence_used_y_gap": float(result["sequence_used_y_gap"]),
        "selection_method": str(result["sequence_method"]),
        "estimated_y_gap": float(result["estimated_y_gap"]),
        "path_score": float(result["path_score"]),
        "endpoint_pruned_top": int(result["endpoint_pruned_top"]),
        "endpoint_pruned_bottom": int(result["endpoint_pruned_bottom"]),
        "gap_filled_count": int(result["gap_filled_count"]),
        "gap_fill_candidate_count": int(result["gap_fill_candidate_count"]),
        "gap_fill_threshold": float(result["gap_fill_threshold"]),
        "endpoint_filled_top": int(result.get("endpoint_filled_top", 0)),
        "endpoint_filled_bottom": int(result.get("endpoint_filled_bottom", 0)),
        "endpoint_fill_candidate_count": int(result.get("endpoint_fill_candidate_count", 0)),
        "endpoint_fill_threshold": float(result.get("endpoint_fill_threshold", 0.0)),
        "false_top": int(zones["false_top"]),
        "false_bottom": int(zones["false_bottom"]),
        "false_inside": int(zones["false_inside"]),
        "missed_top": int(zones["missed_top"]),
        "missed_bottom": int(zones["missed_bottom"]),
        "missed_inside": int(zones["missed_inside"]),
        "selected_indices": [int(i) for i in result["selected_indices"]],
        "overlay_name": overlay_name,
    }


def generate_sequence_overlays(
    model: tf.keras.Model,
    tfdata: ModuleType,
    drawing: ModuleType,
    samples: Sequence[Mapping[str, Any]],
    image_paths: Sequence[Path],
    output_dir: Path,
    num_images: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    min_y_gap: float,
    min_vertebrae: int,
    max_vertebrae: int,
    max_match_distance_px: float,
    selection_method: str = "greedy",
    count_prior: float = 17.0,
    count_prior_weight: float = 0.12,
    candidate_cost: float = 2.0,
    gap_weight: float = 0.65,
    x_jump_weight: float = 0.55,
    size_weight: float = 0.25,
    angle_weight: float = 0.0,
    angle_jump_tolerance_deg: float = 14.0,
    endpoint_pruning: bool = True,
    endpoint_min_score: float = 0.88,
    endpoint_max_gap_factor: float = 1.65,
    endpoint_score_margin: float = 0.08,
    endpoint_score_blend: float = 0.0,
    gap_filling: bool = True,
    gap_fill_threshold: float = 0.6,
    gap_fill_min_gap_factor: float = 1.55,
    gap_fill_max_insertions: int = 4,
    gap_fill_max_x_error: float = 0.07,
    gap_fill_max_size_log_error: float = 0.65,
    threshold_sweep: Sequence[float] | None = None,
    split: str = "validation",
    start_index: int = 0,
) -> dict[str, Any]:
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = select_window(samples, 0, num_images)
    selected_paths = select_window(image_paths, 0, num_images)

    evaluation_thresholds = [float(confidence_threshold)]
    if threshold_sweep:
        for threshold in threshold_sweep:
            threshold_value = float(threshold)
            if not any(abs(threshold_value - existing) < 1e-9 for existing in evaluation_thresholds):
                evaluation_thresholds.append(threshold_value)
    records_by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in evaluation_thresholds
    }

    for index, (sample, image_path) in enumerate(zip(selected_samples, selected_paths)):
        targets = tfdata.sample_to_targets(sample)
        image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
        predictions = model(tf.expand_dims(image, axis=0), training=False)
        prediction_values = {
            "presence": predictions["presence"].numpy()[0],
            "bbox": predictions["bbox"].numpy()[0],
            "points": predictions["points"].numpy()[0],
        }
        if isinstance(predictions, dict) and "cobb_endpoint_score" in predictions:
            prediction_values["cobb_endpoint_score"] = predictions["cobb_endpoint_score"].numpy()[0]

        primary_record: dict[str, Any] | None = None
        for threshold in evaluation_thresholds:
            result = postprocess.postprocess_candidates_sequence(
                presence=prediction_values["presence"],
                bbox=prediction_values["bbox"],
                points=prediction_values["points"],
                cobb_endpoint_score=prediction_values.get("cobb_endpoint_score"),
                confidence_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                min_y_gap=min_y_gap,
                min_vertebrae=min_vertebrae,
                max_vertebrae=max_vertebrae,
                selection_method=selection_method,
                count_prior=count_prior,
                count_prior_weight=count_prior_weight,
                candidate_cost=candidate_cost,
                gap_weight=gap_weight,
                x_jump_weight=x_jump_weight,
                size_weight=size_weight,
                angle_weight=angle_weight,
                angle_jump_tolerance_deg=angle_jump_tolerance_deg,
                endpoint_pruning=endpoint_pruning,
                endpoint_min_score=endpoint_min_score,
                endpoint_max_gap_factor=endpoint_max_gap_factor,
                endpoint_score_margin=endpoint_score_margin,
                endpoint_score_blend=endpoint_score_blend,
                gap_filling=gap_filling,
                gap_fill_threshold=gap_fill_threshold,
                gap_fill_min_gap_factor=gap_fill_min_gap_factor,
                gap_fill_max_insertions=gap_fill_max_insertions,
                gap_fill_max_x_error=gap_fill_max_x_error,
                gap_fill_max_size_log_error=gap_fill_max_size_log_error,
            )

            overlay_name = ""
            if abs(threshold - float(confidence_threshold)) < 1e-9:
                overlay = make_sequence_comparison_overlay(
                    image=image,
                    targets=targets,
                    postprocessed=result,
                    drawing=drawing,
                )
                stem = Path(str(sample["file_name"])).stem
                overlay_name = f"{index:03d}_{stem}_phase5_spatial_offset_sequence.png"
                output_path = output_dir / overlay_name
                tf.io.write_file(
                    str(output_path),
                    tf.io.encode_png(tf.convert_to_tensor(overlay)),
                )

            record = build_evaluation_record(
                sample=sample,
                targets=targets,
                result=result,
                max_match_distance_px=max_match_distance_px,
                overlay_name=overlay_name,
            )
            records_by_threshold[threshold].append(record)
            if abs(threshold - float(confidence_threshold)) < 1e-9:
                primary_record = record

        if primary_record is not None:
            print(
                f"{primary_record['file_name']}: gt={primary_record['gt_count']} "
                f"raw={primary_record['raw_count']} nms={primary_record['nms_count']} "
                f"final={primary_record['final_count']} erro={primary_record['count_error']} "
                f"matched={primary_record['matched_count']} missed={primary_record['missed_gt']} "
                f"false={primary_record['false_pred']} "
                f"filled={primary_record['gap_filled_count']} "
                f"false_top/bottom/inside={primary_record['false_top']}/"
                f"{primary_record['false_bottom']}/{primary_record['false_inside']} "
                f"overlay={primary_record['overlay_name']}"
            )

    summary = summarize_records(
        records=records_by_threshold[float(confidence_threshold)],
        split=split,
        start_index=start_index,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        min_y_gap=min_y_gap,
        max_match_distance_px=max_match_distance_px,
        selection_method=selection_method,
    )
    summary["endpoint_score_blend"] = float(endpoint_score_blend)
    summary_path = output_dir / "phase5_spatial_offset_sequence_summary.json"
    details_path = output_dir / "phase5_spatial_offset_sequence_details.csv"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_details_csv(details_path, records_by_threshold[float(confidence_threshold)])

    if len(evaluation_thresholds) > 1:
        sweep_summaries = [
            summarize_records(
                records=records_by_threshold[threshold],
                split=split,
                start_index=start_index,
                confidence_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                min_y_gap=min_y_gap,
                max_match_distance_px=max_match_distance_px,
                selection_method=selection_method,
            )
            for threshold in evaluation_thresholds
        ]
        for sweep_summary in sweep_summaries:
            sweep_summary["endpoint_score_blend"] = float(endpoint_score_blend)
        sweep_json_path = output_dir / "phase5_spatial_offset_sequence_threshold_sweep.json"
        sweep_csv_path = output_dir / "phase5_spatial_offset_sequence_threshold_sweep.csv"
        with sweep_json_path.open("w", encoding="utf-8") as file:
            json.dump(sweep_summaries, file, indent=2)
        sweep_fields = [
            "confidence_threshold",
            "endpoint_score_blend",
            "mean_abs_count_error",
            "exact_count_accuracy",
            "plausible_count_rate",
            "mean_matched_count",
            "mean_missed_gt",
            "mean_false_pred",
            "mean_points_mae_px",
            "total_false_top",
            "total_false_bottom",
            "total_false_inside",
            "total_missed_top",
            "total_missed_bottom",
            "total_missed_inside",
            "total_gap_filled",
            "mean_gap_filled",
            "total_endpoint_pruned_top",
            "total_endpoint_pruned_bottom",
        ]
        with sweep_csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sweep_fields)
            writer.writeheader()
            for sweep_summary in sweep_summaries:
                writer.writerow({field: sweep_summary.get(field) for field in sweep_fields})
        summary["threshold_sweep"] = sweep_summaries
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera overlays prediction-driven para Fase 5 v3 espacial com offsets."
    )
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_v1.keras"),
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start-index", type=int, default=2000)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Lista opcional de thresholds adicionais. As previsoes do modelo "
            "sao calculadas uma so vez por imagem; apenas o threshold principal "
            "gera overlays."
        ),
    )
    parser.add_argument("--nms-iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-y-gap", type=float, default=0.025)
    parser.add_argument("--selection-method", choices=("path", "greedy"), default="greedy")
    parser.add_argument("--count-prior", type=float, default=17.0)
    parser.add_argument("--count-prior-weight", type=float, default=0.12)
    parser.add_argument("--candidate-cost", type=float, default=2.0)
    parser.add_argument("--gap-weight", type=float, default=0.65)
    parser.add_argument("--x-jump-weight", type=float, default=0.55)
    parser.add_argument("--size-weight", type=float, default=0.25)
    parser.add_argument("--angle-weight", type=float, default=0.0)
    parser.add_argument("--angle-jump-tolerance-deg", type=float, default=14.0)
    parser.add_argument("--disable-endpoint-pruning", action="store_true")
    parser.add_argument("--endpoint-min-score", type=float, default=0.88)
    parser.add_argument("--endpoint-max-gap-factor", type=float, default=1.65)
    parser.add_argument("--endpoint-score-margin", type=float, default=0.08)
    parser.add_argument(
        "--endpoint-score-blend",
        type=float,
        default=0.0,
        help=(
            "Peso [0,1] do output cobb_endpoint_score para pruning/filling dos extremos. "
            "0 preserva o comportamento antigo baseado so em presence."
        ),
    )
    parser.add_argument("--disable-gap-filling", action="store_true")
    parser.add_argument("--gap-fill-threshold", type=float, default=0.6)
    parser.add_argument("--gap-fill-min-gap-factor", type=float, default=1.55)
    parser.add_argument("--gap-fill-max-insertions", type=int, default=4)
    parser.add_argument("--gap-fill-max-x-error", type=float, default=0.07)
    parser.add_argument("--gap-fill-max-size-log-error", type=float, default=0.65)
    parser.add_argument("--min-vertebrae", type=int, default=14)
    parser.add_argument("--max-vertebrae", type=int, default=21)
    parser.add_argument("--max-match-distance-px", type=float, default=32.0)
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "phase5_resnet50_fpn_spatial_offset_sequence"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")

    model_path = resolve_project_path(args.model_path)
    output_dir = resolve_project_path(args.output_dir)

    print(f"A carregar modelo: {model_path}")
    model = load_spatial_offset_model_for_eval(model_path)

    print(f"A carregar split {args.split}...")
    samples, image_paths = phase2_train.load_split_samples_and_paths(tfdata, args.split)
    selected_samples = select_window(samples, args.start_index, args.num_images)
    selected_paths = select_window(image_paths, args.start_index, args.num_images)
    if not selected_samples:
        raise ValueError("Nenhuma imagem selecionada. Verifica --start-index e --num-images.")

    summary = generate_sequence_overlays(
        model=model,
        tfdata=tfdata,
        drawing=phase2_train,
        samples=selected_samples,
        image_paths=selected_paths,
        output_dir=output_dir,
        num_images=args.num_images,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        min_y_gap=args.min_y_gap,
        min_vertebrae=args.min_vertebrae,
        max_vertebrae=args.max_vertebrae,
        max_match_distance_px=args.max_match_distance_px,
        selection_method=args.selection_method,
        count_prior=args.count_prior,
        count_prior_weight=args.count_prior_weight,
        candidate_cost=args.candidate_cost,
        gap_weight=args.gap_weight,
        x_jump_weight=args.x_jump_weight,
        size_weight=args.size_weight,
        angle_weight=args.angle_weight,
        angle_jump_tolerance_deg=args.angle_jump_tolerance_deg,
        endpoint_pruning=not args.disable_endpoint_pruning,
        endpoint_min_score=args.endpoint_min_score,
        endpoint_max_gap_factor=args.endpoint_max_gap_factor,
        endpoint_score_margin=args.endpoint_score_margin,
        endpoint_score_blend=args.endpoint_score_blend,
        gap_filling=not args.disable_gap_filling,
        gap_fill_threshold=args.gap_fill_threshold,
        gap_fill_min_gap_factor=args.gap_fill_min_gap_factor,
        gap_fill_max_insertions=args.gap_fill_max_insertions,
        gap_fill_max_x_error=args.gap_fill_max_x_error,
        gap_fill_max_size_log_error=args.gap_fill_max_size_log_error,
        threshold_sweep=args.threshold_sweep,
        split=args.split,
        start_index=args.start_index,
    )

    print("\nResumo Fase 5 espacial offsets + sequencia")
    print(f"imagens: {summary['num_images']}")
    print(f"threshold confidence: {summary['confidence_threshold']}")
    print(f"threshold NMS IoU: {summary['nms_iou_threshold']}")
    print(f"min_y_gap: {summary['min_y_gap']}")
    print(f"endpoint_score_blend: {summary['endpoint_score_blend']}")
    print(f"erro absoluto medio de contagem: {summary['mean_abs_count_error']}")
    print(f"accuracy contagem exata: {summary['exact_count_accuracy']}")
    print(f"taxa contagem plausivel: {summary['plausible_count_rate']}")
    print(f"media vertebras matched: {summary['mean_matched_count']}")
    print(f"media missed_gt: {summary['mean_missed_gt']}")
    print(f"media false_pred: {summary['mean_false_pred']}")
    print(
        "falsos topo/fundo/interior: "
        f"{summary['total_false_top']}/{summary['total_false_bottom']}/{summary['total_false_inside']}"
    )
    print(
        "missed topo/fundo/interior: "
        f"{summary['total_missed_top']}/{summary['total_missed_bottom']}/{summary['total_missed_inside']}"
    )
    print(
        "endpoint pruning topo/fundo: "
        f"{summary['total_endpoint_pruned_top']}/{summary['total_endpoint_pruned_bottom']}"
    )
    print(f"gap filling inseridos: {summary['total_gap_filled']}")
    print(f"erro medio centroide px: {summary['mean_center_error_px']}")
    print(f"MAE medio pontos px: {summary['mean_points_mae_px']}")
    if "threshold_sweep" in summary:
        print("\nSweep de thresholds")
        for sweep_summary in summary["threshold_sweep"]:
            print(
                f"thr={sweep_summary['confidence_threshold']}: "
                f"MAE_count={sweep_summary['mean_abs_count_error']}, "
                f"exact={sweep_summary['exact_count_accuracy']}, "
                f"missed={sweep_summary['mean_missed_gt']}, "
                f"false={sweep_summary['mean_false_pred']}, "
                f"gap_filled={sweep_summary['total_gap_filled']}"
            )
    print(f"Outputs guardados em: {output_dir}")


if __name__ == "__main__":
    main()
