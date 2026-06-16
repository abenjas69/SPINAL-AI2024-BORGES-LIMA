"""Evaluate colleague centerline model with clean Phase 5 predicted ROIs.

V1 used the annotation points to build the colleague model ROI. That is useful
for diagnosis, but it leaks ground truth into the model input. V3 builds the ROI
only from the Phase 5 selected points produced by the locked current pipeline.

The script also writes a compatible MLP v2 prediction CSV for the same rows, so
the fusion step can be tuned on holdout and applied to test without rejoining
different runs.
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
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

DEFAULT_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_AUX_PHASE5_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_hard_negative_fulltrain_probe_v1_colab.keras"
DEFAULT_AUX_PROFILE = "anatomical_endpoint_safe_v1"
DEFAULT_PHASE7_MODEL = MODELS_DIR / "phase7_bilstm_global_endpoint_safe_v1.keras"
DEFAULT_PHASE8_MODEL = MODELS_DIR / "phase8_aux_angle_head_weighted_endpoint_safe_v2.keras"
DEFAULT_MLP_EXPERIMENT = EXPERIMENTS_DIR / "phase9_cobb_residual_mlp_v2_train12000_cal768_val3192"
DEFAULT_COLLEAGUE_MODEL = MODELS_DIR / "centerline_daniel_unet_baseline_2000_padding_512.keras"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR / "fusion_centerline_model_v3_phase5_roi_holdout3192"

ROI_SOURCE = "phase5_predicted_points"
ANGLE_NAMES = ("PT", "MT", "TLL")
CENTERLINE_FIELDNAMES = [
    "index",
    "file_name",
    "image_path",
    "roi_source",
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
    "phase5_predicted_point_count",
    "phase5_selected_vertebrae",
    "phase5_mean_score",
    "phase5_points_normalized_min",
    "phase5_points_normalized_max",
    "phase5_points_are_normalized",
    "image_width",
    "image_height",
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


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def select_window(items: Sequence[Any], start_index: int, num_items: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_items <= 0:
        return items[start_index:]
    return items[start_index : start_index + num_items]


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


def preferred_centerline_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row.keys())
    return [key for key in CENTERLINE_FIELDNAMES if key in keys] + sorted(keys - set(CENTERLINE_FIELDNAMES))


def predicted_points_to_absolute(group: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    selected_points = np.asarray(group.get("selected_points", []), dtype=np.float32).reshape(-1, 4, 2)
    if selected_points.size == 0:
        raise ValueError("sem pontos Phase 5 selecionados para construir ROI")

    flat_points = selected_points.reshape(-1, 2)
    finite_mask = np.isfinite(flat_points).all(axis=1)
    flat_points = flat_points[finite_mask]
    if flat_points.size == 0:
        raise ValueError("pontos Phase 5 selecionados nao finitos")

    width = float(group["image_width"])
    height = float(group["image_height"])
    max_abs = float(np.max(np.abs(flat_points)))
    normalized = max_abs <= 2.0
    if normalized:
        absolute = flat_points.copy()
        absolute[:, 0] *= width
        absolute[:, 1] *= height
    else:
        absolute = flat_points.copy()

    absolute[:, 0] = np.clip(absolute[:, 0], 0.0, max(width - 1.0, 0.0))
    absolute[:, 1] = np.clip(absolute[:, 1], 0.0, max(height - 1.0, 0.0))
    return absolute.astype(np.float32), normalized


def calculate_phase5_roi(group: Mapping[str, Any], roi_padding: int) -> dict[str, Any]:
    points_abs, points_are_normalized = predicted_points_to_absolute(group)
    width = int(round(float(group["image_width"])))
    height = int(round(float(group["image_height"])))
    padding = int(roi_padding)

    xmin = max(0, int(np.floor(float(np.min(points_abs[:, 0])) - padding)))
    ymin = max(0, int(np.floor(float(np.min(points_abs[:, 1])) - padding)))
    xmax = min(width, int(np.ceil(float(np.max(points_abs[:, 0])) + padding)))
    ymax = min(height, int(np.ceil(float(np.max(points_abs[:, 1])) + padding)))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"ROI Phase 5 invalida: {(xmin, ymin, xmax, ymax)}")

    selected_scores = np.asarray(group.get("selected_scores", []), dtype=np.float32).reshape(-1)
    selected_points = np.asarray(group.get("selected_points", []), dtype=np.float32).reshape(-1)
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "roi_source": ROI_SOURCE,
        "phase5_predicted_point_count": int(points_abs.shape[0]),
        "phase5_selected_vertebrae": int(points_abs.shape[0] // 4),
        "phase5_mean_score": float(np.mean(selected_scores)) if selected_scores.size else 0.0,
        "phase5_points_normalized_min": float(np.min(selected_points)) if selected_points.size else np.nan,
        "phase5_points_normalized_max": float(np.max(selected_points)) if selected_points.size else np.nan,
        "phase5_points_are_normalized": bool(points_are_normalized),
        "image_width": width,
        "image_height": height,
    }


def preprocess_group_with_phase5_roi(
    group: Mapping[str, Any],
    *,
    centerline_v1: ModuleType,
    roi_padding: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path = Path(str(group.get("image_path", "")))
    if not image_path.is_file():
        sample = group.get("sample", {})
        image_path = centerline_v1.resolve_image_path(sample)

    roi = calculate_phase5_roi(group, roi_padding)
    image = centerline_v1.decode_image(image_path)
    crop = image[roi["ymin"] : roi["ymax"], roi["xmin"] : roi["xmax"], :]
    crop_height = int(roi["ymax"] - roi["ymin"])
    crop_width = int(roi["xmax"] - roi["xmin"])
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(f"crop vazio para {group.get('file_name')}")

    image_size = int(centerline_v1.IMAGE_SIZE)
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


def build_locked_mlp_rows(
    *,
    args: argparse.Namespace,
    locked_mlp: ModuleType,
    tfdata: ModuleType,
    phase2_train: ModuleType,
    phase5_sequence: ModuleType,
    phase5_eval: ModuleType,
    phase6_embeddings: ModuleType,
    postprocess: ModuleType,
    oracle_script: ModuleType,
    pair_reranker: ModuleType,
    residual_calibrator: ModuleType,
    residual_mlp: ModuleType,
    phase9_v1: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    feature_names = residual_calibrator.build_feature_names(pair_reranker)
    mlp_models, mean, std, alpha, max_correction_deg, mlp_lock = locked_mlp.load_locked_mlp(
        mlp_experiment_dir=resolve_project_path(args.mlp_experiment_dir),
        residual_mlp=residual_mlp,
        expected_feature_names=feature_names,
    )

    groups, eval_metadata = locked_mlp.build_test_groups(
        args=args,
        tfdata=tfdata,
        phase2_train=phase2_train,
        phase5_sequence=phase5_sequence,
        phase5_eval=phase5_eval,
        phase6_embeddings=phase6_embeddings,
        postprocess=postprocess,
        oracle_script=oracle_script,
        pair_reranker=pair_reranker,
        phase9_v1=phase9_v1,
    )
    if not groups:
        raise ValueError("Nenhuma imagem com candidatos Phase 5/MLP para avaliacao.")

    print("A construir features MLP v2 e prever residuos...")
    features = np.stack(
        [residual_calibrator.build_image_features(group, pair_reranker) for group in groups]
    ).astype(np.float32)
    scaled_features = ((features - mean) / std).astype(np.float32)
    predicted_residuals = residual_mlp.predict_ensemble(
        mlp_models,
        scaled_features,
        batch_size=max(int(args.prediction_batch_size), 1),
    )
    mlp_metrics, mlp_rows = locked_mlp.evaluate_locked_groups(
        groups=groups,
        predicted_residuals=predicted_residuals,
        alpha=alpha,
        max_correction_deg=max_correction_deg,
        residual_calibrator=residual_calibrator,
    )
    mlp_payload = {
        "model_path": str(resolve_project_path(args.model_path)),
        "profile": args.profile,
        "aux_phase5_model_path": str(resolve_project_path(args.aux_phase5_model_path)),
        "aux_profile": args.aux_profile,
        "phase7_model_path": str(resolve_project_path(args.phase7_model_path)),
        "phase8_model_path": str(resolve_project_path(args.phase8_model_path)),
        "mlp_experiment_dir": str(resolve_project_path(args.mlp_experiment_dir)),
        "mlp_lock": mlp_lock,
        "feature_names": feature_names,
        "metrics": mlp_metrics,
    }
    return groups, mlp_rows, eval_metadata, mlp_payload


def process_centerline_batch_safely(
    *,
    centerline_v1: ModuleType,
    model: tf.keras.Model,
    batch_items: Sequence[tuple[int, Mapping[str, Any], np.ndarray, np.ndarray, dict[str, Any]]],
    threshold: float,
    output_dir: Path,
    save_overlays_enabled: bool,
    overlay_limit: int,
    overlay_count: int,
    errors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    try:
        return centerline_v1.process_batch(
            model=model,
            batch_items=batch_items,
            threshold=threshold,
            output_dir=output_dir,
            save_overlays_enabled=save_overlays_enabled,
            overlay_limit=overlay_limit,
            overlay_count=overlay_count,
        )
    except Exception as exc:
        if len(batch_items) == 1:
            global_index, sample, _normalized, _display_image, _meta = batch_items[0]
            errors.append(
                {
                    "index": global_index,
                    "file_name": sample.get("file_name", ""),
                    "error": f"centerline processing failed: {exc}",
                }
            )
            return [], overlay_count

        rows: list[dict[str, Any]] = []
        for item in batch_items:
            item_rows, overlay_count = process_centerline_batch_safely(
                centerline_v1=centerline_v1,
                model=model,
                batch_items=[item],
                threshold=threshold,
                output_dir=output_dir,
                save_overlays_enabled=save_overlays_enabled,
                overlay_limit=overlay_limit,
                overlay_count=overlay_count,
                errors=errors,
            )
            rows.extend(item_rows)
        return rows, overlay_count


def evaluate_centerline_rows(
    *,
    args: argparse.Namespace,
    groups: Sequence[Mapping[str, Any]],
    centerline_v1: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_path = resolve_project_path(args.colleague_model)
    require_file(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[tuple[int, Mapping[str, Any], np.ndarray, np.ndarray, dict[str, Any]]] = []
    overlay_count = 0
    total = len(groups)
    batch_size = max(int(args.centerline_batch_size), 1)
    last_reported = 0

    for offset, group in enumerate(groups):
        global_index = int(group.get("sample_index", offset))
        sample = group.get("sample")
        if not isinstance(sample, Mapping):
            errors.append(
                {
                    "index": global_index,
                    "file_name": group.get("file_name", ""),
                    "error": "grupo sem sample original",
                }
            )
            continue
        try:
            normalized, display_image, meta = preprocess_group_with_phase5_roi(
                group,
                centerline_v1=centerline_v1,
                roi_padding=int(args.roi_padding),
            )
            pending.append((global_index, sample, normalized, display_image, meta))
        except Exception as exc:
            errors.append(
                {
                    "index": global_index,
                    "file_name": group.get("file_name", ""),
                    "error": str(exc),
                }
            )
            continue

        if len(pending) >= batch_size:
            batch_rows, overlay_count = process_centerline_batch_safely(
                centerline_v1=centerline_v1,
                model=model,
                batch_items=list(pending),
                threshold=float(args.centerline_threshold),
                output_dir=resolve_project_path(args.output_dir),
                save_overlays_enabled=bool(args.save_overlays),
                overlay_limit=int(args.overlay_limit),
                overlay_count=overlay_count,
                errors=errors,
            )
            rows.extend(batch_rows)
            pending = []

        processed = len(rows) + len(errors)
        should_report = (
            args.progress_every > 0
            and processed > last_reported
            and (processed - last_reported >= args.progress_every or processed >= total)
        )
        if should_report:
            print(f"centerline processadas {processed}/{total} imagens")
            last_reported = processed

    if pending:
        batch_rows, overlay_count = process_centerline_batch_safely(
            centerline_v1=centerline_v1,
            model=model,
            batch_items=list(pending),
            threshold=float(args.centerline_threshold),
            output_dir=resolve_project_path(args.output_dir),
            save_overlays_enabled=bool(args.save_overlays),
            overlay_limit=int(args.overlay_limit),
            overlay_count=overlay_count,
            errors=errors,
        )
        rows.extend(batch_rows)
        processed = len(rows) + len(errors)
        print(f"centerline processadas {processed}/{total} imagens")

    return rows, errors


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    centerline_v1: ModuleType,
    mlp_metrics: Mapping[str, Any],
    centerline_metrics: Mapping[str, Mapping[str, Any]],
    eval_metadata: Mapping[str, Any],
    centerline_rows: int,
    centerline_errors: int,
) -> None:
    lines = [
        "# Fusion Centerline Model V3 - Phase 5 ROI",
        "",
        "## Scope",
        "",
        "- ROI source: `phase5_predicted_points`.",
        "- Ground truth is used only for metrics, not for crop/ROI construction.",
        "- This run writes aligned MLP and centerline prediction CSVs for the fusion lock step.",
        "",
        "## Setup",
        "",
        f"- split: `{args.split}`",
        f"- start index: `{args.start_index}`",
        f"- num images requested: `{args.num_images}`",
        f"- selected images: `{eval_metadata.get('num_images_selected')}`",
        f"- groups with candidates: `{eval_metadata.get('num_groups_with_candidates')}`",
        f"- skipped without candidates: `{eval_metadata.get('skipped_no_candidates')}`",
        f"- colleague model: `{resolve_project_path(args.colleague_model)}`",
        f"- roi padding: `{args.roi_padding}`",
        f"- centerline threshold: `{args.centerline_threshold}`",
        f"- centerline rows: `{centerline_rows}`",
        f"- centerline errors: `{centerline_errors}`",
        "",
        "## MLP Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        centerline_v1.metric_row(mlp_metrics["calibrated_mlp_v2_original"], "MLP v2 original"),
        centerline_v1.metric_row(mlp_metrics["geometric_max_angle"], "geometry max-angle"),
        centerline_v1.metric_row(mlp_metrics["endpoint_pair_oracle_same_sequence"], "endpoint-pair oracle"),
        "",
        "## Centerline Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("max_cobb", "PT", "MT", "TLL", "agg3"):
        lines.append(centerline_v1.metric_row(centerline_metrics[label], label))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- MLP predictions: `{path.parent / 'mlp_v2_predictions.csv'}`",
            f"- centerline predictions: `{path.parent / 'colleague_centerline_predictions.csv'}`",
            f"- metrics JSON: `{path.parent / 'centerline_v3_phase5_roi_metrics.json'}`",
            f"- errors CSV: `{path.parent / 'colleague_centerline_errors.csv'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_centerline_only_report(
    path: Path,
    *,
    args: argparse.Namespace,
    centerline_v1: ModuleType,
    selected_count: int,
    processed_count: int,
    skipped_resume_count: int,
    metrics: Mapping[str, Mapping[str, Any]],
    error_count: int,
    existing_mlp_rows: int | None,
) -> None:
    lines = [
        "# Centerline V3 Phase 5 ROI - Fast Resume",
        "",
        "## Scope",
        "",
        "- ROI source: `phase5_predicted_points`.",
        "- Fast mode recomputes only Phase 5 predicted ROIs plus the colleague centerline model.",
        "- Existing MLP predictions are reused for the later fusion step.",
        "",
        "## Setup",
        "",
        f"- split: `{args.split}`",
        f"- start index: `{args.start_index}`",
        f"- num images requested: `{args.num_images}`",
        f"- selected images: `{selected_count}`",
        f"- processed centerline rows: `{processed_count}`",
        f"- skipped by resume: `{skipped_resume_count}`",
        f"- errors: `{error_count}`",
        f"- existing MLP rows: `{existing_mlp_rows if existing_mlp_rows is not None else 'not checked'}`",
        f"- Phase 5 model: `{resolve_project_path(args.model_path)}`",
        f"- Phase 5 profile: `{args.profile}`",
        f"- colleague model: `{resolve_project_path(args.colleague_model)}`",
        f"- roi padding: `{args.roi_padding}`",
        f"- centerline threshold: `{args.centerline_threshold}`",
        "",
        "## Centerline Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("max_cobb", "PT", "MT", "TLL", "agg3"):
        lines.append(centerline_v1.metric_row(metrics[label], label))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- centerline predictions: `{path.parent / 'colleague_centerline_predictions.csv'}`",
            f"- metrics JSON: `{path.parent / 'centerline_v3_phase5_roi_metrics.json'}`",
            f"- errors CSV: `{path.parent / 'colleague_centerline_errors.csv'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_centerline_only_fast(args: argparse.Namespace) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "colleague_centerline_predictions.csv"
    errors_path = output_dir / "colleague_centerline_errors.csv"
    metrics_path = output_dir / "centerline_v3_phase5_roi_metrics.json"
    report_path = output_dir / "centerline_v3_phase5_roi_report.md"

    centerline_v1 = import_script("107_eval_colleague_centerline_model_v1.py", "centerline_v1_utils_for_v3_fast")
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader_for_centerline_v3_fast")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train_for_centerline_v3_fast")
    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval_for_centerline_v3_fast")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval_for_centerline_v3_fast")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess_for_centerline_v3_fast")

    model_path = resolve_project_path(args.model_path)
    colleague_model_path = resolve_project_path(args.colleague_model)
    require_file(model_path)
    require_file(colleague_model_path)
    if args.profile not in phase5_sequence.PROFILE_CONFIGS:
        raise ValueError(f"Perfil desconhecido: {args.profile}")

    existing_mlp_rows: int | None = None
    if args.existing_mlp_predictions:
        mlp_path = resolve_project_path(args.existing_mlp_predictions)
        existing_mlp_rows = len(read_existing_rows(mlp_path))

    existing_rows = read_existing_rows(prediction_path) if args.resume else []
    existing_errors = read_existing_rows(errors_path) if args.resume else []
    processed_file_names = {
        str(row.get("file_name", "")).strip()
        for row in [*existing_rows, *existing_errors]
        if str(row.get("file_name", "")).strip()
    }
    if not args.resume and (prediction_path.exists() or errors_path.exists()):
        raise ValueError(
            "Ja existem ficheiros centerline neste output-dir. "
            "Use --resume para continuar ou escolha outro --output-dir."
        )

    samples, image_paths = phase2_train.load_split_samples_and_paths(tfdata, args.split)
    selected_samples = list(select_window(samples, int(args.start_index), int(args.num_images)))
    selected_paths = list(select_window(image_paths, int(args.start_index), int(args.num_images)))
    if not selected_samples:
        raise ValueError("Nenhuma imagem selecionada para avaliacao.")

    remaining_samples: list[Mapping[str, Any]] = []
    remaining_paths: list[Path] = []
    remaining_indices: list[int] = []
    for offset, (sample, image_path) in enumerate(zip(selected_samples, selected_paths)):
        file_name = str(sample.get("file_name", "")).strip()
        if file_name in processed_file_names:
            continue
        remaining_samples.append(sample)
        remaining_paths.append(image_path)
        remaining_indices.append(int(args.start_index + offset))

    print("\n===== CENTERLINE V3 PHASE5 ROI FAST =====")
    print(f"Split: {args.split}")
    print(f"Start index: {args.start_index}")
    print(f"Selected images: {len(selected_samples)}")
    print(f"Already processed by resume: {len(processed_file_names)}")
    print(f"Remaining images: {len(remaining_samples)}")
    print(f"ROI source: {ROI_SOURCE}")
    print(f"Output: {output_dir}")

    main_config = dict(phase5_sequence.PROFILE_CONFIGS[args.profile])
    print(f"A carregar modelo Fase 5 principal: {model_path}")
    main_model = phase5_eval.load_spatial_offset_model_for_eval(model_path)
    print(f"A carregar modelo centerline colega: {colleague_model_path}")
    centerline_model = tf.keras.models.load_model(colleague_model_path, compile=False)

    rows: list[dict[str, Any]] = [dict(row) for row in existing_rows]
    errors: list[dict[str, Any]] = [dict(row) for row in existing_errors]
    overlay_count = 0
    pending: list[tuple[int, Mapping[str, Any], np.ndarray, np.ndarray, dict[str, Any]]] = []
    centerline_batch_size = max(int(args.centerline_batch_size), 1)
    processed_total = len(processed_file_names)
    last_reported = processed_total

    dataset = tfdata.build_dataset(
        remaining_samples,
        remaining_paths,
        batch_size=max(int(args.inference_batch_size), 1),
        shuffle=False,
        cache=False,
    )

    consumed = 0
    for images, _targets in dataset:
        batch_size = int(images.shape[0])
        main_predictions = main_model(images, training=False)
        presence_batch = main_predictions["presence"].numpy()
        bbox_batch = main_predictions["bbox"].numpy()
        points_batch = main_predictions["points"].numpy()
        cobb_endpoint_batch = None
        if isinstance(main_predictions, dict) and "cobb_endpoint_score" in main_predictions:
            cobb_endpoint_batch = main_predictions["cobb_endpoint_score"].numpy()

        for batch_index in range(batch_size):
            sample = remaining_samples[consumed + batch_index]
            image_path = remaining_paths[consumed + batch_index]
            global_index = remaining_indices[consumed + batch_index]
            file_name = str(sample["file_name"])
            try:
                result = postprocess.postprocess_candidates_sequence(
                    presence=presence_batch[batch_index],
                    bbox=bbox_batch[batch_index],
                    points=points_batch[batch_index],
                    cobb_endpoint_score=None if cobb_endpoint_batch is None else cobb_endpoint_batch[batch_index],
                    **main_config,
                )
                selected_points = np.asarray(result["selected_points"], dtype=np.float32).reshape(-1, 8)
                selected_scores = np.asarray(result["selected_scores"], dtype=np.float32).reshape(-1)
                group = {
                    "file_name": file_name,
                    "sample": sample,
                    "image_path": str(image_path),
                    "sample_index": global_index,
                    "image_width": float(sample.get("width", tfdata.IMAGE_SIZE[1])),
                    "image_height": float(sample.get("height", tfdata.IMAGE_SIZE[0])),
                    "selected_points": selected_points.astype(np.float32).tolist(),
                    "selected_scores": selected_scores.astype(np.float32).tolist(),
                }
                normalized, display_image, meta = preprocess_group_with_phase5_roi(
                    group,
                    centerline_v1=centerline_v1,
                    roi_padding=int(args.roi_padding),
                )
                pending.append((global_index, sample, normalized, display_image, meta))
            except Exception as exc:
                error_row = {
                    "index": global_index,
                    "file_name": file_name,
                    "error": str(exc),
                }
                errors.append(error_row)
                append_csv_rows(errors_path, [error_row], ["index", "file_name", "error"])
                processed_total += 1

            if len(pending) >= centerline_batch_size:
                batch_rows, overlay_count = process_centerline_batch_safely(
                    centerline_v1=centerline_v1,
                    model=centerline_model,
                    batch_items=list(pending),
                    threshold=float(args.centerline_threshold),
                    output_dir=output_dir,
                    save_overlays_enabled=bool(args.save_overlays),
                    overlay_limit=int(args.overlay_limit),
                    overlay_count=overlay_count,
                    errors=errors,
                )
                rows.extend(batch_rows)
                append_csv_rows(prediction_path, batch_rows, CENTERLINE_FIELDNAMES)
                pending = []
                processed_total = len(rows) + len(errors)

            should_report = (
                args.progress_every > 0
                and processed_total > last_reported
                and (processed_total - last_reported >= args.progress_every or processed_total >= len(selected_samples))
            )
            if should_report:
                print(f"centerline fast processadas {processed_total}/{len(selected_samples)} imagens")
                last_reported = processed_total

        consumed += batch_size

    if pending:
        batch_rows, overlay_count = process_centerline_batch_safely(
            centerline_v1=centerline_v1,
            model=centerline_model,
            batch_items=list(pending),
            threshold=float(args.centerline_threshold),
            output_dir=output_dir,
            save_overlays_enabled=bool(args.save_overlays),
            overlay_limit=int(args.overlay_limit),
            overlay_count=overlay_count,
            errors=errors,
        )
        rows.extend(batch_rows)
        append_csv_rows(prediction_path, batch_rows, CENTERLINE_FIELDNAMES)
        pending = []

    processed_total = len(rows) + len(errors)
    if processed_total != last_reported:
        print(f"centerline fast processadas {processed_total}/{len(selected_samples)} imagens")

    metrics = centerline_v1.build_metrics(rows)
    payload = {
        "phase": "centerline_model_v3_phase5_roi_fast",
        "roi_source": ROI_SOURCE,
        "leakage_guard": {
            "roi_uses_annotation_points": False,
            "gt_used_for_metrics_only": True,
        },
        "config": vars(args),
        "selected_images": len(selected_samples),
        "processed_rows": len(rows),
        "errors": errors,
        "skipped_by_resume": len(processed_file_names),
        "existing_mlp_rows": existing_mlp_rows,
        "metrics": metrics,
    }
    write_json(metrics_path, payload)
    write_centerline_only_report(
        report_path,
        args=args,
        centerline_v1=centerline_v1,
        selected_count=len(selected_samples),
        processed_count=len(rows),
        skipped_resume_count=len(processed_file_names),
        metrics=metrics,
        error_count=len(errors),
        existing_mlp_rows=existing_mlp_rows,
    )

    print("\n===== METRICS =====")
    centerline_metric = metrics["max_cobb"]
    if int(centerline_metric.get("num_images", 0)) == 0:
        print("centerline max_cobb: sem predicoes validas")
    else:
        print(
            "centerline max_cobb: "
            f"MAE={centerline_metric['mae_deg']:.3f}, "
            f"SMAPE={centerline_metric['paper_smape_pct']:.4f}%, "
            f"within5={centerline_metric['within_5deg_rate']:.4f}, "
            f"falhas>5={centerline_metric['failures_gt5']}"
        )
    print(f"Centerline predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print("===== CONCLUIDO =====")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia centerline V3 com ROI previsto pela Phase 5, sem ROI por GT."
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start-index", type=int, default=12768)
    parser.add_argument("--num-images", type=int, default=3192)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--aux-phase5-model-path", default=str(DEFAULT_AUX_PHASE5_MODEL))
    parser.add_argument("--aux-profile", default=DEFAULT_AUX_PROFILE)
    parser.add_argument("--phase7-model-path", default=str(DEFAULT_PHASE7_MODEL))
    parser.add_argument("--phase8-model-path", default=str(DEFAULT_PHASE8_MODEL))
    parser.add_argument("--mlp-experiment-dir", default=str(DEFAULT_MLP_EXPERIMENT))
    parser.add_argument("--colleague-model", default=str(DEFAULT_COLLEAGUE_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--centerline-batch-size", type=int, default=8)
    parser.add_argument("--centerline-threshold", type=float, default=0.5)
    parser.add_argument("--roi-padding", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=20)
    parser.add_argument(
        "--centerline-only-fast",
        action="store_true",
        help="Recalcula apenas ROIs Phase 5 + centerline, reutilizando CSV MLP existente para a fusao posterior.",
    )
    parser.add_argument(
        "--existing-mlp-predictions",
        default="",
        help="CSV MLP ja existente usado apenas para sanity check no modo --centerline-only-fast.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No modo --centerline-only-fast, continua a partir de CSV centerline/errors ja existentes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if args.centerline_batch_size <= 0:
        raise ValueError("--centerline-batch-size deve ser > 0")
    if args.overlay_limit < 0:
        raise ValueError("--overlay-limit deve ser >= 0")

    tf.keras.utils.set_random_seed(42)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.centerline_only_fast:
        run_centerline_only_fast(args)
        return

    locked_mlp = import_script("106_eval_final_test_subset5_mlp_v2_locked.py", "locked_mlp_v2_for_centerline_v3")
    centerline_v1 = import_script("107_eval_colleague_centerline_model_v1.py", "centerline_v1_utils_for_v3")
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader_for_centerline_v3")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train_for_centerline_v3")
    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval_for_centerline_v3")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval_for_centerline_v3")
    phase6_embeddings = import_script("24_vertebral_embeddings_v1.py", "phase6_embeddings_for_centerline_v3")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess_for_centerline_v3")
    oracle_script = import_script("56_oracle_phase5_sequence_portfolio_cobb_v1.py", "phase5_cobb_oracle_for_centerline_v3")
    pair_reranker = import_script("57_train_phase9_endpoint_pair_reranker_v1.py", "phase9_pair_reranker_for_centerline_v3")
    residual_calibrator = import_script("59_train_phase9_cobb_residual_calibrator_v1.py", "phase9_residual_calibrator_for_centerline_v3")
    residual_mlp = import_script("66_train_phase9_cobb_residual_mlp_v2.py", "phase9_residual_mlp_v2_for_centerline_v3")
    phase9_v1 = import_script("32_eval_phase9_final_cobb_v1.py", "phase9_v1_for_centerline_v3")

    print("\n===== CENTERLINE V3 PHASE5 ROI =====")
    print(f"Split: {args.split}")
    print(f"Start index: {args.start_index}")
    print(f"Num images: {args.num_images}")
    print(f"ROI source: {ROI_SOURCE}")
    print(f"Output: {output_dir}")

    groups, mlp_rows, eval_metadata, mlp_payload = build_locked_mlp_rows(
        args=args,
        locked_mlp=locked_mlp,
        tfdata=tfdata,
        phase2_train=phase2_train,
        phase5_sequence=phase5_sequence,
        phase5_eval=phase5_eval,
        phase6_embeddings=phase6_embeddings,
        postprocess=postprocess,
        oracle_script=oracle_script,
        pair_reranker=pair_reranker,
        residual_calibrator=residual_calibrator,
        residual_mlp=residual_mlp,
        phase9_v1=phase9_v1,
    )

    aux_rows = list(eval_metadata.pop("aux_rows", []))
    if aux_rows:
        write_csv(output_dir / "mlp_aux_predictions.csv", aux_rows, list(aux_rows[0].keys()))
    if mlp_rows:
        write_csv(output_dir / "mlp_v2_predictions.csv", mlp_rows, list(mlp_rows[0].keys()))

    centerline_rows, centerline_errors = evaluate_centerline_rows(
        args=args,
        groups=groups,
        centerline_v1=centerline_v1,
    )
    centerline_metrics = centerline_v1.build_metrics(centerline_rows)

    prediction_path = output_dir / "colleague_centerline_predictions.csv"
    errors_path = output_dir / "colleague_centerline_errors.csv"
    metrics_path = output_dir / "centerline_v3_phase5_roi_metrics.json"
    report_path = output_dir / "centerline_v3_phase5_roi_report.md"

    if centerline_rows:
        write_csv(prediction_path, centerline_rows, preferred_centerline_fieldnames(centerline_rows))
    else:
        write_csv(prediction_path, [], ["index", "file_name"])
    if centerline_errors:
        write_csv(errors_path, centerline_errors, ["index", "file_name", "error"])
    else:
        write_csv(errors_path, [], ["index", "file_name", "error"])

    payload = {
        "phase": "centerline_model_v3_phase5_roi",
        "roi_source": ROI_SOURCE,
        "leakage_guard": {
            "roi_uses_annotation_points": False,
            "gt_used_for_metrics_only": True,
        },
        "config": vars(args),
        "eval": eval_metadata,
        "mlp": mlp_payload,
        "centerline": {
            "colleague_model": str(resolve_project_path(args.colleague_model)),
            "rows": len(centerline_rows),
            "errors": centerline_errors,
            "threshold": float(args.centerline_threshold),
            "roi_padding": int(args.roi_padding),
            "metrics": centerline_metrics,
        },
    }
    write_json(metrics_path, payload)
    write_report(
        report_path,
        args=args,
        centerline_v1=centerline_v1,
        mlp_metrics=mlp_payload["metrics"],
        centerline_metrics=centerline_metrics,
        eval_metadata=eval_metadata,
        centerline_rows=len(centerline_rows),
        centerline_errors=len(centerline_errors),
    )

    print("\n===== METRICS =====")
    mlp_metric = mlp_payload["metrics"]["calibrated_mlp_v2_original"]
    print(
        "MLP v2 original: "
        f"MAE={mlp_metric['mae_deg']:.3f}, "
        f"SMAPE={mlp_metric['paper_smape_pct']:.4f}%, "
        f"within5={mlp_metric['within_5deg_rate']:.4f}, "
        f"falhas>5={mlp_metric['failures_gt5']}"
    )
    centerline_metric = centerline_metrics["max_cobb"]
    if int(centerline_metric.get("num_images", 0)) == 0:
        print("centerline max_cobb: sem predicoes validas")
    else:
        print(
            "centerline max_cobb: "
            f"MAE={centerline_metric['mae_deg']:.3f}, "
            f"SMAPE={centerline_metric['paper_smape_pct']:.4f}%, "
            f"within5={centerline_metric['within_5deg_rate']:.4f}, "
            f"falhas>5={centerline_metric['failures_gt5']}"
        )
    print(f"MLP predictions: {output_dir / 'mlp_v2_predictions.csv'}")
    print(f"Centerline predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
