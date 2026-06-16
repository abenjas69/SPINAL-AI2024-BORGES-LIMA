"""Final locked test-set evaluation for the current Phase 9 MLP v2 pipeline.

This script evaluates the already selected operational pipeline on the cleaned
Spinal-AI2024 test split. It does not train, tune, sweep alpha, or select any
threshold using test-set results.

Locked pipeline:
- Phase 5 hardmining sequence model/profile for Cobb candidates.
- Endpoint-safe Phase 6/7/8 auxiliary angle path to reproduce `aux_cobb`.
- Phase 9 residual MLP v2 with the alpha selected on the internal calibration
  split.
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
REPORTS_DIR = PROJECT_ROOT / "sanity_check" / "reports"

DEFAULT_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_AUX_PHASE5_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_hard_negative_fulltrain_probe_v1_colab.keras"
DEFAULT_AUX_PROFILE = "anatomical_endpoint_safe_v1"
DEFAULT_PHASE7_MODEL = MODELS_DIR / "phase7_bilstm_global_endpoint_safe_v1.keras"
DEFAULT_PHASE8_MODEL = MODELS_DIR / "phase8_aux_angle_head_weighted_endpoint_safe_v2.keras"
DEFAULT_MLP_EXPERIMENT = EXPERIMENTS_DIR / "phase9_cobb_residual_mlp_v2_train12000_cal768_val3192"
DEFAULT_EXPERIMENT_NAME = "final_test_subset5_mlp_v2_locked"
DEFAULT_REPORT = REPORTS_DIR / f"{DEFAULT_EXPERIMENT_NAME}.md"
ANGLE_NAMES = ("PT", "MT", "TLL")


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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def select_window(items: Sequence[Any], start_index: int, num_items: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_items <= 0:
        return items[start_index:]
    return items[start_index : start_index + num_items]


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
        where=denominator > 1.0e-8,
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
    result["r2"] = float(1.0 - float(np.sum((gt - pred) ** 2)) / total) if total > 1.0e-8 else None
    return result


def metric_row(metrics: Mapping[str, Any], label: str) -> str:
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


def tensor_batch_item(outputs: Mapping[str, tf.Tensor], index: int) -> dict[str, tf.Tensor]:
    return {key: value[index : index + 1] for key, value in outputs.items()}


def angle_predictions_from_outputs(model: tf.keras.Model, outputs: Any) -> np.ndarray:
    if isinstance(outputs, dict):
        return np.asarray(outputs["angle_deg"], dtype=np.float32)
    output_values = outputs if isinstance(outputs, list) else [outputs]
    output_map = dict(zip(model.output_names, output_values))
    return np.asarray(output_map["angle_deg"], dtype=np.float32)


def load_locked_mlp(
    *,
    mlp_experiment_dir: Path,
    residual_mlp: ModuleType,
    expected_feature_names: Sequence[str],
) -> tuple[list[tf.keras.Model], np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    metrics_path = mlp_experiment_dir / "phase9_cobb_residual_mlp_metrics.json"
    scaler_path = mlp_experiment_dir / "phase9_cobb_residual_mlp_scaler.npz"
    require_file(metrics_path)
    require_file(scaler_path)

    with metrics_path.open("r", encoding="utf-8") as file:
        mlp_metrics = json.load(file)
    with np.load(scaler_path, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        scaler_feature_names = [str(value) for value in data["feature_names"].tolist()]
        seed_list = [int(value) for value in np.asarray(data.get("seed_list", []), dtype=np.int32).reshape(-1)]

    if not seed_list:
        seed_list = [int(value) for value in mlp_metrics.get("seed_list", [42])]

    if scaler_feature_names != list(expected_feature_names):
        raise ValueError("O scaler do MLP v2 nao corresponde as features esperadas.")
    if mean.reshape(-1).shape[0] != len(expected_feature_names):
        raise ValueError("Dimensao do scaler do MLP v2 invalida.")

    models: list[tf.keras.Model] = []
    for seed in seed_list:
        model_path = mlp_experiment_dir / f"phase9_cobb_residual_mlp_seed{seed}.keras"
        require_file(model_path)
        models.append(tf.keras.models.load_model(model_path, compile=False))

    alpha = float(mlp_metrics["selected_alpha"])
    max_correction_deg = float(mlp_metrics["max_correction_deg"])
    summary = {
        "metrics_path": str(metrics_path),
        "scaler_path": str(scaler_path),
        "seed_list": seed_list,
        "selected_alpha": alpha,
        "max_correction_deg": max_correction_deg,
        "internal_holdout": mlp_metrics.get("holdout", {}),
    }
    return models, mean.reshape(1, -1), std.reshape(1, -1), alpha, max_correction_deg, summary


def build_test_groups(
    *,
    args: argparse.Namespace,
    tfdata: ModuleType,
    phase2_train: ModuleType,
    phase5_sequence: ModuleType,
    phase5_eval: ModuleType,
    phase6_embeddings: ModuleType,
    postprocess: ModuleType,
    oracle_script: ModuleType,
    pair_reranker: ModuleType,
    phase9_v1: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_path = resolve_project_path(args.model_path)
    aux_phase5_model_path = resolve_project_path(args.aux_phase5_model_path)
    phase7_model_path = resolve_project_path(args.phase7_model_path)
    phase8_model_path = resolve_project_path(args.phase8_model_path)
    for path in (model_path, aux_phase5_model_path, phase7_model_path, phase8_model_path):
        require_file(path)

    if args.profile not in phase5_sequence.PROFILE_CONFIGS:
        raise ValueError(f"Perfil desconhecido: {args.profile}")
    if args.aux_profile not in phase5_sequence.PROFILE_CONFIGS:
        raise ValueError(f"Perfil aux desconhecido: {args.aux_profile}")

    samples, image_paths = phase2_train.load_split_samples_and_paths(tfdata, args.split)
    selected_samples = list(select_window(samples, int(args.start_index), int(args.num_images)))
    selected_paths = list(select_window(image_paths, int(args.start_index), int(args.num_images)))
    if not selected_samples:
        raise ValueError("Nenhuma imagem selecionada para avaliacao.")
    annotations_path = tfdata.TRAIN_JSON if args.split == "train" else tfdata.TEST_JSON
    eval_metadata = {
        "eval_mode": "window",
        "split": args.split,
        "absolute_start_index": int(args.start_index),
        "relative_start_index": int(args.start_index),
        "num_images_selected": int(len(selected_samples)),
        "annotations_path": str(annotations_path),
    }

    print(f"A carregar modelo Fase 5 principal: {model_path}")
    main_model = phase5_eval.load_spatial_offset_model_for_eval(model_path)
    print(f"A carregar modelo Fase 5 aux endpoint-safe: {aux_phase5_model_path}")
    aux_phase5_model = phase5_eval.load_spatial_offset_model_for_eval(aux_phase5_model_path)
    aux_feature_model = phase6_embeddings.build_feature_prediction_model(aux_phase5_model)

    print(f"A carregar Fase 7 aux: {phase7_model_path}")
    phase7_models = import_script("25_bilstm_global_model_v1.py", "phase7_models_for_final_test_loading")
    phase7_model = tf.keras.models.load_model(phase7_model_path, compile=False)
    phase7_context_model = phase7_models.build_context_extractor(phase7_model)

    print(f"A carregar Fase 8 aux: {phase8_model_path}")
    import_script("29_aux_angle_head_v1.py", "phase8_models_for_final_test_loading")
    phase8_model = tf.keras.models.load_model(phase8_model_path, compile=False)

    dataset = tfdata.build_dataset(
        selected_samples,
        selected_paths,
        batch_size=max(int(args.inference_batch_size), 1),
        shuffle=False,
        cache=False,
    )

    groups: list[dict[str, Any]] = []
    aux_rows: list[dict[str, Any]] = []
    skipped_no_candidates = 0
    processed = 0
    main_config = dict(phase5_sequence.PROFILE_CONFIGS[args.profile])
    aux_config = dict(phase5_sequence.PROFILE_CONFIGS[args.aux_profile])

    print(
        "A avaliar test set bloqueado: "
        f"imagens={len(selected_samples)}, batch={args.inference_batch_size}, "
        f"perfil={args.profile}"
    )
    for images, _targets in dataset:
        batch_size = int(images.shape[0])

        aux_feature_outputs = aux_feature_model(images, training=False)
        aux_embeddings: list[np.ndarray] = []
        aux_masks: list[np.ndarray] = []
        aux_counts: list[int] = []
        for batch_index in range(batch_size):
            sample = selected_samples[processed + batch_index]
            targets = tfdata.sample_to_targets(sample)
            arrays, _record = phase6_embeddings.extract_sequence_for_image(
                image=images[batch_index],
                sample=sample,
                targets=targets,
                feature_outputs=tensor_batch_item(aux_feature_outputs, batch_index),
                postprocess=postprocess,
                postprocess_config=aux_config,
            )
            aux_embeddings.append(arrays["embeddings"])
            aux_masks.append(arrays["mask"])
            aux_counts.append(int(np.sum(arrays["mask"] > 0.5)))

        aux_embedding_array = np.stack(aux_embeddings).astype(np.float32)
        aux_mask_array = np.stack(aux_masks).astype(np.float32)
        contextual = phase7_context_model(
            {
                "embeddings": tf.convert_to_tensor(aux_embedding_array, dtype=tf.float32),
                "mask": tf.convert_to_tensor(aux_mask_array, dtype=tf.float32),
            },
            training=False,
        )
        aux_outputs = phase8_model(
            {
                "contextual_embeddings": contextual,
                "mask": tf.convert_to_tensor(aux_mask_array, dtype=tf.float32),
            },
            training=False,
        )
        aux_angles_batch = angle_predictions_from_outputs(phase8_model, aux_outputs)
        aux_cobb_batch = np.max(aux_angles_batch[:, :3], axis=1).astype(np.float32)

        main_predictions = main_model(images, training=False)
        presence_batch = main_predictions["presence"].numpy()
        bbox_batch = main_predictions["bbox"].numpy()
        points_batch = main_predictions["points"].numpy()
        cobb_endpoint_batch = None
        if isinstance(main_predictions, dict) and "cobb_endpoint_score" in main_predictions:
            cobb_endpoint_batch = main_predictions["cobb_endpoint_score"].numpy()

        for batch_index in range(batch_size):
            sample_index = processed + batch_index
            sample = selected_samples[sample_index]
            targets = tfdata.sample_to_targets(sample)
            file_name = str(sample["file_name"])
            gt_cobb, gt_major_region = oracle_script.cobb_max_from_targets(targets)
            gt_angles = np.asarray(targets["cobb_angles"], dtype=np.float32).reshape(-1)
            width = float(sample.get("width", tfdata.IMAGE_SIZE[1]))
            height = float(sample.get("height", tfdata.IMAGE_SIZE[0]))

            result = postprocess.postprocess_candidates_sequence(
                presence=presence_batch[batch_index],
                bbox=bbox_batch[batch_index],
                points=points_batch[batch_index],
                cobb_endpoint_score=None if cobb_endpoint_batch is None else cobb_endpoint_batch[batch_index],
                **main_config,
            )
            selected_points = np.asarray(result["selected_points"], dtype=np.float32).reshape(-1, 8)
            selected_scores = np.asarray(result["selected_scores"], dtype=np.float32).reshape(-1)
            aux_angles = aux_angles_batch[batch_index].astype(np.float32)
            aux_cobb = float(aux_cobb_batch[batch_index])
            candidates = pair_reranker.build_pair_candidates(
                phase9_v1=phase9_v1,
                file_name=file_name,
                points=selected_points,
                scores=selected_scores,
                width=width,
                height=height,
                gt_cobb=gt_cobb,
                aux_cobb=aux_cobb,
            )
            aux_rows.append(
                {
                    "file_name": file_name,
                    "aux_PT": f"{float(aux_angles[0]):.6f}",
                    "aux_MT": f"{float(aux_angles[1]):.6f}",
                    "aux_TLL": f"{float(aux_angles[2]):.6f}",
                    "aux_cobb_max": f"{aux_cobb:.6f}",
                    "aux_count": int(aux_counts[batch_index]),
                }
            )
            if not candidates:
                skipped_no_candidates += 1
                continue
            groups.append(
                {
                    "file_name": file_name,
                    "sample": sample,
                    "image_path": str(selected_paths[sample_index]),
                    "sample_index": int(args.start_index + sample_index),
                    "image_width": width,
                    "image_height": height,
                    "gt_cobb": float(gt_cobb),
                    "gt_major_region": gt_major_region,
                    "gt_PT": float(gt_angles[0]),
                    "gt_MT": float(gt_angles[1]),
                    "gt_TLL": float(gt_angles[2]),
                    "aux_cobb": aux_cobb,
                    "aux_available": bool(np.isfinite(aux_cobb)),
                    "aux_PT": float(aux_angles[0]),
                    "aux_MT": float(aux_angles[1]),
                    "aux_TLL": float(aux_angles[2]),
                    "aux_count": int(aux_counts[batch_index]),
                    "main_count": int(selected_points.shape[0]),
                    "main_mean_score": float(np.mean(selected_scores)) if selected_scores.size else 0.0,
                    "selected_points": selected_points.astype(np.float32).tolist(),
                    "selected_scores": selected_scores.astype(np.float32).tolist(),
                    "candidates": candidates,
                }
            )

        processed += batch_size
        if args.progress_every > 0 and (processed % args.progress_every == 0 or processed >= len(selected_samples)):
            print(f"processadas {min(processed, len(selected_samples))}/{len(selected_samples)} imagens")

    eval_metadata["num_groups_with_candidates"] = int(len(groups))
    eval_metadata["skipped_no_candidates"] = int(skipped_no_candidates)
    eval_metadata["aux_rows"] = aux_rows
    return groups, eval_metadata


def evaluate_locked_groups(
    *,
    groups: Sequence[Mapping[str, Any]],
    predicted_residuals: np.ndarray,
    alpha: float,
    max_correction_deg: float,
    residual_calibrator: ModuleType,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_values: list[float] = []
    geom_values: list[float] = []
    calibrated_values: list[float] = []
    oracle_values: list[float] = []
    aux_values: list[float] = []
    rows: list[dict[str, Any]] = []
    corrected_count = 0
    rescued_gt5 = 0
    broken_gt5 = 0
    rescued_gt10 = 0
    broken_gt10 = 0
    improved_ge3 = 0
    worsened_ge3 = 0

    for group, predicted_residual in zip(groups, predicted_residuals):
        candidates = list(group["candidates"])
        geom = residual_calibrator.select_pair_by_flag(candidates, "is_geom_pair")
        oracle = residual_calibrator.select_pair_by_flag(candidates, "is_oracle_pair")
        gt = float(group["gt_cobb"])
        geom_cobb = float(geom["angle_deg"])
        correction = float(
            np.clip(
                float(alpha) * float(predicted_residual),
                -float(max_correction_deg),
                float(max_correction_deg),
            )
        )
        calibrated_cobb = geom_cobb + correction
        oracle_cobb = float(oracle["angle_deg"])
        aux_cobb = finite_float(group.get("aux_cobb", np.nan))
        geom_error = abs(geom_cobb - gt)
        calibrated_error = abs(calibrated_cobb - gt)
        error_delta = calibrated_error - geom_error

        if abs(correction) >= 0.5:
            corrected_count += 1
        if geom_error > 5.0 and calibrated_error <= 5.0:
            rescued_gt5 += 1
        if geom_error <= 5.0 and calibrated_error > 5.0:
            broken_gt5 += 1
        if geom_error > 10.0 and calibrated_error <= 10.0:
            rescued_gt10 += 1
        if geom_error <= 10.0 and calibrated_error > 10.0:
            broken_gt10 += 1
        if error_delta <= -3.0:
            improved_ge3 += 1
        if error_delta >= 3.0:
            worsened_ge3 += 1

        gt_values.append(gt)
        geom_values.append(geom_cobb)
        calibrated_values.append(calibrated_cobb)
        oracle_values.append(oracle_cobb)
        aux_values.append(aux_cobb)
        rows.append(
            {
                "file_name": str(group["file_name"]),
                "sample_index": int(group.get("sample_index", -1)),
                "gt_PT": f"{float(group.get('gt_PT', np.nan)):.6f}",
                "gt_MT": f"{float(group.get('gt_MT', np.nan)):.6f}",
                "gt_TLL": f"{float(group.get('gt_TLL', np.nan)):.6f}",
                "gt_cobb_max": f"{gt:.6f}",
                "gt_major_region": str(group.get("gt_major_region", "")),
                "main_count": int(group.get("main_count", 0)),
                "main_mean_score": f"{float(group.get('main_mean_score', 0.0)):.6f}",
                "aux_count": int(group.get("aux_count", 0)),
                "aux_PT": f"{float(group.get('aux_PT', np.nan)):.6f}",
                "aux_MT": f"{float(group.get('aux_MT', np.nan)):.6f}",
                "aux_TLL": f"{float(group.get('aux_TLL', np.nan)):.6f}",
                "aux_cobb": f"{aux_cobb:.6f}" if np.isfinite(aux_cobb) else "",
                "geom_cobb": f"{geom_cobb:.6f}",
                "geom_abs_error": f"{geom_error:.6f}",
                "calibrated_cobb": f"{calibrated_cobb:.6f}",
                "calibrated_abs_error": f"{calibrated_error:.6f}",
                "error_delta_deg": f"{error_delta:.6f}",
                "predicted_residual_deg": f"{float(predicted_residual):.6f}",
                "applied_correction_deg": f"{correction:.6f}",
                "oracle_cobb": f"{oracle_cobb:.6f}",
                "oracle_abs_error": f"{abs(oracle_cobb - gt):.6f}",
                "rescued_gt5": int(geom_error > 5.0 and calibrated_error <= 5.0),
                "broken_gt5": int(geom_error <= 5.0 and calibrated_error > 5.0),
                "rescued_gt10": int(geom_error > 10.0 and calibrated_error <= 10.0),
                "broken_gt10": int(geom_error <= 10.0 and calibrated_error > 10.0),
                "improved_ge3": int(error_delta <= -3.0),
                "worsened_ge3": int(error_delta >= 3.0),
                "geom_upper_index": int(geom["upper_index"]),
                "geom_lower_index": int(geom["lower_index"]),
                "geom_span": int(geom["span"]),
                "oracle_upper_index": int(oracle["upper_index"]),
                "oracle_lower_index": int(oracle["lower_index"]),
                "oracle_span": int(oracle["span"]),
                "candidate_count": int(len(candidates)),
            }
        )

    metrics = {
        "alpha": float(alpha),
        "geometric_max_angle": metric_bundle(geom_values, gt_values),
        "calibrated_mlp_v2_original": metric_bundle(calibrated_values, gt_values),
        "endpoint_pair_oracle_same_sequence": metric_bundle(oracle_values, gt_values),
        "auxiliary_head_max": metric_bundle(aux_values, gt_values),
        "correction_rate": float(corrected_count / max(len(groups), 1)),
        "corrected_count": int(corrected_count),
        "rescued_gt5": int(rescued_gt5),
        "broken_gt5": int(broken_gt5),
        "rescued_gt10": int(rescued_gt10),
        "broken_gt10": int(broken_gt10),
        "improved_ge3": int(improved_ge3),
        "worsened_ge3": int(worsened_ge3),
    }
    return metrics, rows


def make_report(
    *,
    path: Path,
    payload: Mapping[str, Any],
    experiment_dir: Path,
    prediction_path: Path,
) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Final test subset5 - MLP v2 original locked",
        "",
        "## Scope",
        "",
        "Evaluation-only run on the cleaned `test` split. No training, alpha sweep, threshold tuning, or model selection was run on these results.",
        "",
        "## Locked artefacts",
        "",
        f"- Phase 5 model: `{payload['model_path']}`",
        f"- Phase 5 profile: `{payload['profile']}`",
        f"- Aux Phase 5 model: `{payload['aux_phase5_model_path']}`",
        f"- Aux profile: `{payload['aux_profile']}`",
        f"- Phase 7 model: `{payload['phase7_model_path']}`",
        f"- Phase 8 model: `{payload['phase8_model_path']}`",
        f"- MLP v2 experiment: `{payload['mlp_experiment_dir']}`",
        f"- selected alpha: `{payload['mlp_lock']['selected_alpha']:.3f}`",
        f"- max correction: `{payload['mlp_lock']['max_correction_deg']:.3f}` deg",
        "",
        "## Test split",
        "",
        f"- selected images: `{payload['eval']['num_images_selected']}`",
        f"- images with candidates: `{payload['eval']['num_groups_with_candidates']}`",
        f"- skipped without candidates: `{payload['eval']['skipped_no_candidates']}`",
        "",
        "## Results",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row(metrics["geometric_max_angle"], "geometry max-angle"),
        metric_row(metrics["calibrated_mlp_v2_original"], "MLP v2 original"),
        metric_row(metrics["auxiliary_head_max"], "auxiliary head max"),
        metric_row(metrics["endpoint_pair_oracle_same_sequence"], "endpoint-pair oracle"),
        "",
        "## MLP effect",
        "",
        f"- corrections >=0.5 deg: `{metrics['corrected_count']}`",
        f"- rescued >5: `{metrics['rescued_gt5']}`",
        f"- broken >5: `{metrics['broken_gt5']}`",
        f"- rescued >10: `{metrics['rescued_gt10']}`",
        f"- broken >10: `{metrics['broken_gt10']}`",
        f"- improved >=3 deg: `{metrics['improved_ge3']}`",
        f"- worsened >=3 deg: `{metrics['worsened_ge3']}`",
        "",
        "## Artefacts",
        "",
        f"- metrics JSON: `{experiment_dir / 'final_test_subset5_mlp_v2_metrics.json'}`",
        f"- predictions CSV: `{prediction_path}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o MLP v2 original no test subset5 sem tuning.")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--aux-phase5-model-path", default=str(DEFAULT_AUX_PHASE5_MODEL))
    parser.add_argument("--aux-profile", default=DEFAULT_AUX_PROFILE)
    parser.add_argument("--phase7-model-path", default=str(DEFAULT_PHASE7_MODEL))
    parser.add_argument("--phase8-model-path", default=str(DEFAULT_PHASE8_MODEL))
    parser.add_argument("--mlp-experiment-dir", default=str(DEFAULT_MLP_EXPERIMENT))
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(42)

    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    phase6_embeddings = import_script("24_vertebral_embeddings_v1.py", "phase6_embeddings")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    oracle_script = import_script("56_oracle_phase5_sequence_portfolio_cobb_v1.py", "phase5_cobb_oracle")
    pair_reranker = import_script("57_train_phase9_endpoint_pair_reranker_v1.py", "phase9_pair_reranker")
    residual_calibrator = import_script("59_train_phase9_cobb_residual_calibrator_v1.py", "phase9_residual_calibrator")
    residual_mlp = import_script("66_train_phase9_cobb_residual_mlp_v2.py", "phase9_residual_mlp_v2")
    phase9_v1 = import_script("32_eval_phase9_final_cobb_v1.py", "phase9_v1_for_final_test")

    experiment_dir = EXPERIMENTS_DIR / str(args.experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_path = resolve_project_path(args.report_path)
    prediction_path = experiment_dir / "final_test_subset5_mlp_v2_predictions.csv"
    aux_path = experiment_dir / "final_test_subset5_aux_predictions.csv"
    metrics_path = experiment_dir / "final_test_subset5_mlp_v2_metrics.json"

    feature_names = residual_calibrator.build_feature_names(pair_reranker)
    mlp_models, mean, std, alpha, max_correction_deg, mlp_lock = load_locked_mlp(
        mlp_experiment_dir=resolve_project_path(args.mlp_experiment_dir),
        residual_mlp=residual_mlp,
        expected_feature_names=feature_names,
    )

    groups, eval_metadata = build_test_groups(
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
        raise ValueError("Nenhuma imagem com candidatos no test set.")

    aux_rows = list(eval_metadata.pop("aux_rows", []))
    if aux_rows:
        write_csv(aux_path, aux_rows, list(aux_rows[0].keys()))

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

    metrics, rows = evaluate_locked_groups(
        groups=groups,
        predicted_residuals=predicted_residuals,
        alpha=alpha,
        max_correction_deg=max_correction_deg,
        residual_calibrator=residual_calibrator,
    )
    payload = {
        "phase": "final_test_subset5_mlp_v2_locked",
        "model_path": str(resolve_project_path(args.model_path)),
        "profile": args.profile,
        "aux_phase5_model_path": str(resolve_project_path(args.aux_phase5_model_path)),
        "aux_profile": args.aux_profile,
        "phase7_model_path": str(resolve_project_path(args.phase7_model_path)),
        "phase8_model_path": str(resolve_project_path(args.phase8_model_path)),
        "mlp_experiment_dir": str(resolve_project_path(args.mlp_experiment_dir)),
        "mlp_lock": mlp_lock,
        "eval": eval_metadata,
        "feature_names": feature_names,
        "metrics": {key: jsonable_metric_value(value) for key, value in metrics.items()},
        "config": vars(args),
    }

    write_json(metrics_path, payload)
    if rows:
        write_csv(prediction_path, rows, list(rows[0].keys()))
    make_report(
        path=report_path,
        payload=payload,
        experiment_dir=experiment_dir,
        prediction_path=prediction_path,
    )

    print("\nFinal test subset5 - MLP v2 original locked")
    print(f"imagens avaliadas: {metrics['calibrated_mlp_v2_original']['num_images']}")
    for key, label in [
        ("geometric_max_angle", "geometria"),
        ("calibrated_mlp_v2_original", "MLP v2 original"),
        ("auxiliary_head_max", "head auxiliar"),
        ("endpoint_pair_oracle_same_sequence", "oracle endpoint"),
    ]:
        row = metrics[key]
        print(
            f"{label}: MAE={row['mae_deg']:.3f}, "
            f"SMAPE={row['paper_smape_pct']:.4f}%, "
            f"within5={row['within_5deg_rate']:.4f}, "
            f"within10={row['within_10deg_rate']:.4f}, "
            f"falhas>5={row['failures_gt5']}, "
            f"falhas>10={row['failures_gt10']}"
        )
    print(f"Metricas guardadas em: {metrics_path}")
    print(f"Predicoes guardadas em: {prediction_path}")
    print(f"Relatorio guardado em: {report_path}")


if __name__ == "__main__":
    main()
