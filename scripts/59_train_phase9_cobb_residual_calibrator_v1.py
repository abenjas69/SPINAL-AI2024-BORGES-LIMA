"""Fase 9 experimental: calibrador residual do angulo Cobb.

Este caminho e deliberadamente diferente do endpoint switcher:
- mantem o par max-angle da Fase 5 como geometria principal;
- aprende uma correcao residual pequena para o angulo Cobb;
- escolhe a intensidade da correcao num split de calibracao.

Isto permite corrigir subestimacoes do Cobb, coisa que trocar para outro par de
endpoints nao consegue fazer quando o par max-angle ja e o maior angulo
disponivel.
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
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "sanity_check" / "reports"

DEFAULT_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_V6_PREDICTIONS = (
    EXPERIMENTS_DIR
    / "phase9_confidence_fallback_v6_endpoint_safe"
    / "phase9_confidence_fallback_predictions.csv"
)
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_EXPERIMENT_NAME = "phase9_cobb_residual_calibrator_v1_val500"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / DEFAULT_EXPERIMENT_NAME
DEFAULT_REPORT = REPORTS_DIR / f"{DEFAULT_EXPERIMENT_NAME}.md"

EXTRA_FEATURE_NAMES = [
    "candidate_count_norm",
    "top2_angle_norm",
    "top3_angle_norm",
    "top5_mean_angle_norm",
    "top5_std_angle_norm",
    "top1_top2_gap_norm",
    "top1_top3_gap_norm",
    "aux_available_image",
    "aux_geom_signed_delta_norm_image",
    "aux_geom_abs_delta_norm_image",
]


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


def metric_bundle(pred_values: Sequence[float], gt_values: Sequence[float]) -> dict[str, Any]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    errors = pred - gt
    abs_errors = np.abs(errors)
    result: dict[str, Any] = {
        "num_images": int(pred.size),
        "mae_deg": float(np.mean(abs_errors)),
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "bias_deg": float(np.mean(errors)),
        "median_abs_error_deg": float(np.median(abs_errors)),
        "p90_abs_error_deg": float(np.percentile(abs_errors, 90)),
        "within_3deg_rate": float(np.mean(abs_errors <= 3.0)),
        "within_5deg_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10deg_rate": float(np.mean(abs_errors <= 10.0)),
        "failures_gt5": int(np.sum(abs_errors > 5.0)),
    }
    if pred.size >= 2 and np.ptp(pred) > 1.0e-6 and np.ptp(gt) > 1.0e-6:
        result["pearson"] = float(np.corrcoef(gt, pred)[0, 1])
    else:
        result["pearson"] = None
    return result


def select_pair_by_flag(candidates: Sequence[Mapping[str, Any]], flag: str) -> Mapping[str, Any]:
    for candidate in candidates:
        if int(candidate.get(flag, 0)) == 1:
            return candidate
    raise ValueError(f"Nenhum par marcado com {flag}")


def feature_map(candidate: Mapping[str, Any], pair_reranker: ModuleType) -> dict[str, float]:
    values = np.asarray(candidate["features"], dtype=np.float32).reshape(-1)
    return {name: float(value) for name, value in zip(pair_reranker.FEATURE_NAMES, values)}


def build_feature_names(pair_reranker: ModuleType) -> list[str]:
    return [f"geom_{name}" for name in pair_reranker.FEATURE_NAMES] + EXTRA_FEATURE_NAMES


def build_image_features(group: Mapping[str, Any], pair_reranker: ModuleType) -> np.ndarray:
    candidates = list(group["candidates"])
    geom = select_pair_by_flag(candidates, "is_geom_pair")
    geom_map = feature_map(geom, pair_reranker)
    geom_values = [geom_map[name] for name in pair_reranker.FEATURE_NAMES]

    angles = sorted([float(candidate["angle_deg"]) for candidate in candidates], reverse=True)
    top1 = angles[0] if angles else float(geom["angle_deg"])
    top2 = angles[1] if len(angles) >= 2 else top1
    top3 = angles[2] if len(angles) >= 3 else top2
    top5_values = np.asarray(angles[:5] if angles else [top1], dtype=np.float32)
    aux_cobb = float(group.get("aux_cobb", np.nan))
    geom_angle = float(geom["angle_deg"])
    aux_available = bool(np.isfinite(aux_cobb))
    aux_delta = aux_cobb - geom_angle if aux_available else 0.0
    extra_values = [
        len(candidates) / 100.0,
        top2 / 100.0,
        top3 / 100.0,
        float(np.mean(top5_values)) / 100.0,
        float(np.std(top5_values)) / 100.0,
        (top1 - top2) / 100.0,
        (top1 - top3) / 100.0,
        float(aux_available),
        aux_delta / 100.0,
        abs(aux_delta) / 100.0,
    ]
    return np.asarray(geom_values + extra_values, dtype=np.float32)


def build_calibrator_dataset(
    *,
    groups: Sequence[Mapping[str, Any]],
    pair_reranker: ModuleType,
    target_clip_deg: float,
    failure_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    rows: list[dict[str, Any]] = []
    for group in groups:
        candidates = list(group["candidates"])
        geom = select_pair_by_flag(candidates, "is_geom_pair")
        gt = float(group["gt_cobb"])
        geom_cobb = float(geom["angle_deg"])
        residual = gt - geom_cobb
        target = float(np.clip(residual, -float(target_clip_deg), float(target_clip_deg)))
        geom_abs_error = abs(geom_cobb - gt)
        weight = 1.0 + float(failure_weight) if geom_abs_error > 5.0 else 1.0
        features.append(build_image_features(group, pair_reranker))
        targets.append(target)
        weights.append(weight)
        rows.append(
            {
                "file_name": str(group["file_name"]),
                "gt_cobb_max": gt,
                "geom_cobb": geom_cobb,
                "geom_abs_error": geom_abs_error,
                "target_residual_deg": target,
                "raw_residual_deg": residual,
                "sample_weight": weight,
            }
        )
    if not features:
        raise ValueError("Sem imagens para treinar o calibrador.")
    return (
        np.stack(features).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        rows,
    )


def fit_ridge_residual(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0).astype(np.float32)
    std = np.std(features, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    x_norm = (features - mean.reshape(1, -1)) / std.reshape(1, -1)
    x_design = np.concatenate(
        [x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).astype(np.float64)
    y = targets.astype(np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    xw = x_design * np.sqrt(sample_weights).reshape(-1, 1)
    yw = y * np.sqrt(sample_weights)
    penalty = np.eye(x_design.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    lhs = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return beta.astype(np.float32), mean, std


def predict_residual(features: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_norm = (features - mean.reshape(1, -1)) / std.reshape(1, -1)
    x_design = np.concatenate(
        [x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (x_design @ beta.reshape(-1, 1)).reshape(-1).astype(np.float32)


def evaluate_calibrator(
    *,
    groups: Sequence[Mapping[str, Any]],
    pair_reranker: ModuleType,
    beta: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float,
    max_correction_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if groups:
        features = np.stack([build_image_features(group, pair_reranker) for group in groups]).astype(np.float32)
        predicted_residuals = predict_residual(features, beta, mean, std)
    else:
        predicted_residuals = np.asarray([], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    gt_values: list[float] = []
    geom_values: list[float] = []
    calibrated_values: list[float] = []
    oracle_values: list[float] = []
    rescued = 0
    broken = 0
    improved_ge3 = 0
    worsened_ge3 = 0
    corrected_count = 0
    for group, predicted_residual in zip(groups, predicted_residuals):
        candidates = list(group["candidates"])
        geom = select_pair_by_flag(candidates, "is_geom_pair")
        oracle = select_pair_by_flag(candidates, "is_oracle_pair")
        gt = float(group["gt_cobb"])
        geom_cobb = float(geom["angle_deg"])
        correction = float(np.clip(alpha * float(predicted_residual), -max_correction_deg, max_correction_deg))
        calibrated_cobb = geom_cobb + correction
        geom_error = abs(geom_cobb - gt)
        calibrated_error = abs(calibrated_cobb - gt)
        error_delta = calibrated_error - geom_error
        if abs(correction) >= 0.5:
            corrected_count += 1
        if geom_error > 5.0 and calibrated_error <= 5.0:
            rescued += 1
        if geom_error <= 5.0 and calibrated_error > 5.0:
            broken += 1
        if error_delta <= -3.0:
            improved_ge3 += 1
        if error_delta >= 3.0:
            worsened_ge3 += 1
        gt_values.append(gt)
        geom_values.append(geom_cobb)
        calibrated_values.append(calibrated_cobb)
        oracle_values.append(float(oracle["angle_deg"]))
        rows.append(
            {
                "file_name": str(group["file_name"]),
                "gt_cobb_max": gt,
                "geom_cobb": geom_cobb,
                "geom_abs_error": geom_error,
                "calibrated_cobb": calibrated_cobb,
                "calibrated_abs_error": calibrated_error,
                "error_delta_deg": error_delta,
                "predicted_residual_deg": float(predicted_residual),
                "applied_correction_deg": correction,
                "aux_cobb": float(group.get("aux_cobb", np.nan)),
                "oracle_cobb": float(oracle["angle_deg"]),
                "oracle_abs_error": abs(float(oracle["angle_deg"]) - gt),
                "rescued": int(geom_error > 5.0 and calibrated_error <= 5.0),
                "broken": int(geom_error <= 5.0 and calibrated_error > 5.0),
                "improved_ge3": int(error_delta <= -3.0),
                "worsened_ge3": int(error_delta >= 3.0),
            }
        )

    metrics = {
        "alpha": float(alpha),
        "geometric_max_angle": metric_bundle(geom_values, gt_values),
        "calibrated": metric_bundle(calibrated_values, gt_values),
        "endpoint_pair_oracle": metric_bundle(oracle_values, gt_values),
        "correction_rate": float(corrected_count / max(len(groups), 1)),
        "corrected_count": int(corrected_count),
        "rescued": int(rescued),
        "broken": int(broken),
        "improved_ge3": int(improved_ge3),
        "worsened_ge3": int(worsened_ge3),
    }
    return metrics, rows


def select_alpha(alpha_sweep: Sequence[Mapping[str, Any]]) -> float:
    if not alpha_sweep:
        raise ValueError("Sweep alpha vazio.")
    best = sorted(
        alpha_sweep,
        key=lambda row: (
            -float(row["within_5deg_rate"]),
            int(row["broken"]),
            -int(row["rescued"]),
            float(row["mae_deg"]),
            float(row["correction_rate"]),
            float(row["alpha"]),
        ),
    )[0]
    return float(best["alpha"])


def make_report(path: Path, args: argparse.Namespace, payload: Mapping[str, Any], experiment_dir: Path) -> None:
    holdout = payload["holdout"]
    train = payload["train"]
    lines = [
        "# Fase 9 - Cobb residual calibrator v1",
        "",
        "## Objetivo",
        "",
        "Aprender uma correcao residual conservadora sobre o Cobb max-angle da Fase 5.",
        "",
        "## Configuracao",
        "",
        f"- modelo Fase 5: `{args.model_path}`",
        f"- perfil Fase 5: `{args.profile}`",
        f"- imagens: `{payload['num_images']}`",
        f"- treino/calibracao/holdout: `{payload['num_train_images']}/{payload.get('num_calibration_images', 0)}/{payload['num_holdout_images']}`",
        f"- alpha selecionado em `{payload.get('alpha_split_name', 'train')}`: `{payload['selected_alpha']:.3f}`",
        "",
        "## Holdout interno",
        "",
        "| metodo | MAE | within5 | falhas >5 | correcoes | resgatados | quebrados | melh >=3 | pioras >=3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("geometric_max_angle", "geometria max-angle"),
        ("calibrated", "calibrador residual"),
        ("endpoint_pair_oracle", "oracle endpoint"),
    ]:
        row = holdout[key]
        lines.append(
            "| "
            f"{label} | "
            f"{row.get('mae_deg', 0.0):.3f} | "
            f"{row.get('within_5deg_rate', 0.0):.3f} | "
            f"{row.get('failures_gt5', 0)} | "
            f"{holdout.get('corrected_count', '') if key == 'calibrated' else ''} | "
            f"{holdout.get('rescued', '') if key == 'calibrated' else ''} | "
            f"{holdout.get('broken', '') if key == 'calibrated' else ''} | "
            f"{holdout.get('improved_ge3', '') if key == 'calibrated' else ''} | "
            f"{holdout.get('worsened_ge3', '') if key == 'calibrated' else ''} |"
        )
    lines.extend(
        [
            "",
            "## Treino interno",
            "",
            "| metodo | MAE | within5 | falhas >5 | correcoes | resgatados | quebrados | melh >=3 | pioras >=3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("geometric_max_angle", "geometria max-angle"),
        ("calibrated", "calibrador residual"),
        ("endpoint_pair_oracle", "oracle endpoint"),
    ]:
        row = train[key]
        lines.append(
            "| "
            f"{label} | "
            f"{row.get('mae_deg', 0.0):.3f} | "
            f"{row.get('within_5deg_rate', 0.0):.3f} | "
            f"{row.get('failures_gt5', 0)} | "
            f"{train.get('corrected_count', '') if key == 'calibrated' else ''} | "
            f"{train.get('rescued', '') if key == 'calibrated' else ''} | "
            f"{train.get('broken', '') if key == 'calibrated' else ''} | "
            f"{train.get('improved_ge3', '') if key == 'calibrated' else ''} | "
            f"{train.get('worsened_ge3', '') if key == 'calibrated' else ''} |"
        )
    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            f"- metricas: `{experiment_dir / 'phase9_cobb_residual_calibrator_metrics.json'}`",
            f"- predicoes holdout: `{experiment_dir / 'phase9_cobb_residual_calibrator_holdout_predictions.csv'}`",
            f"- predicoes treino: `{experiment_dir / 'phase9_cobb_residual_calibrator_train_predictions.csv'}`",
            f"- pesos ridge: `{experiment_dir / 'phase9_cobb_residual_calibrator_ridge.npz'}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina calibrador residual Cobb v1.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--phase9-predictions-csv", default=str(DEFAULT_V6_PREDICTIONS))
    parser.add_argument(
        "--candidate-cache-path",
        default="",
        help="Cache jsonl/jsonl.gz criada pelo script 61; se definido, salta a inferencia da Fase 5.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--eval-mode", choices=("validation", "window"), default="validation")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--train-size", type=int, default=12768)
    parser.add_argument("--val-size", type=int, default=3192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--train-images", type=int, default=250)
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=100,
        help="Se >0, escolhe alpha num split separado entre treino e holdout.",
    )
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    parser.add_argument("--failure-weight", type=float, default=4.0)
    parser.add_argument("--target-clip-deg", type=float, default=20.0)
    parser.add_argument("--max-correction-deg", type=float, default=8.0)
    parser.add_argument("--alpha-grid", default="0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00")
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval")
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    oracle_script = import_script("56_oracle_phase5_sequence_portfolio_cobb_v1.py", "phase5_cobb_oracle")
    pair_reranker = import_script("57_train_phase9_endpoint_pair_reranker_v1.py", "phase9_pair_reranker")
    candidate_cache = import_script("61_extract_phase9_cobb_candidate_cache_v1.py", "phase9_candidate_cache")

    if args.profile not in phase5_sequence.PROFILE_CONFIGS:
        raise ValueError(f"Perfil desconhecido: {args.profile}")

    model_path = resolve_project_path(args.model_path)
    aux_csv = resolve_project_path(args.phase9_predictions_csv)
    cache_path = resolve_project_path(args.candidate_cache_path) if args.candidate_cache_path else None
    report_path = resolve_project_path(args.report_path)
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    resolve_project_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    aux_by_file = pair_reranker.load_aux_cobb_by_file(aux_csv)

    if cache_path is not None:
        print(f"A carregar cache de candidatos Fase 9: {cache_path}")
        cache_metadata, groups = candidate_cache.load_candidate_cache(cache_path)
        candidate_cache.validate_candidate_cache(
            metadata=cache_metadata,
            groups=groups,
            expected_profile=args.profile,
            expected_feature_names=pair_reranker.FEATURE_NAMES,
        )
        eval_metadata = dict(cache_metadata.get("eval", {}))
        print(f"cache carregada: imagens com candidatos={len(groups)}")
    else:
        print(f"A carregar modelo Fase 5: {model_path}")
        model = phase5_eval.load_spatial_offset_model_for_eval(model_path)
        selected_samples, selected_paths, eval_metadata = phase5_sequence.select_eval_samples(
            tfdata=tfdata,
            phase2_train=phase2_train,
            eval_mode=args.eval_mode,
            split=args.split,
            train_size=args.train_size,
            val_size=args.val_size,
            start_index=args.start_index,
            num_images=args.num_images,
        )
        print(
            "A gerar dados calibrador residual: "
            f"imagens={len(selected_samples)}, perfil={args.profile}, aux_csv={aux_csv.is_file()}"
        )

        phase9_v1 = pair_reranker.import_script("32_eval_phase9_final_cobb_v1.py", "phase9_v1_for_residual_calibrator")
        groups: list[dict[str, Any]] = []
        for image_index, (sample, image_path) in enumerate(zip(selected_samples, selected_paths), start=1):
            targets = tfdata.sample_to_targets(sample)
            image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
            predictions = model(tf.expand_dims(image, axis=0), training=False)
            result = postprocess.postprocess_candidates_sequence(
                presence=predictions["presence"].numpy()[0],
                bbox=predictions["bbox"].numpy()[0],
                points=predictions["points"].numpy()[0],
                **phase5_sequence.PROFILE_CONFIGS[args.profile],
            )
            file_name = str(sample["file_name"])
            gt_cobb, gt_major_region = oracle_script.cobb_max_from_targets(targets)
            width = float(sample.get("width", tf.shape(image)[1].numpy()))
            height = float(sample.get("height", tf.shape(image)[0].numpy()))
            aux_cobb = aux_by_file.get(file_name, np.nan)
            candidates = pair_reranker.build_pair_candidates(
                phase9_v1=phase9_v1,
                file_name=file_name,
                points=np.asarray(result["selected_points"], dtype=np.float32),
                scores=np.asarray(result["selected_scores"], dtype=np.float32),
                width=width,
                height=height,
                gt_cobb=gt_cobb,
                aux_cobb=aux_cobb,
            )
            if candidates:
                groups.append(
                    {
                        "file_name": file_name,
                        "gt_cobb": float(gt_cobb),
                        "gt_major_region": gt_major_region,
                        "aux_cobb": float(aux_cobb) if np.isfinite(aux_cobb) else np.nan,
                        "aux_available": bool(np.isfinite(aux_cobb)),
                        "candidates": candidates,
                    }
                )
            if args.progress_every > 0 and (image_index % args.progress_every == 0 or image_index == len(selected_samples)):
                print(f"processadas {image_index}/{len(selected_samples)} imagens")

    train_count = min(max(int(args.train_images), 1), len(groups) - 1)
    train_groups = groups[:train_count]
    remaining_groups = groups[train_count:]
    calibration_count = 0
    if int(args.calibration_images) > 0 and len(remaining_groups) > 1:
        calibration_count = min(int(args.calibration_images), len(remaining_groups) - 1)
    calibration_groups = remaining_groups[:calibration_count]
    holdout_groups = remaining_groups[calibration_count:]

    train_features, train_targets, train_weights, train_label_rows = build_calibrator_dataset(
        groups=train_groups,
        pair_reranker=pair_reranker,
        target_clip_deg=args.target_clip_deg,
        failure_weight=args.failure_weight,
    )
    print(
        "dataset calibrador: "
        f"imagens={len(train_groups)}, features={train_features.shape[1]}, "
        f"falhas_geom={int(np.sum(np.abs(train_targets) > 5.0))}"
    )
    beta, mean, std = fit_ridge_residual(
        train_features,
        train_targets,
        train_weights,
        ridge_lambda=float(args.ridge_lambda),
    )

    alphas = [float(part.strip()) for part in str(args.alpha_grid).split(",") if part.strip()]
    alpha_groups = calibration_groups if calibration_groups else train_groups
    alpha_split_name = "calibration" if calibration_groups else "train"
    alpha_sweep: list[dict[str, Any]] = []
    for alpha in alphas:
        sweep_metrics, _ = evaluate_calibrator(
            groups=alpha_groups,
            pair_reranker=pair_reranker,
            beta=beta,
            mean=mean,
            std=std,
            alpha=alpha,
            max_correction_deg=float(args.max_correction_deg),
        )
        alpha_sweep.append(
            {
                "alpha": float(alpha),
                "within_5deg_rate": float(sweep_metrics["calibrated"]["within_5deg_rate"]),
                "mae_deg": float(sweep_metrics["calibrated"]["mae_deg"]),
                "failures_gt5": int(sweep_metrics["calibrated"]["failures_gt5"]),
                "correction_rate": float(sweep_metrics["correction_rate"]),
                "rescued": int(sweep_metrics["rescued"]),
                "broken": int(sweep_metrics["broken"]),
                "improved_ge3": int(sweep_metrics["improved_ge3"]),
                "worsened_ge3": int(sweep_metrics["worsened_ge3"]),
            }
        )
    if args.fixed_alpha is None:
        selected_alpha = select_alpha(alpha_sweep)
    else:
        selected_alpha = float(args.fixed_alpha)
        alpha_split_name = "fixed"

    train_metrics, train_rows = evaluate_calibrator(
        groups=train_groups,
        pair_reranker=pair_reranker,
        beta=beta,
        mean=mean,
        std=std,
        alpha=selected_alpha,
        max_correction_deg=float(args.max_correction_deg),
    )
    calibration_metrics: dict[str, Any] | None = None
    calibration_rows: list[dict[str, Any]] = []
    if calibration_groups:
        calibration_metrics, calibration_rows = evaluate_calibrator(
            groups=calibration_groups,
            pair_reranker=pair_reranker,
            beta=beta,
            mean=mean,
            std=std,
            alpha=selected_alpha,
            max_correction_deg=float(args.max_correction_deg),
        )
    holdout_metrics, holdout_rows = evaluate_calibrator(
        groups=holdout_groups,
        pair_reranker=pair_reranker,
        beta=beta,
        mean=mean,
        std=std,
        alpha=selected_alpha,
        max_correction_deg=float(args.max_correction_deg),
    )

    feature_names = build_feature_names(pair_reranker)
    payload = {
        "phase": "phase9_cobb_residual_calibrator_v1",
        "model_path": str(model_path),
        "phase9_predictions_csv": str(aux_csv),
        "candidate_cache_path": str(cache_path) if cache_path is not None else None,
        "eval": eval_metadata,
        "profile": args.profile,
        "feature_names": feature_names,
        "num_images": int(len(groups)),
        "num_train_images": int(len(train_groups)),
        "num_calibration_images": int(len(calibration_groups)),
        "num_holdout_images": int(len(holdout_groups)),
        "selected_alpha": float(selected_alpha),
        "alpha_split_name": alpha_split_name,
        "alpha_sweep": alpha_sweep,
        "train": train_metrics,
        "calibration": calibration_metrics,
        "holdout": holdout_metrics,
        "config": vars(args),
    }

    np.savez(
        experiment_dir / "phase9_cobb_residual_calibrator_ridge.npz",
        beta=beta,
        mean=mean,
        std=std,
        feature_names=np.asarray(feature_names),
    )
    write_json(experiment_dir / "phase9_cobb_residual_calibrator_metrics.json", payload)
    prediction_fields = [
        "file_name",
        "gt_cobb_max",
        "geom_cobb",
        "geom_abs_error",
        "calibrated_cobb",
        "calibrated_abs_error",
        "error_delta_deg",
        "predicted_residual_deg",
        "applied_correction_deg",
        "aux_cobb",
        "oracle_cobb",
        "oracle_abs_error",
        "rescued",
        "broken",
        "improved_ge3",
        "worsened_ge3",
    ]
    label_fields = [
        "file_name",
        "gt_cobb_max",
        "geom_cobb",
        "geom_abs_error",
        "target_residual_deg",
        "raw_residual_deg",
        "sample_weight",
    ]
    write_csv(experiment_dir / "phase9_cobb_residual_calibrator_train_labels.csv", train_label_rows, label_fields)
    write_csv(experiment_dir / "phase9_cobb_residual_calibrator_train_predictions.csv", train_rows, prediction_fields)
    if calibration_rows:
        write_csv(
            experiment_dir / "phase9_cobb_residual_calibrator_calibration_predictions.csv",
            calibration_rows,
            prediction_fields,
        )
    write_csv(
        experiment_dir / "phase9_cobb_residual_calibrator_holdout_predictions.csv",
        holdout_rows,
        prediction_fields,
    )
    make_report(report_path, args, payload, experiment_dir)

    print("\nResumo Cobb residual calibrator v1")
    print(f"alpha selecionado: {selected_alpha:.3f} via {alpha_split_name}")
    metric_blocks: list[tuple[str, Mapping[str, Any]]] = [("train", train_metrics)]
    if calibration_metrics is not None:
        metric_blocks.append(("calibration", calibration_metrics))
    metric_blocks.append(("holdout", holdout_metrics))
    for split_name, split_metrics in metric_blocks:
        print(f"{split_name}:")
        for key in ("geometric_max_angle", "calibrated", "endpoint_pair_oracle"):
            row = split_metrics[key]
            print(
                f"  {key}: MAE={row['mae_deg']:.3f}, "
                f"within5={row['within_5deg_rate']:.3f}, "
                f"falhas>5={row['failures_gt5']}"
            )
        print(
            f"  correcoes={split_metrics['corrected_count']}, "
            f"resgatados={split_metrics['rescued']}, "
            f"quebrados={split_metrics['broken']}, "
            f"melhorias>=3={split_metrics['improved_ge3']}, "
            f"pioras>=3={split_metrics['worsened_ge3']}"
        )
    print(f"Metricas guardadas em: {experiment_dir / 'phase9_cobb_residual_calibrator_metrics.json'}")
    print(f"Relatorio guardado em: {report_path}")


if __name__ == "__main__":
    main()
