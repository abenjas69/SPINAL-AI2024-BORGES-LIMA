"""Treina um calibrador residual Cobb nao-linear para a Fase 9.

Este script substitui a regressao ridge do `59_train_phase9_cobb_residual_calibrator_v1.py`
por uma MLP pequena sobre as mesmas features de imagem/candidatos. O objetivo e
corrigir a subcorrecao e os erros de direcao que ficaram na auditoria residual,
mantendo a inferencia escalar e independente de uma escolha discreta de endpoint.
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
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "sanity_check" / "reports"

DEFAULT_CACHE = (
    EXPERIMENTS_DIR
    / "phase9_cobb_candidate_cache_v1_train12000_cal768_val3192"
    / "phase9_cobb_candidate_cache.jsonl"
)
DEFAULT_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
DEFAULT_EXPERIMENT_NAME = "phase9_cobb_residual_mlp_v2_train12000_cal768_val3192"
DEFAULT_REPORT = REPORTS_DIR / f"{DEFAULT_EXPERIMENT_NAME}.md"


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


def parse_float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def select_pair_by_flag(candidates: Sequence[Mapping[str, Any]], flag: str) -> Mapping[str, Any]:
    for candidate in candidates:
        if int(candidate.get(flag, 0)) == 1:
            return candidate
    raise ValueError(f"Nenhum par marcado com {flag}")


def build_model(input_dim: int, hidden_units: int, dropout_rate: float, learning_rate: float, huber_delta: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="cobb_residual_features")
    x = tf.keras.layers.Dense(int(hidden_units), activation="relu")(inputs)
    x = tf.keras.layers.Dropout(float(dropout_rate))(x)
    x = tf.keras.layers.Dense(max(int(hidden_units) // 2, 16), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="predicted_residual_deg")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase9_cobb_residual_mlp_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tf.keras.losses.Huber(delta=float(huber_delta)),
    )
    return model


def standardize_features(train_features: np.ndarray, *blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = np.mean(train_features, axis=0).astype(np.float32)
    std = np.std(train_features, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    scaled = [((block - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32) for block in blocks]
    return mean, std, scaled


def predict_ensemble(models: Sequence[tf.keras.Model], features: np.ndarray, batch_size: int) -> np.ndarray:
    predictions = [
        model.predict(features, batch_size=max(int(batch_size), 1), verbose=0).reshape(-1)
        for model in models
    ]
    return np.mean(np.stack(predictions).astype(np.float32), axis=0)


def evaluate_groups(
    *,
    groups: Sequence[Mapping[str, Any]],
    predicted_residuals: np.ndarray,
    alpha: float,
    max_correction_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_values: list[float] = []
    geom_values: list[float] = []
    calibrated_values: list[float] = []
    oracle_values: list[float] = []
    rows: list[dict[str, Any]] = []
    corrected_count = 0
    rescued = 0
    broken = 0
    improved_ge3 = 0
    worsened_ge3 = 0
    for group, predicted_residual in zip(groups, predicted_residuals):
        candidates = list(group["candidates"])
        geom = select_pair_by_flag(candidates, "is_geom_pair")
        oracle = select_pair_by_flag(candidates, "is_oracle_pair")
        gt = float(group["gt_cobb"])
        geom_cobb = float(geom["angle_deg"])
        correction = float(np.clip(float(alpha) * float(predicted_residual), -float(max_correction_deg), float(max_correction_deg)))
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
        "calibrated_mlp": metric_bundle(calibrated_values, gt_values),
        "endpoint_pair_oracle": metric_bundle(oracle_values, gt_values),
        "correction_rate": float(corrected_count / max(len(groups), 1)),
        "corrected_count": int(corrected_count),
        "rescued": int(rescued),
        "broken": int(broken),
        "improved_ge3": int(improved_ge3),
        "worsened_ge3": int(worsened_ge3),
    }
    return metrics, rows


def select_alpha(
    *,
    groups: Sequence[Mapping[str, Any]],
    predicted_residuals: np.ndarray,
    alpha_grid: Sequence[float],
    max_correction_deg: float,
) -> tuple[float, list[dict[str, Any]]]:
    sweep: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        metrics, _ = evaluate_groups(
            groups=groups,
            predicted_residuals=predicted_residuals,
            alpha=float(alpha),
            max_correction_deg=float(max_correction_deg),
        )
        row = {
            "alpha": float(alpha),
            "mae_deg": float(metrics["calibrated_mlp"]["mae_deg"]),
            "within_5deg_rate": float(metrics["calibrated_mlp"]["within_5deg_rate"]),
            "failures_gt5": int(metrics["calibrated_mlp"]["failures_gt5"]),
            "correction_rate": float(metrics["correction_rate"]),
            "rescued": int(metrics["rescued"]),
            "broken": int(metrics["broken"]),
            "improved_ge3": int(metrics["improved_ge3"]),
            "worsened_ge3": int(metrics["worsened_ge3"]),
        }
        sweep.append(row)
    best = sorted(
        sweep,
        key=lambda row: (
            -float(row["within_5deg_rate"]),
            int(row["broken"]),
            -int(row["rescued"]),
            float(row["mae_deg"]),
            float(row["alpha"]),
        ),
    )[0]
    return float(best["alpha"]), sweep


def make_report(path: Path, payload: Mapping[str, Any], experiment_dir: Path) -> None:
    holdout = payload["holdout"]
    lines = [
        "# Fase 9 - Cobb residual MLP v2",
        "",
        "## Configuracao",
        "",
        f"- cache candidatos: `{payload['candidate_cache_path']}`",
        f"- treino/calibracao/holdout: `{payload['num_train_images']}/{payload['num_calibration_images']}/{payload['num_holdout_images']}`",
        f"- seeds: `{payload['seed_list']}`",
        f"- alpha selecionado: `{payload['selected_alpha']:.3f}`",
        "",
        "## Holdout",
        "",
        "| metodo | MAE | within3 | within5 | within10 | falhas >5 | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("geometric_max_angle", "geometria max-angle"),
        ("calibrated_mlp", "calibrador MLP"),
        ("endpoint_pair_oracle", "oracle endpoint"),
    ]:
        row = holdout[key]
        lines.append(
            "| "
            f"{label} | "
            f"{row['mae_deg']:.3f} | "
            f"{row['within_3deg_rate']:.3f} | "
            f"{row['within_5deg_rate']:.3f} | "
            f"{row['within_10deg_rate']:.3f} | "
            f"{row['failures_gt5']} | "
            f"{row['p90_abs_error_deg']:.3f} | "
            f"{row['bias_deg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Calibrador",
            "",
            f"- correcoes >=0.5 deg: `{holdout['corrected_count']}`",
            f"- resgatados: `{holdout['rescued']}`",
            f"- quebrados: `{holdout['broken']}`",
            f"- melhorias >=3 deg: `{holdout['improved_ge3']}`",
            f"- pioras >=3 deg: `{holdout['worsened_ge3']}`",
            "",
            "## Artefactos",
            "",
            f"- metricas: `{experiment_dir / 'phase9_cobb_residual_mlp_metrics.json'}`",
            f"- predicoes holdout: `{experiment_dir / 'phase9_cobb_residual_mlp_holdout_predictions.csv'}`",
            f"- alpha sweep: `{experiment_dir / 'phase9_cobb_residual_mlp_alpha_sweep.csv'}`",
            f"- scaler: `{experiment_dir / 'phase9_cobb_residual_mlp_scaler.npz'}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina calibrador residual MLP v2.")
    parser.add_argument("--candidate-cache-path", default=str(DEFAULT_CACHE))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--train-images", type=int, default=12000)
    parser.add_argument("--calibration-images", type=int, default=768)
    parser.add_argument("--target-clip-deg", type=float, default=25.0)
    parser.add_argument("--failure-weight", type=float, default=5.0)
    parser.add_argument("--max-correction-deg", type=float, default=10.0)
    parser.add_argument("--alpha-grid", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20")
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--seed-list", default="42")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / DEFAULT_EXPERIMENT_NAME))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_cache = import_script("61_extract_phase9_cobb_candidate_cache_v1.py", "phase9_candidate_cache")
    pair_reranker = import_script("57_train_phase9_endpoint_pair_reranker_v1.py", "phase9_pair_reranker")
    residual_calibrator = import_script("59_train_phase9_cobb_residual_calibrator_v1.py", "phase9_residual_calibrator")

    cache_path = resolve_project_path(args.candidate_cache_path)
    experiment_dir = EXPERIMENTS_DIR / str(args.experiment_name)
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"A carregar cache de candidatos: {cache_path}")
    cache_metadata, groups = candidate_cache.load_candidate_cache(cache_path)
    candidate_cache.validate_candidate_cache(
        metadata=cache_metadata,
        groups=groups,
        expected_profile=args.profile,
        expected_feature_names=pair_reranker.FEATURE_NAMES,
    )
    if int(args.num_images) > 0:
        groups = groups[: int(args.num_images)]
    print(f"cache carregada: imagens com candidatos={len(groups)}")

    train_count = min(max(int(args.train_images), 1), len(groups) - 1)
    remaining = groups[train_count:]
    calibration_count = min(max(int(args.calibration_images), 0), max(len(remaining) - 1, 0))
    train_groups = groups[:train_count]
    calibration_groups = remaining[:calibration_count]
    holdout_groups = remaining[calibration_count:]
    if not holdout_groups:
        raise ValueError("Split sem holdout. Reduza --train-images ou --calibration-images.")

    print("A construir features MLP...")
    train_features, train_targets, train_weights, _ = residual_calibrator.build_calibrator_dataset(
        groups=train_groups,
        pair_reranker=pair_reranker,
        target_clip_deg=float(args.target_clip_deg),
        failure_weight=float(args.failure_weight),
    )
    calibration_features, _calibration_targets, calibration_weights, _ = residual_calibrator.build_calibrator_dataset(
        groups=calibration_groups,
        pair_reranker=pair_reranker,
        target_clip_deg=float(args.target_clip_deg),
        failure_weight=float(args.failure_weight),
    )
    holdout_features, _holdout_targets, _holdout_weights, _ = residual_calibrator.build_calibrator_dataset(
        groups=holdout_groups,
        pair_reranker=pair_reranker,
        target_clip_deg=float(args.target_clip_deg),
        failure_weight=float(args.failure_weight),
    )
    mean, std, scaled_blocks = standardize_features(
        train_features,
        train_features,
        calibration_features,
        holdout_features,
    )
    train_scaled, calibration_scaled, holdout_scaled = scaled_blocks

    seed_list = parse_int_list(args.seed_list)
    models: list[tf.keras.Model] = []
    histories: list[dict[str, list[float]]] = []
    for seed in seed_list:
        print(f"A treinar MLP residual seed={seed}")
        tf.keras.utils.set_random_seed(int(seed))
        model = build_model(
            input_dim=train_scaled.shape[1],
            hidden_units=int(args.hidden_units),
            dropout_rate=float(args.dropout_rate),
            learning_rate=float(args.learning_rate),
            huber_delta=float(args.huber_delta),
        )
        callbacks: list[tf.keras.callbacks.Callback] = []
        validation_data = None
        validation_split = float(args.validation_split)
        if int(args.early_stopping_patience) > 0:
            validation_data = (calibration_scaled, _calibration_targets, calibration_weights)
            validation_split = 0.0
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(args.early_stopping_patience),
                    restore_best_weights=True,
                )
            )
        history = model.fit(
            train_scaled,
            train_targets,
            sample_weight=train_weights,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            shuffle=True,
            callbacks=callbacks,
            verbose=int(args.verbose),
        )
        models.append(model)
        histories.append({key: [float(value) for value in values] for key, values in history.history.items()})
        model.save(experiment_dir / f"phase9_cobb_residual_mlp_seed{seed}.keras")

    print("A prever residuos...")
    train_pred = predict_ensemble(models, train_scaled, int(args.prediction_batch_size))
    calibration_pred = predict_ensemble(models, calibration_scaled, int(args.prediction_batch_size))
    holdout_pred = predict_ensemble(models, holdout_scaled, int(args.prediction_batch_size))

    if args.fixed_alpha is None:
        selected_alpha, alpha_sweep = select_alpha(
            groups=calibration_groups,
            predicted_residuals=calibration_pred,
            alpha_grid=parse_float_grid(args.alpha_grid),
            max_correction_deg=float(args.max_correction_deg),
        )
        alpha_split_name = "calibration"
    else:
        selected_alpha = float(args.fixed_alpha)
        alpha_sweep = []
        alpha_split_name = "fixed"

    train_metrics, train_rows = evaluate_groups(
        groups=train_groups,
        predicted_residuals=train_pred,
        alpha=selected_alpha,
        max_correction_deg=float(args.max_correction_deg),
    )
    calibration_metrics, calibration_rows = evaluate_groups(
        groups=calibration_groups,
        predicted_residuals=calibration_pred,
        alpha=selected_alpha,
        max_correction_deg=float(args.max_correction_deg),
    )
    holdout_metrics, holdout_rows = evaluate_groups(
        groups=holdout_groups,
        predicted_residuals=holdout_pred,
        alpha=selected_alpha,
        max_correction_deg=float(args.max_correction_deg),
    )

    feature_names = residual_calibrator.build_feature_names(pair_reranker)
    np.savez(
        experiment_dir / "phase9_cobb_residual_mlp_scaler.npz",
        mean=mean,
        std=std,
        feature_names=np.asarray(feature_names),
        seed_list=np.asarray(seed_list),
    )
    payload = {
        "phase": "phase9_cobb_residual_mlp_v2",
        "candidate_cache_path": str(cache_path),
        "profile": args.profile,
        "eval": cache_metadata.get("eval", {}),
        "feature_names": feature_names,
        "num_images": int(len(groups)),
        "num_train_images": int(len(train_groups)),
        "num_calibration_images": int(len(calibration_groups)),
        "num_holdout_images": int(len(holdout_groups)),
        "seed_list": seed_list,
        "selected_alpha": float(selected_alpha),
        "alpha_split_name": alpha_split_name,
        "max_correction_deg": float(args.max_correction_deg),
        "alpha_sweep": alpha_sweep,
        "train": train_metrics,
        "calibration": calibration_metrics,
        "holdout": holdout_metrics,
        "history": histories,
        "config": vars(args),
    }
    write_json(experiment_dir / "phase9_cobb_residual_mlp_metrics.json", payload)
    prediction_fields = list(holdout_rows[0].keys()) if holdout_rows else []
    write_csv(experiment_dir / "phase9_cobb_residual_mlp_train_predictions.csv", train_rows, prediction_fields)
    write_csv(
        experiment_dir / "phase9_cobb_residual_mlp_calibration_predictions.csv",
        calibration_rows,
        prediction_fields,
    )
    write_csv(experiment_dir / "phase9_cobb_residual_mlp_holdout_predictions.csv", holdout_rows, prediction_fields)
    if alpha_sweep:
        write_csv(experiment_dir / "phase9_cobb_residual_mlp_alpha_sweep.csv", alpha_sweep, list(alpha_sweep[0].keys()))
    make_report(report_path, payload, experiment_dir)

    print("\nResumo Cobb residual MLP v2")
    print(f"alpha selecionado: {selected_alpha:.3f} via {alpha_split_name}")
    for split_name, metrics in [("train", train_metrics), ("calibration", calibration_metrics), ("holdout", holdout_metrics)]:
        print(f"{split_name}:")
        for key in ("geometric_max_angle", "calibrated_mlp", "endpoint_pair_oracle"):
            row = metrics[key]
            print(
                f"  {key}: MAE={row['mae_deg']:.3f}, "
                f"within5={row['within_5deg_rate']:.3f}, "
                f"falhas>5={row['failures_gt5']}"
            )
        print(
            f"  correcoes={metrics['corrected_count']}, "
            f"resgatados={metrics['rescued']}, "
            f"quebrados={metrics['broken']}, "
            f"melh>=3={metrics['improved_ge3']}, "
            f"pioras>=3={metrics['worsened_ge3']}"
        )
    print(f"Metricas guardadas em: {experiment_dir / 'phase9_cobb_residual_mlp_metrics.json'}")
    print(f"Relatorio guardado em: {report_path}")


if __name__ == "__main__":
    main()
