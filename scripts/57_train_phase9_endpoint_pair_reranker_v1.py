"""Treina um reranker supervisionado para pares de endpoints Cobb.

O objetivo e testar se o teto visto no endpoint-pair oracle e aprendivel com
features disponiveis em inferencia. O treino usa apenas um split interno da
janela avaliada; o holdout mede generalizacao dentro dessas 500 imagens.
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
DEFAULT_EXPERIMENT_NAME = "phase9_endpoint_pair_reranker_v1_val500"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / DEFAULT_EXPERIMENT_NAME
DEFAULT_REPORT = REPORTS_DIR / f"{DEFAULT_EXPERIMENT_NAME}.md"

FEATURE_NAMES = [
    "angle_norm",
    "angle_delta_from_max_norm",
    "upper_norm",
    "lower_norm",
    "span_norm",
    "center_norm",
    "count_norm",
    "upper_score",
    "lower_score",
    "endpoint_score_mean",
    "endpoint_score_min",
    "mean_sequence_score",
    "min_sequence_score",
    "score_std",
    "upper_is_extreme",
    "lower_is_extreme",
    "either_extreme",
    "pair_region_pt",
    "pair_region_mt",
    "pair_region_tll",
    "top_angle_sin",
    "top_angle_cos",
    "bottom_angle_sin",
    "bottom_angle_cos",
    "aux_available",
    "aux_cobb_norm",
    "aux_abs_delta_norm",
    "aux_signed_delta_norm",
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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


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


def load_aux_cobb_by_file(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name = str(row.get("file_name", ""))
            if not file_name:
                continue
            aux_value = safe_float(row.get("aux_cobb_max"), default=np.nan)
            if np.isfinite(aux_value):
                values[file_name] = aux_value
    return values


def axial_sin_cos(angle_deg: float) -> tuple[float, float]:
    radians = np.deg2rad(float(angle_deg) * 2.0)
    return float(np.sin(radians)), float(np.cos(radians))


def coarse_region(upper_index: int, lower_index: int, count: int) -> tuple[float, float, float]:
    if count <= 1:
        return 0.0, 0.0, 0.0
    center = (float(upper_index) + float(lower_index)) / 2.0 / float(max(count - 1, 1))
    if center < 0.34:
        return 1.0, 0.0, 0.0
    if center < 0.67:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def build_pair_candidates(
    *,
    phase9_v1: ModuleType,
    file_name: str,
    points: np.ndarray,
    scores: np.ndarray,
    width: float,
    height: float,
    gt_cobb: float,
    aux_cobb: float,
) -> list[dict[str, Any]]:
    valid_points = np.asarray(points, dtype=np.float32).reshape(-1, 8)
    valid_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = int(valid_points.shape[0])
    if count < 2:
        return []

    top_angles: list[float] = []
    bottom_angles: list[float] = []
    for row in valid_points:
        top_angles.append(phase9_v1.line_angle_deg(row[0:2], row[2:4], width, height))
        bottom_angles.append(phase9_v1.line_angle_deg(row[4:6], row[6:8], width, height))

    raw_pairs: list[dict[str, Any]] = []
    max_angle = -1.0
    for upper_index in range(count - 1):
        for lower_index in range(upper_index + 1, count):
            angle = float(phase9_v1.angle_diff_deg(top_angles[upper_index], bottom_angles[lower_index]))
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
    aux_available = float(np.isfinite(aux_cobb))
    aux_value = float(aux_cobb) if np.isfinite(aux_cobb) else 0.0
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
        region_pt, region_mt, region_tll = coarse_region(upper_index, lower_index, count)
        top_sin, top_cos = axial_sin_cos(float(top_angles[upper_index]))
        bottom_sin, bottom_cos = axial_sin_cos(float(bottom_angles[lower_index]))
        aux_signed_delta = angle - aux_value if aux_available > 0.0 else 0.0
        abs_error = abs(angle - float(gt_cobb))
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
                "abs_error": float(abs_error),
                "target_score": float(np.exp(-abs_error / 2.5)),
                "is_oracle_pair": 0,
                "is_geom_pair": 0,
                "is_aux_pair": 0,
                "features": feature_values,
            }
        )

    best_oracle_index = int(np.argmin([float(item["abs_error"]) for item in candidates]))
    best_geom_index = int(np.argmax([float(item["angle_deg"]) for item in candidates]))
    candidates[best_oracle_index]["is_oracle_pair"] = 1
    candidates[best_geom_index]["is_geom_pair"] = 1
    if aux_available > 0.0:
        best_aux_index = int(np.argmin([abs(float(item["angle_deg"]) - aux_value) for item in candidates]))
        candidates[best_aux_index]["is_aux_pair"] = 1
    return candidates


def build_model(input_dim: int, hidden_units: int, dropout_rate: float, learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="pair_features")
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(max(hidden_units // 2, 8), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pair_score")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase9_endpoint_pair_reranker_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


def select_pair_by_flag(candidates: Sequence[Mapping[str, Any]], flag: str) -> Mapping[str, Any]:
    for candidate in candidates:
        if int(candidate.get(flag, 0)) == 1:
            return candidate
    raise ValueError(f"Nenhum par marcado com {flag}")


def evaluate_groups(
    *,
    groups: Sequence[Mapping[str, Any]],
    model: tf.keras.Model,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_features: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        candidates = list(group["candidates"])
        features = np.stack([np.asarray(item["features"], dtype=np.float32) for item in candidates])
        all_features.append(features)
        offsets.append((cursor, cursor + features.shape[0]))
        cursor += features.shape[0]

    feature_matrix = np.concatenate(all_features, axis=0)
    normalized = (feature_matrix - mean.reshape(1, -1)) / std.reshape(1, -1)
    predicted_scores = model.predict(normalized, batch_size=batch_size, verbose=0).reshape(-1)

    rows: list[dict[str, Any]] = []
    gt_values: list[float] = []
    geom_values: list[float] = []
    oracle_values: list[float] = []
    aux_values: list[float] = []
    reranker_values: list[float] = []
    for group, (start, end) in zip(groups, offsets):
        candidates = list(group["candidates"])
        local_scores = predicted_scores[start:end]
        geom = select_pair_by_flag(candidates, "is_geom_pair")
        oracle = select_pair_by_flag(candidates, "is_oracle_pair")
        aux = select_pair_by_flag(candidates, "is_aux_pair") if bool(group["aux_available"]) else None
        reranker = candidates[int(np.argmax(local_scores))]
        gt = float(group["gt_cobb"])

        gt_values.append(gt)
        geom_values.append(float(geom["angle_deg"]))
        oracle_values.append(float(oracle["angle_deg"]))
        reranker_values.append(float(reranker["angle_deg"]))
        if aux is not None:
            aux_values.append(float(aux["angle_deg"]))

        rows.append(
            {
                "file_name": str(group["file_name"]),
                "gt_cobb_max": gt,
                "aux_available": int(bool(group["aux_available"])),
                "aux_cobb_max": float(group["aux_cobb"]) if bool(group["aux_available"]) else "",
                "geom_cobb": float(geom["angle_deg"]),
                "geom_abs_error": abs(float(geom["angle_deg"]) - gt),
                "oracle_cobb": float(oracle["angle_deg"]),
                "oracle_abs_error": abs(float(oracle["angle_deg"]) - gt),
                "aux_closest_cobb": float(aux["angle_deg"]) if aux is not None else "",
                "aux_closest_abs_error": abs(float(aux["angle_deg"]) - gt) if aux is not None else "",
                "reranker_cobb": float(reranker["angle_deg"]),
                "reranker_abs_error": abs(float(reranker["angle_deg"]) - gt),
                "reranker_score": float(np.max(local_scores)),
                "reranker_upper_index": int(reranker["upper_index"]),
                "reranker_lower_index": int(reranker["lower_index"]),
                "reranker_span": int(reranker["span"]),
                "oracle_upper_index": int(oracle["upper_index"]),
                "oracle_lower_index": int(oracle["lower_index"]),
                "geom_upper_index": int(geom["upper_index"]),
                "geom_lower_index": int(geom["lower_index"]),
                "candidate_count": int(len(candidates)),
            }
        )

    metrics = {
        "geometric_max_angle": metric_bundle(geom_values, gt_values),
        "endpoint_pair_oracle": metric_bundle(oracle_values, gt_values),
        "reranker": metric_bundle(reranker_values, gt_values),
    }
    if aux_values and len(aux_values) == len(groups):
        metrics["aux_closest_pair"] = metric_bundle(aux_values, gt_values)
    return metrics, rows


def make_report(
    *,
    path: Path,
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    experiment_dir: Path,
) -> None:
    holdout = metrics["holdout"]
    train = metrics["train"]
    lines = [
        "# Fase 9 - endpoint pair reranker v1",
        "",
        "## Objetivo",
        "",
        "Testar se um reranker pequeno consegue escolher pares de endpoints Cobb melhores do que o par de angulo maximo.",
        "",
        "## Configuracao",
        "",
        f"- modelo Fase 5: `{args.model_path}`",
        f"- perfil Fase 5: `{args.profile}`",
        f"- imagens avaliadas: `{args.num_images}`",
        f"- imagens de treino interno: `{metrics['num_train_images']}`",
        f"- imagens holdout interno: `{metrics['num_holdout_images']}`",
        f"- CSV aux: `{args.phase9_predictions_csv}`",
        "",
        "## Holdout interno",
        "",
        "| metodo | MAE | within5 | falhas >5 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("geometric_max_angle", "geometria max-angle"),
        ("aux_closest_pair", "par mais perto da Fase 8"),
        ("reranker", "reranker v1"),
        ("endpoint_pair_oracle", "oracle de endpoint"),
    ]:
        if key not in holdout:
            continue
        row = holdout[key]
        lines.append(
            "| "
            f"{label} | "
            f"{row.get('mae_deg', 0.0):.3f} | "
            f"{row.get('within_5deg_rate', 0.0):.3f} | "
            f"{row.get('failures_gt5', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Treino interno",
            "",
            "| metodo | MAE | within5 | falhas >5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("geometric_max_angle", "geometria max-angle"),
        ("aux_closest_pair", "par mais perto da Fase 8"),
        ("reranker", "reranker v1"),
        ("endpoint_pair_oracle", "oracle de endpoint"),
    ]:
        if key not in train:
            continue
        row = train[key]
        lines.append(
            "| "
            f"{label} | "
            f"{row.get('mae_deg', 0.0):.3f} | "
            f"{row.get('within_5deg_rate', 0.0):.3f} | "
            f"{row.get('failures_gt5', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            f"- metricas: `{experiment_dir / 'phase9_endpoint_pair_reranker_metrics.json'}`",
            f"- predicoes treino: `{experiment_dir / 'phase9_endpoint_pair_reranker_train_predictions.csv'}`",
            f"- predicoes holdout: `{experiment_dir / 'phase9_endpoint_pair_reranker_holdout_predictions.csv'}`",
            f"- modelo: `{experiment_dir / 'phase9_endpoint_pair_reranker_v1.keras'}`",
            f"- scaler/features: `{experiment_dir / 'phase9_endpoint_pair_reranker_scaler.npz'}`",
            "",
            "## Nota",
            "",
            "Este treino usa labels Cobb reais apenas no split interno de treino. O holdout interno serve como primeira estimativa; nao deve ser tratado como resultado final congelado.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina endpoint pair reranker v1.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--phase9-predictions-csv", default=str(DEFAULT_V6_PREDICTIONS))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--eval-mode", choices=("validation", "window"), default="validation")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--train-size", type=int, default=12768)
    parser.add_argument("--val-size", type=int, default=3192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--train-images", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(int(args.seed))

    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval")
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    phase9_v1 = import_script("32_eval_phase9_final_cobb_v1.py", "phase9_v1")
    oracle_script = import_script("56_oracle_phase5_sequence_portfolio_cobb_v1.py", "phase5_cobb_oracle")

    if args.profile not in phase5_sequence.PROFILE_CONFIGS:
        raise ValueError(f"Perfil desconhecido: {args.profile}")

    model_path = resolve_project_path(args.model_path)
    aux_csv = resolve_project_path(args.phase9_predictions_csv)
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    aux_by_file = load_aux_cobb_by_file(aux_csv)

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
        "A gerar pares endpoint: "
        f"imagens={len(selected_samples)}, perfil={args.profile}, aux_csv={aux_csv.is_file()}"
    )

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
        candidates = build_pair_candidates(
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

    if len(groups) < 4:
        raise ValueError("Poucas imagens com candidatos para treinar o reranker.")
    train_count = min(max(int(args.train_images), 1), len(groups) - 1)
    train_groups = groups[:train_count]
    holdout_groups = groups[train_count:]
    if not holdout_groups:
        raise ValueError("Holdout vazio. Reduz --train-images.")

    train_candidates = [candidate for group in train_groups for candidate in group["candidates"]]
    train_features = np.stack([np.asarray(candidate["features"], dtype=np.float32) for candidate in train_candidates])
    train_targets = np.asarray([float(candidate["target_score"]) for candidate in train_candidates], dtype=np.float32)
    train_weights = np.asarray(
        [0.2 + 2.0 * float(candidate["target_score"]) + 3.0 * int(candidate["is_oracle_pair"]) for candidate in train_candidates],
        dtype=np.float32,
    )
    mean = np.mean(train_features, axis=0).astype(np.float32)
    std = np.std(train_features, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    train_features_norm = (train_features - mean.reshape(1, -1)) / std.reshape(1, -1)

    reranker = build_model(
        input_dim=train_features_norm.shape[1],
        hidden_units=int(args.hidden_units),
        dropout_rate=float(args.dropout_rate),
        learning_rate=float(args.learning_rate),
    )
    history = reranker.fit(
        train_features_norm,
        train_targets,
        sample_weight=train_weights,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=0,
    )

    train_metrics, train_rows = evaluate_groups(
        groups=train_groups,
        model=reranker,
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
    )
    holdout_metrics, holdout_rows = evaluate_groups(
        groups=holdout_groups,
        model=reranker,
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
    )

    metrics = {
        "phase": "phase9_endpoint_pair_reranker_v1",
        "model_path": str(model_path),
        "phase9_predictions_csv": str(aux_csv),
        "eval": eval_metadata,
        "profile": args.profile,
        "feature_names": FEATURE_NAMES,
        "num_images": int(len(groups)),
        "num_train_images": int(len(train_groups)),
        "num_holdout_images": int(len(holdout_groups)),
        "num_train_pairs": int(len(train_candidates)),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }

    reranker.save(experiment_dir / "phase9_endpoint_pair_reranker_v1.keras")
    np.savez(
        experiment_dir / "phase9_endpoint_pair_reranker_scaler.npz",
        mean=mean,
        std=std,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    write_json(experiment_dir / "phase9_endpoint_pair_reranker_metrics.json", metrics)
    row_fields = [
        "file_name",
        "gt_cobb_max",
        "aux_available",
        "aux_cobb_max",
        "geom_cobb",
        "geom_abs_error",
        "oracle_cobb",
        "oracle_abs_error",
        "aux_closest_cobb",
        "aux_closest_abs_error",
        "reranker_cobb",
        "reranker_abs_error",
        "reranker_score",
        "reranker_upper_index",
        "reranker_lower_index",
        "reranker_span",
        "oracle_upper_index",
        "oracle_lower_index",
        "geom_upper_index",
        "geom_lower_index",
        "candidate_count",
    ]
    write_csv(experiment_dir / "phase9_endpoint_pair_reranker_train_predictions.csv", train_rows, row_fields)
    write_csv(experiment_dir / "phase9_endpoint_pair_reranker_holdout_predictions.csv", holdout_rows, row_fields)
    make_report(path=report_path, args=args, metrics=metrics, experiment_dir=experiment_dir)

    print("\nResumo endpoint pair reranker v1")
    print(f"imagens treino/holdout: {len(train_groups)}/{len(holdout_groups)}")
    for split_name, split_metrics in [("train", train_metrics), ("holdout", holdout_metrics)]:
        print(f"{split_name}:")
        for key in ("geometric_max_angle", "aux_closest_pair", "reranker", "endpoint_pair_oracle"):
            if key not in split_metrics:
                continue
            row = split_metrics[key]
            print(
                f"  {key}: MAE={row['mae_deg']:.3f}, "
                f"within5={row['within_5deg_rate']:.3f}, "
                f"falhas>5={row['failures_gt5']}"
            )
    print(f"Metricas guardadas em: {experiment_dir / 'phase9_endpoint_pair_reranker_metrics.json'}")
    print(f"Relatorio guardado em: {report_path}")


if __name__ == "__main__":
    main()
