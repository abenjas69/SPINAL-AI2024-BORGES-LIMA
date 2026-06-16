"""Fase 9: avaliacao integrada do Cobb final.

Esta primeira versao nao re-treina a arquitetura inteira. Ela integra os
artefactos ja validados:

- pontos/quadrilateros finais guardados no NPZ da Fase 7;
- embeddings contextualizados da Fase 7;
- head auxiliar limpa da Fase 8, opcionalmente.

O Cobb final principal e calculado geometricamente a partir dos quatro pontos
previstos por vertebra. A head auxiliar e reportada como referencia global, mas
nao substitui a medicao geometrica.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from html import escape
from pathlib import Path
from types import ModuleType
from typing import Any

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

MAX_VERTEBRAE = 21
ANGLE_NAMES = ("PT", "MT", "TLL")
DEFAULT_NPZ = (
    OUTPUTS_DIR
    / "phase7_bilstm_global_probe_colab_v1_noshuffle"
    / "phase7_contextual_embeddings_probe_colab_v1_noshuffle.npz"
)
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "processed" / "cleaned" / "train_ready_annotations_clean.json"
DEFAULT_PHASE8_MODEL = (
    PROJECT_ROOT
    / "models"
    / "phase8_aux_angle_head_weighted_probe_colab_v2_noshuffle.keras"
)


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


def load_context_npz(path: Path) -> dict[str, np.ndarray]:
    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        required = (
            "contextual_embeddings",
            "mask",
            "points",
            "scores",
            "gt_counts",
            "cobb_angles",
            "file_names",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"O NPZ nao contem as chaves esperadas: {missing}")
        arrays = {name: data[name] for name in data.files}

    arrays["contextual_embeddings"] = arrays["contextual_embeddings"].astype(np.float32)
    arrays["mask"] = arrays["mask"].astype(np.float32)
    arrays["points"] = arrays["points"].astype(np.float32)
    arrays["scores"] = arrays["scores"].astype(np.float32)
    arrays["gt_counts"] = arrays["gt_counts"].astype(np.float32)
    arrays["cobb_angles"] = arrays["cobb_angles"].astype(np.float32)

    num_samples, max_vertebrae, _ = arrays["contextual_embeddings"].shape
    if arrays["mask"].shape != (num_samples, max_vertebrae):
        raise ValueError("mask nao combina com contextual_embeddings")
    if arrays["points"].shape != (num_samples, max_vertebrae, 8):
        raise ValueError("points deve ter shape (N, 21, 8)")
    if arrays["scores"].shape != (num_samples, max_vertebrae):
        raise ValueError("scores deve ter shape (N, 21)")
    if arrays["cobb_angles"].shape != (num_samples, len(ANGLE_NAMES)):
        raise ValueError("cobb_angles deve ter shape (N, 3)")

    return arrays


def load_dimensions_by_file_name(path: Path) -> dict[str, tuple[float, float]]:
    require_file(path)
    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    dimensions: dict[str, tuple[float, float]] = {}
    for sample in samples:
        dimensions[str(sample["file_name"])] = (float(sample["width"]), float(sample["height"]))
    return dimensions


def select_indices(
    num_samples: int,
    train_size: int,
    val_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples, dtype=np.int32)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_count = num_samples if train_size <= 0 else min(train_size, num_samples)
    remaining = max(num_samples - train_count, 0)
    val_count = remaining if val_size <= 0 else min(val_size, remaining)
    if val_count <= 0 and num_samples > train_count:
        val_count = num_samples - train_count
    return indices[:train_count], indices[train_count : train_count + val_count]


def choose_eval_indices(args: argparse.Namespace, num_samples: int) -> np.ndarray:
    mode = args.eval_mode.lower()
    if mode == "all":
        return np.arange(num_samples, dtype=np.int32)
    if mode == "range":
        start = max(int(args.start_index), 0)
        count = num_samples - start if args.num_images <= 0 else int(args.num_images)
        end = min(start + count, num_samples)
        if start >= end:
            raise ValueError("Range de avaliacao vazio.")
        return np.arange(start, end, dtype=np.int32)
    if mode == "validation":
        _, val_indices = select_indices(
            num_samples=num_samples,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed,
            shuffle=not args.no_shuffle,
        )
        if len(val_indices) == 0:
            raise ValueError("Split de validacao vazio.")
        return val_indices
    raise ValueError("--eval-mode deve ser validation, all ou range.")


def line_angle_deg(point_a: np.ndarray, point_b: np.ndarray, width: float, height: float) -> float:
    dx = (float(point_b[0]) - float(point_a[0])) * width
    dy = (float(point_b[1]) - float(point_a[1])) * height
    return math.degrees(math.atan2(dy, dx))


def angle_diff_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(float(angle_a) - float(angle_b)) % 180.0
    return min(diff, 180.0 - diff)


def geometric_final_cobb(
    points: np.ndarray,
    width: float,
    height: float,
) -> dict[str, Any]:
    valid_points = np.asarray(points, dtype=np.float32).reshape(-1, 8)
    count = int(valid_points.shape[0])
    if count < 2:
        return {
            "cobb_deg": None,
            "upper_index": -1,
            "lower_index": -1,
            "upper_angle_deg": None,
            "lower_angle_deg": None,
            "span": 0,
        }

    top_angles: list[float] = []
    bottom_angles: list[float] = []
    for row in valid_points:
        upper_left = row[0:2]
        upper_right = row[2:4]
        lower_left = row[4:6]
        lower_right = row[6:8]
        top_angles.append(line_angle_deg(upper_left, upper_right, width, height))
        bottom_angles.append(line_angle_deg(lower_left, lower_right, width, height))

    best_angle = -1.0
    best_upper = 0
    best_lower = 1
    for upper_index in range(count - 1):
        for lower_index in range(upper_index + 1, count):
            candidate = angle_diff_deg(top_angles[upper_index], bottom_angles[lower_index])
            if candidate > best_angle:
                best_angle = candidate
                best_upper = upper_index
                best_lower = lower_index

    return {
        "cobb_deg": float(best_angle),
        "upper_index": int(best_upper),
        "lower_index": int(best_lower),
        "upper_angle_deg": float(top_angles[best_upper]),
        "lower_angle_deg": float(bottom_angles[best_lower]),
        "span": int(best_lower - best_upper),
    }


def safe_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def scalar_metrics(pred_values: np.ndarray, gt_values: np.ndarray) -> dict[str, float | None]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    errors = pred - gt
    abs_errors = np.abs(errors)
    metrics: dict[str, float | None] = {
        "mae_deg": safe_float(np.mean(abs_errors)),
        "rmse_deg": safe_float(np.sqrt(np.mean(errors**2))),
        "bias_deg": safe_float(np.mean(errors)),
        "median_abs_error_deg": safe_float(np.median(abs_errors)),
        "p90_abs_error_deg": safe_float(np.percentile(abs_errors, 90)),
        "within_5deg_rate": safe_float(np.mean(abs_errors <= 5.0)),
        "within_10deg_rate": safe_float(np.mean(abs_errors <= 10.0)),
        "mean_pred_deg": safe_float(np.mean(pred)),
        "mean_gt_deg": safe_float(np.mean(gt)),
    }
    if pred.size >= 2 and np.ptp(pred) > 1.0e-6 and np.ptp(gt) > 1.0e-6:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(np.corrcoef(gt, pred)[0, 1])
        metrics["pearson"] = safe_float(corr) if np.isfinite(corr) else None
    else:
        metrics["pearson"] = None
    total = float(np.sum((gt - np.mean(gt)) ** 2))
    if total > 1.0e-8:
        metrics["r2"] = safe_float(1.0 - float(np.sum((gt - pred) ** 2)) / total)
    else:
        metrics["r2"] = None
    return metrics


def count_metrics(pred_counts: np.ndarray, gt_counts: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_counts, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_counts, dtype=np.float32).reshape(-1)
    errors = pred - gt
    abs_errors = np.abs(errors)
    return {
        "mae": float(np.mean(abs_errors)),
        "exact_accuracy": float(np.mean(errors == 0.0)),
        "under_count_rate": float(np.mean(errors < 0.0)),
        "over_count_rate": float(np.mean(errors > 0.0)),
        "mean_pred_count": float(np.mean(pred)),
        "mean_gt_count": float(np.mean(gt)),
    }


def predict_aux_angles(
    model_path: Path,
    contextual_embeddings: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    require_file(model_path)
    import_script("29_aux_angle_head_v1.py", "phase8_models_for_loading")
    model = tf.keras.models.load_model(model_path)
    outputs = model.predict(
        {"contextual_embeddings": contextual_embeddings, "mask": mask},
        batch_size=batch_size,
        verbose=0,
    )
    if isinstance(outputs, dict):
        return np.asarray(outputs["angle_deg"], dtype=np.float32)
    output_values = outputs if isinstance(outputs, list) else [outputs]
    output_map = dict(zip(model.output_names, output_values))
    return np.asarray(output_map["angle_deg"], dtype=np.float32)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], include_aux: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "valid_count",
        "gt_count",
        "mean_score",
        "gt_PT",
        "gt_MT",
        "gt_TLL",
        "gt_cobb_max",
        "gt_major_region",
        "geom_cobb",
        "geom_abs_error",
        "geom_upper_index",
        "geom_lower_index",
        "geom_span",
        "geom_upper_angle",
        "geom_lower_angle",
    ]
    if include_aux:
        fieldnames.extend(
            [
                "aux_PT",
                "aux_MT",
                "aux_TLL",
                "aux_cobb_max",
                "aux_abs_error",
                "aux_major_region",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_scatter_svg(
    path: Path,
    gt_values: np.ndarray,
    series: list[tuple[str, np.ndarray, dict[str, float | None]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_width = 360
    plot_size = 235
    left = 62
    top = 62
    gap = 36
    width = left + len(series) * (panel_width + gap) + 30
    height = 380
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#111827}'
        '.title{font-size:16px;font-weight:700}.axis{stroke:#111827;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}.diag{stroke:#dc2626;stroke-width:1.6;stroke-dasharray:6 5}'
        '.point{fill:#1d4ed8;fill-opacity:.35}</style>',
        '<text class="title" x="24" y="30">Fase 9 - Cobb final: predito vs ground truth</text>',
    ]
    for index, (name, pred_values, metrics) in enumerate(series):
        x0 = left + index * (panel_width + gap)
        y0 = top
        gt = np.asarray(gt_values, dtype=np.float32)
        pred = np.asarray(pred_values, dtype=np.float32)
        lower = float(min(np.min(gt), np.min(pred)))
        upper = float(max(np.max(gt), np.max(pred)))
        pad = max((upper - lower) * 0.06, 1.0)
        lower -= pad
        upper += pad
        scale = max(upper - lower, 1.0)

        def sx(value: float) -> float:
            return x0 + (float(value) - lower) / scale * plot_size

        def sy(value: float) -> float:
            return y0 + plot_size - (float(value) - lower) / scale * plot_size

        title = (
            f"{name} MAE={metrics.get('mae_deg'):.2f} "
            f"r={metrics.get('pearson'):.3f}"
            if metrics.get("pearson") is not None
            else f"{name} MAE={metrics.get('mae_deg'):.2f}"
        )
        elements.append(f'<text class="title" x="{x0}" y="{y0 - 18}">{escape(title)}</text>')
        for tick in np.linspace(lower, upper, 5):
            xt = sx(float(tick))
            yt = sy(float(tick))
            elements.append(f'<line class="grid" x1="{xt:.2f}" y1="{y0}" x2="{xt:.2f}" y2="{y0 + plot_size}"/>')
            elements.append(f'<line class="grid" x1="{x0}" y1="{yt:.2f}" x2="{x0 + plot_size}" y2="{yt:.2f}"/>')
        elements.append(f'<rect class="axis" x="{x0}" y="{y0}" width="{plot_size}" height="{plot_size}" fill="none"/>')
        elements.append(
            f'<line class="diag" x1="{sx(lower):.2f}" y1="{sy(lower):.2f}" '
            f'x2="{sx(upper):.2f}" y2="{sy(upper):.2f}"/>'
        )
        for gt_value, pred_value in zip(gt, pred):
            elements.append(
                f'<circle class="point" cx="{sx(float(gt_value)):.2f}" '
                f'cy="{sy(float(pred_value)):.2f}" r="2.0"/>'
            )
        elements.append(f'<text x="{x0}" y="{y0 + plot_size + 34}">Ground truth Cobb max (graus)</text>')
        elements.append(
            f'<text x="{x0 - 46}" y="{y0 + plot_size / 2}" '
            f'transform="rotate(-90 {x0 - 46} {y0 + plot_size / 2})">Predito (graus)</text>'
        )
    elements.append('<text x="24" y="356">Linha vermelha tracejada: previsao perfeita y=x.</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia a Fase 9 Cobb final geometrico.")
    parser.add_argument("--npz-path", default=str(DEFAULT_NPZ))
    parser.add_argument("--annotations-path", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--phase8-model-path", default=str(DEFAULT_PHASE8_MODEL))
    parser.add_argument("--skip-aux-head", action="store_true")
    parser.add_argument("--eval-mode", choices=("validation", "all", "range"), default="validation")
    parser.add_argument("--train-size", type=int, default=12768)
    parser.add_argument("--val-size", type=int, default=3192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", dest="no_shuffle", action="store_false")
    parser.add_argument("--no-shuffle", dest="no_shuffle", action="store_true")
    parser.set_defaults(no_shuffle=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--experiment-name", default="phase9_probe_colab_final_cobb_v1")
    parser.add_argument("--output-dir", default="outputs/phase9_probe_colab_final_cobb_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_project_path(args.npz_path)
    annotations_path = resolve_project_path(args.annotations_path)
    output_dir = resolve_project_path(args.output_dir)
    experiment_dir = resolve_project_path(str(EXPERIMENTS_DIR / args.experiment_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_context_npz(npz_path)
    dimensions = load_dimensions_by_file_name(annotations_path)
    eval_indices = choose_eval_indices(args, arrays["contextual_embeddings"].shape[0])

    aux_predictions: np.ndarray | None = None
    phase8_model_path: Path | None = None
    if not args.skip_aux_head:
        phase8_model_path = resolve_project_path(args.phase8_model_path)
        aux_predictions = predict_aux_angles(
            model_path=phase8_model_path,
            contextual_embeddings=arrays["contextual_embeddings"][eval_indices],
            mask=arrays["mask"][eval_indices],
            batch_size=args.batch_size,
        )

    rows: list[dict[str, Any]] = []
    geom_values: list[float] = []
    gt_max_values: list[float] = []
    aux_max_values: list[float] = []
    valid_counts: list[int] = []
    gt_counts: list[float] = []

    for local_index, sample_index in enumerate(eval_indices):
        file_name = str(arrays["file_names"][sample_index])
        if file_name not in dimensions:
            raise KeyError(f"Dimensoes nao encontradas para {file_name}")
        width, height = dimensions[file_name]
        valid_mask = arrays["mask"][sample_index] > 0.5
        valid_points = arrays["points"][sample_index][valid_mask]
        valid_scores = arrays["scores"][sample_index][valid_mask]
        geom = geometric_final_cobb(valid_points, width=width, height=height)
        if geom["cobb_deg"] is None:
            continue

        gt_angles = arrays["cobb_angles"][sample_index].astype(np.float32)
        gt_max = float(np.max(gt_angles))
        gt_major_index = int(np.argmax(gt_angles))
        geom_cobb = float(geom["cobb_deg"])
        row: dict[str, Any] = {
            "file_name": file_name,
            "valid_count": int(np.sum(valid_mask)),
            "gt_count": int(arrays["gt_counts"][sample_index]),
            "mean_score": f"{float(np.mean(valid_scores)):.6f}" if valid_scores.size else "",
            "gt_PT": f"{float(gt_angles[0]):.6f}",
            "gt_MT": f"{float(gt_angles[1]):.6f}",
            "gt_TLL": f"{float(gt_angles[2]):.6f}",
            "gt_cobb_max": f"{gt_max:.6f}",
            "gt_major_region": ANGLE_NAMES[gt_major_index],
            "geom_cobb": f"{geom_cobb:.6f}",
            "geom_abs_error": f"{abs(geom_cobb - gt_max):.6f}",
            "geom_upper_index": int(geom["upper_index"]),
            "geom_lower_index": int(geom["lower_index"]),
            "geom_span": int(geom["span"]),
            "geom_upper_angle": f"{float(geom['upper_angle_deg']):.6f}",
            "geom_lower_angle": f"{float(geom['lower_angle_deg']):.6f}",
        }

        if aux_predictions is not None:
            aux_angles = aux_predictions[local_index].astype(np.float32)
            aux_max = float(np.max(aux_angles))
            aux_major_index = int(np.argmax(aux_angles))
            aux_max_values.append(aux_max)
            row.update(
                {
                    "aux_PT": f"{float(aux_angles[0]):.6f}",
                    "aux_MT": f"{float(aux_angles[1]):.6f}",
                    "aux_TLL": f"{float(aux_angles[2]):.6f}",
                    "aux_cobb_max": f"{aux_max:.6f}",
                    "aux_abs_error": f"{abs(aux_max - gt_max):.6f}",
                    "aux_major_region": ANGLE_NAMES[aux_major_index],
                }
            )

        rows.append(row)
        geom_values.append(geom_cobb)
        gt_max_values.append(gt_max)
        valid_counts.append(int(np.sum(valid_mask)))
        gt_counts.append(float(arrays["gt_counts"][sample_index]))

    geom_array = np.asarray(geom_values, dtype=np.float32)
    gt_max_array = np.asarray(gt_max_values, dtype=np.float32)
    metrics: dict[str, Any] = {
        "phase": "phase9_final_cobb_v1",
        "npz_path": str(npz_path),
        "annotations_path": str(annotations_path),
        "phase8_model_path": str(phase8_model_path) if phase8_model_path else None,
        "eval_mode": args.eval_mode,
        "num_images": int(len(rows)),
        "index_count_requested": int(len(eval_indices)),
        "geometric_final_cobb": scalar_metrics(geom_array, gt_max_array),
        "sequence_count": count_metrics(np.asarray(valid_counts), np.asarray(gt_counts)),
        "mean_selected_span": float(np.mean([int(row["geom_span"]) for row in rows])) if rows else None,
    }
    series = [("Geometrico", geom_array, metrics["geometric_final_cobb"])]

    if aux_predictions is not None and aux_max_values:
        aux_array = np.asarray(aux_max_values, dtype=np.float32)
        metrics["auxiliary_head_max"] = scalar_metrics(aux_array, gt_max_array)
        series.append(("Head auxiliar", aux_array, metrics["auxiliary_head_max"]))

    write_json(experiment_dir / "metrics.json", metrics)
    write_json(
        experiment_dir / "config.json",
        {
            "args": vars(args),
            "eval_indices": eval_indices.astype(int).tolist(),
        },
    )
    write_predictions_csv(
        experiment_dir / "phase9_predictions.csv",
        rows,
        include_aux=aux_predictions is not None,
    )
    save_scatter_svg(output_dir / "phase9_cobb_scatter.svg", gt_max_array, series)

    print("Resumo Fase 9 - Cobb final")
    print(f"NPZ: {npz_path}")
    print(f"modo avaliacao: {args.eval_mode}")
    print(f"imagens avaliadas: {len(rows)}")
    geom_metrics = metrics["geometric_final_cobb"]
    print(
        "Cobb geometrico: "
        f"MAE={geom_metrics['mae_deg']:.3f} deg, "
        f"RMSE={geom_metrics['rmse_deg']:.3f} deg, "
        f"r={geom_metrics['pearson']:.3f}, "
        f"within10={geom_metrics['within_10deg_rate']:.3f}"
    )
    if "auxiliary_head_max" in metrics:
        aux_metrics = metrics["auxiliary_head_max"]
        print(
            "Head auxiliar max: "
            f"MAE={aux_metrics['mae_deg']:.3f} deg, "
            f"RMSE={aux_metrics['rmse_deg']:.3f} deg, "
            f"r={aux_metrics['pearson']:.3f}, "
            f"within10={aux_metrics['within_10deg_rate']:.3f}"
        )
    count = metrics["sequence_count"]
    print(
        "Contagem sequencia: "
        f"MAE={count['mae']:.3f}, exact={count['exact_accuracy']:.3f}"
    )
    print(f"Metricas guardadas em: {experiment_dir / 'metrics.json'}")
    print(f"Predicoes guardadas em: {experiment_dir / 'phase9_predictions.csv'}")
    print(f"Grafico guardado em: {output_dir / 'phase9_cobb_scatter.svg'}")


if __name__ == "__main__":
    main()
