"""Fase 6: geracao de embeddings vertebrais.

Esta fase recebe as previsoes prediction-driven da Fase 5, aplica o mesmo
pos-processamento anatomico e transforma cada vertebra final numa representacao
compacta para a futura modelacao sequencial da Fase 7.

O modelo treinado da Fase 5 fica congelado e e usado como extrator:

- feature visual local: ROI pooling simples sobre `fpn_p3` e a ultima feature
  convolucional da head;
- feature geometrica explicita: centroide, bbox, pontos, score, posicao da
  celula e inclinacoes das faces vertebrais;
- embedding final: visual L2-normalizado + geometria normalizada.

O output principal e um `.npz` com sequencias padded ate `MAX_VERTEBRAE`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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
MAX_VERTEBRAE = 21
GRID_SIZE = 64
VISUAL_FEATURE_DIM = 128

GEOMETRY_FEATURE_NAMES = (
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "upper_left_x",
    "upper_left_y",
    "upper_right_x",
    "upper_right_y",
    "lower_left_x",
    "lower_left_y",
    "lower_right_x",
    "lower_right_y",
    "score",
    "grid_row",
    "grid_col",
    "sequence_position",
    "upper_slope",
    "lower_slope",
    "left_slope",
    "right_slope",
    "log_aspect",
    "area",
)

GEOMETRY_FEATURE_DIM = len(GEOMETRY_FEATURE_NAMES)
EMBEDDING_DIM = VISUAL_FEATURE_DIM + GEOMETRY_FEATURE_DIM


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


def select_window(items: Sequence[Any], start_index: int, num_items: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_items <= 0:
        return items[start_index:]
    return items[start_index : start_index + num_items]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {path}")


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm < epsilon:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32)


def candidate_grid_position(candidate_index: int, grid_size: int = GRID_SIZE) -> tuple[float, float]:
    if candidate_index < 0:
        return 0.0, 0.0
    row = int(candidate_index) // grid_size
    col = int(candidate_index) % grid_size
    denominator = max(float(grid_size - 1), 1.0)
    return float(row) / denominator, float(col) / denominator


def pool_feature_at_point(
    feature_map: np.ndarray,
    x_norm: float,
    y_norm: float,
    radius: int = 1,
) -> np.ndarray:
    """Extrai uma pequena media local de um mapa HxWxC em coordenadas [0, 1]."""
    feature_values = np.asarray(feature_map, dtype=np.float32)
    height, width, _ = feature_values.shape
    row = int(np.clip(round(float(y_norm) * float(height - 1)), 0, height - 1))
    col = int(np.clip(round(float(x_norm) * float(width - 1)), 0, width - 1))

    row0 = max(0, row - radius)
    row1 = min(height, row + radius + 1)
    col0 = max(0, col - radius)
    col1 = min(width, col + radius + 1)
    return np.mean(feature_values[row0:row1, col0:col1], axis=(0, 1)).astype(np.float32)


def slope_feature(start: np.ndarray, end: np.ndarray) -> float:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    return float(math.atan2(dy, dx) / math.pi)


def geometry_features(
    points: np.ndarray,
    bbox: np.ndarray,
    score: float,
    candidate_index: int,
    sequence_index: int,
    sequence_count: int,
) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32).reshape(4, 2)
    bbox_array = np.asarray(bbox, dtype=np.float32).reshape(4)
    centroid = np.mean(points_array, axis=0)
    grid_row, grid_col = candidate_grid_position(candidate_index)
    sequence_position = (
        float(sequence_index) / float(sequence_count - 1)
        if sequence_count > 1
        else 0.0
    )

    width = max(float(bbox_array[2]), 1e-6)
    height = max(float(bbox_array[3]), 1e-6)
    upper_left, upper_right, lower_left, lower_right = points_array
    values = [
        float(centroid[0]),
        float(centroid[1]),
        float(bbox_array[0]),
        float(bbox_array[1]),
        float(bbox_array[2]),
        float(bbox_array[3]),
        *[float(value) for value in points_array.reshape(-1)],
        float(score),
        grid_row,
        grid_col,
        sequence_position,
        slope_feature(upper_left, upper_right),
        slope_feature(lower_left, lower_right),
        slope_feature(upper_left, lower_left),
        slope_feature(upper_right, lower_right),
        float(np.clip(math.log(width / height), -3.0, 3.0) / 3.0),
        float(np.clip(width * height, 0.0, 1.0)),
    ]
    return np.asarray(values, dtype=np.float32)


def visual_feature_for_candidate(
    p3_map: np.ndarray,
    head_map: np.ndarray,
    points: np.ndarray,
    candidate_index: int,
) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centroid = np.mean(points_array, axis=0)
    grid_row, grid_col = candidate_grid_position(candidate_index)

    p3_roi = pool_feature_at_point(p3_map, float(centroid[0]), float(centroid[1]), radius=1)
    head_roi = pool_feature_at_point(head_map, float(centroid[0]), float(centroid[1]), radius=1)
    head_cell = pool_feature_at_point(head_map, grid_col, grid_row, radius=0)
    stacked = np.stack(
        [
            l2_normalize(p3_roi),
            l2_normalize(head_roi),
            l2_normalize(head_cell),
        ],
        axis=0,
    )
    return l2_normalize(np.mean(stacked, axis=0))


def build_feature_prediction_model(model: tf.keras.Model) -> tf.keras.Model:
    """Cria um modelo auxiliar que devolve features locais e previsoes."""
    required_layers = ("fpn_p3", "spatial_offset_head_conv2", "presence", "bbox", "points")
    missing = [name for name in required_layers if not any(layer.name == name for layer in model.layers)]
    if missing:
        raise ValueError(f"O modelo nao tem as camadas esperadas para a Fase 6: {missing}")

    return tf.keras.Model(
        inputs=model.input,
        outputs={
            "p3": model.get_layer("fpn_p3").output,
            "head_features": model.get_layer("spatial_offset_head_conv2").output,
            "presence": model.get_layer("presence").output,
            "bbox": model.get_layer("bbox").output,
            "points": model.get_layer("points").output,
        },
        name="phase6_feature_prediction_model",
    )


def make_empty_sequence_arrays() -> dict[str, np.ndarray]:
    return {
        "embeddings": np.zeros((MAX_VERTEBRAE, EMBEDDING_DIM), dtype=np.float32),
        "mask": np.zeros((MAX_VERTEBRAE,), dtype=np.float32),
        "points": np.zeros((MAX_VERTEBRAE, 8), dtype=np.float32),
        "bbox": np.zeros((MAX_VERTEBRAE, 4), dtype=np.float32),
        "scores": np.zeros((MAX_VERTEBRAE,), dtype=np.float32),
        "candidate_indices": np.full((MAX_VERTEBRAE,), -1, dtype=np.int32),
    }


def make_sequence_diagram(
    image: tf.Tensor,
    targets: Mapping[str, Any],
    selected_points: np.ndarray,
    drawing: ModuleType,
) -> np.ndarray:
    canvas = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
    gt_count = int(targets["vertebra_count"])
    gt_points = np.asarray(targets["points"][:gt_count], dtype=np.float32).reshape(gt_count, 4, 2) * IMAGE_SIZE
    pred_points = np.asarray(selected_points, dtype=np.float32).reshape(-1, 4, 2) * IMAGE_SIZE

    pred_color = np.array([255, 0, 0], dtype=np.uint8)
    node_color = np.array([115, 60, 180], dtype=np.uint8)
    line_color = np.array([80, 80, 80], dtype=np.uint8)

    drawing.draw_ground_truth_points(canvas, gt_points)
    for vertebra_points in pred_points:
        drawing.draw_quadrilateral(canvas, vertebra_points, pred_color)

    side = np.full((canvas.shape[0], 160, 3), 245, dtype=np.uint8)
    if pred_points.size:
        y_centers = np.mean(pred_points[:, :, 1], axis=1)
        x_center = 80.0
        previous = None
        for y_value in y_centers:
            current = np.asarray([x_center, float(y_value)], dtype=np.float32)
            if previous is not None:
                drawing.draw_line(side, previous, current, line_color, radius=1)
            drawing.draw_point(side, x_center, float(y_value), node_color, radius=5)
            previous = current

    separator = np.full((canvas.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([canvas, separator, side], axis=1)


def extract_sequence_for_image(
    image: tf.Tensor,
    sample: Mapping[str, Any],
    targets: Mapping[str, Any],
    feature_outputs: Mapping[str, tf.Tensor],
    postprocess: ModuleType,
    postprocess_config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del image

    values = {
        key: np.asarray(value.numpy()[0], dtype=np.float32)
        for key, value in feature_outputs.items()
    }
    result = postprocess.postprocess_candidates_sequence(
        presence=values["presence"],
        bbox=values["bbox"],
        points=values["points"],
        **dict(postprocess_config),
    )

    arrays = make_empty_sequence_arrays()
    selected_indices = np.asarray(result["selected_indices"], dtype=np.int32)
    selected_points = np.asarray(result["selected_points"], dtype=np.float32).reshape(-1, 8)
    selected_bbox = np.asarray(result["selected_bbox"], dtype=np.float32).reshape(-1, 4)
    selected_scores = np.asarray(result["selected_scores"], dtype=np.float32).reshape(-1)
    sequence_count = min(int(selected_indices.size), MAX_VERTEBRAE)

    for seq_index in range(sequence_count):
        candidate_index = int(selected_indices[seq_index])
        points = selected_points[seq_index]
        bbox = selected_bbox[seq_index]
        score = float(selected_scores[seq_index])
        visual = visual_feature_for_candidate(
            p3_map=values["p3"],
            head_map=values["head_features"],
            points=points,
            candidate_index=candidate_index,
        )
        geometry = geometry_features(
            points=points,
            bbox=bbox,
            score=score,
            candidate_index=candidate_index,
            sequence_index=seq_index,
            sequence_count=sequence_count,
        )
        arrays["embeddings"][seq_index] = np.concatenate([visual, geometry]).astype(np.float32)
        arrays["mask"][seq_index] = 1.0
        arrays["points"][seq_index] = points
        arrays["bbox"][seq_index] = bbox
        arrays["scores"][seq_index] = score
        arrays["candidate_indices"][seq_index] = candidate_index

    y_centroids = (
        np.mean(selected_points[:sequence_count].reshape(sequence_count, 4, 2)[:, :, 1], axis=1)
        if sequence_count > 0
        else np.asarray([], dtype=np.float32)
    )
    order_is_valid = bool(np.all(np.diff(y_centroids) >= -1e-6))
    gt_count = int(targets["vertebra_count"])
    record = {
        "file_name": str(sample["file_name"]),
        "gt_count": gt_count,
        "embedding_count": sequence_count,
        "count_error": sequence_count - gt_count,
        "raw_count": int(result["raw_count"]),
        "nms_count": int(result["nms_count"]),
        "gap_filled_count": int(result["gap_filled_count"]),
        "endpoint_pruned_top": int(result["endpoint_pruned_top"]),
        "endpoint_pruned_bottom": int(result["endpoint_pruned_bottom"]),
        "endpoint_filled_top": int(result.get("endpoint_filled_top", 0)),
        "endpoint_filled_bottom": int(result.get("endpoint_filled_bottom", 0)),
        "endpoint_fill_candidate_count": int(result.get("endpoint_fill_candidate_count", 0)),
        "mean_score": float(np.mean(selected_scores[:sequence_count])) if sequence_count else 0.0,
        "sequence_order_ok": order_is_valid,
        "sequence_method": str(result.get("sequence_method", "")),
        "path_score": float(result.get("path_score", 0.0)),
        "selected_indices": " ".join(str(int(index)) for index in selected_indices[:sequence_count]),
    }
    return arrays, record


def write_details_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "file_name",
        "gt_count",
        "embedding_count",
        "count_error",
        "raw_count",
        "nms_count",
        "gap_filled_count",
        "endpoint_pruned_top",
        "endpoint_pruned_bottom",
        "endpoint_filled_top",
        "endpoint_filled_bottom",
        "endpoint_fill_candidate_count",
        "mean_score",
        "sequence_order_ok",
        "sequence_method",
        "path_score",
        "selected_indices",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    split: str,
    start_index: int,
    model_path: Path,
    profile: str,
    postprocess_config: Mapping[str, Any],
    inference_batch_size: int,
    output_npz: Path,
) -> dict[str, Any]:
    counts = [int(record["embedding_count"]) for record in records]
    count_errors = [abs(int(record["count_error"])) for record in records]
    order_ok = [bool(record["sequence_order_ok"]) for record in records]
    return {
        "phase": "phase6_vertebral_embeddings_v1",
        "split": split,
        "start_index": start_index,
        "num_images": len(records),
        "model_path": str(model_path),
        "output_npz": str(output_npz),
        "profile": str(profile),
        "postprocess_config": dict(postprocess_config),
        "inference_batch_size": int(inference_batch_size),
        "embedding_dim": EMBEDDING_DIM,
        "visual_feature_dim": VISUAL_FEATURE_DIM,
        "geometry_feature_dim": GEOMETRY_FEATURE_DIM,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "max_vertebrae": MAX_VERTEBRAE,
        "confidence_threshold": float(postprocess_config.get("confidence_threshold", 0.0)),
        "nms_iou_threshold": float(postprocess_config.get("nms_iou_threshold", 0.0)),
        "min_y_gap": float(postprocess_config.get("min_y_gap", 0.0)),
        "selection_method": str(postprocess_config.get("selection_method", "")),
        "angle_weight": float(postprocess_config.get("angle_weight", 0.0)),
        "angle_jump_tolerance_deg": float(postprocess_config.get("angle_jump_tolerance_deg", 0.0)),
        "gap_fill_threshold": float(postprocess_config.get("gap_fill_threshold", 0.0)),
        "mean_embedding_count": float(np.mean(counts)) if counts else None,
        "min_embedding_count": int(np.min(counts)) if counts else None,
        "max_embedding_count": int(np.max(counts)) if counts else None,
        "mean_abs_count_error": float(np.mean(count_errors)) if count_errors else None,
        "sequence_order_error_count": int(np.sum([not value for value in order_ok])),
        "sequence_order_ok_rate": float(np.mean(order_ok)) if order_ok else None,
    }


def build_manual_postprocess_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "confidence_threshold": float(args.confidence_threshold),
        "nms_iou_threshold": float(args.nms_iou_threshold),
        "min_y_gap": float(args.min_y_gap),
        "min_vertebrae": int(args.min_vertebrae),
        "max_vertebrae": int(args.max_vertebrae),
        "selection_method": str(args.selection_method),
        "count_prior": float(args.count_prior),
        "count_prior_weight": float(args.count_prior_weight),
        "candidate_cost": float(args.candidate_cost),
        "gap_weight": float(args.gap_weight),
        "x_jump_weight": float(args.x_jump_weight),
        "size_weight": float(args.size_weight),
        "angle_weight": float(args.angle_weight),
        "angle_jump_tolerance_deg": float(args.angle_jump_tolerance_deg),
        "endpoint_pruning": not bool(args.disable_endpoint_pruning),
        "gap_filling": not bool(args.disable_gap_filling),
        "gap_fill_threshold": float(args.gap_fill_threshold),
    }


def resolve_postprocess_config(
    args: argparse.Namespace,
    phase5_sequence: ModuleType,
) -> tuple[str, dict[str, Any]]:
    profile = str(args.profile).strip()
    if not profile:
        return "manual_cli", build_manual_postprocess_config(args)

    profile_configs = getattr(phase5_sequence, "PROFILE_CONFIGS", {})
    if profile not in profile_configs:
        available = ", ".join(sorted(str(key) for key in profile_configs.keys()))
        raise ValueError(f"Perfil desconhecido: {profile}. Perfis disponiveis: {available}")
    return profile, dict(profile_configs[profile])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera embeddings vertebrais da Fase 6.")
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_hard_negative_balanced_v1.keras"),
    )
    parser.add_argument(
        "--profile",
        default="",
        help=(
            "Perfil anatomico definido em scripts/51_eval_phase5_anatomical_sequence_v1.py. "
            "Quando definido, substitui os parametros manuais de pos-processamento."
        ),
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Numero de imagens a processar. Usa 0 para processar ate ao fim do split.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-y-gap", type=float, default=0.025)
    parser.add_argument("--min-vertebrae", type=int, default=14)
    parser.add_argument("--max-vertebrae", type=int, default=21)
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
    parser.add_argument("--disable-gap-filling", action="store_true")
    parser.add_argument("--gap-fill-threshold", type=float, default=0.6)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--num-diagrams", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "phase6_vertebral_embeddings_v1"),
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Nome opcional do ficheiro .npz. Por defeito e gerado automaticamente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval")

    model_path = resolve_project_path(args.model_path)
    require_file(model_path)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile, postprocess_config = resolve_postprocess_config(args, phase5_sequence)

    print(f"A carregar modelo base da Fase 5: {model_path}")
    model = phase5_eval.load_spatial_offset_model_for_eval(model_path)
    feature_model = build_feature_prediction_model(model)

    print(f"A carregar split {args.split}...")
    samples, image_paths = phase2_train.load_split_samples_and_paths(tfdata, args.split)
    selected_samples = select_window(samples, args.start_index, args.num_images)
    selected_paths = select_window(image_paths, args.start_index, args.num_images)
    if not selected_samples:
        raise ValueError("Nenhuma imagem selecionada. Verifica --start-index e --num-images.")

    requested_count = len(selected_samples)
    output_name = args.output_name
    if not output_name:
        count_label = "all" if args.num_images <= 0 else str(requested_count)
        output_name = f"phase6_vertebral_embeddings_{args.split}_start{args.start_index}_n{count_label}.npz"
    if not output_name.endswith(".npz"):
        output_name += ".npz"
    output_npz = output_dir / output_name

    print(
        "A gerar embeddings: "
        f"imagens={requested_count}, embedding_dim={EMBEDDING_DIM}, "
        f"profile={profile}, threshold={postprocess_config.get('confidence_threshold')}, "
        f"gap_fill={postprocess_config.get('gap_fill_threshold')}, "
        f"batch={max(int(args.inference_batch_size), 1)}"
    )

    embedding_arrays: list[np.ndarray] = []
    mask_arrays: list[np.ndarray] = []
    points_arrays: list[np.ndarray] = []
    bbox_arrays: list[np.ndarray] = []
    score_arrays: list[np.ndarray] = []
    candidate_index_arrays: list[np.ndarray] = []
    gt_counts: list[int] = []
    cobb_angles: list[np.ndarray] = []
    file_names: list[str] = []
    records: list[dict[str, Any]] = []

    inference_batch_size = max(int(args.inference_batch_size), 1)
    for batch_start in range(0, requested_count, inference_batch_size):
        batch_samples = selected_samples[batch_start : batch_start + inference_batch_size]
        batch_paths = selected_paths[batch_start : batch_start + inference_batch_size]
        batch_images: list[tf.Tensor] = []
        batch_targets: list[Mapping[str, Any]] = []
        for sample, image_path in zip(batch_samples, batch_paths):
            targets = tfdata.sample_to_targets(sample)
            image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
            batch_targets.append(targets)
            batch_images.append(image)

        image_batch = tf.stack(batch_images, axis=0)
        batch_feature_outputs = feature_model(image_batch, training=False)

        for local_index, (sample, image, targets) in enumerate(
            zip(batch_samples, batch_images, batch_targets),
            start=0,
        ):
            index = batch_start + local_index + 1
            feature_outputs = {
                key: value[local_index : local_index + 1]
                for key, value in batch_feature_outputs.items()
            }
            arrays, record = extract_sequence_for_image(
                image=image,
                sample=sample,
                targets=targets,
                feature_outputs=feature_outputs,
                postprocess=postprocess,
                postprocess_config=postprocess_config,
            )

            embedding_arrays.append(arrays["embeddings"])
            mask_arrays.append(arrays["mask"])
            points_arrays.append(arrays["points"])
            bbox_arrays.append(arrays["bbox"])
            score_arrays.append(arrays["scores"])
            candidate_index_arrays.append(arrays["candidate_indices"])
            gt_counts.append(int(targets["vertebra_count"]))
            cobb_angles.append(np.asarray(targets["cobb_angles"], dtype=np.float32))
            file_names.append(str(sample["file_name"]))
            records.append(record)

            if index <= args.num_diagrams:
                valid_count = int(record["embedding_count"])
                diagram = make_sequence_diagram(
                    image=image,
                    targets=targets,
                    selected_points=arrays["points"][:valid_count],
                    drawing=phase2_train,
                )
                stem = Path(str(sample["file_name"])).stem
                diagram_path = output_dir / f"{index - 1:03d}_{stem}_phase6_embedding_sequence.png"
                tf.io.write_file(
                    str(diagram_path),
                    tf.io.encode_png(tf.convert_to_tensor(diagram)),
                )

        processed = min(batch_start + len(batch_samples), requested_count)
        if args.progress_every > 0 and (processed % args.progress_every == 0 or processed == requested_count):
            print(f"processadas {processed}/{requested_count} imagens")

    summary = summarize_records(
        records=records,
        split=args.split,
        start_index=args.start_index,
        model_path=model_path,
        profile=profile,
        postprocess_config=postprocess_config,
        inference_batch_size=inference_batch_size,
        output_npz=output_npz,
    )

    np.savez_compressed(
        output_npz,
        embeddings=np.stack(embedding_arrays).astype(np.float32),
        mask=np.stack(mask_arrays).astype(np.float32),
        points=np.stack(points_arrays).astype(np.float32),
        bbox=np.stack(bbox_arrays).astype(np.float32),
        scores=np.stack(score_arrays).astype(np.float32),
        candidate_indices=np.stack(candidate_index_arrays).astype(np.int32),
        gt_counts=np.asarray(gt_counts, dtype=np.int32),
        cobb_angles=np.stack(cobb_angles).astype(np.float32),
        file_names=np.asarray(file_names, dtype="U256"),
        metadata_json=np.asarray(json.dumps(summary, indent=2)),
    )

    summary_path = output_dir / "phase6_vertebral_embeddings_summary.json"
    details_path = output_dir / "phase6_vertebral_embeddings_details.csv"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_details_csv(details_path, records)

    print("\nResumo Fase 6 - embeddings vertebrais")
    print(f"imagens: {summary['num_images']}")
    print(f"embedding_dim: {summary['embedding_dim']}")
    print(f"visual_feature_dim: {summary['visual_feature_dim']}")
    print(f"geometry_feature_dim: {summary['geometry_feature_dim']}")
    print(f"profile: {summary['profile']}")
    print(f"media embeddings por imagem: {summary['mean_embedding_count']}")
    print(f"erro absoluto medio de contagem: {summary['mean_abs_count_error']}")
    print(f"sequencias fora de ordem: {summary['sequence_order_error_count']}")
    print(f"NPZ guardado em: {output_npz}")
    print(f"Resumo guardado em: {summary_path}")
    print(f"Detalhes guardados em: {details_path}")


if __name__ == "__main__":
    main()
