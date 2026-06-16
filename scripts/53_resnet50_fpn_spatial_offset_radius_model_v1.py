"""Fase 5 v5: targets PositiveRadius e hard mining direccionado.

Esta variante reutiliza a arquitectura ResNet50+FPN+offsets locais da Fase 5
v3/v4. A diferenca fica nos targets e na loss de `presence`:

- a celula central continua a ser o positivo forte;
- celulas vizinhas recebem soft targets opcionais;
- positivos em extremos anatomicos podem receber maior peso;
- negativos acima/abaixo da coluna real podem receber maior prioridade.

Por defeito, bbox/points continuam supervisionados apenas na celula central.
Isto segue a opcao conservadora do plano para reduzir o risco de degradar a
geometria dos quadrilateros enquanto se melhora recall/calibracao da presence.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative o ambiente .venv antes de correr este script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_model = import_script("18_resnet50_fpn_spatial_offset_model_v1.py", "phase5_offset_base_model")
hard_model = import_script("22_resnet50_fpn_spatial_offset_hard_negative_model_v1.py", "phase5_hard_model")

IMAGE_SHAPE = base_model.IMAGE_SHAPE
MAX_VERTEBRAE = base_model.MAX_VERTEBRAE
SPATIAL_GRID_SIZE = base_model.SPATIAL_GRID_SIZE
SPATIAL_NUM_CANDIDATES = base_model.SPATIAL_NUM_CANDIDATES
DEFAULT_FPN_CHANNELS = base_model.DEFAULT_FPN_CHANNELS
DEFAULT_HEAD_CHANNELS = base_model.DEFAULT_HEAD_CHANNELS
DEFAULT_POINT_OFFSET_SCALE = base_model.DEFAULT_POINT_OFFSET_SCALE
DEFAULT_PRESENCE_PRIOR = base_model.DEFAULT_PRESENCE_PRIOR

DEFAULT_POSITIVE_RADIUS = 1
DEFAULT_CENTER_TARGET = 1.0
DEFAULT_RADIUS_TARGET = 0.5
DEFAULT_RADIUS2_TARGET = 0.25
DEFAULT_RADIUS_METRIC = "chebyshev"
DEFAULT_NEIGHBOR_REGRESSION_WEIGHT = 0.0
DEFAULT_ENDPOINT_POSITIVE_BOOST = 1.35
DEFAULT_ENDPOINT_VERTEBRAE = 2
DEFAULT_SEVERE_CURVE_BOOST = 1.0
DEFAULT_SEVERE_CURVE_THRESHOLD_DEG = 40.0
DEFAULT_EXTREME_NEGATIVE_MARGIN = 0.03
DEFAULT_EXTREME_NEGATIVE_WEIGHT = 0.7

DEFAULT_FOCAL_GAMMA = hard_model.DEFAULT_FOCAL_GAMMA
DEFAULT_POSITIVE_WEIGHT = hard_model.DEFAULT_POSITIVE_WEIGHT
DEFAULT_HARD_NEGATIVE_RATIO = hard_model.DEFAULT_HARD_NEGATIVE_RATIO
DEFAULT_MAX_HARD_NEGATIVES = 192
DEFAULT_HARD_NEGATIVE_WEIGHT = 0.6
DEFAULT_EASY_NEGATIVE_WEIGHT = hard_model.DEFAULT_EASY_NEGATIVE_WEIGHT
DEFAULT_PRESENCE_LOSS_WEIGHT = hard_model.DEFAULT_PRESENCE_LOSS_WEIGHT


imagenet_weights_cache_path = base_model.imagenet_weights_cache_path
resolve_resnet50_weights = base_model.resolve_resnet50_weights
resolved_weights_label = base_model.resolved_weights_label
build_resnet50_fpn_spatial_offset_model_v1 = base_model.build_resnet50_fpn_spatial_offset_model_v1
configure_loaded_model_for_finetuning = base_model.configure_loaded_model_for_finetuning
count_trainable_layers = base_model.count_trainable_layers


def _radius_offsets(radius: int, metric: str) -> tf.Tensor:
    if radius < 0:
        raise ValueError("positive_radius deve ser >= 0.")
    if metric not in {"chebyshev", "manhattan"}:
        raise ValueError("radius_metric deve ser 'chebyshev' ou 'manhattan'.")

    offsets: list[tuple[int, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = max(abs(dy), abs(dx)) if metric == "chebyshev" else abs(dy) + abs(dx)
            if distance <= radius:
                offsets.append((dy, dx, distance))
    offsets.sort(key=lambda item: (item[2], abs(item[0]) + abs(item[1]), item[0], item[1]))
    return tf.constant(offsets, dtype=tf.int32)


def _scatter_max(
    batch_size: tf.Tensor,
    num_candidates: int,
    indices: tf.Tensor,
    updates: tf.Tensor,
) -> tf.Tensor:
    """Scatter max para tensores [batch, candidates, 1], tolerando indices duplicados."""
    flat_size = batch_size * num_candidates
    flat_indices = indices[:, 0] * num_candidates + indices[:, 1]
    flat_updates = tf.squeeze(tf.cast(updates, tf.float32), axis=-1)
    values = tf.math.unsorted_segment_max(flat_updates, flat_indices, flat_size)
    values = tf.where(values < -1.0e20, tf.zeros_like(values), values)
    return tf.reshape(values, [batch_size, num_candidates, 1])


def _presence_channels(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    if y_true.shape.rank is not None and y_true.shape[-1] is not None and y_true.shape[-1] >= 4:
        target = y_true[..., 0:1]
        positive_weight = y_true[..., 1:2]
        negative_priority = y_true[..., 2:3]
        central_mask = y_true[..., 3:4]
    else:
        target = y_true
        positive_weight = tf.ones_like(target)
        negative_priority = tf.ones_like(target)
        central_mask = tf.cast(target >= 0.5, tf.float32)
    return target, positive_weight, negative_priority, central_mask


def spatial_fpn_radius_targets(
    images: tf.Tensor,
    targets: Dict[str, tf.Tensor],
    grid_size: int = SPATIAL_GRID_SIZE,
    positive_radius: int = DEFAULT_POSITIVE_RADIUS,
    center_target: float = DEFAULT_CENTER_TARGET,
    radius_target: float = DEFAULT_RADIUS_TARGET,
    radius2_target: float = DEFAULT_RADIUS2_TARGET,
    radius_metric: str = DEFAULT_RADIUS_METRIC,
    neighbor_regression_weight: float = DEFAULT_NEIGHBOR_REGRESSION_WEIGHT,
    endpoint_positive_boost: float = DEFAULT_ENDPOINT_POSITIVE_BOOST,
    endpoint_vertebrae: int = DEFAULT_ENDPOINT_VERTEBRAE,
    severe_curve_boost: float = DEFAULT_SEVERE_CURVE_BOOST,
    severe_curve_threshold_deg: float = DEFAULT_SEVERE_CURVE_THRESHOLD_DEG,
    extreme_negative_margin: float = DEFAULT_EXTREME_NEGATIVE_MARGIN,
    extreme_negative_weight: float = DEFAULT_EXTREME_NEGATIVE_WEIGHT,
) -> tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Atribui soft targets de presence em torno da celula central de cada vertebra."""
    if center_target <= 0.0:
        raise ValueError("center_target deve ser > 0.")
    if radius_target < 0.0 or radius2_target < 0.0:
        raise ValueError("targets de radius devem ser >= 0.")
    if neighbor_regression_weight < 0.0:
        raise ValueError("neighbor_regression_weight deve ser >= 0.")
    if endpoint_positive_boost <= 0.0 or severe_curve_boost <= 0.0:
        raise ValueError("boosts de positivos devem ser > 0.")
    if endpoint_vertebrae < 0:
        raise ValueError("endpoint_vertebrae deve ser >= 0.")
    if extreme_negative_margin < 0.0 or extreme_negative_weight < 0.0:
        raise ValueError("parametros de negativos extremos devem ser >= 0.")

    points = tf.cast(targets["points"], tf.float32)
    bbox = tf.cast(targets["bbox"], tf.float32)
    mask = tf.cast(targets["mask"], tf.float32)

    batch_size = tf.shape(points)[0]
    max_vertebrae = tf.shape(points)[1]
    num_candidates = grid_size * grid_size

    points_reshaped = tf.reshape(points, [batch_size, max_vertebrae, 4, 2])
    centroids = tf.reduce_mean(points_reshaped, axis=2)
    cols = tf.cast(tf.floor(centroids[..., 0] * float(grid_size)), tf.int32)
    rows = tf.cast(tf.floor(centroids[..., 1] * float(grid_size)), tf.int32)
    cols = tf.clip_by_value(cols, 0, grid_size - 1)
    rows = tf.clip_by_value(rows, 0, grid_size - 1)
    flat_indices = rows * grid_size + cols

    batch_indices = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1]),
        [1, max_vertebrae],
    )
    scatter_indices = tf.stack([batch_indices, flat_indices], axis=-1)
    valid = tf.squeeze(mask, axis=-1) > 0.5
    valid_indices = tf.boolean_mask(scatter_indices, valid)
    valid_rows = tf.boolean_mask(rows, valid)
    valid_cols = tf.boolean_mask(cols, valid)
    valid_batch = tf.boolean_mask(batch_indices, valid)

    target_mask = tf.zeros([batch_size, num_candidates, 1], dtype=tf.float32)
    center_updates = tf.ones([tf.shape(valid_indices)[0], 1], dtype=tf.float32)
    target_mask = tf.tensor_scatter_nd_update(target_mask, valid_indices, center_updates)

    offsets = _radius_offsets(positive_radius, radius_metric)
    offsets_dy = offsets[:, 0]
    offsets_dx = offsets[:, 1]
    offsets_distance = offsets[:, 2]
    candidate_rows = tf.expand_dims(valid_rows, axis=1) + tf.reshape(offsets_dy, [1, -1])
    candidate_cols = tf.expand_dims(valid_cols, axis=1) + tf.reshape(offsets_dx, [1, -1])
    candidate_batch = tf.tile(tf.expand_dims(valid_batch, axis=1), [1, tf.shape(offsets)[0]])
    candidate_distance = tf.tile(tf.reshape(offsets_distance, [1, -1]), [tf.shape(valid_batch)[0], 1])
    inside_grid = (
        (candidate_rows >= 0)
        & (candidate_rows < grid_size)
        & (candidate_cols >= 0)
        & (candidate_cols < grid_size)
    )

    radius_batch = tf.boolean_mask(candidate_batch, inside_grid)
    radius_rows = tf.boolean_mask(candidate_rows, inside_grid)
    radius_cols = tf.boolean_mask(candidate_cols, inside_grid)
    radius_distance = tf.boolean_mask(candidate_distance, inside_grid)
    radius_indices = tf.stack([radius_batch, radius_rows * grid_size + radius_cols], axis=-1)

    radius_targets = tf.where(
        radius_distance == 0,
        tf.fill(tf.shape(radius_distance), tf.cast(center_target, tf.float32)),
        tf.where(
            radius_distance == 1,
            tf.fill(tf.shape(radius_distance), tf.cast(radius_target, tf.float32)),
            tf.fill(tf.shape(radius_distance), tf.cast(radius2_target, tf.float32)),
        ),
    )
    presence_target = _scatter_max(
        batch_size,
        num_candidates,
        radius_indices,
        tf.expand_dims(radius_targets, axis=-1),
    )
    presence_target = tf.clip_by_value(presence_target, 0.0, 1.0)

    vertebra_positions = tf.tile(
        tf.reshape(tf.range(max_vertebrae, dtype=tf.int32), [1, max_vertebrae]),
        [batch_size, 1],
    )
    valid_counts = tf.reduce_sum(tf.cast(valid, tf.int32), axis=1)
    endpoint_top = vertebra_positions < int(endpoint_vertebrae)
    endpoint_bottom = vertebra_positions >= tf.maximum(valid_counts[:, None] - int(endpoint_vertebrae), 0)
    endpoint_flag = tf.cast(endpoint_top | endpoint_bottom, tf.float32)

    positive_boost = tf.ones([batch_size, max_vertebrae], dtype=tf.float32)
    positive_boost += endpoint_flag * (float(endpoint_positive_boost) - 1.0)
    if "cobb_angles" in targets:
        max_cobb = tf.reduce_max(tf.cast(targets["cobb_angles"], tf.float32), axis=-1)
        severe_flag = tf.cast(max_cobb >= float(severe_curve_threshold_deg), tf.float32)
        positive_boost *= 1.0 + severe_flag[:, None] * (float(severe_curve_boost) - 1.0)

    valid_boost = tf.boolean_mask(positive_boost, valid)
    radius_boost_updates = tf.tile(tf.expand_dims(valid_boost, axis=1), [1, tf.shape(offsets)[0]])
    radius_boost_updates = tf.boolean_mask(radius_boost_updates, inside_grid)
    radius_positive_weight = _scatter_max(
        batch_size,
        num_candidates,
        radius_indices,
        tf.expand_dims(radius_boost_updates, axis=-1),
    )
    positive_weight = tf.where(presence_target > 0.0, tf.maximum(radius_positive_weight, 1.0), 1.0)

    row_centers = (tf.cast(tf.range(grid_size), tf.float32) + 0.5) / float(grid_size)
    candidate_y = tf.reshape(tf.tile(row_centers[:, None], [1, grid_size]), [1, num_candidates, 1])
    valid_centroids_y = tf.where(valid, centroids[..., 1], tf.zeros_like(centroids[..., 1]))
    first_y = tf.reduce_min(tf.where(valid, valid_centroids_y, tf.ones_like(valid_centroids_y)), axis=1)
    last_y = tf.reduce_max(tf.where(valid, valid_centroids_y, tf.zeros_like(valid_centroids_y)), axis=1)
    first_y = tf.reshape(first_y, [batch_size, 1, 1])
    last_y = tf.reshape(last_y, [batch_size, 1, 1])
    extreme_negative = (candidate_y < first_y - float(extreme_negative_margin)) | (
        candidate_y > last_y + float(extreme_negative_margin)
    )
    negative_priority = 1.0 + tf.cast(extreme_negative, tf.float32) * float(extreme_negative_weight)
    negative_priority = tf.where(presence_target > 0.0, tf.ones_like(negative_priority), negative_priority)

    bbox_targets = tf.zeros([batch_size, num_candidates, 4], dtype=tf.float32)
    points_targets = tf.zeros([batch_size, num_candidates, 8], dtype=tf.float32)
    bbox_targets = tf.tensor_scatter_nd_update(
        bbox_targets,
        valid_indices,
        tf.boolean_mask(bbox, valid),
    )
    points_targets = tf.tensor_scatter_nd_update(
        points_targets,
        valid_indices,
        tf.boolean_mask(points, valid),
    )

    regression_mask = target_mask
    if neighbor_regression_weight > 0.0:
        valid_bbox = tf.boolean_mask(bbox, valid)
        valid_points = tf.boolean_mask(points, valid)
        radius_bbox = tf.tile(tf.expand_dims(valid_bbox, axis=1), [1, tf.shape(offsets)[0], 1])
        radius_points = tf.tile(tf.expand_dims(valid_points, axis=1), [1, tf.shape(offsets)[0], 1])
        bbox_targets = tf.tensor_scatter_nd_update(
            bbox_targets,
            radius_indices,
            tf.boolean_mask(radius_bbox, inside_grid),
        )
        points_targets = tf.tensor_scatter_nd_update(
            points_targets,
            radius_indices,
            tf.boolean_mask(radius_points, inside_grid),
        )
        neighbor_mask = tf.where(
            (presence_target > 0.0) & (target_mask < 0.5),
            tf.ones_like(presence_target) * float(neighbor_regression_weight),
            tf.zeros_like(presence_target),
        )
        regression_mask = tf.maximum(target_mask, neighbor_mask)

    labels = {
        "presence": tf.concat([presence_target, positive_weight, negative_priority, target_mask], axis=-1),
        "bbox": tf.concat([bbox_targets, regression_mask], axis=-1),
        "points": tf.concat([points_targets, regression_mask], axis=-1),
    }
    return images, labels


def make_spatial_fpn_radius_targets(
    grid_size: int = SPATIAL_GRID_SIZE,
    positive_radius: int = DEFAULT_POSITIVE_RADIUS,
    center_target: float = DEFAULT_CENTER_TARGET,
    radius_target: float = DEFAULT_RADIUS_TARGET,
    radius2_target: float = DEFAULT_RADIUS2_TARGET,
    radius_metric: str = DEFAULT_RADIUS_METRIC,
    neighbor_regression_weight: float = DEFAULT_NEIGHBOR_REGRESSION_WEIGHT,
    endpoint_positive_boost: float = DEFAULT_ENDPOINT_POSITIVE_BOOST,
    endpoint_vertebrae: int = DEFAULT_ENDPOINT_VERTEBRAE,
    severe_curve_boost: float = DEFAULT_SEVERE_CURVE_BOOST,
    severe_curve_threshold_deg: float = DEFAULT_SEVERE_CURVE_THRESHOLD_DEG,
    extreme_negative_margin: float = DEFAULT_EXTREME_NEGATIVE_MARGIN,
    extreme_negative_weight: float = DEFAULT_EXTREME_NEGATIVE_WEIGHT,
):
    def _mapper(images: tf.Tensor, targets: Dict[str, tf.Tensor]) -> tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        return spatial_fpn_radius_targets(
            images,
            targets,
            grid_size=grid_size,
            positive_radius=positive_radius,
            center_target=center_target,
            radius_target=radius_target,
            radius2_target=radius2_target,
            radius_metric=radius_metric,
            neighbor_regression_weight=neighbor_regression_weight,
            endpoint_positive_boost=endpoint_positive_boost,
            endpoint_vertebrae=endpoint_vertebrae,
            severe_curve_boost=severe_curve_boost,
            severe_curve_threshold_deg=severe_curve_threshold_deg,
            extreme_negative_margin=extreme_negative_margin,
            extreme_negative_weight=extreme_negative_weight,
        )

    return _mapper


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class PresenceFocalRadiusHardMiningLoss(tf.keras.losses.Loss):
    """Focal loss para targets soft com hard negatives priorizados."""

    def __init__(
        self,
        positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
        max_hard_negatives: int = DEFAULT_MAX_HARD_NEGATIVES,
        hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
        easy_negative_weight: float = DEFAULT_EASY_NEGATIVE_WEIGHT,
        name: str = "presence_focal_radius_hard_mining_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if positive_weight <= 0.0:
            raise ValueError("positive_weight deve ser > 0.")
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma deve ser >= 0.")
        if hard_negative_ratio <= 0.0:
            raise ValueError("hard_negative_ratio deve ser > 0.")
        if max_hard_negatives <= 0:
            raise ValueError("max_hard_negatives deve ser > 0.")
        if hard_negative_weight < 0.0 or easy_negative_weight < 0.0:
            raise ValueError("pesos de negativos devem ser >= 0.")

        self.positive_weight = float(positive_weight)
        self.focal_gamma = float(focal_gamma)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.max_hard_negatives = int(max_hard_negatives)
        self.hard_negative_weight = float(hard_negative_weight)
        self.easy_negative_weight = float(easy_negative_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target, positive_weight_map, negative_priority, central_mask = _presence_channels(y_true)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        positive_mask = tf.cast(target > 0.0, tf.float32)
        negative_mask = 1.0 - positive_mask

        bce = -(target * tf.math.log(y_pred) + (1.0 - target) * tf.math.log(1.0 - y_pred))
        p_t = target * y_pred + (1.0 - target) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, self.focal_gamma)
        focal_bce = bce * focal_factor

        positive_loss = focal_bce * positive_mask * positive_weight_map * self.positive_weight
        negative_loss = focal_bce * negative_mask * negative_priority

        negative_flat = tf.squeeze(negative_loss, axis=-1)
        num_candidates = tf.shape(negative_flat)[1]
        max_k = tf.minimum(tf.cast(self.max_hard_negatives, tf.int32), num_candidates)
        top_negative_values = tf.nn.top_k(negative_flat, k=max_k, sorted=False).values

        positives_per_image = tf.reduce_sum(tf.squeeze(central_mask, axis=-1), axis=1)
        soft_positive_counts = tf.reduce_sum(tf.squeeze(positive_mask, axis=-1), axis=1)
        positives_per_image = tf.where(positives_per_image > 0.0, positives_per_image, soft_positive_counts)
        hard_counts = tf.cast(tf.round(positives_per_image * self.hard_negative_ratio), tf.int32)
        hard_counts = tf.clip_by_value(hard_counts, 1, max_k)
        ranks = tf.reshape(tf.range(max_k, dtype=tf.int32), [1, max_k])
        hard_mask = tf.cast(ranks < tf.reshape(hard_counts, [-1, 1]), tf.float32)

        positive_sum = tf.reduce_sum(positive_loss)
        hard_negative_sum = self.hard_negative_weight * tf.reduce_sum(top_negative_values * hard_mask)
        easy_negative_sum = self.easy_negative_weight * tf.reduce_sum(negative_loss)

        positive_denominator = self.positive_weight * tf.reduce_sum(positive_mask * positive_weight_map)
        hard_denominator = self.hard_negative_weight * tf.reduce_sum(hard_mask)
        easy_denominator = self.easy_negative_weight * tf.reduce_sum(negative_mask * negative_priority)
        denominator = tf.maximum(
            positive_denominator + hard_denominator + easy_denominator,
            1.0,
        )
        return (positive_sum + hard_negative_sum + easy_negative_sum) / denominator

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "positive_weight": self.positive_weight,
                "focal_gamma": self.focal_gamma,
                "hard_negative_ratio": self.hard_negative_ratio,
                "max_hard_negatives": self.max_hard_negatives,
                "hard_negative_weight": self.hard_negative_weight,
                "easy_negative_weight": self.easy_negative_weight,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_central_positive_recall_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    _, _, _, central_mask = _presence_channels(y_true)
    selected = tf.cast(y_pred >= 0.8, tf.float32)
    true_positive = tf.reduce_sum(central_mask * selected)
    return true_positive / tf.maximum(tf.reduce_sum(central_mask), 1.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_soft_positive_recall_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    target, _, _, _ = _presence_channels(y_true)
    positives = tf.cast(target > 0.0, tf.float32)
    selected = tf.cast(y_pred >= 0.8, tf.float32)
    true_positive = tf.reduce_sum(positives * selected)
    return true_positive / tf.maximum(tf.reduce_sum(positives), 1.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_false_positives_per_image_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    target, _, _, _ = _presence_channels(y_true)
    negatives = tf.cast(target <= 0.0, tf.float32)
    selected = tf.cast(y_pred >= 0.8, tf.float32)
    false_positives = tf.reduce_sum(negatives * selected, axis=[1, 2])
    return tf.reduce_mean(false_positives)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_selected_count_mae_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    _, _, _, central_mask = _presence_channels(y_true)
    gt_count = tf.reduce_sum(central_mask, axis=[1, 2])
    pred_count = tf.reduce_sum(tf.cast(y_pred >= 0.8, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_count - gt_count))


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_target_mass_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    target, _, _, _ = _presence_channels(y_true)
    gt_mass = tf.reduce_sum(target, axis=[1, 2])
    pred_mass = tf.reduce_sum(tf.cast(y_pred >= 0.5, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_mass - gt_mass))


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def predicted_positive_rate_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    del y_true
    return tf.reduce_mean(tf.cast(y_pred >= 0.8, tf.float32))


def compile_resnet50_fpn_spatial_offset_radius_model_v1(
    model: tf.keras.Model,
    learning_rate: float = 3e-5,
    presence_loss_weight: float = DEFAULT_PRESENCE_LOSS_WEIGHT,
    positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
    max_hard_negatives: int = DEFAULT_MAX_HARD_NEGATIVES,
    hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
    easy_negative_weight: float = DEFAULT_EASY_NEGATIVE_WEIGHT,
) -> tf.keras.Model:
    presence_loss = PresenceFocalRadiusHardMiningLoss(
        positive_weight=positive_weight,
        focal_gamma=focal_gamma,
        hard_negative_ratio=hard_negative_ratio,
        max_hard_negatives=max_hard_negatives,
        hard_negative_weight=hard_negative_weight,
        easy_negative_weight=easy_negative_weight,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "presence": presence_loss,
            "bbox": base_model.masked_bbox_mae_loss,
            "points": base_model.masked_points_geometry_loss,
        },
        loss_weights={"presence": presence_loss_weight, "bbox": 2.0, "points": 8.0},
        metrics={
            "presence": [
                presence_central_positive_recall_08,
                presence_soft_positive_recall_08,
                presence_false_positives_per_image_08,
                presence_selected_count_mae_08,
                presence_target_mass_mae,
                predicted_positive_rate_08,
            ],
            "bbox": [base_model.masked_bbox_mae],
            "points": [base_model.masked_points_mae],
        },
    )
    return model


def custom_objects() -> dict[str, object]:
    objects = hard_model.custom_objects().copy()
    objects.update(
        {
            "PresenceFocalRadiusHardMiningLoss": PresenceFocalRadiusHardMiningLoss,
            "presence_central_positive_recall_08": presence_central_positive_recall_08,
            "presence_soft_positive_recall_08": presence_soft_positive_recall_08,
            "presence_false_positives_per_image_08": presence_false_positives_per_image_08,
            "presence_selected_count_mae_08": presence_selected_count_mae_08,
            "presence_target_mass_mae": presence_target_mass_mae,
            "predicted_positive_rate_08": predicted_positive_rate_08,
        }
    )
    return objects


def load_model_with_batchnorm_renorm_compat(model_path: Path) -> tf.keras.Model:
    batch_norm_classes = [tf.keras.layers.BatchNormalization]
    try:
        import keras

        batch_norm_classes.append(keras.layers.BatchNormalization)
    except Exception:
        pass

    originals = []
    for batch_norm_class in dict.fromkeys(batch_norm_classes):
        original_from_config = batch_norm_class.from_config
        originals.append((batch_norm_class, original_from_config))

        @classmethod
        def compatible_from_config(cls, config, original=original_from_config):
            del cls
            config = dict(config)
            config.pop("renorm", None)
            config.pop("renorm_clipping", None)
            config.pop("renorm_momentum", None)
            return original(config)

        batch_norm_class.from_config = compatible_from_config

    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects())
    finally:
        for batch_norm_class, original_from_config in originals:
            batch_norm_class.from_config = original_from_config


def load_resnet50_fpn_spatial_offset_radius_model_v1(model_path: Path) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    return load_model_with_batchnorm_renorm_compat(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumo do modelo Fase 5 v5 PositiveRadius/HardMining.")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weights", choices=("auto", "none", "imagenet"), default="auto")
    parser.add_argument("--trainable-backbone-layers", type=int, default=0)
    parser.add_argument("--fpn-channels", type=int, default=DEFAULT_FPN_CHANNELS)
    parser.add_argument("--head-channels", type=int, default=DEFAULT_HEAD_CHANNELS)
    parser.add_argument("--presence-prior", type=float, default=DEFAULT_PRESENCE_PRIOR)
    parser.add_argument("--offset-scale", type=float, default=DEFAULT_POINT_OFFSET_SCALE)
    parser.add_argument("--positive-radius", type=int, default=DEFAULT_POSITIVE_RADIUS)
    parser.add_argument("--radius-target", type=float, default=DEFAULT_RADIUS_TARGET)
    parser.add_argument("--radius2-target", type=float, default=DEFAULT_RADIUS2_TARGET)
    parser.add_argument("--radius-metric", choices=("chebyshev", "manhattan"), default=DEFAULT_RADIUS_METRIC)
    parser.add_argument("--presence-loss-weight", type=float, default=DEFAULT_PRESENCE_LOSS_WEIGHT)
    parser.add_argument("--positive-weight", type=float, default=DEFAULT_POSITIVE_WEIGHT)
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    parser.add_argument("--hard-negative-ratio", type=float, default=DEFAULT_HARD_NEGATIVE_RATIO)
    parser.add_argument("--max-hard-negatives", type=int, default=DEFAULT_MAX_HARD_NEGATIVES)
    parser.add_argument("--hard-negative-weight", type=float, default=DEFAULT_HARD_NEGATIVE_WEIGHT)
    parser.add_argument("--easy-negative-weight", type=float, default=DEFAULT_EASY_NEGATIVE_WEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_resnet50_fpn_spatial_offset_model_v1(
        weights=args.weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        fpn_channels=args.fpn_channels,
        head_channels=args.head_channels,
        presence_prior=args.presence_prior,
        offset_scale=args.offset_scale,
    )
    compile_resnet50_fpn_spatial_offset_radius_model_v1(
        model,
        learning_rate=args.learning_rate,
        presence_loss_weight=args.presence_loss_weight,
        positive_weight=args.positive_weight,
        focal_gamma=args.focal_gamma,
        hard_negative_ratio=args.hard_negative_ratio,
        max_hard_negatives=args.max_hard_negatives,
        hard_negative_weight=args.hard_negative_weight,
        easy_negative_weight=args.easy_negative_weight,
    )
    model.summary()
    print(f"grid_size: {SPATIAL_GRID_SIZE}")
    print(f"num_candidates: {SPATIAL_NUM_CANDIDATES}")
    print(
        "positive radius: "
        f"radius={args.positive_radius}, metric={args.radius_metric}, "
        f"r1={args.radius_target}, r2={args.radius2_target}"
    )


if __name__ == "__main__":
    main()
