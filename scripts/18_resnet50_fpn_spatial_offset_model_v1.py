"""Fase 5 v3: ResNet50 + FPN + head espacial com offsets locais.

Esta variante mantem a ideia da Fase 5 v2: uma celula do mapa FPN P3 gera um
candidato vertebral. A diferenca principal e que os 4 pontos deixam de ser
coordenadas absolutas livres e passam a ser descodificados como offsets locais
em torno da celula P3 positiva.

Isto reduz deslocamentos grandes quando a celula escolhida esta correta, mas a
regressao absoluta dos pontos deriva horizontalmente. A bbox e derivada dos
pontos previstos, mantendo a geometria interna consistente.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative o ambiente .venv antes de correr este script."
    ) from exc


IMAGE_SHAPE = (512, 512, 3)
MAX_VERTEBRAE = 21
SPATIAL_GRID_SIZE = 64
SPATIAL_NUM_CANDIDATES = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE
DEFAULT_FPN_CHANNELS = 128
DEFAULT_HEAD_CHANNELS = 128
DEFAULT_POINT_OFFSET_SCALE = 0.12
DEFAULT_PRESENCE_POS_WEIGHT = 50.0
DEFAULT_PRESENCE_PRIOR = MAX_VERTEBRAE / SPATIAL_NUM_CANDIDATES
IMAGENET_NOTOP_WEIGHTS = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


def imagenet_weights_cache_path() -> Path:
    return Path.home() / ".keras" / "models" / IMAGENET_NOTOP_WEIGHTS


def resolve_resnet50_weights(weights: str) -> str | None:
    """Resolve a opcao de pesos sem fazer download quando weights='auto'."""
    normalized = weights.strip().lower()
    if normalized == "none":
        return None
    if normalized == "imagenet":
        return "imagenet"
    if normalized == "auto":
        return "imagenet" if imagenet_weights_cache_path().is_file() else None
    raise ValueError(f"Opcao de weights invalida: {weights}")


def resolved_weights_label(weights: str) -> str:
    resolved = resolve_resnet50_weights(weights)
    return "imagenet" if resolved == "imagenet" else "none"


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class ResNet50Preprocess(tf.keras.layers.Layer):
    """Converte imagens 0-1 do loader para o preprocess_input da ResNet50."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class CoordinateChannels(tf.keras.layers.Layer):
    """Acrescenta canais x/y normalizados para a head conhecer a posicao."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        y_coords = tf.linspace(0.0, 1.0, height)
        x_coords = tf.linspace(0.0, 1.0, width)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing="ij")
        coords = tf.stack([xx, yy], axis=-1)
        coords = tf.expand_dims(coords, axis=0)
        coords = tf.tile(coords, [batch_size, 1, 1, 1])
        coords = tf.cast(coords, inputs.dtype)
        return tf.concat([inputs, coords], axis=-1)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class OffsetPointsDecode(tf.keras.layers.Layer):
    """Descodifica deltas locais da celula P3 para pontos absolutos [0, 1]."""

    def __init__(
        self,
        grid_size: int = SPATIAL_GRID_SIZE,
        offset_scale: float = DEFAULT_POINT_OFFSET_SCALE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if grid_size <= 0:
            raise ValueError("grid_size deve ser positivo.")
        if offset_scale <= 0.0:
            raise ValueError("offset_scale deve ser positivo.")
        self.grid_size = int(grid_size)
        self.offset_scale = float(offset_scale)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        deltas = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(deltas)[0]
        height = tf.shape(deltas)[1]
        width = tf.shape(deltas)[2]

        y_centers = (tf.cast(tf.range(height), tf.float32) + 0.5) / tf.cast(height, tf.float32)
        x_centers = (tf.cast(tf.range(width), tf.float32) + 0.5) / tf.cast(width, tf.float32)
        yy, xx = tf.meshgrid(y_centers, x_centers, indexing="ij")
        centers = tf.stack([xx, yy], axis=-1)
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, 4, 1])

        point_offsets = tf.reshape(deltas, [batch_size, height, width, 4, 2])
        points = centers + point_offsets * self.offset_scale
        points = tf.clip_by_value(points, 0.0, 1.0)
        return tf.reshape(points, [batch_size, height * width, 8])

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, int, int]:
        height = input_shape[1] or self.grid_size
        width = input_shape[2] or self.grid_size
        return (input_shape[0], int(height) * int(width), 8)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"grid_size": self.grid_size, "offset_scale": self.offset_scale})
        return config


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class PointsToBbox(tf.keras.layers.Layer):
    """Converte 4 pontos normalizados para bbox [x, y, w, h]."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        points = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(points)[0]
        num_candidates = tf.shape(points)[1]
        points_xy = tf.reshape(points, [batch_size, num_candidates, 4, 2])

        x_min = tf.reduce_min(points_xy[..., 0], axis=-1)
        y_min = tf.reduce_min(points_xy[..., 1], axis=-1)
        x_max = tf.reduce_max(points_xy[..., 0], axis=-1)
        y_max = tf.reduce_max(points_xy[..., 1], axis=-1)
        width = tf.maximum(x_max - x_min, 0.0)
        height = tf.maximum(y_max - y_min, 0.0)
        return tf.stack([x_min, y_min, width, height], axis=-1)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, int | None, int]:
        return (input_shape[0], input_shape[1], 4)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def weighted_presence_bce_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    positive_weight: float = DEFAULT_PRESENCE_POS_WEIGHT,
) -> tf.Tensor:
    """BCE ponderada para compensar o desequilibrio 21 positivos vs 4096 celulas."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    weights = 1.0 + y_true * (positive_weight - 1.0)
    return tf.reduce_sum(bce * weights) / tf.maximum(tf.reduce_sum(weights), 1.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def masked_bbox_mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE de bbox apenas nas celulas positivas."""
    bbox_true = y_true[..., :4]
    mask = y_true[..., 4:5]
    error = tf.abs(bbox_true - y_pred) * mask
    denominator = tf.maximum(tf.reduce_sum(mask) * 4.0, 1.0)
    return tf.reduce_sum(error) / denominator


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def masked_points_mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE dos 8 valores dos 4 pontos apenas nas celulas positivas."""
    points_true = y_true[..., :8]
    mask = y_true[..., 8:9]
    error = tf.abs(points_true - y_pred) * mask
    denominator = tf.maximum(tf.reduce_sum(mask) * 8.0, 1.0)
    return tf.reduce_sum(error) / denominator


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def masked_points_geometry_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE dos pontos com termo extra para alinhar centroides vertebrais."""
    points_mae = masked_points_mae_loss(y_true, y_pred)

    points_true = y_true[..., :8]
    mask = y_true[..., 8:9]
    batch_size = tf.shape(points_true)[0]
    num_candidates = tf.shape(points_true)[1]

    true_xy = tf.reshape(points_true, [batch_size, num_candidates, 4, 2])
    pred_xy = tf.reshape(y_pred, [batch_size, num_candidates, 4, 2])
    true_centers = tf.reduce_mean(true_xy, axis=2)
    pred_centers = tf.reduce_mean(pred_xy, axis=2)
    center_error = tf.abs(true_centers - pred_centers) * mask
    denominator = tf.maximum(tf.reduce_sum(mask) * 2.0, 1.0)
    center_mae = tf.reduce_sum(center_error) / denominator
    return points_mae + 0.5 * center_mae


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def masked_bbox_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return masked_bbox_mae_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def masked_points_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return masked_points_mae_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_positive_recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    positives = tf.cast(y_true >= 0.5, tf.float32)
    selected = tf.cast(y_pred >= 0.5, tf.float32)
    true_positive = tf.reduce_sum(positives * selected)
    return true_positive / tf.maximum(tf.reduce_sum(positives), 1.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def predicted_positive_rate(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    del y_true
    return tf.reduce_mean(tf.cast(y_pred >= 0.5, tf.float32))


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def positive_count_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    gt_count = tf.reduce_sum(tf.cast(y_true >= 0.5, tf.float32), axis=[1, 2])
    pred_count = tf.reduce_sum(tf.cast(y_pred >= 0.5, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_count - gt_count))


def spatial_fpn_targets(
    images: tf.Tensor,
    targets: Dict[str, tf.Tensor],
    grid_size: int = SPATIAL_GRID_SIZE,
) -> tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Atribui cada vertebra a celula P3 onde cai o seu centroide."""
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

    presence = tf.zeros([batch_size, num_candidates, 1], dtype=tf.float32)
    bbox_targets = tf.zeros([batch_size, num_candidates, 4], dtype=tf.float32)
    points_targets = tf.zeros([batch_size, num_candidates, 8], dtype=tf.float32)
    target_mask = tf.zeros([batch_size, num_candidates, 1], dtype=tf.float32)

    positive_updates = tf.ones([tf.shape(valid_indices)[0], 1], dtype=tf.float32)
    presence = tf.tensor_scatter_nd_update(presence, valid_indices, positive_updates)
    target_mask = tf.tensor_scatter_nd_update(target_mask, valid_indices, positive_updates)
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

    labels = {
        "presence": presence,
        "bbox": tf.concat([bbox_targets, target_mask], axis=-1),
        "points": tf.concat([points_targets, target_mask], axis=-1),
    }
    return images, labels


def set_backbone_trainability(
    backbone: tf.keras.Model,
    trainable_backbone_layers: int,
) -> None:
    if trainable_backbone_layers < 0:
        raise ValueError("--trainable-backbone-layers deve ser >= 0")

    if trainable_backbone_layers == 0:
        backbone.trainable = False
        return

    backbone.trainable = True
    frozen_until = max(0, len(backbone.layers) - trainable_backbone_layers)
    for layer in backbone.layers[:frozen_until]:
        layer.trainable = False
    for layer in backbone.layers[frozen_until:]:
        layer.trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_resnet50_feature_extractor(
    image_shape: tuple[int, int, int],
    weights: str,
    trainable_backbone_layers: int,
) -> tf.keras.Model:
    resolved_weights = resolve_resnet50_weights(weights)
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=resolved_weights,
        input_shape=image_shape,
        name="resnet50_base",
    )
    backbone = tf.keras.Model(
        inputs=base.input,
        outputs=[
            base.get_layer("conv3_block4_out").output,
            base.get_layer("conv4_block6_out").output,
            base.get_layer("conv5_block3_out").output,
        ],
        name="resnet50_backbone",
    )
    set_backbone_trainability(backbone, trainable_backbone_layers)
    return backbone


def build_fpn(
    c3: tf.Tensor,
    c4: tf.Tensor,
    c5: tf.Tensor,
    channels: int = DEFAULT_FPN_CHANNELS,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    p5 = tf.keras.layers.Conv2D(channels, 1, padding="same", name="fpn_c5_lateral")(c5)

    c4_lateral = tf.keras.layers.Conv2D(channels, 1, padding="same", name="fpn_c4_lateral")(c4)
    p5_up = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_p5_upsample")(p5)
    p4 = tf.keras.layers.Add(name="fpn_p4_add")([c4_lateral, p5_up])

    c3_lateral = tf.keras.layers.Conv2D(channels, 1, padding="same", name="fpn_c3_lateral")(c3)
    p4_up = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_p4_upsample")(p4)
    p3 = tf.keras.layers.Add(name="fpn_p3_add")([c3_lateral, p4_up])

    p3 = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name="fpn_p3")(p3)
    p4 = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name="fpn_p4")(p4)
    p5 = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name="fpn_p5")(p5)
    return p3, p4, p5


def build_spatial_offset_candidate_head(
    p3: tf.Tensor,
    head_channels: int = DEFAULT_HEAD_CHANNELS,
    presence_prior: float = DEFAULT_PRESENCE_PRIOR,
    offset_scale: float = DEFAULT_POINT_OFFSET_SCALE,
) -> dict[str, tf.Tensor]:
    """Head convolucional: presence por celula e pontos por offset local."""
    if not 0.0 < presence_prior < 1.0:
        raise ValueError("presence_prior deve estar entre 0 e 1.")

    presence_bias = math.log(presence_prior / (1.0 - presence_prior))
    x = CoordinateChannels(name="p3_coord_channels")(p3)
    x = tf.keras.layers.Conv2D(
        head_channels,
        3,
        padding="same",
        activation="relu",
        name="spatial_offset_head_conv1",
    )(x)
    x = tf.keras.layers.Conv2D(
        head_channels,
        3,
        padding="same",
        activation="relu",
        name="spatial_offset_head_conv2",
    )(x)

    presence_map = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(presence_bias),
        name="presence_map",
    )(x)
    point_delta_map = tf.keras.layers.Conv2D(
        8,
        1,
        activation="tanh",
        name="point_delta_map",
    )(x)

    presence = tf.keras.layers.Reshape((SPATIAL_NUM_CANDIDATES, 1), name="presence")(presence_map)
    points = OffsetPointsDecode(
        grid_size=SPATIAL_GRID_SIZE,
        offset_scale=offset_scale,
        name="points",
    )(point_delta_map)
    bbox = PointsToBbox(name="bbox")(points)
    return {"presence": presence, "bbox": bbox, "points": points}


def build_resnet50_fpn_spatial_offset_model_v1(
    image_shape: tuple[int, int, int] = IMAGE_SHAPE,
    weights: str = "auto",
    trainable_backbone_layers: int = 0,
    fpn_channels: int = DEFAULT_FPN_CHANNELS,
    head_channels: int = DEFAULT_HEAD_CHANNELS,
    presence_prior: float = DEFAULT_PRESENCE_PRIOR,
    offset_scale: float = DEFAULT_POINT_OFFSET_SCALE,
) -> tf.keras.Model:
    """Cria ResNet50 + FPN + head espacial com offsets locais."""
    inputs = tf.keras.Input(shape=image_shape, name="image")
    x = ResNet50Preprocess(name="resnet50_preprocess")(inputs)

    backbone = build_resnet50_feature_extractor(
        image_shape=image_shape,
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    c3, c4, c5 = backbone(x, training=trainable_backbone_layers > 0)
    p3, _, _ = build_fpn(c3, c4, c5, channels=fpn_channels)
    outputs = build_spatial_offset_candidate_head(
        p3,
        head_channels=head_channels,
        presence_prior=presence_prior,
        offset_scale=offset_scale,
    )

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="phase5_resnet50_fpn_spatial_offset_model_v1",
    )


def compile_resnet50_fpn_spatial_offset_model_v1(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    presence_loss_weight: float = 1.0,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "presence": weighted_presence_bce_loss,
            "bbox": masked_bbox_mae_loss,
            "points": masked_points_geometry_loss,
        },
        loss_weights={"presence": presence_loss_weight, "bbox": 2.0, "points": 8.0},
        metrics={
            "presence": [
                presence_positive_recall,
                predicted_positive_rate,
                positive_count_mae,
            ],
            "bbox": [masked_bbox_mae],
            "points": [masked_points_mae],
        },
    )
    return model


def custom_objects() -> dict[str, object]:
    return {
        "ResNet50Preprocess": ResNet50Preprocess,
        "CoordinateChannels": CoordinateChannels,
        "OffsetPointsDecode": OffsetPointsDecode,
        "PointsToBbox": PointsToBbox,
        "weighted_presence_bce_loss": weighted_presence_bce_loss,
        "masked_bbox_mae_loss": masked_bbox_mae_loss,
        "masked_points_mae_loss": masked_points_mae_loss,
        "masked_points_geometry_loss": masked_points_geometry_loss,
        "masked_bbox_mae": masked_bbox_mae,
        "masked_points_mae": masked_points_mae,
        "presence_positive_recall": presence_positive_recall,
        "predicted_positive_rate": predicted_positive_rate,
        "positive_count_mae": positive_count_mae,
    }


def load_resnet50_fpn_spatial_offset_model_v1(model_path: Path) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects())


def configure_loaded_model_for_finetuning(
    model: tf.keras.Model,
    trainable_backbone_layers: int,
) -> None:
    backbone = model.get_layer("resnet50_backbone")
    set_backbone_trainability(backbone, trainable_backbone_layers)


def iter_nested_layers(layer: tf.keras.layers.Layer) -> list[tf.keras.layers.Layer]:
    layers: list[tf.keras.layers.Layer] = []
    seen: set[int] = set()

    def visit(current: tf.keras.layers.Layer) -> None:
        identifier = id(current)
        if identifier in seen:
            return
        seen.add(identifier)
        layers.append(current)
        for child in getattr(current, "layers", []):
            visit(child)

    visit(layer)
    return layers


def count_trainable_layers(model: tf.keras.Model) -> int:
    return sum(1 for layer in iter_nested_layers(model) if layer.trainable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumo do modelo espacial com offsets da Fase 5.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--weights",
        choices=("auto", "none", "imagenet"),
        default="auto",
        help=(
            "auto usa ImageNet apenas se os pesos ja estiverem em cache local; "
            "imagenet pode tentar fazer download via Keras."
        ),
    )
    parser.add_argument("--trainable-backbone-layers", type=int, default=0)
    parser.add_argument("--fpn-channels", type=int, default=DEFAULT_FPN_CHANNELS)
    parser.add_argument("--head-channels", type=int, default=DEFAULT_HEAD_CHANNELS)
    parser.add_argument("--presence-prior", type=float, default=DEFAULT_PRESENCE_PRIOR)
    parser.add_argument("--offset-scale", type=float, default=DEFAULT_POINT_OFFSET_SCALE)
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
    compile_resnet50_fpn_spatial_offset_model_v1(model, learning_rate=args.learning_rate)

    print(f"weights resolvidos: {resolved_weights_label(args.weights)}")
    if args.weights == "auto" and resolved_weights_label(args.weights) == "none":
        print(f"cache ImageNet nao encontrada em: {imagenet_weights_cache_path()}")
    print(f"grid_size: {SPATIAL_GRID_SIZE}")
    print(f"num_candidates espaciais: {SPATIAL_NUM_CANDIDATES}")
    print(f"fpn_channels: {args.fpn_channels}")
    print(f"head_channels: {args.head_channels}")
    print(f"presence_prior: {args.presence_prior:.6f}")
    print(f"offset_scale: {args.offset_scale}")
    print(f"camadas treinaveis no modelo: {count_trainable_layers(model)}")
    model.summary()

    dummy = tf.zeros((2, *IMAGE_SHAPE), dtype=tf.float32)
    outputs = model(dummy, training=False)
    print("\nSmoke forward pass")
    print(f"presence: {tuple(outputs['presence'].shape)}")
    print(f"bbox: {tuple(outputs['bbox'].shape)}")
    print(f"points: {tuple(outputs['points'].shape)}")


if __name__ == "__main__":
    main()
