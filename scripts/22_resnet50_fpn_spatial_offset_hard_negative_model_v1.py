"""Fase 5 v4: variante hard-negative para a presence da head espacial.

Esta variante reutiliza a arquitetura da Fase 5 v3, que ja mostrou boa
regressao dos quatro pontos. A mudanca fica concentrada na loss de `presence`:
focal loss + hard-negative mining por imagem, para ensinar a head a rejeitar
candidatos falsos mas visualmente plausiveis.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

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

IMAGE_SHAPE = base_model.IMAGE_SHAPE
MAX_VERTEBRAE = base_model.MAX_VERTEBRAE
SPATIAL_GRID_SIZE = base_model.SPATIAL_GRID_SIZE
SPATIAL_NUM_CANDIDATES = base_model.SPATIAL_NUM_CANDIDATES
DEFAULT_FPN_CHANNELS = base_model.DEFAULT_FPN_CHANNELS
DEFAULT_HEAD_CHANNELS = base_model.DEFAULT_HEAD_CHANNELS
DEFAULT_POINT_OFFSET_SCALE = base_model.DEFAULT_POINT_OFFSET_SCALE
DEFAULT_PRESENCE_PRIOR = base_model.DEFAULT_PRESENCE_PRIOR

DEFAULT_FOCAL_GAMMA = 1.5
DEFAULT_POSITIVE_WEIGHT = 45.0
DEFAULT_HARD_NEGATIVE_RATIO = 6.0
DEFAULT_MAX_HARD_NEGATIVES = 160
DEFAULT_HARD_NEGATIVE_WEIGHT = 0.45
DEFAULT_EASY_NEGATIVE_WEIGHT = 0.01
DEFAULT_PRESENCE_LOSS_WEIGHT = 1.2


imagenet_weights_cache_path = base_model.imagenet_weights_cache_path
resolve_resnet50_weights = base_model.resolve_resnet50_weights
resolved_weights_label = base_model.resolved_weights_label
spatial_fpn_targets = base_model.spatial_fpn_targets
build_resnet50_fpn_spatial_offset_model_v1 = base_model.build_resnet50_fpn_spatial_offset_model_v1
configure_loaded_model_for_finetuning = base_model.configure_loaded_model_for_finetuning
count_trainable_layers = base_model.count_trainable_layers


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
class PresenceFocalHardNegativeLoss(tf.keras.losses.Loss):
    """Focal loss com selecao dos negativos mais dificeis em cada imagem."""

    def __init__(
        self,
        positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
        max_hard_negatives: int = DEFAULT_MAX_HARD_NEGATIVES,
        hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
        easy_negative_weight: float = DEFAULT_EASY_NEGATIVE_WEIGHT,
        name: str = "presence_focal_hard_negative_loss",
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
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        positive_mask = tf.cast(y_true >= 0.5, tf.float32)
        negative_mask = 1.0 - positive_mask

        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, self.focal_gamma)
        focal_bce = bce * focal_factor

        positive_loss = focal_bce * positive_mask * self.positive_weight
        negative_loss = focal_bce * negative_mask

        negative_flat = tf.squeeze(negative_loss, axis=-1)
        num_candidates = tf.shape(negative_flat)[1]
        max_k = tf.minimum(tf.cast(self.max_hard_negatives, tf.int32), num_candidates)
        top_negative_values = tf.nn.top_k(negative_flat, k=max_k, sorted=False).values

        positives_per_image = tf.reduce_sum(tf.squeeze(positive_mask, axis=-1), axis=1)
        hard_counts = tf.cast(
            tf.round(positives_per_image * self.hard_negative_ratio),
            tf.int32,
        )
        hard_counts = tf.clip_by_value(hard_counts, 1, max_k)
        ranks = tf.reshape(tf.range(max_k, dtype=tf.int32), [1, max_k])
        hard_mask = tf.cast(ranks < tf.reshape(hard_counts, [-1, 1]), tf.float32)

        positive_sum = tf.reduce_sum(positive_loss)
        hard_negative_sum = self.hard_negative_weight * tf.reduce_sum(top_negative_values * hard_mask)
        easy_negative_sum = self.easy_negative_weight * tf.reduce_sum(negative_loss)

        positive_denominator = self.positive_weight * tf.reduce_sum(positive_mask)
        hard_denominator = self.hard_negative_weight * tf.reduce_sum(hard_mask)
        easy_denominator = self.easy_negative_weight * tf.reduce_sum(negative_mask)
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
def presence_positive_recall_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    positives = tf.cast(y_true >= 0.5, tf.float32)
    selected = tf.cast(y_pred >= 0.8, tf.float32)
    true_positive = tf.reduce_sum(positives * selected)
    return true_positive / tf.maximum(tf.reduce_sum(positives), 1.0)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_false_positives_per_image_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    negatives = tf.cast(y_true < 0.5, tf.float32)
    selected = tf.cast(y_pred >= 0.8, tf.float32)
    false_positives = tf.reduce_sum(negatives * selected, axis=[1, 2])
    return tf.reduce_mean(false_positives)


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def presence_selected_count_mae_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    gt_count = tf.reduce_sum(tf.cast(y_true >= 0.5, tf.float32), axis=[1, 2])
    pred_count = tf.reduce_sum(tf.cast(y_pred >= 0.8, tf.float32), axis=[1, 2])
    return tf.reduce_mean(tf.abs(pred_count - gt_count))


@tf.keras.utils.register_keras_serializable(package="SpinalAI2024")
def predicted_positive_rate_08(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    del y_true
    return tf.reduce_mean(tf.cast(y_pred >= 0.8, tf.float32))


def compile_resnet50_fpn_spatial_offset_hard_negative_model_v1(
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
    presence_loss = PresenceFocalHardNegativeLoss(
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
                base_model.presence_positive_recall,
                base_model.predicted_positive_rate,
                base_model.positive_count_mae,
                presence_positive_recall_08,
                presence_false_positives_per_image_08,
                presence_selected_count_mae_08,
                predicted_positive_rate_08,
            ],
            "bbox": [base_model.masked_bbox_mae],
            "points": [base_model.masked_points_mae],
        },
    )
    return model


def custom_objects() -> dict[str, object]:
    objects = base_model.custom_objects().copy()
    objects.update(
        {
            "PresenceFocalHardNegativeLoss": PresenceFocalHardNegativeLoss,
            "presence_positive_recall_08": presence_positive_recall_08,
            "presence_false_positives_per_image_08": presence_false_positives_per_image_08,
            "presence_selected_count_mae_08": presence_selected_count_mae_08,
            "predicted_positive_rate_08": predicted_positive_rate_08,
        }
    )
    return objects


def load_model_with_batchnorm_renorm_compat(model_path: Path) -> tf.keras.Model:
    """Carrega modelos antigos em Keras que ja nao aceita args `renorm`.

    Alguns checkpoints guardados em `.keras` incluem `renorm=False`,
    `renorm_clipping=None` e `renorm_momentum=0.99` nas BatchNormalization do
    ResNet50. Versoes recentes do Keras no Colab podem rejeitar estes campos.
    Como o valor guardado e `renorm=False`, remover esses campos preserva o
    comportamento efetivo da layer.
    """

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


def load_resnet50_fpn_spatial_offset_hard_negative_model_v1(model_path: Path) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    return load_model_with_batchnorm_renorm_compat(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumo do modelo Fase 5 v4 hard-negative presence.")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weights", choices=("auto", "none", "imagenet"), default="auto")
    parser.add_argument("--trainable-backbone-layers", type=int, default=0)
    parser.add_argument("--fpn-channels", type=int, default=DEFAULT_FPN_CHANNELS)
    parser.add_argument("--head-channels", type=int, default=DEFAULT_HEAD_CHANNELS)
    parser.add_argument("--presence-prior", type=float, default=DEFAULT_PRESENCE_PRIOR)
    parser.add_argument("--offset-scale", type=float, default=DEFAULT_POINT_OFFSET_SCALE)
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
    compile_resnet50_fpn_spatial_offset_hard_negative_model_v1(
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

    print(f"weights resolvidos: {resolved_weights_label(args.weights)}")
    print(f"grid_size: {SPATIAL_GRID_SIZE}")
    print(f"num_candidates espaciais: {SPATIAL_NUM_CANDIDATES}")
    print(
        "presence loss: "
        f"loss_weight={args.presence_loss_weight}, focal_gamma={args.focal_gamma}, "
        f"positive_weight={args.positive_weight}"
    )
    print(
        "hard negatives: "
        f"ratio={args.hard_negative_ratio}, max={args.max_hard_negatives}, "
        f"hard_weight={args.hard_negative_weight}, easy_weight={args.easy_negative_weight}"
    )
    print(f"camadas treinaveis no modelo: {count_trainable_layers(model)}")

    dummy = tf.zeros((2, *IMAGE_SHAPE), dtype=tf.float32)
    outputs = model(dummy, training=False)
    print("\nSmoke forward pass")
    print(f"presence: {tuple(outputs['presence'].shape)}")
    print(f"bbox: {tuple(outputs['bbox'].shape)}")
    print(f"points: {tuple(outputs['points'].shape)}")


if __name__ == "__main__":
    main()
