"""Fase 8: head auxiliar para os angulos regionais PT, MT e TLL.

Esta versao recebe os embeddings contextualizados exportados pela Fase 7 e
treina uma pequena head global multi-output. A regressao dos angulos e uma
supervisao auxiliar; nao substitui o Cobb geometrico final previsto para a
Fase 9.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative o ambiente .venv antes de correr este script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_VERTEBRAE = 21
DEFAULT_CONTEXT_DIM = 128
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT_RATE = 0.25
ANGLE_NAMES = ("PT", "MT", "TLL")


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
class MaskedMeanPooling(tf.keras.layers.Layer):
    """Media temporal ignorando posicoes padded."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor]) -> tf.Tensor:
        values, mask_values = inputs
        values = tf.cast(values, tf.float32)
        mask_float = tf.cast(mask_values, tf.float32)
        mask_expanded = tf.expand_dims(mask_float, axis=-1)
        summed = tf.reduce_sum(values * mask_expanded, axis=1)
        denominator = tf.maximum(tf.reduce_sum(mask_expanded, axis=1), 1.0)
        return summed / denominator

    def compute_mask(self, inputs: Any, mask: Any = None) -> None:
        return None


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
class MaskedMaxPooling(tf.keras.layers.Layer):
    """Max pooling temporal ignorando posicoes padded."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor]) -> tf.Tensor:
        values, mask_values = inputs
        values = tf.cast(values, tf.float32)
        mask_float = tf.cast(mask_values, tf.float32)
        mask_expanded = tf.expand_dims(mask_float, axis=-1)
        very_negative = tf.ones_like(values) * tf.constant(-1.0e9, dtype=tf.float32)
        masked_values = tf.where(mask_expanded > 0.5, values, very_negative)
        pooled = tf.reduce_max(masked_values, axis=1)
        has_values = tf.reduce_any(mask_float > 0.5, axis=1, keepdims=True)
        return tf.where(has_values, pooled, tf.zeros_like(pooled))

    def compute_mask(self, inputs: Any, mask: Any = None) -> None:
        return None


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
class MaskToCountNorm(tf.keras.layers.Layer):
    """Calcula a contagem normalizada a partir da mask da sequencia."""

    def __init__(self, max_vertebrae: int = MAX_VERTEBRAE, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_vertebrae = int(max_vertebrae)

    def call(self, mask_values: tf.Tensor) -> tf.Tensor:
        mask_float = tf.cast(mask_values, tf.float32)
        count = tf.reduce_sum(mask_float, axis=1, keepdims=True)
        return count / float(self.max_vertebrae)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"max_vertebrae": self.max_vertebrae})
        return config


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
def count_mae_vertebrae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE da contagem convertido da escala normalizada para vertebras."""

    return tf.reduce_mean(tf.abs(y_true - y_pred)) * float(MAX_VERTEBRAE)


def _angle_bias_initializer(
    angle_bias: Sequence[float] | None,
) -> str | tf.keras.initializers.Initializer:
    if angle_bias is None:
        return "zeros"
    values = [float(value) for value in angle_bias]
    if len(values) != len(ANGLE_NAMES):
        raise ValueError(f"angle_bias deve ter {len(ANGLE_NAMES)} valores.")
    return tf.keras.initializers.Constant(values)


def _count_bias_initializer(
    count_bias_norm: float | None,
) -> str | tf.keras.initializers.Initializer:
    if count_bias_norm is None:
        return "zeros"
    clipped = min(max(float(count_bias_norm), 1.0e-4), 1.0 - 1.0e-4)
    logit = math.log(clipped / (1.0 - clipped))
    return tf.keras.initializers.Constant(logit)


def build_aux_angle_head_v1(
    max_vertebrae: int = MAX_VERTEBRAE,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    angle_bias: Sequence[float] | None = None,
    count_bias_norm: float | None = None,
) -> tf.keras.Model:
    """Constroi a head global da Fase 8 sobre embeddings contextualizados."""

    contextual_input = tf.keras.Input(
        shape=(max_vertebrae, context_dim),
        name="contextual_embeddings",
    )
    mask_input = tf.keras.Input(shape=(max_vertebrae,), name="mask")

    raw_count_norm = MaskToCountNorm(max_vertebrae, name="raw_count_norm")(mask_input)

    x = tf.keras.layers.LayerNormalization(name="context_norm")(contextual_input)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="slot_projection")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="slot_dropout")(x)

    mean_pool = MaskedMeanPooling(name="context_masked_mean")([x, mask_input])
    max_pool = MaskedMaxPooling(name="context_masked_max")([x, mask_input])
    global_features = tf.keras.layers.Concatenate(name="global_features")(
        [mean_pool, max_pool, raw_count_norm]
    )

    shared = tf.keras.layers.Dense(hidden_dim, activation="relu", name="shared_dense_1")(
        global_features
    )
    shared = tf.keras.layers.Dropout(dropout_rate, name="shared_dropout_1")(shared)
    shared = tf.keras.layers.Dense(hidden_dim // 2, activation="relu", name="shared_dense_2")(
        shared
    )

    angle_hidden = tf.keras.layers.Dropout(dropout_rate, name="angle_dropout")(shared)
    angle_deg = tf.keras.layers.Dense(
        len(ANGLE_NAMES),
        activation="linear",
        bias_initializer=_angle_bias_initializer(angle_bias),
        name="angle_deg",
    )(angle_hidden)

    count_hidden = tf.keras.layers.Dropout(dropout_rate, name="count_dropout")(shared)
    count_norm = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=_count_bias_initializer(count_bias_norm),
        name="count_norm",
    )(count_hidden)

    return tf.keras.Model(
        inputs={"contextual_embeddings": contextual_input, "mask": mask_input},
        outputs={"angle_deg": angle_deg, "count_norm": count_norm},
        name="phase8_aux_angle_head_v1",
    )


def compile_aux_angle_model(
    model: tf.keras.Model,
    learning_rate: float = 1.0e-3,
    angle_loss_weight: float = 1.0,
    count_loss_weight: float = 0.2,
) -> tf.keras.Model:
    """Compila o modelo multi-task da Fase 8."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "angle_deg": tf.keras.losses.Huber(delta=5.0),
            "count_norm": tf.keras.losses.Huber(delta=1.0 / MAX_VERTEBRAE),
        },
        loss_weights={
            "angle_deg": float(angle_loss_weight),
            "count_norm": float(count_loss_weight),
        },
        metrics={
            "angle_deg": [tf.keras.metrics.MeanAbsoluteError(name="mae_deg")],
            "count_norm": [count_mae_vertebrae],
        },
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica a head auxiliar da Fase 8.")
    parser.add_argument("--max-vertebrae", type=int, default=MAX_VERTEBRAE)
    parser.add_argument("--context-dim", type=int, default=DEFAULT_CONTEXT_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_aux_angle_head_v1(
        max_vertebrae=args.max_vertebrae,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
    )
    compile_aux_angle_model(model)

    dummy_context = tf.zeros((2, args.max_vertebrae, args.context_dim), dtype=tf.float32)
    dummy_mask = tf.concat(
        [
            tf.ones((2, args.max_vertebrae - 3), dtype=tf.float32),
            tf.zeros((2, 3), dtype=tf.float32),
        ],
        axis=1,
    )
    outputs = model({"contextual_embeddings": dummy_context, "mask": dummy_mask})

    model.summary()
    print("\nFase 8 OK: modelo criado.")
    print(f"angle_deg output: {tuple(outputs['angle_deg'].shape)}")
    print(f"count_norm output: {tuple(outputs['count_norm'].shape)}")
    print(f"angulos: {', '.join(ANGLE_NAMES)}")


if __name__ == "__main__":
    main()
