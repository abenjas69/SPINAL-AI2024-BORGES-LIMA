"""Fase 7: modelos sequenciais sobre embeddings vertebrais.

A Fase 7 recebe as sequencias de embeddings da Fase 6 e adiciona contexto
bidirecional ao longo da coluna. Esta primeira versao usa uma tarefa objetiva e
auditavel: prever a contagem real de vertebras da imagem. Assim conseguimos
comparar diretamente:

- contagem bruta herdada da Fase 6;
- baseline sem contexto sequencial;
- BiLSTM com contexto top-down e bottom-up.

O modelo tambem expoe a camada `context_projection`, que serve como embedding
vertebral contextualizado para fases seguintes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative o ambiente .venv antes de correr este script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_VERTEBRAE = 21
DEFAULT_EMBEDDING_DIM = 152
DEFAULT_PROJECTION_DIM = 128
DEFAULT_LSTM_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.2


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
class SequenceMask(tf.keras.layers.Layer):
    """Converte a mascara float da Fase 6 numa mascara booleana Keras."""

    def call(self, mask_values: tf.Tensor) -> tf.Tensor:
        return tf.greater(tf.cast(mask_values, tf.float32), 0.5)


@tf.keras.utils.register_keras_serializable(package="SpinalAI")
class MaskToCountNorm(tf.keras.layers.Layer):
    """Calcula a contagem normalizada a partir da mascara de sequencia."""

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
def count_mae_vertebrae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE da contagem convertido da escala normalizada para vertebras."""

    return tf.reduce_mean(tf.abs(y_true - y_pred)) * float(MAX_VERTEBRAE)


def compile_count_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"count_norm": tf.keras.losses.Huber(delta=1.0 / MAX_VERTEBRAE)},
        metrics={"count_norm": [count_mae_vertebrae]},
    )
    return model


def build_no_context_count_model_v1(
    max_vertebrae: int = MAX_VERTEBRAE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> tf.keras.Model:
    """Baseline controlada sem BiLSTM, usando apenas pooling mascarado."""

    embeddings_input = tf.keras.Input(
        shape=(max_vertebrae, embedding_dim),
        name="embeddings",
    )
    mask_input = tf.keras.Input(shape=(max_vertebrae,), name="mask")

    x = tf.keras.layers.LayerNormalization(name="embedding_norm")(embeddings_input)
    x = tf.keras.layers.Dense(projection_dim, activation="relu", name="slot_projection")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="slot_dropout")(x)
    pooled = MaskedMeanPooling(name="no_context_masked_mean")([x, mask_input])
    raw_count_norm = MaskToCountNorm(max_vertebrae, name="raw_count_norm")(mask_input)
    global_features = tf.keras.layers.Concatenate(name="no_context_global_features")(
        [pooled, raw_count_norm]
    )
    x = tf.keras.layers.Dense(64, activation="relu", name="count_dense")(global_features)
    x = tf.keras.layers.Dropout(dropout_rate, name="count_dropout")(x)
    count_norm = tf.keras.layers.Dense(1, activation="sigmoid", name="count_norm")(x)

    return tf.keras.Model(
        inputs={"embeddings": embeddings_input, "mask": mask_input},
        outputs={"count_norm": count_norm},
        name="phase7_no_context_count_v1",
    )


def build_bilstm_global_model_v1(
    max_vertebrae: int = MAX_VERTEBRAE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> tf.keras.Model:
    """Modelo BiLSTM da Fase 7 para contexto global vertebral."""

    embeddings_input = tf.keras.Input(
        shape=(max_vertebrae, embedding_dim),
        name="embeddings",
    )
    mask_input = tf.keras.Input(shape=(max_vertebrae,), name="mask")

    sequence_mask = SequenceMask(name="sequence_mask")(mask_input)
    raw_count_norm = MaskToCountNorm(max_vertebrae, name="raw_count_norm")(mask_input)

    x = tf.keras.layers.LayerNormalization(name="embedding_norm")(embeddings_input)
    x = tf.keras.layers.Dense(projection_dim, activation="relu", name="embedding_projection")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="embedding_dropout")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=0.0,
        ),
        name="bilstm_context",
    )(x, mask=sequence_mask)
    context = tf.keras.layers.Dense(
        projection_dim,
        activation="relu",
        name="context_projection",
    )(x)

    pooled = MaskedMeanPooling(name="context_masked_mean")([context, mask_input])
    global_features = tf.keras.layers.Concatenate(name="context_global_features")(
        [pooled, raw_count_norm]
    )
    x = tf.keras.layers.Dense(64, activation="relu", name="count_dense")(global_features)
    x = tf.keras.layers.Dropout(dropout_rate, name="count_dropout")(x)
    count_norm = tf.keras.layers.Dense(1, activation="sigmoid", name="count_norm")(x)

    return tf.keras.Model(
        inputs={"embeddings": embeddings_input, "mask": mask_input},
        outputs={"count_norm": count_norm},
        name="phase7_bilstm_global_v1",
    )


def build_context_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Devolve um modelo auxiliar para exportar embeddings contextualizados."""

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("context_projection").output,
        name="phase7_context_extractor_v1",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica modelos sequenciais da Fase 7.")
    parser.add_argument("--max-vertebrae", type=int, default=MAX_VERTEBRAE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument("--lstm-units", type=int, default=DEFAULT_LSTM_UNITS)
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_context = build_no_context_count_model_v1(
        max_vertebrae=args.max_vertebrae,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout_rate=args.dropout_rate,
    )
    bilstm = build_bilstm_global_model_v1(
        max_vertebrae=args.max_vertebrae,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
    )

    no_context.summary()
    bilstm.summary()
    print("\nFase 7 OK: modelos criados.")
    print(f"Baseline sem contexto output: {no_context.output_shape}")
    print(f"BiLSTM output: {bilstm.output_shape}")


if __name__ == "__main__":
    main()
