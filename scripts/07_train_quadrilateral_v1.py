"""Fase 2: treino do modelo bbox + 4 pontos por vertebra."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime
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
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a Fase 2: bbox + quadrilateros UL, UR, LL, LR."
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument(
        "--val-source",
        choices=("train", "test"),
        default="train",
        help=(
            "Origem da validacao. Por defeito usa um subset disjunto de "
            "train_clean; test deve ser usado apenas de forma explicita."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--model-name", default="phase2_quadrilateral_v1.keras")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--num-overlays", type=int, default=8)
    parser.add_argument(
        "--overlay-dir",
        default=str(OUTPUTS_DIR / "phase2_quadrilateral_v1"),
    )
    return parser.parse_args()


def select_subset(items: Sequence[Any], size: int) -> Sequence[Any]:
    if size <= 0:
        return items
    return items[: min(size, len(items))]


def load_split_samples_and_paths(
    tfdata: ModuleType,
    split: str,
) -> tuple[Sequence[Any], Sequence[Path]]:
    if split == "train":
        samples = tfdata.load_annotations(tfdata.TRAIN_JSON)
    else:
        samples = tfdata.load_annotations(tfdata.TEST_JSON)

    image_index = tfdata.build_image_index(split)
    image_paths = tfdata.resolve_all_image_paths(samples, image_index, split)
    return samples, image_paths


def split_train_validation_from_train(
    samples: Sequence[Any],
    image_paths: Sequence[Path],
    train_size: int,
    val_size: int,
) -> tuple[Sequence[Any], Sequence[Path], Sequence[Any], Sequence[Path]]:
    train_count = len(samples) if train_size <= 0 else min(train_size, len(samples))
    val_start = train_count
    remaining = len(samples) - val_start
    if remaining <= 0:
        raise ValueError(
            "Nao ha samples restantes para validacao em train_clean. "
            "Reduz --train-size ou usa --val-source test explicitamente."
        )

    val_count = remaining if val_size <= 0 else min(val_size, remaining)
    val_end = val_start + val_count
    return (
        samples[:train_count],
        image_paths[:train_count],
        samples[val_start:val_end],
        image_paths[val_start:val_end],
    )


def history_to_jsonable(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


GROUND_TRUTH_COLOR = np.array([0, 220, 0], dtype=np.uint8)


def draw_point(canvas: np.ndarray, x: float, y: float, color: np.ndarray, radius: int = 1) -> None:
    x_int = int(np.clip(round(x), 0, canvas.shape[1] - 1))
    y_int = int(np.clip(round(y), 0, canvas.shape[0] - 1))
    x0 = max(0, x_int - radius)
    x1 = min(canvas.shape[1], x_int + radius + 1)
    y0 = max(0, y_int - radius)
    y1 = min(canvas.shape[0], y_int + radius + 1)
    canvas[y0:y1, x0:x1] = color


def draw_line(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: np.ndarray,
    radius: int = 0,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for x, y in zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps)):
        draw_point(canvas, float(x), float(y), color, radius=radius)


def draw_quadrilateral(canvas: np.ndarray, points: np.ndarray, color: np.ndarray) -> None:
    # points chegam em ordem semantica UL, UR, LL, LR; para desenhar o contorno
    # sem cruzamentos, a ordem geometrica e UL, UR, LR, LL.
    ordered_points = points[[0, 1, 3, 2]]
    closed = np.vstack([ordered_points, ordered_points[0]])
    for start, end in zip(closed[:-1], closed[1:]):
        draw_line(canvas, start, end, color)


def draw_ground_truth_points(canvas: np.ndarray, points: np.ndarray) -> None:
    for vertebra_points in points:
        for x, y in vertebra_points:
            draw_point(canvas, float(x), float(y), GROUND_TRUTH_COLOR, radius=1)


def make_overlay(
    image: tf.Tensor,
    targets: Mapping[str, Any],
    predictions: Mapping[str, tf.Tensor],
) -> np.ndarray:
    canvas = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
    vertebra_count = int(targets["vertebra_count"])

    gt_points = targets["points"][:vertebra_count].reshape(vertebra_count, 4, 2) * 512.0
    pred_points = predictions["points"].numpy()[0, :vertebra_count].reshape(vertebra_count, 4, 2) * 512.0

    pred_color = np.array([255, 0, 0], dtype=np.uint8)

    draw_ground_truth_points(canvas, gt_points)
    for points in pred_points:
        draw_quadrilateral(canvas, points, pred_color)

    return canvas


def generate_overlays(
    model: tf.keras.Model,
    tfdata: ModuleType,
    samples: Sequence[Mapping[str, Any]],
    image_paths: Sequence[Path],
    output_dir: Path,
    num_images: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = min(num_images, len(samples))
    if total <= 0:
        return

    print(f"\nA gerar {total} overlays de quadrilateros em: {output_dir}")
    for index in range(total):
        sample = samples[index]
        image_path = image_paths[index]
        targets = tfdata.sample_to_targets(sample)
        image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
        predictions = model(tf.expand_dims(image, axis=0), training=False)
        overlay = make_overlay(image, targets, predictions)

        stem = Path(str(sample["file_name"])).stem
        output_path = output_dir / f"{index:03d}_{stem}_quadrilateral_overlay.png"
        tf.io.write_file(
            str(output_path),
            tf.io.encode_png(tf.convert_to_tensor(overlay)),
        )
        print(f"{sample['file_name']}: {output_path.name}")

    print("Legenda: verde=pontos ground truth, vermelho=quadrilatero previsto.")


def main() -> None:
    args = parse_args()
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    quadrilateral = import_script("06_quadrilateral_model_v1.py", "phase2_quadrilateral")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = args.experiment_name or datetime.now().strftime("phase2_%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / run_name
    experiment_dir.mkdir(parents=True, exist_ok=False)

    print("A carregar, indexar e verificar train_clean...")
    all_train_samples, all_train_paths = load_split_samples_and_paths(tfdata, "train")

    if args.val_source == "train":
        train_samples, train_paths, val_samples, val_paths = split_train_validation_from_train(
            all_train_samples,
            all_train_paths,
            train_size=args.train_size,
            val_size=args.val_size,
        )
        val_description = "subset disjunto de train_clean"
    else:
        print("A carregar, indexar e verificar test_clean para validacao explicita...")
        train_samples = select_subset(all_train_samples, args.train_size)
        train_paths = select_subset(all_train_paths, args.train_size)
        test_samples, test_paths = load_split_samples_and_paths(tfdata, "test")
        val_samples = select_subset(test_samples, args.val_size)
        val_paths = select_subset(test_paths, args.val_size)
        val_description = "test_clean usado explicitamente"

    print(f"train samples usados: {len(train_samples)}")
    print(f"val samples usados: {len(val_samples)}")
    print(f"validacao: {val_description}")

    train_dataset = tfdata.build_dataset(
        samples=train_samples,
        image_paths=train_paths,
        batch_size=args.batch_size,
        shuffle=True,
        cache=args.cache,
    ).map(quadrilateral.quadrilateral_targets, num_parallel_calls=tf.data.AUTOTUNE).repeat()

    val_dataset = tfdata.build_dataset(
        samples=val_samples,
        image_paths=val_paths,
        batch_size=args.batch_size,
        shuffle=False,
        cache=args.cache,
    ).map(quadrilateral.quadrilateral_targets, num_parallel_calls=tf.data.AUTOTUNE).repeat()

    model = quadrilateral.build_quadrilateral_model_v1()
    quadrilateral.compile_quadrilateral_model_v1(model, learning_rate=args.learning_rate)

    images, labels = next(iter(train_dataset))
    outputs = model(images, training=False)
    print("\nSmoke batch antes do treino")
    print(f"images: {tuple(images.shape)}")
    print(f"label presence: {tuple(labels['presence'].shape)}")
    print(f"label bbox+mask: {tuple(labels['bbox'].shape)}")
    print(f"label points+mask: {tuple(labels['points'].shape)}")
    print(f"pred presence: {tuple(outputs['presence'].shape)}")
    print(f"pred bbox: {tuple(outputs['bbox'].shape)}")
    print(f"pred points: {tuple(outputs['points'].shape)}")

    model_path = MODELS_DIR / args.model_name
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(experiment_dir / "training_log.csv"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    validation_steps = math.ceil(len(val_samples) / args.batch_size)

    print("\nA iniciar treino da Fase 2...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save(model_path)

    history_path = experiment_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history_to_jsonable(history.history), file, indent=2)

    config_path = experiment_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    generate_overlays(
        model=model,
        tfdata=tfdata,
        samples=val_samples,
        image_paths=val_paths,
        output_dir=Path(args.overlay_dir),
        num_images=args.num_overlays,
    )

    print("\nSUCESSO: treino da Fase 2 concluido.")
    print(f"Modelo guardado em: {model_path}")
    print(f"Historico guardado em: {history_path}")


if __name__ == "__main__":
    main()
