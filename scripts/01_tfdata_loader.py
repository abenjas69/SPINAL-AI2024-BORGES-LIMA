"""Fase 0: tf.data loader para Spinal-AI2024.

Este script apenas valida a infraestrutura de dados:
- le annotations clean;
- verifica caminhos locais das imagens;
- carrega e normaliza imagens;
- prepara targets com padding ate MAX_VERTEBRAE;
- cria batches tf.data e imprime um batch de debug.

Nao implementa treino real, ResNet50, FPN, BiLSTM ou heads finais.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow nao esta instalado no Python ativo. "
        "Ative um ambiente com TensorFlow/Keras antes de correr este script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "processed" / "cleaned"
TRAIN_JSON = CLEAN_DIR / "train_ready_annotations_clean.json"
TEST_JSON = CLEAN_DIR / "test_ready_annotations_clean.json"
IMAGE_ROOT = PROJECT_ROOT / "raw" / "images"

IMAGE_SIZE = (512, 512)
MAX_VERTEBRAE = 21
DEFAULT_BATCH_SIZE = 4
DEFAULT_SUBSET_SIZE = 50
POINT_ORDER = ("upper_left", "upper_right", "lower_left", "lower_right")
COBB_ORDER = ("PT", "MT", "TLL")


Targets = Dict[str, np.ndarray | np.int32 | str]
Sample = Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test do tf.data loader para Spinal-AI2024."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Aplica dataset.cache() ao subset usado no smoke test.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Desativa shuffle no dataset de treino de debug.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro obrigatorio nao encontrado: {path}")


def load_annotations(path: Path) -> List[Sample]:
    require_file(path)
    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError(f"O JSON deve conter uma lista de samples: {path}")

    return samples


def build_image_index(split: str) -> Dict[str, Path]:
    split_root = IMAGE_ROOT / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {split_root}")

    image_index: Dict[str, Path] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        for image_path in split_root.rglob(suffix):
            image_index[image_path.name] = image_path

    if not image_index:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {split_root}")

    return image_index


def resolve_image_path(sample: Sample, image_index: Mapping[str, Path]) -> Path:
    raw_path = str(sample.get("image_path", ""))
    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_file():
            return candidate

    file_name = str(sample["file_name"])
    indexed_path = image_index.get(file_name)
    if indexed_path is not None and indexed_path.is_file():
        return indexed_path

    split = str(sample.get("split", "")).strip()
    direct_candidate = IMAGE_ROOT / split / file_name
    if direct_candidate.is_file():
        return direct_candidate

    raise FileNotFoundError(
        f"Imagem nao encontrada para sample {file_name}. "
        f"Caminho no JSON: {raw_path or '<vazio>'}"
    )


def resolve_all_image_paths(
    samples: Sequence[Sample],
    image_index: Mapping[str, Path],
    split: str,
) -> List[Path]:
    resolved_paths: List[Path] = []
    missing: List[str] = []

    for sample in samples:
        try:
            resolved_paths.append(resolve_image_path(sample, image_index))
        except FileNotFoundError as exc:
            missing.append(str(exc))
            if len(missing) >= 10:
                break

    if missing:
        details = "\n".join(f"- {message}" for message in missing)
        raise FileNotFoundError(
            f"Falha ao resolver imagens do split {split}. Exemplos:\n{details}"
        )

    return resolved_paths


def normalize_bbox(bbox: Sequence[float], width: float, height: float) -> np.ndarray:
    x, y, box_width, box_height = [float(value) for value in bbox]
    return np.asarray(
        [x / width, y / height, box_width / width, box_height / height],
        dtype=np.float32,
    )


def normalize_points(points: Mapping[str, Sequence[float]], width: float, height: float) -> np.ndarray:
    normalized: List[float] = []
    for point_name in POINT_ORDER:
        x, y = points[point_name]
        normalized.extend([float(x) / width, float(y) / height])
    return np.asarray(normalized, dtype=np.float32)


def cobb_to_array(cobb_angles: Mapping[str, Any]) -> np.ndarray:
    values: List[float] = []
    for key in COBB_ORDER:
        if key == "TLL":
            value = cobb_angles.get("TLL", cobb_angles.get("TL/L"))
        else:
            value = cobb_angles.get(key)

        if value is None:
            raise ValueError(f"Cobb angle em falta para chave {key}: {cobb_angles}")

        values.append(float(value))

    return np.asarray(values, dtype=np.float32)


def sample_to_targets(sample: Sample) -> Targets:
    width = float(sample["width"])
    height = float(sample["height"])
    vertebrae = sorted(
        sample["vertebrae"],
        key=lambda vertebra: float(vertebra.get("y_centroid", 0.0)),
    )

    if len(vertebrae) > MAX_VERTEBRAE:
        raise ValueError(
            f"{sample['file_name']} tem {len(vertebrae)} vertebras; "
            f"MAX_VERTEBRAE={MAX_VERTEBRAE}."
        )

    presence = np.zeros((MAX_VERTEBRAE, 1), dtype=np.float32)
    bbox = np.zeros((MAX_VERTEBRAE, 4), dtype=np.float32)
    points = np.zeros((MAX_VERTEBRAE, 8), dtype=np.float32)
    mask = np.zeros((MAX_VERTEBRAE, 1), dtype=np.float32)

    for index, vertebra in enumerate(vertebrae):
        presence[index, 0] = 1.0
        mask[index, 0] = 1.0
        bbox[index] = normalize_bbox(vertebra["bbox"], width, height)
        points[index] = normalize_points(vertebra["points"], width, height)

    return {
        "presence": presence,
        "bbox": bbox,
        "points": points,
        "mask": mask,
        "cobb_angles": cobb_to_array(sample["cobb_angles"]),
        "vertebra_count": np.int32(sample.get("vertebra_count", len(vertebrae))),
        "file_name": str(sample["file_name"]),
    }


def make_generator(
    samples: Sequence[Sample],
    image_paths: Sequence[Path],
) -> Iterable[Tuple[str, Targets]]:
    def generator() -> Iterator[Tuple[str, Targets]]:
        for sample, image_path in zip(samples, image_paths):
            yield str(image_path), sample_to_targets(sample)

    return generator


def output_signature() -> Tuple[tf.TensorSpec, Dict[str, tf.TensorSpec]]:
    return (
        tf.TensorSpec(shape=(), dtype=tf.string),
        {
            "presence": tf.TensorSpec(shape=(MAX_VERTEBRAE, 1), dtype=tf.float32),
            "bbox": tf.TensorSpec(shape=(MAX_VERTEBRAE, 4), dtype=tf.float32),
            "points": tf.TensorSpec(shape=(MAX_VERTEBRAE, 8), dtype=tf.float32),
            "mask": tf.TensorSpec(shape=(MAX_VERTEBRAE, 1), dtype=tf.float32),
            "cobb_angles": tf.TensorSpec(shape=(3,), dtype=tf.float32),
            "vertebra_count": tf.TensorSpec(shape=(), dtype=tf.int32),
            "file_name": tf.TensorSpec(shape=(), dtype=tf.string),
        },
    )


def load_image(image_path: tf.Tensor, targets: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    image_bytes = tf.io.read_file(image_path)
    # channels=3 replica radiografias grayscale para 3 canais quando necessario.
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, targets


def build_dataset(
    samples: Sequence[Sample],
    image_paths: Sequence[Path],
    batch_size: int,
    shuffle: bool,
    cache: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        make_generator(samples, image_paths),
        output_signature=output_signature(),
    )

    if shuffle:
        buffer_size = min(len(samples), 512)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=False)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def print_dataset_summary(name: str, samples: Sequence[Sample], image_paths: Sequence[Path]) -> None:
    counts = [int(sample["vertebra_count"]) for sample in samples]
    print(f"{name}: {len(samples)} samples")
    print(f"{name}: vertebra_count min={min(counts)}, max={max(counts)}")
    print(f"{name}: primeiro caminho resolvido={image_paths[0]}")


def debug_batch(dataset: tf.data.Dataset) -> None:
    images, targets = next(iter(dataset))

    print("\nBatch debug")
    print(f"images: {tuple(images.shape)}")
    print(f"presence: {tuple(targets['presence'].shape)}")
    print(f"bbox: {tuple(targets['bbox'].shape)}")
    print(f"points: {tuple(targets['points'].shape)}")
    print(f"mask: {tuple(targets['mask'].shape)}")
    print(f"cobb_angles: {tuple(targets['cobb_angles'].shape)}")
    print(f"vertebra_count: {tuple(targets['vertebra_count'].shape)}")
    print(f"image min/max: {tf.reduce_min(images).numpy():.6f} / {tf.reduce_max(images).numpy():.6f}")

    file_name = targets["file_name"][0].numpy().decode("utf-8")
    vertebra_count = int(targets["vertebra_count"][0].numpy())
    cobb_angles = targets["cobb_angles"][0].numpy()
    mask_head = targets["mask"][0, :10, 0].numpy()
    first_bbox = targets["bbox"][0, 0].numpy()
    first_points = targets["points"][0, 0].numpy()

    print("\nPrimeiro sample do batch")
    print(f"file_name: {file_name}")
    print(f"vertebra_count: {vertebra_count}")
    print(f"cobb_angles [PT, MT, TLL]: {cobb_angles}")
    print(f"mask primeiras 10 posicoes: {mask_head}")
    print(f"primeira bbox normalizada [x, y, w, h]: {first_bbox}")
    print(
        "primeiros 4 pontos normalizados "
        "[ULx, ULy, URx, URy, LLx, LLy, LRx, LRy]: "
        f"{first_points}"
    )


def main() -> None:
    args = parse_args()

    require_file(TRAIN_JSON)
    require_file(TEST_JSON)

    print("A carregar annotations clean...")
    train_samples = load_annotations(TRAIN_JSON)
    test_samples = load_annotations(TEST_JSON)

    print("A indexar imagens locais...")
    train_index = build_image_index("train")
    test_index = build_image_index("test")

    print("A verificar caminhos das imagens...")
    train_paths = resolve_all_image_paths(train_samples, train_index, "train")
    test_paths = resolve_all_image_paths(test_samples, test_index, "test")

    print_dataset_summary("train_clean", train_samples, train_paths)
    print_dataset_summary("test_clean", test_samples, test_paths)

    subset_size = min(args.subset_size, len(train_samples))
    subset_samples = train_samples[:subset_size]
    subset_paths = train_paths[:subset_size]

    print(
        f"\nA criar tf.data.Dataset de smoke test com {subset_size} imagens, "
        f"batch_size={args.batch_size}, shuffle={not args.no_shuffle}, cache={args.cache}."
    )
    train_dataset = build_dataset(
        samples=subset_samples,
        image_paths=subset_paths,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        cache=args.cache,
    )

    debug_batch(train_dataset)

    print("\nSUCESSO: Fase 0 loader validado com shapes consistentes.")


if __name__ == "__main__":
    main()
