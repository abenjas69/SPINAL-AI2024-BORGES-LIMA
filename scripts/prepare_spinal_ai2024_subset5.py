"""Prepare local SPINAL-AI2024 subset5 data for image-level evaluations.

The public portfolio repository does not redistribute the dataset. This helper
expects a local clone or extracted copy of the upstream dataset repository and
creates the local layout used by the evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGE_DIR = PROJECT_ROOT / "raw" / "images" / "test" / "Spinal-AI2024-subset5"
TARGET_ANNOTATIONS = PROJECT_ROOT / "processed" / "cleaned" / "test_ready_annotations_clean.json"
UPSTREAM_SUBSET_DIR = "Spinal-AI2024-subset5"
UPSTREAM_ANNOTATION_ZIP = "Spinal_AI2024_test_annotation.zip"
UPSTREAM_ANNOTATION_JSON = "Spinal_AI2024_test_annotation.json"
UPSTREAM_COBB_GT = "Cobb_spinal-AI2024-test_gt.txt"
EXPECTED_SOURCE_IMAGES = 4000
EXPECTED_CLEAN_SAMPLES = 3988
DEFAULT_MIN_VERTEBRAE = 14
POINT_ORDER = ("upper_left", "upper_right", "lower_left", "lower_right")


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_coco_from_zip(zip_path: Path) -> dict[str, Any]:
    require_path(zip_path, "upstream test annotation zip")
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.replace("\\", "/").endswith(UPSTREAM_ANNOTATION_JSON)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"{zip_path} does not contain {UPSTREAM_ANNOTATION_JSON}"
            )
        with archive.open(candidates[0]) as file:
            return json.loads(file.read().decode("utf-8"))


def load_cobb_gt(path: Path) -> dict[str, dict[str, float]]:
    require_path(path, "upstream test Cobb ground truth")
    values: dict[str, dict[str, float]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid Cobb GT line {line_number}: {raw_line!r}")
        file_name, pt, mt, tll = parts
        values[file_name] = {"PT": float(pt), "MT": float(mt), "TLL": float(tll)}
    return values


def points_from_segmentation(annotation: dict[str, Any]) -> dict[str, list[float]]:
    segmentation = annotation.get("segmentation")
    if not segmentation or not isinstance(segmentation, list):
        raise ValueError(f"Annotation {annotation.get('id')} has no segmentation")
    flat = segmentation[0]
    if len(flat) != 8:
        raise ValueError(
            f"Annotation {annotation.get('id')} segmentation must have 8 values"
        )
    return {
        point_name: [float(flat[index * 2]), float(flat[index * 2 + 1])]
        for index, point_name in enumerate(POINT_ORDER)
    }


def y_centroid(points: dict[str, list[float]]) -> float:
    return sum(point[1] for point in points.values()) / len(points)


def build_clean_annotations(
    coco: dict[str, Any],
    cobb_gt: dict[str, dict[str, float]],
    min_vertebrae: int,
) -> list[dict[str, Any]]:
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    samples: list[dict[str, Any]] = []
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        file_name = str(image["file_name"])
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) < min_vertebrae:
            continue
        if file_name not in cobb_gt:
            raise KeyError(f"Cobb GT missing for {file_name}")

        vertebrae = []
        for annotation in annotations:
            points = points_from_segmentation(annotation)
            vertebrae.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "category_id": int(annotation.get("category_id", 1)),
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "points": points,
                    "y_centroid": y_centroid(points),
                }
            )
        vertebrae.sort(key=lambda vertebra: float(vertebra["y_centroid"]))

        samples.append(
            {
                "split": "test",
                "image_id": image_id,
                "file_name": file_name,
                "image_path": f"raw/images/test/{UPSTREAM_SUBSET_DIR}/{file_name}",
                "height": int(image["height"]),
                "width": int(image["width"]),
                "vertebrae": vertebrae,
                "vertebra_count": len(vertebrae),
                "cobb_angles": cobb_gt[file_name],
            }
        )
    return samples


def copy_images(upstream_subset_dir: Path, target_dir: Path, overwrite: bool) -> int:
    require_path(upstream_subset_dir, "upstream subset5 image directory")
    source_images = sorted(upstream_subset_dir.glob("*.jpg"))
    if len(source_images) != EXPECTED_SOURCE_IMAGES:
        print(
            f"Warning: expected {EXPECTED_SOURCE_IMAGES} upstream images, "
            f"found {len(source_images)} in {upstream_subset_dir}",
            file=sys.stderr,
        )

    if upstream_subset_dir.resolve() == target_dir.resolve():
        return len(source_images)

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in source_images:
        target_path = target_dir / source_path.name
        if target_path.exists() and not overwrite:
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def write_clean_annotations(samples: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(samples, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def count_local_images() -> int:
    if not TARGET_IMAGE_DIR.is_dir():
        return 0
    return len(list(TARGET_IMAGE_DIR.glob("*.jpg")))


def check_local_data() -> int:
    problems: list[str] = []
    image_count = count_local_images()
    if image_count != EXPECTED_SOURCE_IMAGES:
        problems.append(
            f"Expected {EXPECTED_SOURCE_IMAGES} images in "
            f"{project_relative(TARGET_IMAGE_DIR)}, found {image_count}."
        )

    if not TARGET_ANNOTATIONS.is_file():
        problems.append(f"Missing {project_relative(TARGET_ANNOTATIONS)}.")
        sample_count = 0
    else:
        data = json.loads(TARGET_ANNOTATIONS.read_text(encoding="utf-8"))
        sample_count = len(data)
        if sample_count != EXPECTED_CLEAN_SAMPLES:
            problems.append(
                f"Expected {EXPECTED_CLEAN_SAMPLES} cleaned samples in "
                f"{project_relative(TARGET_ANNOTATIONS)}, found {sample_count}."
            )

    print("SPINAL-AI2024 subset5 local data check")
    print(f"- images: {image_count}/{EXPECTED_SOURCE_IMAGES}")
    print(f"- cleaned annotations: {sample_count}/{EXPECTED_CLEAN_SAMPLES}")

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"- {problem}")
        print(
            "\nRun:\n"
            "python scripts/prepare_spinal_ai2024_subset5.py "
            "--upstream <path-to-Ernestchenchen-Spinal-AI2024>"
        )
        return 1

    print("OK: local subset5 data is ready for image-level evaluation.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local SPINAL-AI2024 subset5 data for evaluation."
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        help=(
            "Path to a local clone or extracted copy of "
            "https://github.com/Ernestchenchen/Spinal-AI2024"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate the local expected dataset placement.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Generate annotations without copying images.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite already copied local subset5 images.",
    )
    parser.add_argument(
        "--min-vertebrae",
        type=int,
        default=DEFAULT_MIN_VERTEBRAE,
        help="Minimum vertebra annotations required to keep a sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        return check_local_data()

    if args.upstream is None:
        raise SystemExit("--upstream is required unless --check-only is used")

    upstream_root = args.upstream.resolve()
    require_path(upstream_root, "upstream SPINAL-AI2024 directory")

    if not args.skip_images:
        copied = copy_images(
            upstream_root / UPSTREAM_SUBSET_DIR,
            TARGET_IMAGE_DIR,
            overwrite=args.overwrite_images,
        )
        print(
            f"Images ready in {project_relative(TARGET_IMAGE_DIR)} "
            f"({copied} copied, {count_local_images()} present)."
        )

    coco = load_coco_from_zip(upstream_root / UPSTREAM_ANNOTATION_ZIP)
    cobb_gt = load_cobb_gt(upstream_root / UPSTREAM_COBB_GT)
    samples = build_clean_annotations(coco, cobb_gt, min_vertebrae=args.min_vertebrae)
    write_clean_annotations(samples, TARGET_ANNOTATIONS)
    print(
        f"Cleaned annotations written to {project_relative(TARGET_ANNOTATIONS)} "
        f"({len(samples)} samples)."
    )
    if len(samples) != EXPECTED_CLEAN_SAMPLES:
        print(
            f"Warning: expected {EXPECTED_CLEAN_SAMPLES} samples with "
            f"--min-vertebrae {DEFAULT_MIN_VERTEBRAE}.",
            file=sys.stderr,
        )
    return check_local_data()


if __name__ == "__main__":
    raise SystemExit(main())
