"""Convenience runner for the SPINAL-AI2024 evaluation package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(args: list[str]) -> None:
    printable = " ".join(str(a) for a in args)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def common_window(args: argparse.Namespace) -> list[str]:
    return [
        "--start-index",
        str(args.start_index),
        "--num-images",
        str(args.num_images),
        "--progress-every",
        str(args.progress_every),
    ]


def maybe_overlays(args: argparse.Namespace) -> list[str]:
    if not args.save_overlays:
        return []
    return ["--save-overlays", "--overlay-limit", str(args.overlay_limit)]


def subset5_diogo(args: argparse.Namespace) -> None:
    run(
        [
            PYTHON,
            script("106_eval_final_test_subset5_mlp_v2_locked.py"),
            "--split",
            "test",
            *common_window(args),
            "--experiment-name",
            "eval_subset5_diogo",
            "--report-path",
            "outputs/reports/subset5_diogo.md",
        ]
    )


def subset5_daniel(args: argparse.Namespace) -> None:
    run(
        [
            PYTHON,
            script("111_eval_colleague_centerline_model_v3_phase5_roi.py"),
            "--split",
            "test",
            *common_window(args),
            "--output-dir",
            "outputs/subset5_daniel",
            "--centerline-only-fast",
            *maybe_overlays(args),
        ]
    )


def subset5_fusion(args: argparse.Namespace) -> None:
    input_dir = "outputs/subset5_fusion_inputs"
    run(
        [
            PYTHON,
            script("111_eval_colleague_centerline_model_v3_phase5_roi.py"),
            "--split",
            "test",
            *common_window(args),
            "--output-dir",
            input_dir,
            *maybe_overlays(args),
        ]
    )
    run(
        [
            PYTHON,
            script("112_eval_fusion_centerline_mlp_v3.py"),
            "--mlp-predictions",
            f"{input_dir}/mlp_v2_predictions.csv",
            "--centerline-predictions",
            f"{input_dir}/colleague_centerline_predictions.csv",
            "--output-dir",
            "outputs/subset5_fusion",
            "--lock-input",
            "experiments/fusion_centerline_mlp_v3_holdout3192/fusion_centerline_mlp_v3_lock.json",
            "--require-same-files",
        ]
    )


def metric_at(path: str, *keys: str) -> dict:
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for key in keys:
        data = data[key]
    return data


def format_n(metrics: dict) -> str:
    total = metrics.get("total_images")
    covered = metrics.get("num_images")
    if total is not None and covered is not None and total != covered:
        return f"{covered}/{total}"
    return str(covered if covered is not None else total)


def metrics_summary(args: argparse.Namespace) -> None:
    del args
    rows = [
        (
            "Landmark MLP v2, subset5",
            metric_at(
                "experiments/reference/subset5_diogo/final_test_subset5_mlp_v2_metrics.json",
                "metrics",
                "calibrated_mlp_v2_original",
            ),
        ),
        (
            "Centerline raw max Cobb, subset5",
            metric_at(
                "experiments/reference/subset5_centerline/centerline_v3_phase5_roi_metrics.json",
                "metrics",
                "max_cobb",
            ),
        ),
        (
            "Centerline bias-corrected, subset5",
            metric_at(
                "experiments/reference/subset5_fusion/fusion_centerline_mlp_v3_metrics.json",
                "metrics",
                "centerline_bias_corrected",
            ),
        ),
        (
            "Locked fusion v3, subset5",
            metric_at(
                "experiments/reference/subset5_fusion/fusion_centerline_mlp_v3_metrics.json",
                "metrics",
                "fusion_v3_locked",
            ),
        ),
        (
            "AASCE fusion, zero-shot",
            metric_at(
                "experiments/reference/aasce_fusion/ascee_aasce2019_fusion_v3_locked_metrics.json",
                "metrics",
                "fusion_v3_locked",
            ),
        ),
        (
            "AASCE GT-landmark geometry audit",
            metric_at(
                "experiments/reference/aasce_fusion/ascee_aasce2019_fusion_v3_locked_metrics.json",
                "metrics",
                "gt_landmark_geometry_audit",
            ),
        ),
    ]
    print("Reference aggregate metrics")
    print("-" * 92)
    print(f"{'Evaluation':42} {'N':>9} {'MAE':>10} {'SMAPE':>10} {'within5':>10}")
    print("-" * 92)
    for label, metrics in rows:
        print(
            f"{label:42} "
            f"{format_n(metrics):>9} "
            f"{metrics['mae_deg']:>9.4f}d "
            f"{metrics['paper_smape_pct']:>9.4f}% "
            f"{metrics['within_5deg_rate'] * 100:>9.2f}%"
        )


def aasce_fusion(args: argparse.Namespace) -> None:
    run(
        [
            PYTHON,
            script("119_eval_ascee_aasce2019_fusion_v3_locked.py"),
            "--manifest-path",
            "external_datasets/ascee_aasce2019/processed/ascee_aasce2019_manifest.jsonl",
            "--output-dir",
            "outputs/aasce_fusion",
            "--report-path",
            "outputs/reports/aasce_fusion.md",
            *common_window(args),
            *maybe_overlays(args),
        ]
    )


def all_smoke(args: argparse.Namespace) -> None:
    smoke = argparse.Namespace(
        start_index=args.start_index,
        num_images=args.num_images,
        progress_every=1,
        save_overlays=False,
        overlay_limit=2,
    )
    subset5_diogo(smoke)
    subset5_daniel(smoke)
    subset5_fusion(smoke)
    aasce_fusion(smoke)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SPINAL-AI2024 model and fusion evaluations."
    )
    parser.add_argument(
        "command",
        choices=(
            "subset5-diogo",
            "subset5-daniel",
            "subset5-fusion",
            "metrics-summary",
            "aasce-fusion",
            "all-smoke",
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First sample index. Use 0 for the official start.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of images to evaluate. Use 0 for the full dataset/split.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dispatch = {
        "subset5-diogo": subset5_diogo,
        "subset5-daniel": subset5_daniel,
        "subset5-fusion": subset5_fusion,
        "metrics-summary": metrics_summary,
        "aasce-fusion": aasce_fusion,
        "all-smoke": all_smoke,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
