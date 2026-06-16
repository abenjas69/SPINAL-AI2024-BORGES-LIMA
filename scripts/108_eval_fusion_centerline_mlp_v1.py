"""Evaluate simple fusion rules between the locked MLP v2 and centerline model.

The script is deliberately offline: it reads prediction CSV files already
produced by the locked Phase 9 MLP v2 evaluation and by script 107, joins them
by file_name, and evaluates fixed, non-trained fusion rules.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLP_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "final_test_subset5_mlp_v2_locked"
    / "final_test_subset5_mlp_v2_predictions.csv"
)
DEFAULT_CENTERLINE_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "fusion_centerline_model_v1_full_test"
    / "colleague_centerline_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "fusion_centerline_mlp_v1"
EPS = 1.0e-8


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def jsonable_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metric_bundle(pred_values: Sequence[float], gt_values: Sequence[float]) -> dict[str, Any]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[mask]
    gt = gt[mask]
    if pred.size == 0:
        return {"num_images": 0}

    errors = pred - gt
    abs_errors = np.abs(errors)
    denominator = np.abs(pred) + np.abs(gt)
    smape_terms = np.divide(
        abs_errors,
        denominator,
        out=np.zeros_like(abs_errors, dtype=np.float32),
        where=denominator > EPS,
    )

    result: dict[str, Any] = {
        "num_images": int(pred.size),
        "mae_deg": float(np.mean(abs_errors)),
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "bias_deg": float(np.mean(errors)),
        "median_abs_error_deg": float(np.median(abs_errors)),
        "p90_abs_error_deg": float(np.percentile(abs_errors, 90)),
        "p95_abs_error_deg": float(np.percentile(abs_errors, 95)),
        "p99_abs_error_deg": float(np.percentile(abs_errors, 99)),
        "max_abs_error_deg": float(np.max(abs_errors)),
        "within_3deg_rate": float(np.mean(abs_errors <= 3.0)),
        "within_5deg_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10deg_rate": float(np.mean(abs_errors <= 10.0)),
        "failures_gt3": int(np.sum(abs_errors > 3.0)),
        "failures_gt5": int(np.sum(abs_errors > 5.0)),
        "failures_gt10": int(np.sum(abs_errors > 10.0)),
        "paper_smape_pct": float(100.0 * np.mean(smape_terms)),
        "standard_2x_smape_pct_not_for_curvnet": float(200.0 * np.mean(smape_terms)),
        "smape_term_median_pct": float(100.0 * np.median(smape_terms)),
        "smape_term_p90_pct": float(100.0 * np.percentile(smape_terms, 90)),
        "mean_pred_deg": float(np.mean(pred)),
        "mean_gt_deg": float(np.mean(gt)),
    }
    if pred.size >= 2 and np.ptp(pred) > 1.0e-6 and np.ptp(gt) > 1.0e-6:
        result["pearson"] = float(np.corrcoef(gt, pred)[0, 1])
    else:
        result["pearson"] = None
    total = float(np.sum((gt - np.mean(gt)) ** 2))
    result["r2"] = float(1.0 - float(np.sum((gt - pred) ** 2)) / total) if total > EPS else None
    return result


def metric_row(metrics: Mapping[str, Any], label: str) -> str:
    if int(metrics.get("num_images", 0)) == 0:
        return f"| {label} | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |"
    return (
        "| "
        f"{label} | "
        f"{int(metrics['num_images'])} | "
        f"{float(metrics['mae_deg']):.3f} | "
        f"{float(metrics['paper_smape_pct']):.4f}% | "
        f"{float(metrics['within_3deg_rate']):.3f} | "
        f"{float(metrics['within_5deg_rate']):.3f} | "
        f"{float(metrics['within_10deg_rate']):.3f} | "
        f"{int(metrics['failures_gt5'])} | "
        f"{int(metrics['failures_gt10'])} | "
        f"{float(metrics['rmse_deg']):.3f} | "
        f"{float(metrics['p90_abs_error_deg']):.3f} | "
        f"{float(metrics['bias_deg']):.3f} |"
    )


def index_by_file_name(rows: Sequence[Mapping[str, str]], label: str) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            raise ValueError(f"Linha sem file_name em {label}")
        if file_name in indexed:
            duplicates.append(file_name)
        indexed[file_name] = row
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"{label} contem file_name duplicados: {preview}")
    return indexed


def fuse_row(
    file_name: str,
    mlp_row: Mapping[str, str],
    centerline_row: Mapping[str, str],
    *,
    gate_diff_deg: float,
    gate_min_centerline_points: int,
) -> dict[str, Any]:
    gt = finite_float(mlp_row.get("gt_cobb_max"))
    centerline_gt = finite_float(centerline_row.get("gt_max_cobb"))
    if np.isfinite(gt) and np.isfinite(centerline_gt) and abs(gt - centerline_gt) > 1.0e-3:
        raise ValueError(
            f"GT desalinhado para {file_name}: mlp={gt}, centerline={centerline_gt}"
        )

    mlp_pred = finite_float(mlp_row.get("calibrated_cobb"))
    geom_pred = finite_float(mlp_row.get("geom_cobb"))
    centerline_pred = finite_float(centerline_row.get("pred_max_cobb"))
    centerline_points = int(finite_float(centerline_row.get("centerline_points"), 0.0))
    centerline_status = str(centerline_row.get("status", ""))
    delta_models = (
        abs(mlp_pred - centerline_pred)
        if np.isfinite(mlp_pred) and np.isfinite(centerline_pred)
        else np.nan
    )
    centerline_usable = (
        np.isfinite(centerline_pred)
        and centerline_status == "ok"
        and centerline_points >= gate_min_centerline_points
    )
    safe_gate_uses_centerline = bool(centerline_usable and np.isfinite(delta_models) and delta_models <= gate_diff_deg)

    predictions = {
        "mlp_only": mlp_pred,
        "centerline_only": centerline_pred,
        "mean_50_50": 0.5 * mlp_pred + 0.5 * centerline_pred
        if np.isfinite(mlp_pred) and np.isfinite(centerline_pred)
        else mlp_pred,
        "weighted_75_25": 0.75 * mlp_pred + 0.25 * centerline_pred
        if np.isfinite(mlp_pred) and np.isfinite(centerline_pred)
        else mlp_pred,
        "safe_gate": 0.5 * mlp_pred + 0.5 * centerline_pred if safe_gate_uses_centerline else mlp_pred,
    }

    row: dict[str, Any] = {
        "file_name": file_name,
        "gt_cobb_max": gt,
        "mlp_calibrated_cobb": mlp_pred,
        "mlp_geom_cobb": geom_pred,
        "centerline_pred_max_cobb": centerline_pred,
        "centerline_status": centerline_status,
        "centerline_points": centerline_points,
        "centerline_mask_pixels": centerline_row.get("mask_pixels_gt_threshold", ""),
        "model_delta_abs": delta_models,
        "safe_gate_uses_centerline": int(safe_gate_uses_centerline),
        "safe_gate_reason": "blend" if safe_gate_uses_centerline else "mlp_fallback",
    }
    for method, pred in predictions.items():
        row[f"pred_{method}"] = pred
        row[f"abs_error_{method}"] = abs(pred - gt) if np.isfinite(pred) and np.isfinite(gt) else np.nan
    return row


def build_metrics(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    gt_values = [finite_float(row["gt_cobb_max"]) for row in rows]
    for method in methods:
        metrics[method] = metric_bundle(
            [finite_float(row[f"pred_{method}"]) for row in rows],
            gt_values,
        )
    return metrics


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    metrics: Mapping[str, Mapping[str, Any]],
    common_count: int,
    mlp_only_count: int,
    centerline_only_count: int,
    missing_centerline: Sequence[str],
    missing_mlp: Sequence[str],
    safe_gate_blend_count: int,
) -> None:
    lines = [
        "# Fusion Centerline + MLP v1",
        "",
        "## Setup",
        "",
        f"- mlp predictions: `{resolve_project_path(args.mlp_predictions)}`",
        f"- centerline predictions: `{resolve_project_path(args.centerline_predictions)}`",
        f"- common rows: `{common_count}`",
        f"- mlp-only rows: `{mlp_only_count}`",
        f"- centerline-only rows: `{centerline_only_count}`",
        f"- missing centerline rows: `{len(missing_centerline)}`",
        f"- missing mlp rows: `{len(missing_mlp)}`",
        f"- safe-gate diff threshold: `{args.gate_diff_deg}` deg",
        f"- safe-gate min centerline points: `{args.gate_min_centerline_points}`",
        f"- safe-gate blended rows: `{safe_gate_blend_count}`",
        "",
        "## Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("mlp_only", "centerline_only", "mean_50_50", "weighted_75_25", "safe_gate"):
        lines.append(metric_row(metrics[method], method))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- predictions CSV: `{path.parent / 'fusion_centerline_mlp_predictions.csv'}`",
            f"- metrics JSON: `{path.parent / 'fusion_centerline_mlp_metrics.json'}`",
            "",
            "## Notes",
            "",
            "- Fusion rules are fixed and do not learn or tune weights from this evaluation set.",
            "- `safe_gate` only blends with centerline when both models agree within the fixed threshold.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia fusoes offline entre MLP v2 e centerline.")
    parser.add_argument("--mlp-predictions", default=str(DEFAULT_MLP_PREDICTIONS))
    parser.add_argument("--centerline-predictions", default=str(DEFAULT_CENTERLINE_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gate-diff-deg", type=float, default=3.0)
    parser.add_argument("--gate-min-centerline-points", type=int, default=20)
    parser.add_argument(
        "--require-same-files",
        action="store_true",
        help="Falha se os CSVs nao tiverem exatamente o mesmo conjunto de file_name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlp_path = resolve_project_path(args.mlp_predictions)
    centerline_path = resolve_project_path(args.centerline_predictions)
    output_dir = resolve_project_path(args.output_dir)
    if args.gate_diff_deg < 0:
        raise ValueError("--gate-diff-deg deve ser >= 0")
    if args.gate_min_centerline_points < 0:
        raise ValueError("--gate-min-centerline-points deve ser >= 0")

    mlp_rows = read_csv(mlp_path)
    centerline_rows = read_csv(centerline_path)
    mlp_by_file = index_by_file_name(mlp_rows, "mlp")
    centerline_by_file = index_by_file_name(centerline_rows, "centerline")
    mlp_files = set(mlp_by_file)
    centerline_files = set(centerline_by_file)
    common_files = sorted(mlp_files & centerline_files)
    missing_centerline = sorted(mlp_files - centerline_files)
    missing_mlp = sorted(centerline_files - mlp_files)

    if not common_files:
        raise ValueError("Nao existem file_name comuns entre MLP e centerline.")
    if args.require_same_files and (missing_centerline or missing_mlp):
        raise ValueError(
            "CSVs desalinhados: "
            f"missing_centerline={len(missing_centerline)}, missing_mlp={len(missing_mlp)}"
        )

    rows = [
        fuse_row(
            file_name,
            mlp_by_file[file_name],
            centerline_by_file[file_name],
            gate_diff_deg=args.gate_diff_deg,
            gate_min_centerline_points=args.gate_min_centerline_points,
        )
        for file_name in common_files
    ]
    methods = ("mlp_only", "centerline_only", "mean_50_50", "weighted_75_25", "safe_gate")
    metrics = build_metrics(rows, methods)
    safe_gate_blend_count = int(sum(int(row["safe_gate_uses_centerline"]) for row in rows))

    metrics_payload = {
        "phase": "fusion_centerline_mlp_v1",
        "mlp_predictions": str(mlp_path),
        "centerline_predictions": str(centerline_path),
        "common_rows": len(common_files),
        "mlp_rows": len(mlp_rows),
        "centerline_rows": len(centerline_rows),
        "missing_centerline_count": len(missing_centerline),
        "missing_mlp_count": len(missing_mlp),
        "missing_centerline_preview": missing_centerline[:20],
        "missing_mlp_preview": missing_mlp[:20],
        "gate_diff_deg": float(args.gate_diff_deg),
        "gate_min_centerline_points": int(args.gate_min_centerline_points),
        "safe_gate_blend_count": safe_gate_blend_count,
        "metrics": {
            key: {metric_key: jsonable_metric_value(metric_value) for metric_key, metric_value in value.items()}
            for key, value in metrics.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "fusion_centerline_mlp_predictions.csv"
    metrics_path = output_dir / "fusion_centerline_mlp_metrics.json"
    report_path = output_dir / "fusion_centerline_mlp_report.md"
    fieldnames = [
        "file_name",
        "gt_cobb_max",
        "mlp_calibrated_cobb",
        "mlp_geom_cobb",
        "centerline_pred_max_cobb",
        "centerline_status",
        "centerline_points",
        "centerline_mask_pixels",
        "model_delta_abs",
        "safe_gate_uses_centerline",
        "safe_gate_reason",
    ]
    for method in methods:
        fieldnames.extend([f"pred_{method}", f"abs_error_{method}"])
    write_csv(prediction_path, rows, fieldnames)
    write_json(metrics_path, metrics_payload)
    write_report(
        report_path,
        args=args,
        metrics=metrics,
        common_count=len(common_files),
        mlp_only_count=len(mlp_rows),
        centerline_only_count=len(centerline_rows),
        missing_centerline=missing_centerline,
        missing_mlp=missing_mlp,
        safe_gate_blend_count=safe_gate_blend_count,
    )

    print("\n===== FUSION CENTERLINE + MLP V1 =====")
    print(f"MLP rows: {len(mlp_rows)}")
    print(f"Centerline rows: {len(centerline_rows)}")
    print(f"Common rows: {len(common_files)}")
    print(f"Missing centerline: {len(missing_centerline)}")
    print(f"Missing MLP: {len(missing_mlp)}")
    print(f"Safe-gate blended rows: {safe_gate_blend_count}")
    print("\n===== METRICS =====")
    for method in methods:
        row = metrics[method]
        print(
            f"{method}: MAE={row['mae_deg']:.3f}, "
            f"SMAPE={row['paper_smape_pct']:.4f}%, "
            f"within5={row['within_5deg_rate']:.4f}, "
            f"failures_gt5={row['failures_gt5']}"
        )
    print(f"\nPredictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
