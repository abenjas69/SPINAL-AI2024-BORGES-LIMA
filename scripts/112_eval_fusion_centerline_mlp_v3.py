"""Tune/apply locked V3 fusion between MLP v2 and clean centerline V3.

Use without --lock-input on the holdout split to fit only:
- centerline additive bias correction;
- centerline blend weight.

Use with --lock-input on the test split. In that mode no parameter is fitted
from the input CSVs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_HOLDOUT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "fusion_centerline_mlp_v3_holdout3192"
DEFAULT_MLP_PREDICTIONS = (
    PROJECT_ROOT / "experiments" / "fusion_centerline_model_v3_phase5_roi_holdout3192" / "mlp_v2_predictions.csv"
)
DEFAULT_CENTERLINE_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "fusion_centerline_model_v3_phase5_roi_holdout3192"
    / "colleague_centerline_predictions.csv"
)
EPS = 1.0e-8
METHODS = ("mlp_only", "centerline_only", "centerline_bias_corrected", "fusion_v3_locked")


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fusion_v1 = import_script("108_eval_fusion_centerline_mlp_v1.py", "fusion_v1_utils_for_v3")


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
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def validate_weight(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("centerline weight deve estar entre 0 e 1")
    return value


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


def load_joined_inputs(
    *,
    mlp_predictions: Path,
    centerline_predictions: Path,
    require_same_files: bool,
) -> tuple[list[tuple[str, Mapping[str, str], Mapping[str, str]]], dict[str, Any]]:
    mlp_rows = read_csv(mlp_predictions)
    centerline_rows = read_csv(centerline_predictions)
    mlp_by_file = index_by_file_name(mlp_rows, "mlp")
    centerline_by_file = index_by_file_name(centerline_rows, "centerline")
    mlp_files = set(mlp_by_file)
    centerline_files = set(centerline_by_file)
    common_files = sorted(mlp_files & centerline_files)
    missing_centerline = sorted(mlp_files - centerline_files)
    missing_mlp = sorted(centerline_files - mlp_files)
    if not common_files:
        raise ValueError("Nao existem file_name comuns entre MLP e centerline.")
    if require_same_files and (missing_centerline or missing_mlp):
        raise ValueError(
            "CSVs desalinhados: "
            f"missing_centerline={len(missing_centerline)}, missing_mlp={len(missing_mlp)}"
        )
    joined = [(file_name, mlp_by_file[file_name], centerline_by_file[file_name]) for file_name in common_files]
    metadata = {
        "common_rows": len(common_files),
        "mlp_rows": len(mlp_rows),
        "centerline_rows": len(centerline_rows),
        "missing_centerline_rows": len(missing_centerline),
        "missing_mlp_rows": len(missing_mlp),
    }
    return joined, metadata


def fit_centerline_bias(joined_rows: Sequence[tuple[str, Mapping[str, str], Mapping[str, str]]]) -> float:
    corrections: list[float] = []
    for _file_name, mlp_row, centerline_row in joined_rows:
        gt = finite_float(mlp_row.get("gt_cobb_max"))
        centerline_gt = finite_float(centerline_row.get("gt_max_cobb"))
        if np.isfinite(gt) and np.isfinite(centerline_gt) and abs(gt - centerline_gt) > 1.0e-3:
            raise ValueError(
                "GT desalinhado durante fit de bias: "
                f"mlp={gt}, centerline={centerline_gt}"
            )
        centerline_pred = finite_float(centerline_row.get("pred_max_cobb"))
        if np.isfinite(gt) and np.isfinite(centerline_pred):
            corrections.append(gt - centerline_pred)
    if not corrections:
        raise ValueError("Sem linhas finitas para estimar bias centerline no holdout.")
    return float(np.mean(np.asarray(corrections, dtype=np.float32)))


def fuse_joined_rows(
    joined_rows: Sequence[tuple[str, Mapping[str, str], Mapping[str, str]]],
    *,
    centerline_bias_correction_deg: float,
    centerline_weight: float,
) -> list[dict[str, Any]]:
    weight = validate_weight(centerline_weight)
    rows: list[dict[str, Any]] = []
    for file_name, mlp_row, centerline_row in joined_rows:
        gt = finite_float(mlp_row.get("gt_cobb_max"))
        centerline_gt = finite_float(centerline_row.get("gt_max_cobb"))
        if np.isfinite(gt) and np.isfinite(centerline_gt) and abs(gt - centerline_gt) > 1.0e-3:
            raise ValueError(
                f"GT desalinhado para {file_name}: mlp={gt}, centerline={centerline_gt}"
            )

        mlp_pred = finite_float(mlp_row.get("calibrated_cobb"))
        geom_pred = finite_float(mlp_row.get("geom_cobb"))
        centerline_pred = finite_float(centerline_row.get("pred_max_cobb"))
        centerline_corrected = (
            centerline_pred + centerline_bias_correction_deg
            if np.isfinite(centerline_pred)
            else np.nan
        )
        fusion_pred = (
            (1.0 - weight) * mlp_pred + weight * centerline_corrected
            if np.isfinite(mlp_pred) and np.isfinite(centerline_corrected)
            else mlp_pred
        )
        predictions = {
            "mlp_only": mlp_pred,
            "centerline_only": centerline_pred,
            "centerline_bias_corrected": centerline_corrected,
            "fusion_v3_locked": fusion_pred,
        }

        mlp_error = abs(mlp_pred - gt) if np.isfinite(mlp_pred) and np.isfinite(gt) else np.nan
        fusion_error = abs(fusion_pred - gt) if np.isfinite(fusion_pred) and np.isfinite(gt) else np.nan
        row: dict[str, Any] = {
            "file_name": file_name,
            "gt_cobb_max": gt,
            "mlp_calibrated_cobb": mlp_pred,
            "mlp_geom_cobb": geom_pred,
            "centerline_pred_max_cobb": centerline_pred,
            "centerline_bias_corrected": centerline_corrected,
            "fusion_v3_locked": fusion_pred,
            "centerline_status": centerline_row.get("status", ""),
            "centerline_roi_source": centerline_row.get("roi_source", ""),
            "centerline_points": centerline_row.get("centerline_points", ""),
            "centerline_mask_pixels": centerline_row.get("mask_pixels_gt_threshold", ""),
            "centerline_bias_correction_deg": centerline_bias_correction_deg,
            "centerline_weight": weight,
            "mlp_weight": 1.0 - weight,
            "model_delta_abs_after_biascorr": abs(mlp_pred - centerline_corrected)
            if np.isfinite(mlp_pred) and np.isfinite(centerline_corrected)
            else np.nan,
            "fusion_error_delta_vs_mlp": fusion_error - mlp_error
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else np.nan,
            "fusion_improved_vs_mlp": int(fusion_error < mlp_error)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
            "fusion_rescued_gt5": int(mlp_error > 5.0 and fusion_error <= 5.0)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
            "fusion_broken_gt5": int(mlp_error <= 5.0 and fusion_error > 5.0)
            if np.isfinite(fusion_error) and np.isfinite(mlp_error)
            else "",
        }
        for method, pred in predictions.items():
            row[f"pred_{method}"] = pred
            row[f"abs_error_{method}"] = abs(pred - gt) if np.isfinite(pred) and np.isfinite(gt) else np.nan
        rows.append(row)
    return rows


def build_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    gt_values = [finite_float(row["gt_cobb_max"]) for row in rows]
    return {
        method: fusion_v1.metric_bundle(
            [finite_float(row[f"pred_{method}"]) for row in rows],
            gt_values,
        )
        for method in METHODS
    }


def objective_tuple(metrics: Mapping[str, Any], selection_objective: str) -> tuple[float, float, float]:
    if int(metrics.get("num_images", 0)) == 0:
        return (float("inf"), float("inf"), float("inf"))
    if selection_objective == "failures_gt5":
        return (
            float(metrics["failures_gt5"]),
            float(metrics["mae_deg"]),
            float(metrics["p90_abs_error_deg"]),
        )
    if selection_objective == "smape":
        return (
            float(metrics["paper_smape_pct"]),
            float(metrics["mae_deg"]),
            float(metrics["failures_gt5"]),
        )
    return (
        float(metrics["mae_deg"]),
        float(metrics["failures_gt5"]),
        float(metrics["p90_abs_error_deg"]),
    )


def tune_weight(
    joined_rows: Sequence[tuple[str, Mapping[str, str], Mapping[str, str]]],
    *,
    centerline_bias_correction_deg: float,
    weight_step: float,
    selection_objective: str,
) -> tuple[float, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if weight_step <= 0.0 or weight_step > 1.0:
        raise ValueError("--weight-step deve estar no intervalo (0, 1].")
    weights = np.arange(0.0, 1.0 + EPS, float(weight_step), dtype=np.float32)
    grid_rows: list[dict[str, Any]] = []
    best_weight = 0.0
    best_score = (float("inf"), float("inf"), float("inf"))
    best_metrics: dict[str, dict[str, Any]] = {}

    for raw_weight in weights:
        weight = float(min(max(raw_weight, 0.0), 1.0))
        rows = fuse_joined_rows(
            joined_rows,
            centerline_bias_correction_deg=centerline_bias_correction_deg,
            centerline_weight=weight,
        )
        metrics = build_metrics(rows)
        fusion_metrics = metrics["fusion_v3_locked"]
        score = objective_tuple(fusion_metrics, selection_objective)
        grid_rows.append(
            {
                "centerline_weight": weight,
                "mlp_weight": 1.0 - weight,
                "mae_deg": fusion_metrics.get("mae_deg", np.nan),
                "paper_smape_pct": fusion_metrics.get("paper_smape_pct", np.nan),
                "within_5deg_rate": fusion_metrics.get("within_5deg_rate", np.nan),
                "failures_gt5": fusion_metrics.get("failures_gt5", ""),
                "failures_gt10": fusion_metrics.get("failures_gt10", ""),
                "rmse_deg": fusion_metrics.get("rmse_deg", np.nan),
                "p90_abs_error_deg": fusion_metrics.get("p90_abs_error_deg", np.nan),
                "bias_deg": fusion_metrics.get("bias_deg", np.nan),
            }
        )
        if score < best_score:
            best_score = score
            best_weight = weight
            best_metrics = metrics

    return best_weight, grid_rows, best_metrics


def load_lock(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "centerline_bias_correction_deg" not in payload or "centerline_weight" not in payload:
        raise ValueError(f"Lock invalido: {path}")
    payload["centerline_bias_correction_deg"] = float(payload["centerline_bias_correction_deg"])
    payload["centerline_weight"] = validate_weight(float(payload["centerline_weight"]))
    return payload


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    metrics: Mapping[str, Mapping[str, Any]],
    input_metadata: Mapping[str, Any],
    lock_payload: Mapping[str, Any],
    improved_count: int,
    rescued_count: int,
    broken_count: int,
) -> None:
    lock_mode = "apply-locked" if args.lock_input else "tune-holdout"
    lines = [
        "# Fusion Centerline + MLP V3",
        "",
        "## Setup",
        "",
        f"- mode: `{lock_mode}`",
        f"- mlp predictions: `{resolve_project_path(args.mlp_predictions)}`",
        f"- centerline predictions: `{resolve_project_path(args.centerline_predictions)}`",
        f"- common rows: `{input_metadata['common_rows']}`",
        f"- mlp rows: `{input_metadata['mlp_rows']}`",
        f"- centerline rows: `{input_metadata['centerline_rows']}`",
        f"- missing centerline rows: `{input_metadata['missing_centerline_rows']}`",
        f"- missing mlp rows: `{input_metadata['missing_mlp_rows']}`",
        f"- centerline bias correction: `{float(lock_payload['centerline_bias_correction_deg']):.6f}` deg",
        f"- centerline weight: `{float(lock_payload['centerline_weight']):.4f}`",
        f"- mlp weight: `{1.0 - float(lock_payload['centerline_weight']):.4f}`",
        f"- selection objective: `{lock_payload.get('selection_objective', 'locked')}`",
        f"- lock source: `{lock_payload.get('selection_source', '')}`",
        f"- fusion improved rows: `{improved_count}`",
        f"- fusion rescued >5: `{rescued_count}`",
        f"- fusion broken >5: `{broken_count}`",
        "",
        "## Metrics",
        "",
        "| method | N | MAE | SMAPE CurvNet | within3 | within5 | within10 | failures >5 | failures >10 | RMSE | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(fusion_v1.metric_row(metrics[method], method))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- predictions CSV: `{path.parent / 'fusion_centerline_mlp_v3_predictions.csv'}`",
            f"- metrics JSON: `{path.parent / 'fusion_centerline_mlp_v3_metrics.json'}`",
            f"- lock JSON: `{path.parent / 'fusion_centerline_mlp_v3_lock.json'}`",
            "",
            "## Notes",
            "",
            "- In apply-locked mode, this script does not fit bias or sweep weights on the input CSVs.",
            "- For the final test set, use the lock JSON created on holdout.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Afina/aplica a fusao V3 entre MLP v2 e centerline com ROI Phase 5."
    )
    parser.add_argument("--mlp-predictions", default=str(DEFAULT_MLP_PREDICTIONS))
    parser.add_argument("--centerline-predictions", default=str(DEFAULT_CENTERLINE_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_HOLDOUT_OUTPUT_DIR))
    parser.add_argument(
        "--lock-input",
        default="",
        help="JSON de lock criado no holdout. Se definido, aplica sem tuning.",
    )
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument(
        "--selection-objective",
        choices=("mae", "failures_gt5", "smape"),
        default="mae",
        help="Objetivo usado apenas quando --lock-input nao e fornecido.",
    )
    parser.add_argument("--require-same-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlp_path = resolve_project_path(args.mlp_predictions)
    centerline_path = resolve_project_path(args.centerline_predictions)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joined_rows, input_metadata = load_joined_inputs(
        mlp_predictions=mlp_path,
        centerline_predictions=centerline_path,
        require_same_files=bool(args.require_same_files),
    )

    tuning_grid: list[dict[str, Any]] = []
    if args.lock_input:
        lock_payload = load_lock(resolve_project_path(args.lock_input))
        lock_payload = {
            **lock_payload,
            "applied_lock_input": str(resolve_project_path(args.lock_input)),
            "selection_source": lock_payload.get("selection_source", "holdout"),
        }
    else:
        bias = fit_centerline_bias(joined_rows)
        weight, tuning_grid, _best_metrics = tune_weight(
            joined_rows,
            centerline_bias_correction_deg=bias,
            weight_step=float(args.weight_step),
            selection_objective=str(args.selection_objective),
        )
        lock_payload = {
            "phase": "fusion_centerline_mlp_v3_lock",
            "selection_source": "holdout",
            "selection_objective": str(args.selection_objective),
            "weight_step": float(args.weight_step),
            "centerline_bias_correction_deg": float(bias),
            "centerline_weight": float(weight),
            "mlp_weight": float(1.0 - weight),
            "mlp_predictions": str(mlp_path),
            "centerline_predictions": str(centerline_path),
            "common_rows": int(input_metadata["common_rows"]),
        }

    rows = fuse_joined_rows(
        joined_rows,
        centerline_bias_correction_deg=float(lock_payload["centerline_bias_correction_deg"]),
        centerline_weight=float(lock_payload["centerline_weight"]),
    )
    metrics = build_metrics(rows)
    improved_count = int(sum(int(row["fusion_improved_vs_mlp"]) for row in rows if row["fusion_improved_vs_mlp"] != ""))
    rescued_count = int(sum(int(row["fusion_rescued_gt5"]) for row in rows if row["fusion_rescued_gt5"] != ""))
    broken_count = int(sum(int(row["fusion_broken_gt5"]) for row in rows if row["fusion_broken_gt5"] != ""))

    prediction_path = output_dir / "fusion_centerline_mlp_v3_predictions.csv"
    metrics_path = output_dir / "fusion_centerline_mlp_v3_metrics.json"
    lock_path = output_dir / "fusion_centerline_mlp_v3_lock.json"
    grid_path = output_dir / "fusion_centerline_mlp_v3_weight_grid.csv"
    report_path = output_dir / "fusion_centerline_mlp_v3_report.md"

    if rows:
        write_csv(prediction_path, rows, list(rows[0].keys()))
    if tuning_grid:
        write_csv(grid_path, tuning_grid, list(tuning_grid[0].keys()))
    write_json(lock_path, lock_payload)
    payload = {
        "phase": "fusion_centerline_mlp_v3",
        "mode": "apply_locked" if args.lock_input else "tune_holdout",
        "config": vars(args),
        "inputs": input_metadata,
        "lock": lock_payload,
        "improved_count": improved_count,
        "rescued_gt5": rescued_count,
        "broken_gt5": broken_count,
        "metrics": metrics,
        "tuning_grid": tuning_grid,
    }
    write_json(metrics_path, payload)
    write_report(
        report_path,
        args=args,
        metrics=metrics,
        input_metadata=input_metadata,
        lock_payload=lock_payload,
        improved_count=improved_count,
        rescued_count=rescued_count,
        broken_count=broken_count,
    )

    print("\n===== FUSION V3 =====")
    print(f"mode: {'apply-locked' if args.lock_input else 'tune-holdout'}")
    print(f"common rows: {input_metadata['common_rows']}")
    print(f"centerline bias correction: {float(lock_payload['centerline_bias_correction_deg']):.6f}")
    print(f"centerline weight: {float(lock_payload['centerline_weight']):.4f}")
    for method in METHODS:
        row = metrics[method]
        print(
            f"{method}: MAE={row['mae_deg']:.3f}, "
            f"SMAPE={row['paper_smape_pct']:.4f}%, "
            f"within5={row['within_5deg_rate']:.4f}, "
            f"falhas>5={row['failures_gt5']}, "
            f"falhas>10={row['failures_gt10']}"
        )
    print(f"improved rows: {improved_count}")
    print(f"rescued >5: {rescued_count}")
    print(f"broken >5: {broken_count}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Lock: {lock_path}")
    print(f"Report: {report_path}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
