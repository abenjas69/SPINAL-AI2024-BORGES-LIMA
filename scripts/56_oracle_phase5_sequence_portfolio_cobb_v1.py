"""Oracle Cobb-aware para portefolio de sequencias da Fase 5.

Este script mede o teto de um futuro reranker: para cada imagem, gera varias
sequencias Fase 5 candidatas, calcula o Cobb geometrico de cada uma e reporta
qual seria o erro se um seletor perfeito escolhesse a melhor sequencia.

O oracle usa ground truth apenas para auditoria. Nao e um modo de inferencia.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
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
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "sanity_check" / "reports"

DEFAULT_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_radius_hardmining_v1.keras"
DEFAULT_EXPERIMENT_NAME = "phase5_sequence_portfolio_cobb_oracle_v1"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / DEFAULT_EXPERIMENT_NAME
DEFAULT_REPORT = REPORTS_DIR / f"{DEFAULT_EXPERIMENT_NAME}.md"
DEFAULT_PROFILES = [
    "anatomical_endpoint_safe_v1",
    "anatomical_endpoint_recovery_fill_inner_safe_v2",
    "anatomical_endpoint_recovery_fill_strict_v2",
    "anatomical_endpoint_recovery_fill_conservative_v2",
    "anatomical_endpoint_recovery_balanced_v1",
    "anatomical_endpoint_fallback_moderate_v1",
]


def import_script(file_name: str, module_name: str) -> ModuleType:
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finite_metrics(pred_values: Sequence[float], gt_values: Sequence[float]) -> dict[str, Any]:
    pred = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[mask]
    gt = gt[mask]
    if pred.size == 0:
        return {"num_images": 0}
    errors = pred - gt
    abs_errors = np.abs(errors)
    result: dict[str, Any] = {
        "num_images": int(pred.size),
        "mae_deg": float(np.mean(abs_errors)),
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "bias_deg": float(np.mean(errors)),
        "median_abs_error_deg": float(np.median(abs_errors)),
        "p90_abs_error_deg": float(np.percentile(abs_errors, 90)),
        "within_3deg_rate": float(np.mean(abs_errors <= 3.0)),
        "within_5deg_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10deg_rate": float(np.mean(abs_errors <= 10.0)),
        "failures_gt5": int(np.sum(abs_errors > 5.0)),
        "mean_pred_deg": float(np.mean(pred)),
        "mean_gt_deg": float(np.mean(gt)),
    }
    if pred.size >= 2 and np.ptp(pred) > 1.0e-6 and np.ptp(gt) > 1.0e-6:
        result["pearson"] = float(np.corrcoef(gt, pred)[0, 1])
    else:
        result["pearson"] = None
    return result


def count_summary(values: Sequence[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(values).most_common()}


def cobb_max_from_targets(targets: Mapping[str, Any]) -> tuple[float, str]:
    angles = np.asarray(targets["cobb_angles"], dtype=np.float32).reshape(-1)
    if angles.size < 3:
        raise ValueError("cobb_angles deve conter PT, MT e TLL.")
    max_index = int(np.argmax(angles[:3]))
    names = ("PT", "MT", "TLL")
    return float(angles[max_index]), names[max_index]


def geometric_cobb_for_result(
    *,
    phase9_v1: ModuleType,
    result: Mapping[str, Any],
    width: float,
    height: float,
) -> dict[str, Any]:
    geom = phase9_v1.geometric_final_cobb(
        np.asarray(result["selected_points"], dtype=np.float32),
        width=float(width),
        height=float(height),
    )
    cobb_value = geom.get("cobb_deg")
    if cobb_value is None:
        geom["cobb_deg"] = np.nan
    return geom


def endpoint_pair_oracle_for_points(
    *,
    phase9_v1: ModuleType,
    points: np.ndarray,
    width: float,
    height: float,
    target_cobb: float,
) -> dict[str, Any]:
    valid_points = np.asarray(points, dtype=np.float32).reshape(-1, 8)
    count = int(valid_points.shape[0])
    if count < 2:
        return {
            "cobb_deg": np.nan,
            "abs_error": np.nan,
            "upper_index": -1,
            "lower_index": -1,
            "span": 0,
        }

    top_angles: list[float] = []
    bottom_angles: list[float] = []
    for row in valid_points:
        upper_left = row[0:2]
        upper_right = row[2:4]
        lower_left = row[4:6]
        lower_right = row[6:8]
        top_angles.append(phase9_v1.line_angle_deg(upper_left, upper_right, width, height))
        bottom_angles.append(phase9_v1.line_angle_deg(lower_left, lower_right, width, height))

    best: dict[str, Any] | None = None
    best_error = float("inf")
    for upper_index in range(count - 1):
        for lower_index in range(upper_index + 1, count):
            candidate_cobb = float(
                phase9_v1.angle_diff_deg(
                    top_angles[upper_index],
                    bottom_angles[lower_index],
                )
            )
            candidate_error = abs(candidate_cobb - float(target_cobb))
            if candidate_error < best_error:
                best_error = candidate_error
                best = {
                    "cobb_deg": candidate_cobb,
                    "abs_error": float(candidate_error),
                    "upper_index": int(upper_index),
                    "lower_index": int(lower_index),
                    "span": int(lower_index - upper_index),
                }
    if best is None:
        raise ValueError("Nao foi possivel gerar pares de endpoints.")
    return best


def make_profile_row(
    *,
    profile: str,
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    geom: Mapping[str, Any],
    pair_oracle: Mapping[str, Any],
    gt_cobb_max: float,
    gt_major_region: str,
) -> dict[str, Any]:
    pred_cobb = float(geom["cobb_deg"]) if np.isfinite(float(geom["cobb_deg"])) else np.nan
    abs_error = abs(pred_cobb - float(gt_cobb_max)) if np.isfinite(pred_cobb) else np.nan
    pair_oracle_cobb = float(pair_oracle["cobb_deg"])
    pair_oracle_abs_error = float(pair_oracle["abs_error"])
    return {
        "profile": profile,
        "file_name": str(sample["file_name"]),
        "gt_cobb_max": float(gt_cobb_max),
        "gt_major_region": gt_major_region,
        "pred_cobb": pred_cobb,
        "signed_error": pred_cobb - float(gt_cobb_max) if np.isfinite(pred_cobb) else np.nan,
        "abs_error": abs_error,
        "within5": int(np.isfinite(abs_error) and abs_error <= 5.0),
        "geom_upper_index": int(geom.get("upper_index", -1)),
        "geom_lower_index": int(geom.get("lower_index", -1)),
        "geom_span": int(geom.get("span", 0)),
        "pair_oracle_cobb": pair_oracle_cobb,
        "pair_oracle_abs_error": pair_oracle_abs_error,
        "pair_oracle_within5": int(np.isfinite(pair_oracle_abs_error) and pair_oracle_abs_error <= 5.0),
        "pair_oracle_upper_index": int(pair_oracle.get("upper_index", -1)),
        "pair_oracle_lower_index": int(pair_oracle.get("lower_index", -1)),
        "pair_oracle_span": int(pair_oracle.get("span", 0)),
        "final_count": int(record["final_count"]),
        "count_error": int(record["count_error"]),
        "matched_count": int(record["matched_count"]),
        "missed_gt": int(record["missed_gt"]),
        "false_pred": int(record["false_pred"]),
        "false_extreme": int(record["false_extreme"]),
        "false_inside": int(record["false_inside"]),
        "missed_inside": int(record["missed_inside"]),
        "mean_final_score": float(record["mean_final_score"]),
        "endpoint_filled_top": int(record.get("endpoint_filled_top", 0)),
        "endpoint_filled_bottom": int(record.get("endpoint_filled_bottom", 0)),
        "gap_filled_count": int(record.get("gap_filled_count", 0)),
    }


def summarize_profile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_images": 0}
    metrics = finite_metrics(
        [float(row["pred_cobb"]) for row in rows],
        [float(row["gt_cobb_max"]) for row in rows],
    )
    metrics.update(
        {
            "pair_oracle": finite_metrics(
                [float(row["pair_oracle_cobb"]) for row in rows],
                [float(row["gt_cobb_max"]) for row in rows],
            ),
            "mean_abs_count_error": float(np.mean([abs(int(row["count_error"])) for row in rows])),
            "exact_count_accuracy": float(np.mean([int(row["count_error"]) == 0 for row in rows])),
            "mean_missed_gt": float(np.mean([int(row["missed_gt"]) for row in rows])),
            "mean_false_pred": float(np.mean([int(row["false_pred"]) for row in rows])),
            "mean_false_extreme": float(np.mean([int(row["false_extreme"]) for row in rows])),
            "mean_false_inside": float(np.mean([int(row["false_inside"]) for row in rows])),
            "mean_missed_inside": float(np.mean([int(row["missed_inside"]) for row in rows])),
        }
    )
    return metrics


def make_report(
    *,
    path: Path,
    args: argparse.Namespace,
    eval_metadata: Mapping[str, Any],
    profile_summaries: Mapping[str, Mapping[str, Any]],
    oracle_summary: Mapping[str, Any],
    experiment_dir: Path,
    output_dir: Path,
) -> None:
    baseline = profile_summaries[args.baseline_profile]
    lines = [
        "# Fase 5 - oracle Cobb-aware de portefolio",
        "",
        "## Objetivo",
        "",
        "Medir o teto de um reranker que escolhe entre varias sequencias Fase 5 candidatas por imagem.",
        "O oracle usa o Cobb real apenas para auditoria; nao e uma regra de inferencia.",
        "",
        "## Configuracao",
        "",
        f"- modelo: `{args.model_path}`",
        f"- modo de avaliacao: `{eval_metadata['eval_mode']}`",
        f"- split: `{eval_metadata['split']}`",
        f"- imagens: `{oracle_summary.get('num_images', 0)}`",
        f"- baseline: `{args.baseline_profile}`",
        f"- perfis: `{', '.join(args.profiles)}`",
        "",
        "## Resultados Cobb por perfil",
        "",
        "| perfil | MAE | within5 | falhas >5 | count MAE | missed | falsos extremos | falsos interiores |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, summary in profile_summaries.items():
        lines.append(
            "| "
            f"{profile} | "
            f"{summary.get('mae_deg', 0.0):.3f} | "
            f"{summary.get('within_5deg_rate', 0.0):.3f} | "
            f"{summary.get('failures_gt5', 0)} | "
            f"{summary.get('mean_abs_count_error', 0.0):.3f} | "
            f"{summary.get('mean_missed_gt', 0.0):.3f} | "
            f"{summary.get('mean_false_extreme', 0.0):.3f} | "
            f"{summary.get('mean_false_inside', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle de endpoints dentro da sequencia",
            "",
            "| perfil | endpoint oracle within5 | endpoint oracle MAE | endpoint oracle falhas >5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for profile, summary in profile_summaries.items():
        endpoint_oracle = summary.get("pair_oracle", {})
        lines.append(
            "| "
            f"{profile} | "
            f"{endpoint_oracle.get('within_5deg_rate', 0.0):.3f} | "
            f"{endpoint_oracle.get('mae_deg', 0.0):.3f} | "
            f"{endpoint_oracle.get('failures_gt5', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Oracle de portefolio",
            "",
            f"- baseline within5: `{baseline.get('within_5deg_rate', 0.0):.4f}`",
            f"- oracle within5: `{oracle_summary.get('within_5deg_rate', 0.0):.4f}`",
            f"- oracle endpoint+portefolio within5: `{oracle_summary.get('pair_oracle_within_5deg_rate', 0.0):.4f}`",
            f"- ganho within5: `{oracle_summary.get('delta_within5_vs_baseline', 0.0):+.4f}`",
            f"- baseline MAE: `{baseline.get('mae_deg', 0.0):.3f}`",
            f"- oracle MAE: `{oracle_summary.get('mae_deg', 0.0):.3f}`",
            f"- oracle endpoint+portefolio MAE: `{oracle_summary.get('pair_oracle_mae_deg', 0.0):.3f}`",
            f"- casos resgataveis pela escolha de perfil: `{oracle_summary.get('rescued_failures_vs_baseline', 0)}`",
            f"- casos resgataveis por endpoint+portefolio: `{oracle_summary.get('pair_oracle_rescued_failures_vs_baseline', 0)}`",
            "",
            "## Perfis escolhidos pelo oracle",
            "",
        ]
    )
    for profile, count in oracle_summary.get("selected_profile_counts", {}).items():
        lines.append(f"- `{profile}`: {count}")
    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            f"- resumo JSON: `{experiment_dir / 'phase5_sequence_portfolio_cobb_oracle_summary.json'}`",
            f"- CSV por perfil: `{experiment_dir / 'phase5_sequence_portfolio_cobb_profile_rows.csv'}`",
            f"- CSV oracle: `{experiment_dir / 'phase5_sequence_portfolio_cobb_oracle_rows.csv'}`",
            f"- outputs: `{output_dir}`",
            "",
            "## Interpretacao",
            "",
            "Se o oracle nao se aproximar de 90% within5, um reranker que apenas escolhe entre estes perfis nao chega a meta. Nesse caso, o proximo passo deve gerar mais sequencias candidatas ou alterar a deteccao/candidatos.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle Cobb-aware de portefolio de sequencias Fase 5.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--eval-mode", choices=("validation", "window"), default="validation")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--train-size", type=int, default=12768)
    parser.add_argument("--val-size", type=int, default=3192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--baseline-profile", default="anatomical_endpoint_safe_v1")
    parser.add_argument("--max-match-distance-px", type=float, default=32.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase5_sequence = import_script("51_eval_phase5_anatomical_sequence_v1.py", "phase5_sequence_eval")
    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")
    phase9_v1 = import_script("32_eval_phase9_final_cobb_v1.py", "phase9_v1")

    available_profiles = set(phase5_sequence.PROFILE_CONFIGS.keys())
    missing_profiles = [profile for profile in args.profiles if profile not in available_profiles]
    if missing_profiles:
        raise ValueError(f"Perfis desconhecidos: {missing_profiles}")
    if args.baseline_profile not in args.profiles:
        raise ValueError("--baseline-profile deve estar incluido em --profiles")

    model_path = resolve_project_path(args.model_path)
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"A carregar modelo Fase 5: {model_path}")
    model = phase5_eval.load_spatial_offset_model_for_eval(model_path)

    selected_samples, selected_paths, eval_metadata = phase5_sequence.select_eval_samples(
        tfdata=tfdata,
        phase2_train=phase2_train,
        eval_mode=args.eval_mode,
        split=args.split,
        train_size=args.train_size,
        val_size=args.val_size,
        start_index=args.start_index,
        num_images=args.num_images,
    )
    print(
        "A avaliar oracle Cobb-aware: "
        f"imagens={len(selected_samples)}, perfis={', '.join(args.profiles)}"
    )

    profile_rows: list[dict[str, Any]] = []
    rows_by_profile: dict[str, list[dict[str, Any]]] = {profile: [] for profile in args.profiles}
    oracle_rows: list[dict[str, Any]] = []

    for image_index, (sample, image_path) in enumerate(zip(selected_samples, selected_paths), start=1):
        targets = tfdata.sample_to_targets(sample)
        image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
        predictions = model(tf.expand_dims(image, axis=0), training=False)
        prediction_values = {
            "presence": predictions["presence"].numpy()[0],
            "bbox": predictions["bbox"].numpy()[0],
            "points": predictions["points"].numpy()[0],
        }
        gt_cobb_max, gt_major_region = cobb_max_from_targets(targets)
        width = float(sample.get("width", tf.shape(image)[1].numpy()))
        height = float(sample.get("height", tf.shape(image)[0].numpy()))

        image_rows: list[dict[str, Any]] = []
        for profile in args.profiles:
            result = postprocess.postprocess_candidates_sequence(
                presence=prediction_values["presence"],
                bbox=prediction_values["bbox"],
                points=prediction_values["points"],
                **phase5_sequence.PROFILE_CONFIGS[profile],
            )
            record = phase5_eval.build_evaluation_record(
                sample=sample,
                targets=targets,
                result=result,
                max_match_distance_px=args.max_match_distance_px,
                overlay_name="",
            )
            record["false_extreme"] = phase5_sequence.false_extreme(record)
            geom = geometric_cobb_for_result(
                phase9_v1=phase9_v1,
                result=result,
                width=width,
                height=height,
            )
            pair_oracle = endpoint_pair_oracle_for_points(
                phase9_v1=phase9_v1,
                points=np.asarray(result["selected_points"], dtype=np.float32),
                width=width,
                height=height,
                target_cobb=gt_cobb_max,
            )
            row = make_profile_row(
                profile=profile,
                sample=sample,
                record=record,
                geom=geom,
                pair_oracle=pair_oracle,
                gt_cobb_max=gt_cobb_max,
                gt_major_region=gt_major_region,
            )
            profile_rows.append(row)
            rows_by_profile[profile].append(row)
            image_rows.append(row)

        baseline_row = next(row for row in image_rows if row["profile"] == args.baseline_profile)
        best_row = min(image_rows, key=lambda row: float(row["abs_error"]))
        best_pair_row = min(image_rows, key=lambda row: float(row["pair_oracle_abs_error"]))
        oracle_rows.append(
            {
                "file_name": str(sample["file_name"]),
                "gt_cobb_max": float(gt_cobb_max),
                "gt_major_region": gt_major_region,
                "baseline_profile": args.baseline_profile,
                "baseline_cobb": float(baseline_row["pred_cobb"]),
                "baseline_abs_error": float(baseline_row["abs_error"]),
                "baseline_within5": int(baseline_row["within5"]),
                "oracle_profile": str(best_row["profile"]),
                "oracle_cobb": float(best_row["pred_cobb"]),
                "oracle_abs_error": float(best_row["abs_error"]),
                "oracle_within5": int(best_row["within5"]),
                "rescued_vs_baseline": int(
                    int(baseline_row["within5"]) == 0 and int(best_row["within5"]) == 1
                ),
                "pair_oracle_profile": str(best_pair_row["profile"]),
                "pair_oracle_cobb": float(best_pair_row["pair_oracle_cobb"]),
                "pair_oracle_abs_error": float(best_pair_row["pair_oracle_abs_error"]),
                "pair_oracle_within5": int(best_pair_row["pair_oracle_within5"]),
                "pair_oracle_rescued_vs_baseline": int(
                    int(baseline_row["within5"]) == 0 and int(best_pair_row["pair_oracle_within5"]) == 1
                ),
                "oracle_final_count": int(best_row["final_count"]),
                "oracle_count_error": int(best_row["count_error"]),
                "oracle_false_extreme": int(best_row["false_extreme"]),
                "oracle_missed_gt": int(best_row["missed_gt"]),
            }
        )

        if args.progress_every > 0 and (image_index % args.progress_every == 0 or image_index == len(selected_samples)):
            print(f"processadas {image_index}/{len(selected_samples)} imagens")

    profile_summaries = {
        profile: summarize_profile_rows(rows_by_profile[profile])
        for profile in args.profiles
    }
    baseline_summary = profile_summaries[args.baseline_profile]
    oracle_metrics = finite_metrics(
        [float(row["oracle_cobb"]) for row in oracle_rows],
        [float(row["gt_cobb_max"]) for row in oracle_rows],
    )
    oracle_metrics.update(
        {
            "selected_profile_counts": count_summary([str(row["oracle_profile"]) for row in oracle_rows]),
            "pair_oracle_selected_profile_counts": count_summary(
                [str(row["pair_oracle_profile"]) for row in oracle_rows]
            ),
            "rescued_failures_vs_baseline": int(np.sum([int(row["rescued_vs_baseline"]) for row in oracle_rows])),
            "pair_oracle_rescued_failures_vs_baseline": int(
                np.sum([int(row["pair_oracle_rescued_vs_baseline"]) for row in oracle_rows])
            ),
            "pair_oracle": finite_metrics(
                [float(row["pair_oracle_cobb"]) for row in oracle_rows],
                [float(row["gt_cobb_max"]) for row in oracle_rows],
            ),
            "delta_within5_vs_baseline": float(
                float(oracle_metrics.get("within_5deg_rate", 0.0))
                - float(baseline_summary.get("within_5deg_rate", 0.0))
            ),
            "delta_mae_vs_baseline": float(
                float(oracle_metrics.get("mae_deg", 0.0))
                - float(baseline_summary.get("mae_deg", 0.0))
            ),
        }
    )
    oracle_metrics["pair_oracle_within_5deg_rate"] = float(
        oracle_metrics["pair_oracle"].get("within_5deg_rate", 0.0)
    )
    oracle_metrics["pair_oracle_mae_deg"] = float(oracle_metrics["pair_oracle"].get("mae_deg", 0.0))
    oracle_metrics["pair_oracle_failures_gt5"] = int(oracle_metrics["pair_oracle"].get("failures_gt5", 0))

    summary_payload = {
        "phase": "phase5_sequence_portfolio_cobb_oracle_v1",
        "model_path": str(model_path),
        "eval": eval_metadata,
        "profiles": {profile: phase5_sequence.PROFILE_CONFIGS[profile] for profile in args.profiles},
        "baseline_profile": args.baseline_profile,
        "profile_summaries": profile_summaries,
        "portfolio_oracle": oracle_metrics,
    }

    profile_fields = [
        "profile",
        "file_name",
        "gt_cobb_max",
        "gt_major_region",
        "pred_cobb",
        "signed_error",
        "abs_error",
        "within5",
        "geom_upper_index",
        "geom_lower_index",
        "geom_span",
        "pair_oracle_cobb",
        "pair_oracle_abs_error",
        "pair_oracle_within5",
        "pair_oracle_upper_index",
        "pair_oracle_lower_index",
        "pair_oracle_span",
        "final_count",
        "count_error",
        "matched_count",
        "missed_gt",
        "false_pred",
        "false_extreme",
        "false_inside",
        "missed_inside",
        "mean_final_score",
        "endpoint_filled_top",
        "endpoint_filled_bottom",
        "gap_filled_count",
    ]
    oracle_fields = [
        "file_name",
        "gt_cobb_max",
        "gt_major_region",
        "baseline_profile",
        "baseline_cobb",
        "baseline_abs_error",
        "baseline_within5",
        "oracle_profile",
        "oracle_cobb",
        "oracle_abs_error",
        "oracle_within5",
        "rescued_vs_baseline",
        "pair_oracle_profile",
        "pair_oracle_cobb",
        "pair_oracle_abs_error",
        "pair_oracle_within5",
        "pair_oracle_rescued_vs_baseline",
        "oracle_final_count",
        "oracle_count_error",
        "oracle_false_extreme",
        "oracle_missed_gt",
    ]
    summary_path = experiment_dir / "phase5_sequence_portfolio_cobb_oracle_summary.json"
    profile_csv = experiment_dir / "phase5_sequence_portfolio_cobb_profile_rows.csv"
    oracle_csv = experiment_dir / "phase5_sequence_portfolio_cobb_oracle_rows.csv"
    write_json(summary_path, summary_payload)
    write_csv(profile_csv, profile_rows, profile_fields)
    write_csv(oracle_csv, oracle_rows, oracle_fields)
    make_report(
        path=report_path,
        args=args,
        eval_metadata=eval_metadata,
        profile_summaries=profile_summaries,
        oracle_summary=oracle_metrics,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
    )

    print("\nResumo oracle Cobb-aware")
    for profile, summary in profile_summaries.items():
        print(
            f"{profile}: MAE={summary['mae_deg']:.3f}, "
            f"within5={summary['within_5deg_rate']:.3f}, "
            f"falhas>5={summary['failures_gt5']}, "
            f"count_MAE={summary['mean_abs_count_error']:.3f}"
        )
    print(
        "portfolio_oracle: "
        f"MAE={oracle_metrics['mae_deg']:.3f}, "
        f"within5={oracle_metrics['within_5deg_rate']:.3f}, "
        f"falhas>5={oracle_metrics['failures_gt5']}, "
        f"resgatados={oracle_metrics['rescued_failures_vs_baseline']}"
    )
    print(
        "endpoint_pair_oracle: "
        f"MAE={oracle_metrics['pair_oracle_mae_deg']:.3f}, "
        f"within5={oracle_metrics['pair_oracle_within_5deg_rate']:.3f}, "
        f"falhas>5={oracle_metrics['pair_oracle_failures_gt5']}, "
        f"resgatados={oracle_metrics['pair_oracle_rescued_failures_vs_baseline']}"
    )
    print(f"Resumo guardado em: {summary_path}")
    print(f"Relatorio guardado em: {report_path}")


if __name__ == "__main__":
    main()
