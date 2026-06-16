"""Fase 5 - avaliacao de seleccao anatomica de sequencia.

Implementa a Fase B do plano:
Spinal-AI2024_ajuda/Metodologia/Melhorias para 90-95/
Plano_Fase5_Deteccao_Sequencia_Para_90_95.md

O script compara a sequencia atual da Fase 5 com perfis anatomicos baseados em
caminho top-to-bottom. A inferencia continua prediction-driven; o ground truth
e usado apenas para auditoria e metricas.
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

DEFAULT_MODEL = MODELS_DIR / "phase5_resnet50_fpn_spatial_offset_hard_negative_fulltrain_probe_v1_colab.keras"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "phase5_anatomical_sequence_v1"
DEFAULT_EXPERIMENT_NAME = "phase5_anatomical_sequence_v1"
DEFAULT_REPORT = REPORTS_DIR / "phase5_anatomical_sequence_v1.md"


PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_current": {
        "confidence_threshold": 0.75,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "greedy",
        "count_prior": 17.0,
        "count_prior_weight": 0.12,
        "candidate_cost": 2.0,
        "gap_weight": 0.65,
        "x_jump_weight": 0.55,
        "size_weight": 0.25,
        "angle_weight": 0.0,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.88,
        "endpoint_max_gap_factor": 1.65,
        "endpoint_score_margin": 0.08,
        "gap_filling": True,
        "gap_fill_threshold": 0.6,
        "gap_fill_min_gap_factor": 1.55,
        "gap_fill_max_insertions": 4,
        "gap_fill_max_x_error": 0.07,
        "gap_fill_max_size_log_error": 0.65,
        "gap_fill_min_neighbor_gap_factor": 0.45,
    },
    "anatomical_path_v1": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.30,
        "candidate_cost": 1.20,
        "gap_weight": 1.00,
        "x_jump_weight": 0.80,
        "size_weight": 0.35,
        "angle_weight": 0.20,
        "angle_jump_tolerance_deg": 12.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.86,
        "endpoint_max_gap_factor": 1.55,
        "endpoint_score_margin": 0.06,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 5,
        "gap_fill_max_x_error": 0.075,
        "gap_fill_max_size_log_error": 0.70,
        "gap_fill_min_neighbor_gap_factor": 0.42,
    },
    "anatomical_recall_v1": {
        "confidence_threshold": 0.65,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.35,
        "candidate_cost": 1.10,
        "gap_weight": 1.10,
        "x_jump_weight": 0.90,
        "size_weight": 0.40,
        "angle_weight": 0.25,
        "angle_jump_tolerance_deg": 11.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.84,
        "endpoint_max_gap_factor": 1.50,
        "endpoint_score_margin": 0.05,
        "gap_filling": True,
        "gap_fill_threshold": 0.45,
        "gap_fill_min_gap_factor": 1.40,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
    },
    "anatomical_balanced_v1": {
        "confidence_threshold": 0.75,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.45,
        "candidate_cost": 1.00,
        "gap_weight": 0.90,
        "x_jump_weight": 0.75,
        "size_weight": 0.30,
        "angle_weight": 0.15,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.84,
        "endpoint_max_gap_factor": 1.75,
        "endpoint_score_margin": 0.04,
        "gap_filling": True,
        "gap_fill_threshold": 0.55,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 5,
        "gap_fill_max_x_error": 0.075,
        "gap_fill_max_size_log_error": 0.70,
        "gap_fill_min_neighbor_gap_factor": 0.42,
    },
    "anatomical_endpoint_safe_v1": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
    },
    "anatomical_endpoint_fallback_v1": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.60,
        "endpoint_fill_max_per_side": 2,
        "endpoint_fill_max_gap_factor": 2.10,
        "endpoint_fill_max_x_error": 0.090,
        "endpoint_fill_max_size_log_error": 0.85,
        "endpoint_fill_min_gap_factor": 0.35,
    },
    "anatomical_endpoint_fallback_moderate_v1": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.65,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.75,
        "endpoint_fill_max_x_error": 0.080,
        "endpoint_fill_max_size_log_error": 0.75,
        "endpoint_fill_min_gap_factor": 0.45,
    },
    "anatomical_endpoint_fallback_strict_v1": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.70,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.60,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.50,
    },
    "anatomical_endpoint_fallback_balanced_v2": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.66,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.65,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.48,
    },
    "anatomical_endpoint_fallback_geometry_v2": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.64,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.55,
        "endpoint_fill_max_x_error": 0.070,
        "endpoint_fill_max_size_log_error": 0.65,
        "endpoint_fill_min_gap_factor": 0.55,
    },
    "anatomical_endpoint_fallback_recall_v2": {
        "confidence_threshold": 0.70,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.50,
        "candidate_cost": 0.80,
        "gap_weight": 0.80,
        "x_jump_weight": 0.60,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.80,
        "endpoint_max_gap_factor": 1.90,
        "endpoint_score_margin": 0.03,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.62,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.50,
        "endpoint_fill_max_x_error": 0.068,
        "endpoint_fill_max_size_log_error": 0.62,
        "endpoint_fill_min_gap_factor": 0.58,
    },
    "anatomical_endpoint_recovery_v1": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.55,
        "candidate_cost": 0.75,
        "gap_weight": 0.75,
        "x_jump_weight": 0.55,
        "size_weight": 0.25,
        "angle_weight": 0.10,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.72,
        "endpoint_max_gap_factor": 2.25,
        "endpoint_score_margin": 0.02,
        "gap_filling": True,
        "gap_fill_threshold": 0.42,
        "gap_fill_min_gap_factor": 1.30,
        "gap_fill_max_insertions": 8,
        "gap_fill_max_x_error": 0.090,
        "gap_fill_max_size_log_error": 0.85,
        "gap_fill_min_neighbor_gap_factor": 0.35,
    },
    "anatomical_endpoint_recovery_balanced_v1": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.46,
        "gap_fill_min_gap_factor": 1.36,
        "gap_fill_max_insertions": 7,
        "gap_fill_max_x_error": 0.085,
        "gap_fill_max_size_log_error": 0.80,
        "gap_fill_min_neighbor_gap_factor": 0.38,
    },
    "anatomical_endpoint_recovery_fill_strict_v2": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.46,
        "gap_fill_min_gap_factor": 1.36,
        "gap_fill_max_insertions": 7,
        "gap_fill_max_x_error": 0.085,
        "gap_fill_max_size_log_error": 0.80,
        "gap_fill_min_neighbor_gap_factor": 0.38,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.70,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.60,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.50,
    },
    "anatomical_endpoint_recovery_fill_balanced_v2": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.46,
        "gap_fill_min_gap_factor": 1.36,
        "gap_fill_max_insertions": 7,
        "gap_fill_max_x_error": 0.085,
        "gap_fill_max_size_log_error": 0.80,
        "gap_fill_min_neighbor_gap_factor": 0.38,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.66,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.65,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.48,
    },
    "anatomical_endpoint_recovery_fill_moderate_v2": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.46,
        "gap_fill_min_gap_factor": 1.36,
        "gap_fill_max_insertions": 7,
        "gap_fill_max_x_error": 0.085,
        "gap_fill_max_size_log_error": 0.80,
        "gap_fill_min_neighbor_gap_factor": 0.38,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.65,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.75,
        "endpoint_fill_max_x_error": 0.080,
        "endpoint_fill_max_size_log_error": 0.75,
        "endpoint_fill_min_gap_factor": 0.45,
    },
    "anatomical_endpoint_recovery_fill_inner_safe_v2": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.70,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.60,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.50,
    },
    "anatomical_endpoint_recovery_fill_conf65_v2": {
        "confidence_threshold": 0.65,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.76,
        "endpoint_max_gap_factor": 2.10,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.46,
        "gap_fill_min_gap_factor": 1.36,
        "gap_fill_max_insertions": 7,
        "gap_fill_max_x_error": 0.085,
        "gap_fill_max_size_log_error": 0.80,
        "gap_fill_min_neighbor_gap_factor": 0.38,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.70,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.60,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.50,
    },
    "anatomical_endpoint_recovery_fill_conservative_v2": {
        "confidence_threshold": 0.65,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.58,
        "candidate_cost": 0.85,
        "gap_weight": 0.82,
        "x_jump_weight": 0.62,
        "size_weight": 0.30,
        "angle_weight": 0.12,
        "angle_jump_tolerance_deg": 14.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.78,
        "endpoint_max_gap_factor": 2.00,
        "endpoint_score_margin": 0.025,
        "gap_filling": True,
        "gap_fill_threshold": 0.50,
        "gap_fill_min_gap_factor": 1.45,
        "gap_fill_max_insertions": 6,
        "gap_fill_max_x_error": 0.080,
        "gap_fill_max_size_log_error": 0.75,
        "gap_fill_min_neighbor_gap_factor": 0.40,
        "endpoint_filling": True,
        "endpoint_fill_threshold": 0.70,
        "endpoint_fill_max_per_side": 1,
        "endpoint_fill_max_gap_factor": 1.60,
        "endpoint_fill_max_x_error": 0.075,
        "endpoint_fill_max_size_log_error": 0.70,
        "endpoint_fill_min_gap_factor": 0.50,
    },
    "anatomical_endpoint_recovery_mid_v1": {
        "confidence_threshold": 0.60,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.56,
        "candidate_cost": 0.80,
        "gap_weight": 0.78,
        "x_jump_weight": 0.58,
        "size_weight": 0.27,
        "angle_weight": 0.11,
        "angle_jump_tolerance_deg": 15.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.74,
        "endpoint_max_gap_factor": 2.18,
        "endpoint_score_margin": 0.02,
        "gap_filling": True,
        "gap_fill_threshold": 0.44,
        "gap_fill_min_gap_factor": 1.33,
        "gap_fill_max_insertions": 8,
        "gap_fill_max_x_error": 0.088,
        "gap_fill_max_size_log_error": 0.83,
        "gap_fill_min_neighbor_gap_factor": 0.36,
    },
    "anatomical_endpoint_recovery_loose_v1": {
        "confidence_threshold": 0.55,
        "nms_iou_threshold": 0.1,
        "min_y_gap": 0.025,
        "min_vertebrae": 14,
        "max_vertebrae": 21,
        "selection_method": "path",
        "count_prior": 17.0,
        "count_prior_weight": 0.60,
        "candidate_cost": 0.65,
        "gap_weight": 0.70,
        "x_jump_weight": 0.50,
        "size_weight": 0.22,
        "angle_weight": 0.08,
        "angle_jump_tolerance_deg": 16.0,
        "endpoint_pruning": True,
        "endpoint_min_score": 0.68,
        "endpoint_max_gap_factor": 2.40,
        "endpoint_score_margin": 0.015,
        "gap_filling": True,
        "gap_fill_threshold": 0.40,
        "gap_fill_min_gap_factor": 1.25,
        "gap_fill_max_insertions": 9,
        "gap_fill_max_x_error": 0.100,
        "gap_fill_max_size_log_error": 0.90,
        "gap_fill_min_neighbor_gap_factor": 0.33,
    },
}

_ENDPOINT_AUX_BASE_PROFILE = "anatomical_endpoint_recovery_fill_inner_safe_v2"
for _suffix, _blend in (("25", 0.25), ("50", 0.50), ("75", 0.75)):
    PROFILE_CONFIGS[f"{_ENDPOINT_AUX_BASE_PROFILE}_cobb_endpoint_blend{_suffix}"] = {
        **PROFILE_CONFIGS[_ENDPOINT_AUX_BASE_PROFILE],
        "endpoint_score_blend": _blend,
    }


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


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def select_window(items: Sequence[Any], start_index: int, num_items: int) -> Sequence[Any]:
    if start_index < 0:
        raise ValueError("--start-index deve ser >= 0")
    if num_items <= 0:
        return items[start_index:]
    return items[start_index : start_index + num_items]


def select_eval_samples(
    tfdata: ModuleType,
    phase2_train: ModuleType,
    eval_mode: str,
    split: str,
    train_size: int,
    val_size: int,
    start_index: int,
    num_images: int,
) -> tuple[Sequence[Any], Sequence[Path], dict[str, Any]]:
    samples, image_paths = phase2_train.load_split_samples_and_paths(tfdata, split)
    if eval_mode == "validation":
        if split != "train":
            raise ValueError("--eval-mode validation so e valido com --split train")
        val_start = min(max(int(train_size), 0), len(samples))
        val_end = len(samples) if val_size <= 0 else min(val_start + int(val_size), len(samples))
        validation_samples = samples[val_start:val_end]
        validation_paths = image_paths[val_start:val_end]
        selected_samples = select_window(validation_samples, start_index, num_images)
        selected_paths = select_window(validation_paths, start_index, num_images)
        metadata = {
            "eval_mode": eval_mode,
            "split": split,
            "train_size": train_size,
            "val_size": val_size,
            "absolute_start_index": val_start + start_index,
            "relative_start_index": start_index,
        }
    elif eval_mode == "window":
        selected_samples = select_window(samples, start_index, num_images)
        selected_paths = select_window(image_paths, start_index, num_images)
        metadata = {
            "eval_mode": eval_mode,
            "split": split,
            "absolute_start_index": start_index,
            "relative_start_index": start_index,
        }
    else:
        raise ValueError("--eval-mode deve ser validation ou window")

    if not selected_samples:
        raise ValueError("Nenhuma imagem selecionada. Verifica os indices e --num-images.")
    return selected_samples, selected_paths, metadata


def false_extreme(record: Mapping[str, Any]) -> int:
    return int(record["false_top"]) + int(record["false_bottom"])


def summarize_profile(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_images": 0}

    count_errors = [int(record["count_error"]) for record in records]
    abs_count_errors = [abs(value) for value in count_errors]
    gt_counts = [max(int(record["gt_count"]), 1) for record in records]
    matched = [int(record["matched_count"]) for record in records]
    match_rates = [match / gt for match, gt in zip(matched, gt_counts)]
    return {
        "num_images": len(records),
        "mean_abs_count_error": float(np.mean(abs_count_errors)),
        "exact_count_accuracy": float(np.mean([value == 0 for value in count_errors])),
        "undercount_rate": float(np.mean([value < 0 for value in count_errors])),
        "overcount_rate": float(np.mean([value > 0 for value in count_errors])),
        "mean_matched_count": float(np.mean(matched)),
        "mean_match_rate": float(np.mean(match_rates)),
        "mean_missed_gt": float(np.mean([int(record["missed_gt"]) for record in records])),
        "mean_false_pred": float(np.mean([int(record["false_pred"]) for record in records])),
        "mean_false_extreme": float(np.mean([false_extreme(record) for record in records])),
        "mean_false_inside": float(np.mean([int(record["false_inside"]) for record in records])),
        "mean_missed_inside": float(np.mean([int(record["missed_inside"]) for record in records])),
        "total_false_top": int(np.sum([int(record["false_top"]) for record in records])),
        "total_false_bottom": int(np.sum([int(record["false_bottom"]) for record in records])),
        "total_false_inside": int(np.sum([int(record["false_inside"]) for record in records])),
        "total_missed_top": int(np.sum([int(record["missed_top"]) for record in records])),
        "total_missed_bottom": int(np.sum([int(record["missed_bottom"]) for record in records])),
        "total_missed_inside": int(np.sum([int(record["missed_inside"]) for record in records])),
        "total_gap_filled": int(np.sum([int(record["gap_filled_count"]) for record in records])),
        "total_endpoint_pruned_top": int(np.sum([int(record["endpoint_pruned_top"]) for record in records])),
        "total_endpoint_pruned_bottom": int(
            np.sum([int(record["endpoint_pruned_bottom"]) for record in records])
        ),
        "total_endpoint_filled_top": int(
            np.sum([int(record.get("endpoint_filled_top", 0)) for record in records])
        ),
        "total_endpoint_filled_bottom": int(
            np.sum([int(record.get("endpoint_filled_bottom", 0)) for record in records])
        ),
        "mean_center_error_px": mean(
            [
                float(record["mean_center_error_px"])
                for record in records
                if record["mean_center_error_px"] not in (None, "")
            ]
        ),
        "mean_points_mae_px": mean(
            [
                float(record["mean_points_mae_px"])
                for record in records
                if record["mean_points_mae_px"] not in (None, "")
            ]
        ),
    }


def compare_against_baseline(
    rows_by_profile: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_profile: str,
    candidate_profile: str,
) -> list[dict[str, Any]]:
    baseline_rows = rows_by_profile[baseline_profile]
    candidate_rows = rows_by_profile[candidate_profile]
    rows: list[dict[str, Any]] = []
    for file_name, base in baseline_rows.items():
        candidate = candidate_rows.get(file_name)
        if candidate is None:
            continue
        delta_matched = int(candidate["matched_count"]) - int(base["matched_count"])
        delta_missed = int(candidate["missed_gt"]) - int(base["missed_gt"])
        delta_false = int(candidate["false_pred"]) - int(base["false_pred"])
        delta_false_extreme = false_extreme(candidate) - false_extreme(base)
        base_abs = abs(int(base["count_error"]))
        candidate_abs = abs(int(candidate["count_error"]))
        rows.append(
            {
                "file_name": file_name,
                "baseline_profile": baseline_profile,
                "candidate_profile": candidate_profile,
                "gt_count": int(base["gt_count"]),
                "baseline_final_count": int(base["final_count"]),
                "candidate_final_count": int(candidate["final_count"]),
                "delta_abs_count_error": candidate_abs - base_abs,
                "delta_matched_count": delta_matched,
                "delta_missed_gt": delta_missed,
                "delta_false_pred": delta_false,
                "delta_false_extreme": delta_false_extreme,
                "delta_false_inside": int(candidate["false_inside"]) - int(base["false_inside"]),
                "delta_missed_inside": int(candidate["missed_inside"]) - int(base["missed_inside"]),
                "improved_detection": int(
                    delta_matched > 0
                    or delta_missed < 0
                    or delta_false_extreme < 0
                    or candidate_abs < base_abs
                ),
                "degraded_detection": int(
                    delta_matched < 0
                    or delta_missed > 0
                    or delta_false > 0
                    or candidate_abs > base_abs
                ),
            }
        )
    return rows


def summarize_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_images": 0}
    return {
        "num_images": len(rows),
        "improved_detection_rate": float(np.mean([int(row["improved_detection"]) for row in rows])),
        "degraded_detection_rate": float(np.mean([int(row["degraded_detection"]) for row in rows])),
        "mean_delta_abs_count_error": float(np.mean([int(row["delta_abs_count_error"]) for row in rows])),
        "mean_delta_matched_count": float(np.mean([int(row["delta_matched_count"]) for row in rows])),
        "mean_delta_missed_gt": float(np.mean([int(row["delta_missed_gt"]) for row in rows])),
        "mean_delta_false_pred": float(np.mean([int(row["delta_false_pred"]) for row in rows])),
        "mean_delta_false_extreme": float(np.mean([int(row["delta_false_extreme"]) for row in rows])),
        "mean_delta_false_inside": float(np.mean([int(row["delta_false_inside"]) for row in rows])),
        "mean_delta_missed_inside": float(np.mean([int(row["delta_missed_inside"]) for row in rows])),
    }


def svg_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_profile_svg(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    metrics = [
        ("mean_missed_gt", "missed"),
        ("mean_false_extreme", "false extreme"),
        ("mean_false_inside", "false inside"),
        ("mean_abs_count_error", "count MAE"),
    ]
    profiles = list(summaries.keys())
    width = 980
    height = 120 + 82 * len(metrics)
    max_value = max(
        [float(summaries[p].get(key, 0.0) or 0.0) for p in profiles for key, _ in metrics] + [1.0]
    )
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">Fase 5 - comparacao de sequencia anatomica</text>',
    ]
    chart_x = 210
    bar_max = 620
    y = 74
    for metric_index, (metric_key, label) in enumerate(metrics):
        elements.append(
            f'<text x="24" y="{y + 22}" font-family="Arial" font-size="15" font-weight="700">{svg_text(label)}</text>'
        )
        for profile_index, profile in enumerate(profiles):
            value = float(summaries[profile].get(metric_key, 0.0) or 0.0)
            bar_width = 0.0 if max_value <= 0 else bar_max * value / max_value
            bar_y = y + profile_index * 18
            color = colors[profile_index % len(colors)]
            elements.append(f'<rect x="{chart_x}" y="{bar_y}" width="{bar_width:.1f}" height="13" fill="{color}"/>')
            elements.append(
                f'<text x="{chart_x + bar_width + 8:.1f}" y="{bar_y + 11}" font-family="Arial" font-size="12">{value:.3f}</text>'
            )
            elements.append(
                f'<text x="{chart_x - 150}" y="{bar_y + 11}" font-family="Arial" font-size="12">{svg_text(profile)}</text>'
            )
        y += 82
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def make_report(
    path: Path,
    args: argparse.Namespace,
    eval_metadata: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    comparison_summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    experiment_dir: Path,
) -> None:
    lines = [
        "# Fase 5 - seleccao anatomica de sequencia v1",
        "",
        "## Objetivo",
        "",
        "Implementar a Fase B do plano de melhorias 90-95: comparar a sequencia atual com perfis anatomicos que recuperam candidatos de score mais baixo e penalizam saltos verticais, horizontais, de escala e de angulo.",
        "",
        "## Configuracao",
        "",
        f"- modelo: `{args.model_path}`",
        f"- modo de avaliacao: `{eval_metadata['eval_mode']}`",
        f"- split: `{eval_metadata['split']}`",
        f"- imagens avaliadas: `{next(iter(summaries.values())).get('num_images', 0) if summaries else 0}`",
        f"- perfis: `{', '.join(args.profiles)}`",
        "",
        "## Resultados por perfil",
        "",
        "| perfil | count MAE | exact | matched medio | missed medio | falsos medio | falso extremo medio | missed interior medio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, summary in summaries.items():
        lines.append(
            "| "
            f"{profile} | "
            f"{summary.get('mean_abs_count_error', 0.0):.4f} | "
            f"{summary.get('exact_count_accuracy', 0.0):.4f} | "
            f"{summary.get('mean_matched_count', 0.0):.4f} | "
            f"{summary.get('mean_missed_gt', 0.0):.4f} | "
            f"{summary.get('mean_false_pred', 0.0):.4f} | "
            f"{summary.get('mean_false_extreme', 0.0):.4f} | "
            f"{summary.get('mean_missed_inside', 0.0):.4f} |"
        )

    lines.extend(["", "## Comparacao contra baseline", ""])
    if comparison_summaries:
        lines.extend(
            [
                "| perfil candidato | delta count MAE | delta matched | delta missed | delta falsos | delta falso extremo | melhorou | degradou |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for profile, summary in comparison_summaries.items():
            lines.append(
                "| "
                f"{profile} | "
                f"{summary.get('mean_delta_abs_count_error', 0.0):.4f} | "
                f"{summary.get('mean_delta_matched_count', 0.0):.4f} | "
                f"{summary.get('mean_delta_missed_gt', 0.0):.4f} | "
                f"{summary.get('mean_delta_false_pred', 0.0):.4f} | "
                f"{summary.get('mean_delta_false_extreme', 0.0):.4f} | "
                f"{summary.get('improved_detection_rate', 0.0):.4f} | "
                f"{summary.get('degraded_detection_rate', 0.0):.4f} |"
            )
    else:
        lines.append("Sem perfil candidato para comparar.")

    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            f"- resumo JSON: `{experiment_dir / 'phase5_anatomical_sequence_summary.json'}`",
            f"- CSV de detalhes: `{experiment_dir / 'phase5_anatomical_sequence_details.csv'}`",
            f"- CSV de comparacao: `{experiment_dir / 'phase5_anatomical_sequence_comparison.csv'}`",
            f"- grafico: `{output_dir / 'phase5_anatomical_sequence_profile_comparison.svg'}`",
            f"- overlays: `{output_dir}`",
            "",
            "## Interpretacao esperada",
            "",
            "Promover um perfil anatomico apenas se reduzir missing vertebrae e falsas extremas sem degradar de forma relevante os casos que ja estavam bons. O passo seguinte, depois desta avaliacao, e regenerar embeddings da Fase 6 com o perfil escolhido e medir o impacto no Cobb final.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia perfis anatomicos de sequencia da Fase 5.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--eval-mode", choices=("validation", "window"), default="validation")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--train-size", type=int, default=12768)
    parser.add_argument("--val-size", type=int, default=3192)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-images",
        type=int,
        default=64,
        help="Numero de imagens a avaliar. Usa 0 para avaliar toda a janela escolhida.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_CONFIGS.keys()),
        default=["baseline_current", "anatomical_path_v1"],
    )
    parser.add_argument("--baseline-profile", default="baseline_current", choices=tuple(PROFILE_CONFIGS.keys()))
    parser.add_argument("--max-match-distance-px", type=float, default=32.0)
    parser.add_argument("--num-overlays", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline_profile not in args.profiles:
        raise ValueError("--baseline-profile deve estar incluido em --profiles")

    tfdata = import_script("01_tfdata_loader.py", "phase0_tfdata_loader")
    phase2_train = import_script("07_train_quadrilateral_v1.py", "phase2_train")
    phase5_eval = import_script("21_eval_resnet50_fpn_spatial_offset_sequence.py", "phase5_eval")
    postprocess = import_script("19_postprocess_sequence_v1.py", "phase5_sequence_postprocess")

    model_path = resolve_project_path(args.model_path)
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"A carregar modelo Fase 5: {model_path}")
    model = phase5_eval.load_spatial_offset_model_for_eval(model_path)

    selected_samples, selected_paths, eval_metadata = select_eval_samples(
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
        "A avaliar sequencias: "
        f"imagens={len(selected_samples)}, perfis={', '.join(args.profiles)}"
    )

    details: list[dict[str, Any]] = []
    rows_by_profile: dict[str, dict[str, dict[str, Any]]] = {profile: {} for profile in args.profiles}

    for image_index, (sample, image_path) in enumerate(zip(selected_samples, selected_paths), start=1):
        targets = tfdata.sample_to_targets(sample)
        image, _ = tfdata.load_image(tf.constant(str(image_path)), {})
        predictions = model(tf.expand_dims(image, axis=0), training=False)
        prediction_values = {
            "presence": predictions["presence"].numpy()[0],
            "bbox": predictions["bbox"].numpy()[0],
            "points": predictions["points"].numpy()[0],
        }
        if isinstance(predictions, dict) and "cobb_endpoint_score" in predictions:
            prediction_values["cobb_endpoint_score"] = predictions["cobb_endpoint_score"].numpy()[0]

        for profile in args.profiles:
            config = PROFILE_CONFIGS[profile]
            result = postprocess.postprocess_candidates_sequence(
                presence=prediction_values["presence"],
                bbox=prediction_values["bbox"],
                points=prediction_values["points"],
                cobb_endpoint_score=prediction_values.get("cobb_endpoint_score"),
                **config,
            )

            overlay_name = ""
            if image_index <= args.num_overlays:
                overlay = phase5_eval.make_sequence_comparison_overlay(
                    image=image,
                    targets=targets,
                    postprocessed=result,
                    drawing=phase2_train,
                )
                stem = Path(str(sample["file_name"])).stem
                overlay_name = f"{image_index - 1:03d}_{stem}_{profile}_sequence.png"
                tf.io.write_file(
                    str(output_dir / overlay_name),
                    tf.io.encode_png(tf.convert_to_tensor(overlay)),
                )

            record = phase5_eval.build_evaluation_record(
                sample=sample,
                targets=targets,
                result=result,
                max_match_distance_px=args.max_match_distance_px,
                overlay_name=overlay_name,
            )
            record["profile"] = profile
            record["false_extreme"] = false_extreme(record)
            record["match_rate"] = (
                float(record["matched_count"]) / max(float(record["gt_count"]), 1.0)
            )
            details.append(record)
            rows_by_profile[profile][str(record["file_name"])] = record

        if args.progress_every > 0 and (image_index % args.progress_every == 0 or image_index == len(selected_samples)):
            print(f"processadas {image_index}/{len(selected_samples)} imagens")

    summaries = {
        profile: summarize_profile(list(rows_by_profile[profile].values()))
        for profile in args.profiles
    }
    comparisons: dict[str, list[dict[str, Any]]] = {}
    comparison_summaries: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        if profile == args.baseline_profile:
            continue
        rows = compare_against_baseline(
            rows_by_profile=rows_by_profile,
            baseline_profile=args.baseline_profile,
            candidate_profile=profile,
        )
        comparisons[profile] = rows
        comparison_summaries[profile] = summarize_comparison(rows)

    comparison_rows = [row for rows in comparisons.values() for row in rows]
    summary_payload = {
        "phase": "phase5_anatomical_sequence_v1",
        "model_path": str(model_path),
        "eval": eval_metadata,
        "profiles": {profile: PROFILE_CONFIGS[profile] for profile in args.profiles},
        "baseline_profile": args.baseline_profile,
        "summaries": summaries,
        "comparison_summaries": comparison_summaries,
    }

    details_fields = [
        "profile",
        "file_name",
        "gt_count",
        "raw_count",
        "nms_count",
        "final_count",
        "count_error",
        "abs_count_error",
        "matched_count",
        "match_rate",
        "missed_gt",
        "false_pred",
        "false_extreme",
        "false_top",
        "false_bottom",
        "false_inside",
        "missed_top",
        "missed_bottom",
        "missed_inside",
        "mean_center_error_px",
        "mean_points_mae_px",
        "mean_final_score",
        "mean_endpoint_decision_score",
        "endpoint_score_used",
        "endpoint_score_blend",
        "selection_method",
        "estimated_y_gap",
        "path_score",
        "endpoint_pruned_top",
        "endpoint_pruned_bottom",
        "endpoint_filled_top",
        "endpoint_filled_bottom",
        "endpoint_fill_candidate_count",
        "gap_filled_count",
        "gap_fill_candidate_count",
        "selected_indices",
        "overlay_name",
    ]
    comparison_fields = [
        "file_name",
        "baseline_profile",
        "candidate_profile",
        "gt_count",
        "baseline_final_count",
        "candidate_final_count",
        "delta_abs_count_error",
        "delta_matched_count",
        "delta_missed_gt",
        "delta_false_pred",
        "delta_false_extreme",
        "delta_false_inside",
        "delta_missed_inside",
        "improved_detection",
        "degraded_detection",
    ]

    write_json(experiment_dir / "phase5_anatomical_sequence_summary.json", summary_payload)
    write_csv(experiment_dir / "phase5_anatomical_sequence_details.csv", details, details_fields)
    write_csv(experiment_dir / "phase5_anatomical_sequence_comparison.csv", comparison_rows, comparison_fields)
    write_json(experiment_dir / "config.json", vars(args))
    save_profile_svg(output_dir / "phase5_anatomical_sequence_profile_comparison.svg", summaries)
    make_report(
        path=report_path,
        args=args,
        eval_metadata=eval_metadata,
        summaries=summaries,
        comparison_summaries=comparison_summaries,
        output_dir=output_dir,
        experiment_dir=experiment_dir,
    )

    print("\nResumo Fase 5 - seleccao anatomica")
    for profile, summary in summaries.items():
        print(
            f"{profile}: "
            f"count_MAE={summary['mean_abs_count_error']:.3f}, "
            f"exact={summary['exact_count_accuracy']:.3f}, "
            f"missed={summary['mean_missed_gt']:.3f}, "
            f"false={summary['mean_false_pred']:.3f}, "
            f"false_extreme={summary['mean_false_extreme']:.3f}, "
            f"missed_inside={summary['mean_missed_inside']:.3f}"
        )
    for profile, summary in comparison_summaries.items():
        print(
            f"delta {profile} vs {args.baseline_profile}: "
            f"count_MAE={summary['mean_delta_abs_count_error']:.3f}, "
            f"matched={summary['mean_delta_matched_count']:.3f}, "
            f"missed={summary['mean_delta_missed_gt']:.3f}, "
            f"false_extreme={summary['mean_delta_false_extreme']:.3f}, "
            f"melhorou={summary['improved_detection_rate']:.3f}, "
            f"degradou={summary['degraded_detection_rate']:.3f}"
        )
    print(f"Resumo guardado em: {experiment_dir / 'phase5_anatomical_sequence_summary.json'}")
    print(f"Relatorio guardado em: {report_path}")
    print(f"Outputs guardados em: {output_dir}")


if __name__ == "__main__":
    main()
