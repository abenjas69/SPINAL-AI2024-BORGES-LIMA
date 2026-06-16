"""Fase 5 v3: pos-processamento NMS + selecao sequencial anatomica.

Esta etapa complementa o NMS classico com uma regra simples de anatomia da
coluna. Depois de remover candidatos sobrepostos, a sequencia final e escolhida
por melhor caminho top-to-bottom, penalizando duplicados, saltos horizontais,
gaps verticais estranhos e mudancas bruscas de escala. O resultado continua a
ser prediction-driven e nao usa o `vertebra_count` real para cortar previsoes.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_NMS_IOU_THRESHOLD = 0.1
DEFAULT_MIN_Y_GAP = 0.025
DEFAULT_MIN_VERTEBRAE = 14
DEFAULT_MAX_VERTEBRAE = 21
DEFAULT_SELECTION_METHOD = "greedy"
DEFAULT_COUNT_PRIOR = 17.0
DEFAULT_COUNT_PRIOR_WEIGHT = 0.12
DEFAULT_CANDIDATE_COST = 2.0
DEFAULT_GAP_WEIGHT = 0.65
DEFAULT_X_JUMP_WEIGHT = 0.55
DEFAULT_SIZE_WEIGHT = 0.25
DEFAULT_ANGLE_WEIGHT = 0.0
DEFAULT_ANGLE_JUMP_TOLERANCE_DEG = 14.0
DEFAULT_ENDPOINT_PRUNING = True
DEFAULT_ENDPOINT_MIN_SCORE = 0.88
DEFAULT_ENDPOINT_MAX_GAP_FACTOR = 1.65
DEFAULT_ENDPOINT_SCORE_MARGIN = 0.08
DEFAULT_ENDPOINT_SCORE_BLEND = 0.0
DEFAULT_GAP_FILLING = True
DEFAULT_GAP_FILL_THRESHOLD = 0.6
DEFAULT_GAP_FILL_MIN_GAP_FACTOR = 1.55
DEFAULT_GAP_FILL_MAX_INSERTIONS = 4
DEFAULT_GAP_FILL_MAX_X_ERROR = 0.07
DEFAULT_GAP_FILL_MAX_SIZE_LOG_ERROR = 0.65
DEFAULT_GAP_FILL_MIN_NEIGHBOR_GAP_FACTOR = 0.45
DEFAULT_ENDPOINT_FILLING = False
DEFAULT_ENDPOINT_FILL_THRESHOLD = 0.6
DEFAULT_ENDPOINT_FILL_MAX_PER_SIDE = 1
DEFAULT_ENDPOINT_FILL_MAX_GAP_FACTOR = 2.1
DEFAULT_ENDPOINT_FILL_MAX_X_ERROR = 0.09
DEFAULT_ENDPOINT_FILL_MAX_SIZE_LOG_ERROR = 0.85
DEFAULT_ENDPOINT_FILL_MIN_GAP_FACTOR = 0.35
EPSILON = 1e-7


def clip_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def points_to_xyxy(points: np.ndarray) -> np.ndarray:
    """Converte pontos normalizados (N, 8) para boxes xyxy normalizadas."""
    points_array = clip_unit(np.asarray(points, dtype=np.float32)).reshape(-1, 4, 2)
    x_min = np.min(points_array[:, :, 0], axis=1)
    y_min = np.min(points_array[:, :, 1], axis=1)
    x_max = np.max(points_array[:, :, 0], axis=1)
    y_max = np.max(points_array[:, :, 1], axis=1)
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def y_centroids_from_points(points: np.ndarray) -> np.ndarray:
    points_array = clip_unit(np.asarray(points, dtype=np.float32)).reshape(-1, 4, 2)
    return np.mean(points_array[:, :, 1], axis=1)


def x_centroids_from_points(points: np.ndarray) -> np.ndarray:
    points_array = clip_unit(np.asarray(points, dtype=np.float32)).reshape(-1, 4, 2)
    return np.mean(points_array[:, :, 0], axis=1)


def width_height_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = points_to_xyxy(points)
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], EPSILON)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], EPSILON)
    return widths, heights


def line_angle_deg(point_a: np.ndarray, point_b: np.ndarray) -> float:
    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def signed_angle_delta_deg(angle_a: float, angle_b: float) -> float:
    return float((float(angle_a) - float(angle_b) + 90.0) % 180.0 - 90.0)


def circular_mean_axial_deg(angle_a: float, angle_b: float) -> float:
    radians = np.deg2rad([float(angle_a) * 2.0, float(angle_b) * 2.0])
    mean = np.arctan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
    return float(np.rad2deg(mean) / 2.0)


def mean_plate_angles_from_points(points: np.ndarray) -> np.ndarray:
    """Calcula uma inclinacao axial media por quadrilatero."""
    rows = clip_unit(np.asarray(points, dtype=np.float32)).reshape(-1, 4, 2)
    if rows.size == 0:
        return np.asarray([], dtype=np.float32)

    mean_angles: list[float] = []
    for row in rows:
        top_angle = line_angle_deg(row[0], row[1])
        bottom_angle = line_angle_deg(row[2], row[3])
        mean_angles.append(circular_mean_axial_deg(top_angle, bottom_angle))
    return np.asarray(mean_angles, dtype=np.float32)


def bbox_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calcula IoU entre uma box xyxy e varias boxes xyxy."""
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection_w = np.maximum(0.0, x_max - x_min)
    intersection_h = np.maximum(0.0, y_max - y_min)
    intersection = intersection_w * intersection_h

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - intersection
    return np.divide(intersection, np.maximum(union, 1e-7))


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    max_detections: int | None = None,
) -> np.ndarray:
    """NMS classico sobre boxes xyxy; devolve indices locais mantidos."""
    if boxes_xyxy.size == 0:
        return np.asarray([], dtype=np.int64)

    boxes = clip_unit(np.asarray(boxes_xyxy, dtype=np.float32))
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(score_values)[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if max_detections is not None and len(keep) >= max_detections:
            break
        if order.size == 1:
            break

        remaining = order[1:]
        ious = bbox_iou_xyxy(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def select_by_vertical_spacing(
    scores: np.ndarray,
    y_centroids: np.ndarray,
    min_y_gap: float = DEFAULT_MIN_Y_GAP,
    min_detections: int = DEFAULT_MIN_VERTEBRAE,
    max_detections: int = DEFAULT_MAX_VERTEBRAE,
) -> tuple[np.ndarray, float]:
    """Escolhe candidatos fortes evitando multiplas deteccoes no mesmo nivel y."""
    if scores.size == 0:
        return np.asarray([], dtype=np.int64), float(min_y_gap)

    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    y_values = np.asarray(y_centroids, dtype=np.float32).reshape(-1)
    order = np.argsort(score_values)[::-1]
    gaps = [float(min_y_gap), float(min_y_gap) * 0.75, float(min_y_gap) * 0.5, 0.0]

    best_selected: list[int] = []
    best_gap = gaps[-1]
    for gap in gaps:
        selected: list[int] = []
        for candidate in order:
            candidate_index = int(candidate)
            if all(abs(float(y_values[candidate_index] - y_values[kept])) >= gap for kept in selected):
                selected.append(candidate_index)
                if len(selected) >= max_detections:
                    break
        best_selected = selected
        best_gap = gap
        if len(selected) >= min_detections or gap == 0.0:
            break

    return np.asarray(best_selected, dtype=np.int64), float(best_gap)


def logit_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float32), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def make_endpoint_decision_scores(
    presence_scores: np.ndarray,
    cobb_endpoint_score: np.ndarray | None,
    endpoint_score_blend: float = DEFAULT_ENDPOINT_SCORE_BLEND,
) -> tuple[np.ndarray, bool, float]:
    """Combina presence com score auxiliar Cobb apenas para decisoes nos extremos."""
    scores = np.asarray(presence_scores, dtype=np.float32).reshape(-1)
    blend = float(np.clip(float(endpoint_score_blend), 0.0, 1.0))
    if cobb_endpoint_score is None or blend <= 0.0:
        return scores.copy(), False, blend

    endpoint_scores = np.asarray(cobb_endpoint_score, dtype=np.float32).reshape(-1)
    if endpoint_scores.shape != scores.shape:
        raise ValueError("cobb_endpoint_score deve ter o mesmo numero de candidatos que presence.")

    endpoint_scores = np.clip(endpoint_scores, 0.0, 1.0)
    blended = (1.0 - blend) * scores + blend * endpoint_scores
    return np.clip(blended.astype(np.float32), 0.0, 1.0), True, blend


def estimate_anatomical_y_gap(
    y_centroids: np.ndarray,
    min_y_gap: float,
    count_prior: float = DEFAULT_COUNT_PRIOR,
) -> float:
    """Estima gap vertical esperado entre centroides adjacentes."""
    y_values = np.asarray(y_centroids, dtype=np.float32).reshape(-1)
    if y_values.size < 2:
        return float(max(min_y_gap, 0.04))

    y_span = float(np.max(y_values) - np.min(y_values))
    prior_denominator = max(float(count_prior) - 1.0, 1.0)
    span_gap = y_span / prior_denominator

    positive_diffs = np.diff(np.sort(y_values))
    positive_diffs = positive_diffs[positive_diffs >= min_y_gap * 0.75]
    median_gap = float(np.median(positive_diffs)) if positive_diffs.size > 0 else span_gap

    estimated = 0.65 * span_gap + 0.35 * median_gap
    return float(np.clip(estimated, min_y_gap, 0.075))


def anatomical_transition_score(
    previous_index: int,
    current_index: int,
    x_centroids: np.ndarray,
    y_centroids: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    mean_plate_angles: np.ndarray | None,
    estimated_y_gap: float,
    min_y_gap: float,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
    x_jump_weight: float = DEFAULT_X_JUMP_WEIGHT,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    angle_weight: float = DEFAULT_ANGLE_WEIGHT,
    angle_jump_tolerance_deg: float = DEFAULT_ANGLE_JUMP_TOLERANCE_DEG,
) -> float:
    """Pontua a coerencia anatomica entre dois candidatos consecutivos."""
    y_gap = float(y_centroids[current_index] - y_centroids[previous_index])
    if y_gap < min_y_gap:
        return float("-inf")

    x_jump = abs(float(x_centroids[current_index] - x_centroids[previous_index]))
    max_x_jump = max(0.055, estimated_y_gap * 1.8)

    gap_ratio = abs(y_gap - estimated_y_gap) / max(estimated_y_gap, EPSILON)
    gap_penalty = gap_weight * max(0.0, gap_ratio - 0.35)
    x_penalty = x_jump_weight * max(0.0, x_jump - max_x_jump) / max(max_x_jump, EPSILON)

    width_ratio = abs(np.log(widths[current_index] / widths[previous_index]))
    height_ratio = abs(np.log(heights[current_index] / heights[previous_index]))
    size_penalty = size_weight * float(width_ratio + height_ratio)

    angle_penalty = 0.0
    if angle_weight > 0.0 and mean_plate_angles is not None and mean_plate_angles.size:
        angle_jump = abs(
            signed_angle_delta_deg(
                float(mean_plate_angles[current_index]),
                float(mean_plate_angles[previous_index]),
            )
        )
        excess = max(0.0, angle_jump - float(angle_jump_tolerance_deg))
        angle_penalty = float(angle_weight) * (excess / max(float(angle_jump_tolerance_deg), EPSILON))

    return -(gap_penalty + x_penalty + size_penalty + angle_penalty)


def select_best_anatomical_path(
    scores: np.ndarray,
    points: np.ndarray,
    min_y_gap: float = DEFAULT_MIN_Y_GAP,
    min_detections: int = DEFAULT_MIN_VERTEBRAE,
    max_detections: int = DEFAULT_MAX_VERTEBRAE,
    count_prior: float = DEFAULT_COUNT_PRIOR,
    count_prior_weight: float = DEFAULT_COUNT_PRIOR_WEIGHT,
    candidate_cost: float = DEFAULT_CANDIDATE_COST,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
    x_jump_weight: float = DEFAULT_X_JUMP_WEIGHT,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    angle_weight: float = DEFAULT_ANGLE_WEIGHT,
    angle_jump_tolerance_deg: float = DEFAULT_ANGLE_JUMP_TOLERANCE_DEG,
) -> tuple[np.ndarray, float, float]:
    """Seleciona a melhor subsequencia top-to-bottom por programacao dinamica."""
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    num_candidates = int(score_values.size)
    if num_candidates == 0:
        return np.asarray([], dtype=np.int64), float(min_y_gap), 0.0

    y_values = y_centroids_from_points(points_array)
    x_values = x_centroids_from_points(points_array)
    widths, heights = width_height_from_points(points_array)
    mean_plate_angles = mean_plate_angles_from_points(points_array)
    order = np.argsort(y_values)

    sorted_scores = score_values[order]
    sorted_y = y_values[order]
    sorted_x = x_values[order]
    sorted_widths = widths[order]
    sorted_heights = heights[order]
    sorted_mean_plate_angles = mean_plate_angles[order]

    estimated_y_gap = estimate_anatomical_y_gap(
        sorted_y,
        min_y_gap=min_y_gap,
        count_prior=count_prior,
    )
    unary = logit_scores(sorted_scores) - float(candidate_cost)

    max_length = min(max_detections, num_candidates)
    min_length = min(min_detections, max_length)
    if min_length <= 0:
        min_length = 1

    dp = np.full((max_length, num_candidates), -np.inf, dtype=np.float32)
    previous = np.full((max_length, num_candidates), -1, dtype=np.int64)
    dp[0, :] = unary

    for length_index in range(1, max_length):
        for current in range(num_candidates):
            best_score = -np.inf
            best_previous = -1
            for prev in range(current):
                transition = anatomical_transition_score(
                    previous_index=prev,
                    current_index=current,
                    x_centroids=sorted_x,
                    y_centroids=sorted_y,
                    widths=sorted_widths,
                    heights=sorted_heights,
                    mean_plate_angles=sorted_mean_plate_angles,
                    estimated_y_gap=estimated_y_gap,
                    min_y_gap=min_y_gap,
                    gap_weight=gap_weight,
                    x_jump_weight=x_jump_weight,
                    size_weight=size_weight,
                    angle_weight=angle_weight,
                    angle_jump_tolerance_deg=angle_jump_tolerance_deg,
                )
                if not np.isfinite(transition):
                    continue
                candidate_score = dp[length_index - 1, prev] + transition + unary[current]
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = prev
            dp[length_index, current] = best_score
            previous[length_index, current] = best_previous

    best_objective = -np.inf
    best_length_index = 0
    best_end = int(np.argmax(dp[0, :]))
    for length in range(min_length, max_length + 1):
        length_index = length - 1
        end = int(np.argmax(dp[length_index, :]))
        path_score = float(dp[length_index, end])
        if not np.isfinite(path_score):
            continue
        count_penalty = count_prior_weight * abs(float(length) - float(count_prior))
        objective = path_score - count_penalty
        if objective > best_objective:
            best_objective = objective
            best_length_index = length_index
            best_end = end

    if not np.isfinite(best_objective):
        fallback_indices, fallback_gap = select_by_vertical_spacing(
            scores=score_values,
            y_centroids=y_values,
            min_y_gap=min_y_gap,
            min_detections=min_detections,
            max_detections=max_detections,
        )
        return fallback_indices.astype(np.int64), float(fallback_gap), 0.0

    path_sorted_indices: list[int] = []
    current = best_end
    for length_index in range(best_length_index, -1, -1):
        if current < 0:
            break
        path_sorted_indices.append(int(current))
        current = int(previous[length_index, current]) if length_index > 0 else -1
    path_sorted_indices.reverse()

    selected_local_indices = order[np.asarray(path_sorted_indices, dtype=np.int64)]
    return selected_local_indices.astype(np.int64), float(estimated_y_gap), float(best_objective)


def endpoint_is_incoherent(
    endpoint_index: int,
    neighbor_index: int,
    scores: np.ndarray,
    x_centroids: np.ndarray,
    y_centroids: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    median_score: float,
    estimated_y_gap: float,
    endpoint_min_score: float = DEFAULT_ENDPOINT_MIN_SCORE,
    endpoint_max_gap_factor: float = DEFAULT_ENDPOINT_MAX_GAP_FACTOR,
    endpoint_score_margin: float = DEFAULT_ENDPOINT_SCORE_MARGIN,
) -> bool:
    y_gap = abs(float(y_centroids[neighbor_index] - y_centroids[endpoint_index]))
    x_jump = abs(float(x_centroids[neighbor_index] - x_centroids[endpoint_index]))
    max_x_jump = max(0.075, estimated_y_gap * 2.2)

    width_ratio = abs(np.log(widths[endpoint_index] / widths[neighbor_index]))
    height_ratio = abs(np.log(heights[endpoint_index] / heights[neighbor_index]))
    size_jump = float(width_ratio + height_ratio)

    score = float(scores[endpoint_index])
    low_score = score < endpoint_min_score or score < median_score - endpoint_score_margin
    too_far = y_gap > max(endpoint_max_gap_factor * estimated_y_gap, DEFAULT_MIN_Y_GAP * 1.3)
    too_shifted = x_jump > max_x_jump
    size_changed = size_jump > 0.75

    return (too_far and (low_score or too_shifted or size_changed)) or (
        low_score and (too_shifted or size_changed)
    )


def prune_endpoint_outliers(
    selected_local_indices: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    estimated_y_gap: float,
    min_detections: int,
    endpoint_min_score: float = DEFAULT_ENDPOINT_MIN_SCORE,
    endpoint_max_gap_factor: float = DEFAULT_ENDPOINT_MAX_GAP_FACTOR,
    endpoint_score_margin: float = DEFAULT_ENDPOINT_SCORE_MARGIN,
) -> tuple[np.ndarray, int, int]:
    """Remove endpoints pouco coerentes quando a sequencia ainda fica plausivel."""
    selected = np.asarray(selected_local_indices, dtype=np.int64).copy()
    if selected.size <= min_detections or selected.size < 3:
        return selected, 0, 0

    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    y_values = y_centroids_from_points(points_array)
    x_values = x_centroids_from_points(points_array)
    widths, heights = width_height_from_points(points_array)

    pruned_top = 0
    pruned_bottom = 0
    changed = True
    while changed and selected.size > min_detections and selected.size >= 3:
        changed = False
        order = np.argsort(y_values[selected])
        selected = selected[order]
        median_score = float(np.median(score_values[selected]))

        first = int(selected[0])
        second = int(selected[1])
        if endpoint_is_incoherent(
            endpoint_index=first,
            neighbor_index=second,
            scores=score_values,
            x_centroids=x_values,
            y_centroids=y_values,
            widths=widths,
            heights=heights,
            median_score=median_score,
            estimated_y_gap=estimated_y_gap,
            endpoint_min_score=endpoint_min_score,
            endpoint_max_gap_factor=endpoint_max_gap_factor,
            endpoint_score_margin=endpoint_score_margin,
        ):
            selected = selected[1:]
            pruned_top += 1
            changed = True
            continue

        last = int(selected[-1])
        before_last = int(selected[-2])
        if endpoint_is_incoherent(
            endpoint_index=last,
            neighbor_index=before_last,
            scores=score_values,
            x_centroids=x_values,
            y_centroids=y_values,
            widths=widths,
            heights=heights,
            median_score=median_score,
            estimated_y_gap=estimated_y_gap,
            endpoint_min_score=endpoint_min_score,
            endpoint_max_gap_factor=endpoint_max_gap_factor,
            endpoint_score_margin=endpoint_score_margin,
        ):
            selected = selected[:-1]
            pruned_bottom += 1
            changed = True

    return selected.astype(np.int64), int(pruned_top), int(pruned_bottom)


def nms_indices_above_threshold(
    scores: np.ndarray,
    points: np.ndarray,
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> np.ndarray:
    """Devolve indices globais apos threshold + NMS baseado nos pontos."""
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    raw_indices = np.flatnonzero(score_values >= confidence_threshold)
    if raw_indices.size == 0:
        return raw_indices.astype(np.int64)

    raw_scores = score_values[raw_indices]
    raw_points = points_array[raw_indices]
    raw_boxes_xyxy = points_to_xyxy(raw_points)
    kept_local_indices = non_max_suppression(
        boxes_xyxy=raw_boxes_xyxy,
        scores=raw_scores,
        iou_threshold=nms_iou_threshold,
    )
    return raw_indices[kept_local_indices].astype(np.int64)


def max_iou_with_selected(
    candidate_box: np.ndarray,
    selected_boxes: np.ndarray,
) -> float:
    if selected_boxes.size == 0:
        return 0.0
    return float(np.max(bbox_iou_xyxy(candidate_box, selected_boxes)))


def fill_internal_gaps(
    selected_indices: np.ndarray,
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    estimated_y_gap: float,
    min_y_gap: float,
    max_detections: int,
    nms_iou_threshold: float,
    gap_fill_min_gap_factor: float = DEFAULT_GAP_FILL_MIN_GAP_FACTOR,
    gap_fill_max_insertions: int = DEFAULT_GAP_FILL_MAX_INSERTIONS,
    gap_fill_max_x_error: float = DEFAULT_GAP_FILL_MAX_X_ERROR,
    gap_fill_max_size_log_error: float = DEFAULT_GAP_FILL_MAX_SIZE_LOG_ERROR,
    gap_fill_min_neighbor_gap_factor: float = DEFAULT_GAP_FILL_MIN_NEIGHBOR_GAP_FACTOR,
) -> tuple[np.ndarray, int, int]:
    """Insere candidatos secundarios apenas em gaps internos anatomicos."""
    selected = np.asarray(selected_indices, dtype=np.int64).copy()
    candidates = np.asarray(candidate_indices, dtype=np.int64).copy()
    if (
        selected.size < 2
        or selected.size >= max_detections
        or candidates.size == 0
        or gap_fill_max_insertions <= 0
    ):
        return selected, 0, int(candidates.size)

    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    y_values = y_centroids_from_points(points_array)
    x_values = x_centroids_from_points(points_array)
    widths, heights = width_height_from_points(points_array)
    boxes_xyxy = points_to_xyxy(points_array)

    selected_set = {int(index) for index in selected}
    available_candidates = np.asarray(
        [int(index) for index in candidates if int(index) not in selected_set],
        dtype=np.int64,
    )
    if available_candidates.size == 0:
        return selected, 0, 0

    min_neighbor_gap = max(
        float(min_y_gap) * 0.75,
        float(estimated_y_gap) * float(gap_fill_min_neighbor_gap_factor),
    )
    large_gap_threshold = max(
        float(estimated_y_gap) * float(gap_fill_min_gap_factor),
        float(min_y_gap) * 1.35,
    )
    max_overlap = max(0.22, float(nms_iou_threshold) * 1.5)

    insertions = 0
    while insertions < gap_fill_max_insertions and selected.size < max_detections:
        order = np.argsort(y_values[selected])
        selected = selected[order]
        selected_set = {int(index) for index in selected}
        selected_boxes = boxes_xyxy[selected]

        best_candidate: int | None = None
        best_objective = float("-inf")
        for left_position in range(selected.size - 1):
            left_index = int(selected[left_position])
            right_index = int(selected[left_position + 1])
            left_y = float(y_values[left_index])
            right_y = float(y_values[right_index])
            y_gap = right_y - left_y
            if y_gap < large_gap_threshold:
                continue

            missing_slots = max(1, int(round(y_gap / max(float(estimated_y_gap), EPSILON))) - 1)
            for slot in range(1, missing_slots + 1):
                fraction = slot / float(missing_slots + 1)
                target_y = left_y + y_gap * fraction
                target_x = float(x_values[left_index]) + (
                    float(x_values[right_index]) - float(x_values[left_index])
                ) * fraction
                target_width = max(
                    float(widths[left_index]) + (float(widths[right_index]) - float(widths[left_index])) * fraction,
                    EPSILON,
                )
                target_height = max(
                    float(heights[left_index]) + (float(heights[right_index]) - float(heights[left_index])) * fraction,
                    EPSILON,
                )

                for candidate in available_candidates:
                    candidate_index = int(candidate)
                    if candidate_index in selected_set:
                        continue
                    candidate_y = float(y_values[candidate_index])
                    if not (left_y + min_neighbor_gap <= candidate_y <= right_y - min_neighbor_gap):
                        continue

                    y_error = abs(candidate_y - target_y)
                    max_y_error = max(float(estimated_y_gap) * 0.65, float(min_y_gap) * 0.8)
                    if y_error > max_y_error:
                        continue

                    x_error = abs(float(x_values[candidate_index]) - target_x)
                    if x_error > gap_fill_max_x_error:
                        continue

                    size_error = abs(np.log(float(widths[candidate_index]) / target_width)) + abs(
                        np.log(float(heights[candidate_index]) / target_height)
                    )
                    if size_error > gap_fill_max_size_log_error:
                        continue

                    overlap = max_iou_with_selected(boxes_xyxy[candidate_index], selected_boxes)
                    if overlap > max_overlap:
                        continue

                    objective = (
                        logit_scores(np.asarray([score_values[candidate_index]], dtype=np.float32))[0]
                        - 1.25 * (y_error / max_y_error)
                        - 0.80 * (x_error / max(gap_fill_max_x_error, EPSILON))
                        - 0.55 * (size_error / max(gap_fill_max_size_log_error, EPSILON))
                    )
                    if objective > best_objective:
                        best_objective = float(objective)
                        best_candidate = candidate_index

        if best_candidate is None:
            break

        selected = np.append(selected, best_candidate).astype(np.int64)
        insertions += 1

    selected = selected[np.argsort(y_values[selected])]
    return selected.astype(np.int64), int(insertions), int(available_candidates.size)


def endpoint_target(
    side: str,
    selected: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    estimated_y_gap: float,
) -> tuple[int, float, float, float, float]:
    order = np.argsort(y_values[selected])
    ordered = selected[order]
    if side == "top":
        endpoint = int(ordered[0])
        neighbor = int(ordered[1])
        direction = -1.0
    elif side == "bottom":
        endpoint = int(ordered[-1])
        neighbor = int(ordered[-2])
        direction = 1.0
    else:
        raise ValueError("side deve ser 'top' ou 'bottom'.")

    target_y = float(y_values[endpoint]) + direction * float(estimated_y_gap)
    target_x = float(x_values[endpoint]) + (float(x_values[endpoint]) - float(x_values[neighbor]))
    target_width = float(widths[endpoint])
    target_height = float(heights[endpoint])
    return endpoint, target_y, target_x, target_width, target_height


def best_endpoint_extension_candidate(
    side: str,
    selected: np.ndarray,
    available_candidates: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    estimated_y_gap: float,
    min_y_gap: float,
    nms_iou_threshold: float,
    endpoint_fill_max_gap_factor: float,
    endpoint_fill_max_x_error: float,
    endpoint_fill_max_size_log_error: float,
    endpoint_fill_min_gap_factor: float,
) -> int | None:
    if selected.size < 2 or available_candidates.size == 0:
        return None

    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    y_values = y_centroids_from_points(points_array)
    x_values = x_centroids_from_points(points_array)
    widths, heights = width_height_from_points(points_array)
    boxes_xyxy = points_to_xyxy(points_array)
    selected_set = {int(index) for index in selected}
    selected_boxes = boxes_xyxy[selected]
    endpoint, target_y, target_x, target_width, target_height = endpoint_target(
        side=side,
        selected=selected,
        x_values=x_values,
        y_values=y_values,
        widths=widths,
        heights=heights,
        estimated_y_gap=estimated_y_gap,
    )

    endpoint_y = float(y_values[endpoint])
    direction = -1.0 if side == "top" else 1.0
    min_extension_gap = max(float(min_y_gap) * 0.75, float(estimated_y_gap) * float(endpoint_fill_min_gap_factor))
    max_extension_gap = max(
        float(estimated_y_gap) * float(endpoint_fill_max_gap_factor),
        float(min_y_gap) * 1.3,
    )
    max_y_error = max(float(estimated_y_gap) * 0.75, float(min_y_gap) * 0.8)
    max_overlap = max(0.22, float(nms_iou_threshold) * 1.5)

    best_candidate: int | None = None
    best_objective = float("-inf")
    for candidate in available_candidates:
        candidate_index = int(candidate)
        if candidate_index in selected_set:
            continue

        candidate_y = float(y_values[candidate_index])
        extension_gap = (endpoint_y - candidate_y) if side == "top" else (candidate_y - endpoint_y)
        if extension_gap < min_extension_gap or extension_gap > max_extension_gap:
            continue
        if direction * (candidate_y - endpoint_y) <= 0.0:
            continue

        y_error = abs(candidate_y - target_y)
        if y_error > max_y_error:
            continue

        x_error = abs(float(x_values[candidate_index]) - target_x)
        if x_error > endpoint_fill_max_x_error:
            continue

        size_error = abs(np.log(float(widths[candidate_index]) / max(target_width, EPSILON))) + abs(
            np.log(float(heights[candidate_index]) / max(target_height, EPSILON))
        )
        if size_error > endpoint_fill_max_size_log_error:
            continue

        overlap = max_iou_with_selected(boxes_xyxy[candidate_index], selected_boxes)
        if overlap > max_overlap:
            continue

        objective = (
            logit_scores(np.asarray([score_values[candidate_index]], dtype=np.float32))[0]
            - 1.15 * (y_error / max(max_y_error, EPSILON))
            - 0.85 * (x_error / max(endpoint_fill_max_x_error, EPSILON))
            - 0.60 * (size_error / max(endpoint_fill_max_size_log_error, EPSILON))
        )
        if objective > best_objective:
            best_objective = float(objective)
            best_candidate = candidate_index

    return best_candidate


def fill_endpoint_gaps(
    selected_indices: np.ndarray,
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    estimated_y_gap: float,
    min_y_gap: float,
    max_detections: int,
    nms_iou_threshold: float,
    endpoint_fill_max_per_side: int = DEFAULT_ENDPOINT_FILL_MAX_PER_SIDE,
    endpoint_fill_max_gap_factor: float = DEFAULT_ENDPOINT_FILL_MAX_GAP_FACTOR,
    endpoint_fill_max_x_error: float = DEFAULT_ENDPOINT_FILL_MAX_X_ERROR,
    endpoint_fill_max_size_log_error: float = DEFAULT_ENDPOINT_FILL_MAX_SIZE_LOG_ERROR,
    endpoint_fill_min_gap_factor: float = DEFAULT_ENDPOINT_FILL_MIN_GAP_FACTOR,
) -> tuple[np.ndarray, int, int, int]:
    """Insere candidatos apenas antes do primeiro e depois do ultimo selecionado."""
    selected = np.asarray(selected_indices, dtype=np.int64).copy()
    candidates = np.asarray(candidate_indices, dtype=np.int64).copy()
    if (
        selected.size < 2
        or selected.size >= max_detections
        or candidates.size == 0
        or endpoint_fill_max_per_side <= 0
    ):
        return selected, 0, 0, int(candidates.size)

    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))
    y_values = y_centroids_from_points(points_array)
    selected = selected[np.argsort(y_values[selected])]
    selected_set = {int(index) for index in selected}
    available_candidates = np.asarray(
        [int(index) for index in candidates if int(index) not in selected_set],
        dtype=np.int64,
    )

    filled_top = 0
    filled_bottom = 0
    for side in ("top", "bottom"):
        side_insertions = 0
        while (
            side_insertions < endpoint_fill_max_per_side
            and selected.size < max_detections
            and available_candidates.size > 0
        ):
            selected = selected[np.argsort(y_values[selected])]
            best_candidate = best_endpoint_extension_candidate(
                side=side,
                selected=selected,
                available_candidates=available_candidates,
                scores=scores,
                points=points_array,
                estimated_y_gap=estimated_y_gap,
                min_y_gap=min_y_gap,
                nms_iou_threshold=nms_iou_threshold,
                endpoint_fill_max_gap_factor=endpoint_fill_max_gap_factor,
                endpoint_fill_max_x_error=endpoint_fill_max_x_error,
                endpoint_fill_max_size_log_error=endpoint_fill_max_size_log_error,
                endpoint_fill_min_gap_factor=endpoint_fill_min_gap_factor,
            )
            if best_candidate is None:
                break
            selected = np.append(selected, best_candidate).astype(np.int64)
            available_candidates = np.asarray(
                [int(index) for index in available_candidates if int(index) != int(best_candidate)],
                dtype=np.int64,
            )
            side_insertions += 1
            if side == "top":
                filled_top += 1
            else:
                filled_bottom += 1

    selected = selected[np.argsort(y_values[selected])]
    return selected.astype(np.int64), int(filled_top), int(filled_bottom), int(available_candidates.size)


def postprocess_candidates_sequence(
    presence: np.ndarray,
    bbox: np.ndarray,
    points: np.ndarray,
    cobb_endpoint_score: np.ndarray | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    min_y_gap: float = DEFAULT_MIN_Y_GAP,
    min_vertebrae: int = DEFAULT_MIN_VERTEBRAE,
    max_vertebrae: int = DEFAULT_MAX_VERTEBRAE,
    selection_method: str = DEFAULT_SELECTION_METHOD,
    count_prior: float = DEFAULT_COUNT_PRIOR,
    count_prior_weight: float = DEFAULT_COUNT_PRIOR_WEIGHT,
    candidate_cost: float = DEFAULT_CANDIDATE_COST,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
    x_jump_weight: float = DEFAULT_X_JUMP_WEIGHT,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    angle_weight: float = DEFAULT_ANGLE_WEIGHT,
    angle_jump_tolerance_deg: float = DEFAULT_ANGLE_JUMP_TOLERANCE_DEG,
    endpoint_pruning: bool = DEFAULT_ENDPOINT_PRUNING,
    endpoint_min_score: float = DEFAULT_ENDPOINT_MIN_SCORE,
    endpoint_max_gap_factor: float = DEFAULT_ENDPOINT_MAX_GAP_FACTOR,
    endpoint_score_margin: float = DEFAULT_ENDPOINT_SCORE_MARGIN,
    endpoint_score_blend: float = DEFAULT_ENDPOINT_SCORE_BLEND,
    gap_filling: bool = DEFAULT_GAP_FILLING,
    gap_fill_threshold: float = DEFAULT_GAP_FILL_THRESHOLD,
    gap_fill_min_gap_factor: float = DEFAULT_GAP_FILL_MIN_GAP_FACTOR,
    gap_fill_max_insertions: int = DEFAULT_GAP_FILL_MAX_INSERTIONS,
    gap_fill_max_x_error: float = DEFAULT_GAP_FILL_MAX_X_ERROR,
    gap_fill_max_size_log_error: float = DEFAULT_GAP_FILL_MAX_SIZE_LOG_ERROR,
    gap_fill_min_neighbor_gap_factor: float = DEFAULT_GAP_FILL_MIN_NEIGHBOR_GAP_FACTOR,
    endpoint_filling: bool = DEFAULT_ENDPOINT_FILLING,
    endpoint_fill_threshold: float = DEFAULT_ENDPOINT_FILL_THRESHOLD,
    endpoint_fill_max_per_side: int = DEFAULT_ENDPOINT_FILL_MAX_PER_SIDE,
    endpoint_fill_max_gap_factor: float = DEFAULT_ENDPOINT_FILL_MAX_GAP_FACTOR,
    endpoint_fill_max_x_error: float = DEFAULT_ENDPOINT_FILL_MAX_X_ERROR,
    endpoint_fill_max_size_log_error: float = DEFAULT_ENDPOINT_FILL_MAX_SIZE_LOG_ERROR,
    endpoint_fill_min_gap_factor: float = DEFAULT_ENDPOINT_FILL_MIN_GAP_FACTOR,
) -> dict[str, Any]:
    """Filtra candidatos e devolve uma sequencia top-to-bottom mais estavel."""
    scores = np.asarray(presence, dtype=np.float32).reshape(-1)
    bbox_array = clip_unit(np.asarray(bbox, dtype=np.float32).reshape(-1, 4))
    points_array = clip_unit(np.asarray(points, dtype=np.float32).reshape(-1, 8))

    if not (len(scores) == len(bbox_array) == len(points_array)):
        raise ValueError("presence, bbox e points devem ter o mesmo numero de candidatos.")
    endpoint_decision_scores, endpoint_score_used, endpoint_score_blend = make_endpoint_decision_scores(
        presence_scores=scores,
        cobb_endpoint_score=cobb_endpoint_score,
        endpoint_score_blend=endpoint_score_blend,
    )

    raw_indices = np.flatnonzero(scores >= confidence_threshold)
    raw_scores = scores[raw_indices]
    raw_endpoint_scores = endpoint_decision_scores[raw_indices]
    raw_bbox = bbox_array[raw_indices]
    raw_points = points_array[raw_indices]
    raw_boxes_xyxy = points_to_xyxy(raw_points)

    empty = np.asarray([], dtype=np.float32)
    if raw_indices.size == 0:
        return {
            "raw_indices": raw_indices,
            "raw_scores": raw_scores,
            "raw_endpoint_scores": raw_endpoint_scores,
            "raw_bbox": raw_bbox,
            "raw_points": raw_points,
            "raw_boxes_xyxy": raw_boxes_xyxy,
            "nms_indices": raw_indices,
            "nms_scores": raw_scores,
            "nms_endpoint_scores": raw_endpoint_scores,
            "nms_bbox": raw_bbox,
            "nms_points": raw_points,
            "nms_boxes_xyxy": raw_boxes_xyxy,
            "selected_indices": raw_indices,
            "selected_scores": raw_scores,
            "selected_endpoint_scores": raw_endpoint_scores,
            "selected_bbox": raw_bbox,
            "selected_points": raw_points,
            "selected_boxes_xyxy": raw_boxes_xyxy,
            "selected_y_centroids": empty,
            "raw_count": 0,
            "nms_count": 0,
            "final_count": 0,
            "plausible_count": False,
            "sequence_min_y_gap": float(min_y_gap),
            "sequence_used_y_gap": float(min_y_gap),
            "sequence_method": selection_method,
            "estimated_y_gap": float(min_y_gap),
            "path_score": 0.0,
            "endpoint_pruned_top": 0,
            "endpoint_pruned_bottom": 0,
            "endpoint_score_used": bool(endpoint_score_used),
            "endpoint_score_blend": float(endpoint_score_blend),
            "gap_filled_count": 0,
            "gap_fill_candidate_count": 0,
            "gap_fill_threshold": float(gap_fill_threshold),
            "endpoint_filled_top": 0,
            "endpoint_filled_bottom": 0,
            "endpoint_fill_candidate_count": 0,
            "endpoint_fill_threshold": float(endpoint_fill_threshold),
        }

    kept_local_indices = non_max_suppression(
        boxes_xyxy=raw_boxes_xyxy,
        scores=raw_scores,
        iou_threshold=nms_iou_threshold,
    )

    nms_indices = raw_indices[kept_local_indices]
    nms_scores = scores[nms_indices]
    nms_endpoint_scores = endpoint_decision_scores[nms_indices]
    nms_bbox = bbox_array[nms_indices]
    nms_points = points_array[nms_indices]
    nms_boxes_xyxy = points_to_xyxy(nms_points)
    nms_y_centroids = y_centroids_from_points(nms_points)

    normalized_method = selection_method.strip().lower()
    if normalized_method == "greedy":
        selected_local_indices, used_gap = select_by_vertical_spacing(
            scores=nms_scores,
            y_centroids=nms_y_centroids,
            min_y_gap=min_y_gap,
            min_detections=min_vertebrae,
            max_detections=max_vertebrae,
        )
        estimated_y_gap = float(used_gap)
        path_score = 0.0
    elif normalized_method == "path":
        selected_local_indices, estimated_y_gap, path_score = select_best_anatomical_path(
            scores=nms_scores,
            points=nms_points,
            min_y_gap=min_y_gap,
            min_detections=min_vertebrae,
            max_detections=max_vertebrae,
            count_prior=count_prior,
            count_prior_weight=count_prior_weight,
            candidate_cost=candidate_cost,
            gap_weight=gap_weight,
            x_jump_weight=x_jump_weight,
            size_weight=size_weight,
            angle_weight=angle_weight,
            angle_jump_tolerance_deg=angle_jump_tolerance_deg,
        )
        used_gap = float(min_y_gap)
    else:
        raise ValueError("--selection-method deve ser 'path' ou 'greedy'.")

    selected_indices = nms_indices[selected_local_indices]

    if selected_indices.size >= 2:
        estimated_y_gap = estimate_anatomical_y_gap(
            y_centroids_from_points(points_array[selected_indices]),
            min_y_gap=min_y_gap,
            count_prior=count_prior,
        )

    gap_filled_count = 0
    gap_fill_candidate_count = 0
    if gap_filling and selected_indices.size >= 2 and selected_indices.size < max_vertebrae:
        secondary_threshold = min(float(confidence_threshold), float(gap_fill_threshold))
        secondary_nms_indices = nms_indices_above_threshold(
            scores=scores,
            points=points_array,
            confidence_threshold=secondary_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        selected_indices, gap_filled_count, gap_fill_candidate_count = fill_internal_gaps(
            selected_indices=selected_indices,
            candidate_indices=secondary_nms_indices,
            scores=scores,
            points=points_array,
            estimated_y_gap=estimated_y_gap,
            min_y_gap=min_y_gap,
            max_detections=max_vertebrae,
            nms_iou_threshold=nms_iou_threshold,
            gap_fill_min_gap_factor=gap_fill_min_gap_factor,
            gap_fill_max_insertions=gap_fill_max_insertions,
            gap_fill_max_x_error=gap_fill_max_x_error,
            gap_fill_max_size_log_error=gap_fill_max_size_log_error,
            gap_fill_min_neighbor_gap_factor=gap_fill_min_neighbor_gap_factor,
        )

    endpoint_pruned_top = 0
    endpoint_pruned_bottom = 0
    if endpoint_pruning and selected_indices.size >= 3:
        selected_indices, endpoint_pruned_top, endpoint_pruned_bottom = prune_endpoint_outliers(
            selected_local_indices=selected_indices,
            scores=endpoint_decision_scores,
            points=points_array,
            estimated_y_gap=estimated_y_gap,
            min_detections=min_vertebrae,
            endpoint_min_score=endpoint_min_score,
            endpoint_max_gap_factor=endpoint_max_gap_factor,
            endpoint_score_margin=endpoint_score_margin,
        )

    endpoint_filled_top = 0
    endpoint_filled_bottom = 0
    endpoint_fill_candidate_count = 0
    if endpoint_filling and selected_indices.size >= 2 and selected_indices.size < max_vertebrae:
        endpoint_nms_indices = nms_indices_above_threshold(
            scores=endpoint_decision_scores,
            points=points_array,
            confidence_threshold=endpoint_fill_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        selected_indices, endpoint_filled_top, endpoint_filled_bottom, endpoint_fill_candidate_count = (
            fill_endpoint_gaps(
                selected_indices=selected_indices,
                candidate_indices=endpoint_nms_indices,
                scores=endpoint_decision_scores,
                points=points_array,
                estimated_y_gap=estimated_y_gap,
                min_y_gap=min_y_gap,
                max_detections=max_vertebrae,
                nms_iou_threshold=nms_iou_threshold,
                endpoint_fill_max_per_side=endpoint_fill_max_per_side,
                endpoint_fill_max_gap_factor=endpoint_fill_max_gap_factor,
                endpoint_fill_max_x_error=endpoint_fill_max_x_error,
                endpoint_fill_max_size_log_error=endpoint_fill_max_size_log_error,
                endpoint_fill_min_gap_factor=endpoint_fill_min_gap_factor,
            )
        )

    selected_scores = scores[selected_indices]
    selected_endpoint_scores = endpoint_decision_scores[selected_indices]
    selected_bbox = bbox_array[selected_indices]
    selected_points = points_array[selected_indices]
    selected_boxes_xyxy = points_to_xyxy(selected_points)
    selected_y_centroids = y_centroids_from_points(selected_points)
    order = np.argsort(selected_y_centroids)

    selected_indices = selected_indices[order]
    selected_scores = selected_scores[order]
    selected_endpoint_scores = selected_endpoint_scores[order]
    selected_bbox = selected_bbox[order]
    selected_points = selected_points[order]
    selected_boxes_xyxy = selected_boxes_xyxy[order]
    selected_y_centroids = selected_y_centroids[order]
    final_count = int(selected_indices.size)

    return {
        "raw_indices": raw_indices,
        "raw_scores": raw_scores,
        "raw_endpoint_scores": raw_endpoint_scores,
        "raw_bbox": raw_bbox,
        "raw_points": raw_points,
        "raw_boxes_xyxy": raw_boxes_xyxy,
        "nms_indices": nms_indices,
        "nms_scores": nms_scores,
        "nms_endpoint_scores": nms_endpoint_scores,
        "nms_bbox": nms_bbox,
        "nms_points": nms_points,
        "nms_boxes_xyxy": nms_boxes_xyxy,
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
        "selected_endpoint_scores": selected_endpoint_scores,
        "selected_bbox": selected_bbox,
        "selected_points": selected_points,
        "selected_boxes_xyxy": selected_boxes_xyxy,
        "selected_y_centroids": selected_y_centroids,
        "raw_count": int(raw_indices.size),
        "nms_count": int(nms_indices.size),
        "final_count": final_count,
        "plausible_count": bool(min_vertebrae <= final_count <= max_vertebrae),
        "sequence_min_y_gap": float(min_y_gap),
        "sequence_used_y_gap": float(used_gap),
        "sequence_method": normalized_method,
        "estimated_y_gap": float(estimated_y_gap),
        "path_score": float(path_score),
        "endpoint_pruned_top": int(endpoint_pruned_top),
        "endpoint_pruned_bottom": int(endpoint_pruned_bottom),
        "endpoint_score_used": bool(endpoint_score_used),
        "endpoint_score_blend": float(endpoint_score_blend),
        "gap_filled_count": int(gap_filled_count),
        "gap_fill_candidate_count": int(gap_fill_candidate_count),
        "gap_fill_threshold": float(gap_fill_threshold),
        "endpoint_filled_top": int(endpoint_filled_top),
        "endpoint_filled_bottom": int(endpoint_filled_bottom),
        "endpoint_fill_candidate_count": int(endpoint_fill_candidate_count),
        "endpoint_fill_threshold": float(endpoint_fill_threshold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test do pos-processamento sequencial da Fase 5.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-y-gap", type=float, default=DEFAULT_MIN_Y_GAP)
    parser.add_argument("--selection-method", choices=("path", "greedy"), default=DEFAULT_SELECTION_METHOD)
    parser.add_argument("--angle-weight", type=float, default=DEFAULT_ANGLE_WEIGHT)
    parser.add_argument("--angle-jump-tolerance-deg", type=float, default=DEFAULT_ANGLE_JUMP_TOLERANCE_DEG)
    parser.add_argument("--disable-endpoint-pruning", action="store_true")
    parser.add_argument("--disable-gap-filling", action="store_true")
    parser.add_argument("--gap-fill-threshold", type=float, default=DEFAULT_GAP_FILL_THRESHOLD)
    parser.add_argument("--gap-fill-min-gap-factor", type=float, default=DEFAULT_GAP_FILL_MIN_GAP_FACTOR)
    parser.add_argument("--gap-fill-max-insertions", type=int, default=DEFAULT_GAP_FILL_MAX_INSERTIONS)
    parser.add_argument("--gap-fill-max-x-error", type=float, default=DEFAULT_GAP_FILL_MAX_X_ERROR)
    parser.add_argument("--gap-fill-max-size-log-error", type=float, default=DEFAULT_GAP_FILL_MAX_SIZE_LOG_ERROR)
    parser.add_argument("--enable-endpoint-filling", action="store_true")
    parser.add_argument("--endpoint-fill-threshold", type=float, default=DEFAULT_ENDPOINT_FILL_THRESHOLD)
    parser.add_argument("--endpoint-fill-max-per-side", type=int, default=DEFAULT_ENDPOINT_FILL_MAX_PER_SIDE)
    parser.add_argument("--endpoint-fill-max-gap-factor", type=float, default=DEFAULT_ENDPOINT_FILL_MAX_GAP_FACTOR)
    parser.add_argument("--endpoint-fill-max-x-error", type=float, default=DEFAULT_ENDPOINT_FILL_MAX_X_ERROR)
    parser.add_argument(
        "--endpoint-fill-max-size-log-error",
        type=float,
        default=DEFAULT_ENDPOINT_FILL_MAX_SIZE_LOG_ERROR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presence = np.asarray([0.95, 0.88, 0.91, 0.62, 0.90, 0.30], dtype=np.float32)
    bbox = np.zeros((6, 4), dtype=np.float32)
    points = np.asarray(
        [
            [0.40, 0.20, 0.50, 0.20, 0.40, 0.25, 0.50, 0.25],
            [0.405, 0.205, 0.505, 0.205, 0.405, 0.255, 0.505, 0.255],
            [0.42, 0.35, 0.52, 0.35, 0.42, 0.40, 0.52, 0.40],
            [0.43, 0.47, 0.53, 0.47, 0.43, 0.52, 0.53, 0.52],
            [0.45, 0.59, 0.55, 0.59, 0.45, 0.64, 0.55, 0.64],
            [0.60, 0.70, 0.68, 0.70, 0.60, 0.75, 0.68, 0.75],
        ],
        dtype=np.float32,
    )
    result = postprocess_candidates_sequence(
        presence=presence,
        bbox=bbox,
        points=points,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        min_y_gap=args.min_y_gap,
        min_vertebrae=1,
        max_vertebrae=21,
        selection_method=args.selection_method,
        angle_weight=args.angle_weight,
        angle_jump_tolerance_deg=args.angle_jump_tolerance_deg,
        endpoint_pruning=not args.disable_endpoint_pruning,
        gap_filling=not args.disable_gap_filling,
        gap_fill_threshold=args.gap_fill_threshold,
        gap_fill_min_gap_factor=args.gap_fill_min_gap_factor,
        gap_fill_max_insertions=args.gap_fill_max_insertions,
        gap_fill_max_x_error=args.gap_fill_max_x_error,
        gap_fill_max_size_log_error=args.gap_fill_max_size_log_error,
        endpoint_filling=args.enable_endpoint_filling,
        endpoint_fill_threshold=args.endpoint_fill_threshold,
        endpoint_fill_max_per_side=args.endpoint_fill_max_per_side,
        endpoint_fill_max_gap_factor=args.endpoint_fill_max_gap_factor,
        endpoint_fill_max_x_error=args.endpoint_fill_max_x_error,
        endpoint_fill_max_size_log_error=args.endpoint_fill_max_size_log_error,
    )
    print("Smoke pos-processamento sequencial Fase 5 v3")
    print(f"candidatos acima do threshold: {result['raw_count']}")
    print(f"candidatos apos NMS: {result['nms_count']}")
    print(f"candidatos finais: {result['final_count']}")
    print(f"indices selecionados: {result['selected_indices'].tolist()}")
    print(f"gap y usado: {result['sequence_used_y_gap']:.4f}")
    print(f"metodo: {result['sequence_method']}")
    print(f"endpoint pruning topo/fundo: {result['endpoint_pruned_top']}/{result['endpoint_pruned_bottom']}")
    print(f"gap filling inseridos/candidatos: {result['gap_filled_count']}/{result['gap_fill_candidate_count']}")
    print(
        "endpoint filling topo/fundo/candidatos: "
        f"{result['endpoint_filled_top']}/{result['endpoint_filled_bottom']}/"
        f"{result['endpoint_fill_candidate_count']}"
    )


if __name__ == "__main__":
    main()
