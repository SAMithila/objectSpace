"""
Association utilities for multi-object tracking.

This module provides functions for computing similarity metrics
and solving the assignment problem between detections and tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box_a: First box [x1, y1, x2, y2]
        box_b: Second box [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1]
    """
    # Intersection coordinates
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / max(union_area, 1e-8)


def compute_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    Vectorized implementation for efficiency with large numbers of boxes.

    Args:
        boxes_a: First set of boxes, shape (N, 4)
        boxes_b: Second set of boxes, shape (M, 4)

    Returns:
        IoU matrix of shape (N, M)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Compute areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Compute intersection coordinates
    # boxes_a[:, None, :2] has shape (N, 1, 2)
    # boxes_b[:, :2] has shape (M, 2)
    # Result has shape (N, M, 2)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    # Intersection dimensions (N, M, 2)
    wh = np.maximum(0, rb - lt)

    # Intersection area (N, M)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union area (N, M)
    union = area_a[:, None] + area_b - inter

    return inter / np.maximum(union, 1e-8)


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float = 0.7
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve linear assignment problem using Hungarian algorithm.

    Args:
        cost_matrix: Cost matrix of shape (N, M) where N is number of 
                     existing tracks and M is number of new detections.
                     Lower cost = better match.
        threshold: Maximum cost to accept a match. Matches with cost
                   above threshold are rejected.

    Returns:
        matches: List of (track_idx, detection_idx) tuples
        unmatched_tracks: List of track indices without matches
        unmatched_detections: List of detection indices without matches
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1]))
        )

    # Solve assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    matches = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_detections = set(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= threshold:
            matches.append((row, col))
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)

    return matches, list(unmatched_tracks), list(unmatched_detections)


def associate_detections_to_tracks(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate detections to existing tracks based on IoU.

    This is the main function called by the tracker to determine
    which detections correspond to which existing tracks.

    Args:
        detections: Detection bounding boxes, shape (N, 4)
        tracks: Predicted track bounding boxes, shape (M, 4)
        iou_threshold: Minimum IoU for valid association

    Returns:
        matches: List of (track_idx, detection_idx) tuples
        unmatched_tracks: Track indices without matches
        unmatched_detections: Detection indices without matches
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    # Compute IoU matrix
    iou_matrix = compute_iou_batch(tracks, detections)

    # Convert to cost matrix (lower = better)
    cost_matrix = 1 - iou_matrix

    # Solve assignment with IoU threshold
    # Cost threshold = 1 - iou_threshold
    return linear_assignment(cost_matrix, threshold=1 - iou_threshold)
