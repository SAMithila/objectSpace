"""
Tracking module for Smart Desk Monitor.

This module provides multi-object tracking using the SORT algorithm
with Kalman filtering for motion prediction.

Example:
    >>> from smart_desk_monitor.tracking import SORTTracker
    >>> tracker = SORTTracker()
    >>> result = tracker.update(boxes, class_ids, frame_idx=0)
"""

from .kalman import KalmanBoxTracker, KalmanState
from .association import (
    compute_iou,
    compute_iou_batch,
    linear_assignment,
    associate_detections_to_tracks,
)
from .sort_tracker import SORTTracker, Track, TrackState, TrackingResult

__all__ = [
    # Kalman filter
    "KalmanBoxTracker",
    "KalmanState",
    # Association
    "compute_iou",
    "compute_iou_batch",
    "linear_assignment",
    "associate_detections_to_tracks",
    # SORT tracker
    "SORTTracker",
    "Track",
    "TrackState",
    "TrackingResult",
]
