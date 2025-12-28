"""
SORT (Simple Online and Realtime Tracking) implementation.

This module implements the SORT algorithm for multi-object tracking,
using Kalman filtering for state estimation and Hungarian algorithm
for data association.

Reference:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .kalman import KalmanBoxTracker
from .association import associate_detections_to_tracks
from ..config import TrackerConfig

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Lifecycle state of a track."""
    TENTATIVE = auto()  # Not yet confirmed (< min_hits)
    CONFIRMED = auto()  # Actively tracked
    DELETED = auto()    # Marked for removal


@dataclass
class Track:
    """
    Represents a tracked object.

    Attributes:
        track_id: Unique identifier for this track
        class_id: COCO class ID of the tracked object
        kalman: Kalman filter for state estimation
        state: Current lifecycle state
        history: List of historical bounding boxes
    """
    track_id: int
    class_id: int
    kalman: KalmanBoxTracker
    state: TrackState = TrackState.TENTATIVE
    history: List[np.ndarray] = field(default_factory=list)

    @property
    def bbox(self) -> np.ndarray:
        """Current bounding box estimate."""
        return self.kalman.get_bbox()

    @property
    def time_since_update(self) -> int:
        """Frames since last detection match."""
        return self.kalman.time_since_update

    @property
    def hits(self) -> int:
        """Number of successful updates."""
        return self.kalman.hits

    @property
    def age(self) -> int:
        """Total frames this track has existed."""
        return self.kalman.age


@dataclass
class TrackingResult:
    """
    Container for tracking results from a single frame.

    Attributes:
        tracks: Dictionary mapping class_id to list of (track_id, bbox) tuples
        frame_idx: Index of the processed frame
    """
    tracks: Dict[int, List[Tuple[int, np.ndarray]]]
    frame_idx: int

    def get_all_tracks(self) -> List[Tuple[int, int, np.ndarray]]:
        """
        Get all tracks as flat list.

        Returns:
            List of (class_id, track_id, bbox) tuples
        """
        result = []
        for class_id, track_list in self.tracks.items():
            for track_id, bbox in track_list:
                result.append((class_id, track_id, bbox))
        return result


class SORTTracker:
    """
    SORT multi-object tracker with per-class tracking.

    This tracker maintains separate tracking state for each object class,
    allowing different objects to have the same track ID if they belong
    to different classes.

    Args:
        config: Tracker configuration

    Example:
        >>> config = TrackerConfig(max_age=8, iou_threshold=0.3)
        >>> tracker = SORTTracker(config)
        >>> 
        >>> for frame_idx, detections in enumerate(all_detections):
        ...     boxes, class_ids, _ = detections.to_numpy()
        ...     result = tracker.update(boxes, class_ids, frame_idx)
        ...     for class_id, track_id, bbox in result.get_all_tracks():
        ...         print(f"Track {track_id} ({class_id}): {bbox}")
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """Initialize the SORT tracker."""
        self._config = config or TrackerConfig()
        self._tracks: Dict[int, List[Track]] = {}  # class_id -> list of tracks
        self._next_id: Dict[int, int] = {}  # class_id -> next track ID
        self._frame_count = 0

        logger.info(
            f"Initialized SORTTracker (max_age={self._config.max_age}, "
            f"min_hits={self._config.min_hits}, "
            f"iou_threshold={self._config.iou_threshold})"
        )

    def _get_next_id(self, class_id: int) -> int:
        """Get next available track ID for a class."""
        if class_id not in self._next_id:
            self._next_id[class_id] = 0

        track_id = self._next_id[class_id]
        self._next_id[class_id] += 1
        return track_id

    def _create_track(self, bbox: np.ndarray, class_id: int) -> Track:
        """Create a new track from a detection."""
        track_id = self._get_next_id(class_id)

        kalman = KalmanBoxTracker(
            bbox=bbox,
            process_noise=self._config.process_noise,
            measurement_noise=self._config.measurement_noise
        )

        track = Track(
            track_id=track_id,
            class_id=class_id,
            kalman=kalman,
            state=TrackState.TENTATIVE,
            history=[bbox.copy()]
        )

        logger.debug(f"Created track {track_id} for class {class_id}")
        return track

    def _predict_tracks(self, class_id: int) -> np.ndarray:
        """
        Run prediction step for all tracks of a class.

        Returns:
            Predicted bounding boxes, shape (N, 4)
        """
        if class_id not in self._tracks:
            return np.empty((0, 4), dtype=np.float32)

        predictions = []
        for track in self._tracks[class_id]:
            pred_bbox = track.kalman.predict()
            predictions.append(pred_bbox)

        if not predictions:
            return np.empty((0, 4), dtype=np.float32)

        return np.array(predictions, dtype=np.float32)

    def _update_class(
        self,
        detections: np.ndarray,
        class_id: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracking for a single class.

        Args:
            detections: Detection boxes for this class, shape (N, 4)
            class_id: COCO class ID

        Returns:
            List of (track_id, bbox) for confirmed tracks
        """
        if class_id not in self._tracks:
            self._tracks[class_id] = []

        # Get active tracks (not deleted, within max_age)
        active_tracks = [
            t for t in self._tracks[class_id]
            if t.state != TrackState.DELETED
            and t.time_since_update < self._config.max_age
        ]

        # Predict positions
        if active_tracks:
            predicted_boxes = np.array(
                [t.bbox for t in active_tracks],
                dtype=np.float32
            )
        else:
            predicted_boxes = np.empty((0, 4), dtype=np.float32)

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            detections=detections,
            tracks=predicted_boxes,
            iou_threshold=self._config.iou_threshold
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            det_bbox = detections[det_idx]
            track.kalman.update(det_bbox)
            track.history.append(det_bbox.copy())

            # Promote to confirmed if enough hits
            if track.hits >= self._config.min_hits:
                track.state = TrackState.CONFIRMED

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = self._create_track(detections[det_idx], class_id)
            self._tracks[class_id].append(new_track)

        # Mark old tracks as deleted
        for track in self._tracks[class_id]:
            if track.time_since_update >= self._config.max_age:
                track.state = TrackState.DELETED

        # Clean up deleted tracks
        self._tracks[class_id] = [
            t for t in self._tracks[class_id]
            if t.state != TrackState.DELETED
        ]

        # Return confirmed tracks
        results = []
        for track in self._tracks[class_id]:
            if track.state == TrackState.CONFIRMED and track.time_since_update == 0:
                results.append((track.track_id, track.bbox.copy()))

        return results

    def update(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        frame_idx: int
    ) -> TrackingResult:
        """
        Update tracker with new detections.

        Args:
            boxes: Detection bounding boxes, shape (N, 4)
            class_ids: Class ID for each detection, shape (N,)
            frame_idx: Current frame index

        Returns:
            TrackingResult containing all confirmed tracks
        """
        self._frame_count = frame_idx

        # Predict all existing tracks
        for class_id in self._tracks:
            self._predict_tracks(class_id)

        # Group detections by class
        results: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        unique_classes = np.unique(class_ids)

        for class_id in unique_classes:
            class_id = int(class_id)
            mask = class_ids == class_id
            class_boxes = boxes[mask]

            class_results = self._update_class(class_boxes, class_id)
            if class_results:
                results[class_id] = class_results

        return TrackingResult(tracks=results, frame_idx=frame_idx)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_id.clear()
        self._frame_count = 0
        logger.info("Tracker reset")

    def get_track_count(self) -> Dict[int, int]:
        """Get number of active tracks per class."""
        return {
            class_id: len([t for t in tracks if t.state != TrackState.DELETED])
            for class_id, tracks in self._tracks.items()
        }
