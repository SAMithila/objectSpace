"""
Kalman filter implementation for object tracking.

This module provides an 8-dimensional Kalman filter optimized for
bounding box tracking, using center position, area, and aspect ratio
as the state representation.
"""

import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalman
from dataclasses import dataclass
from typing import Tuple


@dataclass
class KalmanState:
    """
    Represents the state of a Kalman filter.

    State vector: [cx, cy, area, ratio, vx, vy, v_area, v_ratio]
    where:
        - cx, cy: center position
        - area: bounding box area
        - ratio: aspect ratio (width/height)
        - vx, vy, v_area, v_ratio: velocities of the above
    """
    center: np.ndarray  # [cx, cy]
    area: float
    aspect_ratio: float
    velocity: np.ndarray  # [vx, vy, v_area, v_ratio]

    def to_bbox(self) -> np.ndarray:
        """
        Convert state to bounding box [x1, y1, x2, y2].

        Returns:
            Bounding box coordinates as numpy array
        """
        cx, cy = self.center
        w = np.sqrt(self.area * self.aspect_ratio)
        h = self.area / max(w, 1e-6)

        return np.array([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ])


class KalmanBoxTracker:
    """
    8D Kalman filter for tracking bounding boxes.

    Uses a constant velocity model with state:
    [cx, cy, area, ratio, vx, vy, v_area, v_ratio]

    Observations are [cx, cy, area, ratio].

    Args:
        bbox: Initial bounding box [x1, y1, x2, y2]
        process_noise: Process noise multiplier (default: 0.1)
        measurement_noise: Measurement noise multiplier (default: 5.0)

    Example:
        >>> tracker = KalmanBoxTracker([100, 100, 200, 200])
        >>> tracker.predict()
        >>> tracker.update([105, 102, 205, 203])
        >>> state = tracker.get_state()
    """

    def __init__(
        self,
        bbox: np.ndarray,
        process_noise: float = 0.1,
        measurement_noise: float = 5.0
    ):
        self._kf = self._create_filter(process_noise, measurement_noise)
        self._initialize_state(bbox)
        self._time_since_update = 0
        self._hits = 1
        self._age = 0

    def _create_filter(
        self,
        process_noise: float,
        measurement_noise: float
    ) -> FilterPyKalman:
        """Create and configure the Kalman filter."""
        kf = FilterPyKalman(dim_x=8, dim_z=4)
        dt = 1.0

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=np.float64)

        # Measurement matrix (observe position, area, ratio)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance
        kf.R *= measurement_noise

        # Process noise covariance
        kf.Q = np.eye(8, dtype=np.float64) * process_noise

        # Initial state covariance (high uncertainty for velocities)
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        return kf

    def _initialize_state(self, bbox: np.ndarray) -> None:
        """Initialize state from bounding box."""
        measurement = self._bbox_to_measurement(bbox)
        self._kf.x = np.zeros((8, 1), dtype=np.float64)
        self._kf.x[:4, 0] = measurement

    @staticmethod
    def _bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box to measurement vector.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Measurement [cx, cy, area, ratio]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        cx = x1 + w / 2
        cy = y1 + h / 2
        area = w * h
        ratio = w / max(h, 1e-6)

        return np.array([cx, cy, area, ratio])

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        """
        Convert state vector to bounding box.

        Args:
            state: [cx, cy, area, ratio, ...]

        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        cx, cy, area, ratio = state[:4]

        # Ensure positive values
        area = max(area, 1e-6)
        ratio = max(ratio, 1e-6)

        w = np.sqrt(area * ratio)
        h = area / max(w, 1e-6)

        return np.array([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ])

    def predict(self) -> np.ndarray:
        """
        Advance state prediction by one time step.

        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Handle potential negative area
        if self._kf.x[2, 0] + self._kf.x[6, 0] <= 0:
            self._kf.x[6, 0] = 0.0

        self._kf.predict()
        self._age += 1

        if self._time_since_update > 0:
            self._hits = 0

        self._time_since_update += 1

        return self._state_to_bbox(self._kf.x[:, 0])

    def update(self, bbox: np.ndarray) -> None:
        """
        Update state with new observation.

        Args:
            bbox: Observed bounding box [x1, y1, x2, y2]
        """
        self._time_since_update = 0
        self._hits += 1

        measurement = self._bbox_to_measurement(bbox)
        self._kf.update(measurement.reshape(4, 1))

    def get_state(self) -> KalmanState:
        """Get current state as KalmanState object."""
        state = self._kf.x[:, 0]
        return KalmanState(
            center=state[:2].copy(),
            area=float(state[2]),
            aspect_ratio=float(state[3]),
            velocity=state[4:].copy()
        )

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return self._state_to_bbox(self._kf.x[:, 0])

    @property
    def time_since_update(self) -> int:
        """Frames since last successful update."""
        return self._time_since_update

    @property
    def hits(self) -> int:
        """Number of successful updates."""
        return self._hits

    @property
    def age(self) -> int:
        """Total frames this tracker has existed."""
        return self._age
