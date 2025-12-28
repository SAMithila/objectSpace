"""
Abstract base class for object detectors.

This module defines the interface that all detector implementations must follow,
enabling easy swapping between different detection backends (Mask R-CNN, YOLO, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """
    Represents a single object detection.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates
        class_id: COCO class ID
        confidence: Detection confidence score [0, 1]
        mask: Optional segmentation mask (H, W) binary array
    """
    bbox: np.ndarray  # Shape: (4,) - [x1, y1, x2, y2]
    class_id: int
    confidence: float
    mask: Optional[np.ndarray] = None  # Shape: (H, W) if present

    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        """Center point of bounding box as [cx, cy]."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        return self.width / max(self.height, 1e-6)


@dataclass
class DetectionResult:
    """
    Container for all detections in a single frame.

    Attributes:
        detections: List of Detection objects
        frame_shape: Original frame shape as (H, W, C)
    """
    detections: List[Detection]
    frame_shape: tuple

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def filter_by_class(self, class_ids: List[int]) -> "DetectionResult":
        """Return new result containing only specified classes."""
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return DetectionResult(detections=filtered, frame_shape=self.frame_shape)

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Return new result containing only detections above threshold."""
        filtered = [
            d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(detections=filtered, frame_shape=self.frame_shape)

    def to_numpy(self) -> tuple:
        """
        Convert to numpy arrays for batch processing.

        Returns:
            boxes: (N, 4) array of bounding boxes
            class_ids: (N,) array of class IDs
            confidences: (N,) array of confidence scores
        """
        if not self.detections:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32)
            )

        boxes = np.array([d.bbox for d in self.detections], dtype=np.float32)
        class_ids = np.array(
            [d.class_id for d in self.detections], dtype=np.int32)
        confidences = np.array(
            [d.confidence for d in self.detections], dtype=np.float32)

        return boxes, class_ids, confidences


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    All detector implementations (Mask R-CNN, YOLO, etc.) must inherit
    from this class and implement the required methods.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: RGB image as numpy array, shape (H, W, 3)

        Returns:
            DetectionResult containing all detections
        """
        pass

    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of RGB images

        Returns:
            List of DetectionResult, one per frame
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device being used (cuda/cpu)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
