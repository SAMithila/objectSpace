"""
Detection module for Smart Desk Monitor.

This module provides object detection capabilities with a pluggable
architecture supporting different detection backends.

Available Detectors:
    - MaskRCNNDetector: Instance segmentation using Mask R-CNN

Example:
    >>> from smart_desk_monitor.detection import MaskRCNNDetector
    >>> detector = MaskRCNNDetector()
    >>> result = detector.detect(rgb_frame)
"""

from .base import BaseDetector, Detection, DetectionResult
from .mask_rcnn import MaskRCNNDetector

__all__ = [
    "BaseDetector",
    "Detection",
    "DetectionResult",
    "MaskRCNNDetector",
]
