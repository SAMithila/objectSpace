"""
Smart Desk Monitor - Object Detection and Tracking Pipeline.

A production-quality video analysis system for detecting and tracking
objects in desk/workspace monitoring scenarios.

Example:
    >>> from smart_desk_monitor import DetectionTrackingPipeline
    >>> pipeline = DetectionTrackingPipeline()
    >>> results = pipeline.process_video("video.mp4")

For quick usage:
    >>> from smart_desk_monitor import run_pipeline
    >>> results = run_pipeline("videos/", output_dir="output/")
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import (
    PipelineConfig,
    DetectorConfig,
    TrackerConfig,
    VideoConfig,
    OutputConfig,
    COCO_CLASSES,
    get_default_config,
)
from .pipeline import DetectionTrackingPipeline, run_pipeline

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Configuration
    "PipelineConfig",
    "DetectorConfig",
    "TrackerConfig",
    "VideoConfig",
    "OutputConfig",
    "COCO_CLASSES",
    "get_default_config",
    # Pipeline
    "DetectionTrackingPipeline",
    "run_pipeline",
]
