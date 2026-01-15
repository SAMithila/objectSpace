"""
Centralized configuration management for Smart Desk Monitor.

This module provides typed, validated configuration using dataclasses.
Configuration can be loaded from YAML files or constructed programmatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# =============================================================================
# COCO Class Definitions
# =============================================================================

COCO_CLASSES: Dict[int, str] = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter',
    14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
    25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
    30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
    35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard',
    39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork',
    44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple',
    49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog',
    54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch',
    59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv',
    64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone',
    69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator',
    74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear',
    79: 'hair drier', 80: 'toothbrush'
}


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DetectorConfig:
    """Configuration for object detection."""

    model_name: str = "maskrcnn_resnet50_fpn"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    default_confidence: float = 0.3

    # Per-class confidence thresholds (class_id -> threshold)
    class_thresholds: Dict[int, float] = field(default_factory=lambda: {
        1: 0.6,   # person - higher threshold to reduce false positives
        64: 0.2,  # laptop
        46: 0.2,  # bowl
        42: 0.2,  # cup
        40: 0.3,  # bottle
    })

    def get_threshold(self, class_id: int) -> float:
        """Get confidence threshold for a specific class."""
        return self.class_thresholds.get(class_id, self.default_confidence)


@dataclass
class TrackerConfig:
    """Configuration for SORT tracker."""

    max_age: int = 8  # Frames to keep track alive without detection
    min_hits: int = 3  # Minimum detections before track is confirmed
    iou_threshold: float = 0.3  # Minimum IoU for association

    # Kalman filter parameters
    process_noise: float = 0.1
    measurement_noise: float = 5.0


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    target_width: int = 1920
    target_height: int = 1080
    max_frames: int = 100  # Maximum frames to sample per video
    supported_formats: Tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov")


@dataclass
class OutputConfig:
    """Configuration for output generation."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_tracked_frames: bool = True
    save_coco_json: bool = True
    visualization_line_thickness: int = 2
    visualization_font_scale: float = 0.6


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            detector=DetectorConfig(**data.get('detector', {})),
            tracker=TrackerConfig(**data.get('tracker', {})),
            video=VideoConfig(**data.get('video', {})),
            output=OutputConfig(
                output_dir=Path(data.get('output', {}).get(
                    'output_dir', 'output')),
                **{k: v for k, v in data.get('output', {}).items() if k != 'output_dir'}
            ),
            log_level=data.get('log_level', 'INFO'),
            log_file=Path(data['log_file']) if data.get('log_file') else None,
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            'detector': {
                'model_name': self.detector.model_name,
                'device': self.detector.device,
                'default_confidence': self.detector.default_confidence,
                'class_thresholds': self.detector.class_thresholds,
            },
            'tracker': {
                'max_age': self.tracker.max_age,
                'min_hits': self.tracker.min_hits,
                'iou_threshold': self.tracker.iou_threshold,
                'process_noise': self.tracker.process_noise,
                'measurement_noise': self.tracker.measurement_noise,
            },
            'video': {
                'target_width': self.video.target_width,
                'target_height': self.video.target_height,
                'max_frames': self.video.max_frames,
                'supported_formats': list(self.video.supported_formats),
            },
            'output': {
                'output_dir': str(self.output.output_dir),
                'save_tracked_frames': self.output.save_tracked_frames,
                'save_coco_json': self.output.save_coco_json,
                'visualization_line_thickness': self.output.visualization_line_thickness,
                'visualization_font_scale': self.output.visualization_font_scale,
            },
            'log_level': self.log_level,
            'log_file': str(self.log_file) if self.log_file else None,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Default Configuration Factory
# =============================================================================

def get_default_config() -> PipelineConfig:
    """Create default configuration suitable for most use cases."""
    return PipelineConfig()
