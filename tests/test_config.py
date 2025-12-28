"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from src.config import (
    PipelineConfig,
    DetectorConfig,
    TrackerConfig,
    VideoConfig,
    OutputConfig,
    COCO_CLASSES,
)


class TestDetectorConfig:
    """Tests for DetectorConfig."""
    
    def test_default_values(self):
        config = DetectorConfig()
        assert config.model_name == "maskrcnn_resnet50_fpn"
        assert config.device == "cuda"
        assert config.default_confidence == 0.3
    
    def test_custom_thresholds(self):
        config = DetectorConfig(class_thresholds={1: 0.8, 2: 0.5})
        assert config.class_thresholds[1] == 0.8
        assert config.class_thresholds[2] == 0.5
    
    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            DetectorConfig(default_confidence=1.5)
        with pytest.raises(ValueError):
            DetectorConfig(default_confidence=-0.1)


class TestTrackerConfig:
    """Tests for TrackerConfig."""
    
    def test_default_values(self):
        config = TrackerConfig()
        assert config.iou_threshold == 0.3
        assert config.max_age == 8
        assert config.min_hits == 3
    
    def test_invalid_max_age(self):
        with pytest.raises(ValueError):
            TrackerConfig(max_age=0)


class TestVideoConfig:
    """Tests for VideoConfig."""
    
    def test_default_resolution(self):
        config = VideoConfig()
        assert config.target_width == 1920
        assert config.target_height == 1080
    
    def test_supported_formats(self):
        config = VideoConfig()
        assert ".mp4" in config.supported_formats


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_nested_configs(self):
        config = PipelineConfig()
        assert isinstance(config.detector, DetectorConfig)
        assert isinstance(config.tracker, TrackerConfig)
        assert isinstance(config.video, VideoConfig)
        assert isinstance(config.output, OutputConfig)
    
    def test_yaml_roundtrip(self):
        config = PipelineConfig(
            detector=DetectorConfig(default_confidence=0.5),
            tracker=TrackerConfig(max_age=10),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)
            
            loaded = PipelineConfig.from_yaml(yaml_path)
            assert loaded.detector.default_confidence == 0.5
            assert loaded.tracker.max_age == 10


class TestCOCOClasses:
    """Tests for COCO class definitions."""
    
    def test_has_common_classes(self):
        assert 1 in COCO_CLASSES  # person
        assert COCO_CLASSES[1] == "person"
        assert 64 in COCO_CLASSES  # laptop
        assert 65 in COCO_CLASSES  # mouse
    
    def test_class_count(self):
        assert len(COCO_CLASSES) == 81  # 80 classes + background
