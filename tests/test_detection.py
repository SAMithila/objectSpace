"""
Unit tests for detection module.
"""

import numpy as np
import pytest

from smart_desk_monitor.detection import Detection, DetectionResult


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_properties(self):
        det = Detection(
            bbox=np.array([100, 100, 200, 150]),
            class_id=1,
            confidence=0.9
        )
        
        assert det.width == 100
        assert det.height == 50
        assert det.area == 5000
        np.testing.assert_array_equal(det.center, [150, 125])
        assert det.aspect_ratio == pytest.approx(2.0)
    
    def test_optional_mask(self):
        det = Detection(
            bbox=np.array([0, 0, 100, 100]),
            class_id=1,
            confidence=0.8,
            mask=np.ones((100, 100), dtype=np.uint8)
        )
        
        assert det.mask is not None
        assert det.mask.shape == (100, 100)


class TestDetectionResult:
    """Tests for DetectionResult container."""
    
    def test_empty_result(self):
        result = DetectionResult(detections=[], frame_shape=(480, 640, 3))
        
        assert len(result) == 0
        boxes, class_ids, confidences = result.to_numpy()
        assert boxes.shape == (0, 4)
        assert class_ids.shape == (0,)
    
    def test_filter_by_class(self):
        detections = [
            Detection(np.array([0, 0, 50, 50]), class_id=1, confidence=0.9),
            Detection(np.array([100, 100, 150, 150]), class_id=2, confidence=0.8),
            Detection(np.array([200, 200, 250, 250]), class_id=1, confidence=0.7),
        ]
        result = DetectionResult(detections=detections, frame_shape=(480, 640, 3))
        
        filtered = result.filter_by_class([1])
        assert len(filtered) == 2
        
        for det in filtered:
            assert det.class_id == 1
    
    def test_filter_by_confidence(self):
        detections = [
            Detection(np.array([0, 0, 50, 50]), class_id=1, confidence=0.9),
            Detection(np.array([100, 100, 150, 150]), class_id=1, confidence=0.5),
            Detection(np.array([200, 200, 250, 250]), class_id=1, confidence=0.3),
        ]
        result = DetectionResult(detections=detections, frame_shape=(480, 640, 3))
        
        filtered = result.filter_by_confidence(0.6)
        assert len(filtered) == 1
        assert filtered.detections[0].confidence == 0.9
    
    def test_to_numpy(self):
        detections = [
            Detection(np.array([0, 0, 50, 50]), class_id=1, confidence=0.9),
            Detection(np.array([100, 100, 150, 150]), class_id=2, confidence=0.8),
        ]
        result = DetectionResult(detections=detections, frame_shape=(480, 640, 3))
        
        boxes, class_ids, confidences = result.to_numpy()
        
        assert boxes.shape == (2, 4)
        assert class_ids.shape == (2,)
        assert confidences.shape == (2,)
        
        np.testing.assert_array_equal(class_ids, [1, 2])
        np.testing.assert_array_almost_equal(confidences, [0.9, 0.8])
