"""
Unit tests for tracking module.
"""

import numpy as np
import pytest

from smart_desk_monitor.tracking import (
    KalmanBoxTracker,
    compute_iou,
    compute_iou_batch,
    associate_detections_to_tracks,
    SORTTracker,
    TrackState,
)
from smart_desk_monitor.config import TrackerConfig


class TestIoU:
    """Tests for IoU computation."""
    
    def test_identical_boxes(self):
        box = np.array([0, 0, 100, 100])
        assert compute_iou(box, box) == pytest.approx(1.0)
    
    def test_no_overlap(self):
        box_a = np.array([0, 0, 50, 50])
        box_b = np.array([100, 100, 150, 150])
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)
    
    def test_partial_overlap(self):
        box_a = np.array([0, 0, 100, 100])
        box_b = np.array([50, 50, 150, 150])
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert compute_iou(box_a, box_b) == pytest.approx(expected)
    
    def test_batch_iou_shape(self):
        boxes_a = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
        boxes_b = np.array([[0, 0, 100, 100], [100, 100, 200, 200], [0, 0, 50, 50]])
        
        iou_matrix = compute_iou_batch(boxes_a, boxes_b)
        assert iou_matrix.shape == (2, 3)
    
    def test_batch_iou_empty(self):
        boxes_a = np.empty((0, 4))
        boxes_b = np.array([[0, 0, 100, 100]])
        
        iou_matrix = compute_iou_batch(boxes_a, boxes_b)
        assert iou_matrix.shape == (0, 1)


class TestAssociation:
    """Tests for detection-to-track association."""
    
    def test_perfect_match(self):
        detections = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        tracks = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            detections, tracks, iou_threshold=0.5
        )
        
        assert len(matches) == 2
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0
    
    def test_no_match(self):
        detections = np.array([[0, 0, 50, 50]])
        tracks = np.array([[500, 500, 600, 600]])
        
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            detections, tracks, iou_threshold=0.3
        )
        
        assert len(matches) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_dets) == 1
    
    def test_empty_detections(self):
        detections = np.empty((0, 4))
        tracks = np.array([[0, 0, 100, 100]])
        
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            detections, tracks, iou_threshold=0.3
        )
        
        assert len(matches) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_dets) == 0


class TestKalmanBoxTracker:
    """Tests for Kalman filter tracker."""
    
    def test_initialization(self):
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        result_bbox = tracker.get_bbox()
        np.testing.assert_array_almost_equal(result_bbox, bbox, decimal=1)
    
    def test_predict(self):
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        # Predict without update (should stay roughly in place)
        predicted = tracker.predict()
        assert predicted.shape == (4,)
    
    def test_update(self):
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        # Move the box
        new_bbox = np.array([110, 110, 210, 210])
        tracker.predict()
        tracker.update(new_bbox)
        
        result = tracker.get_bbox()
        # Should be close to the new observation
        assert abs(result[0] - new_bbox[0]) < 20
    
    def test_time_since_update(self):
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        assert tracker.time_since_update == 0
        
        tracker.predict()
        assert tracker.time_since_update == 1
        
        tracker.predict()
        assert tracker.time_since_update == 2
        
        tracker.update(bbox)
        assert tracker.time_since_update == 0


class TestSORTTracker:
    """Tests for SORT multi-object tracker."""
    
    def test_initialization(self):
        config = TrackerConfig()
        tracker = SORTTracker(config)
        assert tracker.get_track_count() == {}
    
    def test_single_detection(self):
        tracker = SORTTracker(TrackerConfig(min_hits=1))
        
        boxes = np.array([[100, 100, 200, 200]])
        class_ids = np.array([1])
        
        result = tracker.update(boxes, class_ids, frame_idx=0)
        
        # First detection creates a track
        all_tracks = result.get_all_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0][0] == 1  # class_id
        assert all_tracks[0][1] == 0  # track_id
    
    def test_track_continuity(self):
        """Test that tracks maintain ID across frames."""
        tracker = SORTTracker(TrackerConfig(min_hits=1))
        
        # Frame 0
        boxes = np.array([[100, 100, 200, 200]])
        class_ids = np.array([1])
        result1 = tracker.update(boxes, class_ids, frame_idx=0)
        
        # Frame 1 - slightly moved
        boxes = np.array([[105, 105, 205, 205]])
        result2 = tracker.update(boxes, class_ids, frame_idx=1)
        
        tracks1 = result1.get_all_tracks()
        tracks2 = result2.get_all_tracks()
        
        # Same track ID should persist
        assert tracks1[0][1] == tracks2[0][1]
    
    def test_reset(self):
        tracker = SORTTracker(TrackerConfig(min_hits=1))
        
        boxes = np.array([[100, 100, 200, 200]])
        class_ids = np.array([1])
        tracker.update(boxes, class_ids, frame_idx=0)
        
        assert len(tracker.get_track_count()) > 0
        
        tracker.reset()
        assert tracker.get_track_count() == {}
