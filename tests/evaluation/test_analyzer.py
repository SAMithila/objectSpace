"""Tests for the TrackingAnalyzer."""

import pytest
import time

from smart_desk_monitor.evaluation.analyzer import (
    TrackingAnalyzer,
    TimingContext,
    compute_iou,
    box_center,
    euclidean_distance,
)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_compute_iou_no_overlap(self):
        """IoU is 0 for non-overlapping boxes."""
        box1 = [0, 0, 10, 10]  # x, y, w, h
        box2 = [20, 20, 10, 10]
        assert compute_iou(box1, box2) == 0.0
    
    def test_compute_iou_perfect_overlap(self):
        """IoU is 1 for identical boxes."""
        box1 = [10, 10, 20, 20]
        box2 = [10, 10, 20, 20]
        assert compute_iou(box1, box2) == 1.0
    
    def test_compute_iou_partial_overlap(self):
        """IoU computed correctly for partial overlap."""
        # box1: (0,0) to (10,10), area = 100
        # box2: (5,5) to (15,15), area = 100
        # intersection: (5,5) to (10,10), area = 25
        # union: 100 + 100 - 25 = 175
        # iou: 25/175 ≈ 0.143
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 10, 10]
        iou = compute_iou(box1, box2)
        assert abs(iou - 0.143) < 0.01
    
    def test_compute_iou_zero_area(self):
        """IoU is 0 when a box has zero area."""
        box1 = [0, 0, 0, 0]
        box2 = [0, 0, 10, 10]
        assert compute_iou(box1, box2) == 0.0
    
    def test_box_center(self):
        """Box center calculated correctly."""
        box = [10, 20, 30, 40]  # x, y, w, h
        cx, cy = box_center(box)
        assert cx == 25.0  # 10 + 30/2
        assert cy == 40.0  # 20 + 40/2
    
    def test_euclidean_distance(self):
        """Euclidean distance calculated correctly."""
        p1 = (0, 0)
        p2 = (3, 4)
        assert euclidean_distance(p1, p2) == 5.0
    
    def test_euclidean_distance_same_point(self):
        """Distance is 0 for same point."""
        p1 = (5, 5)
        assert euclidean_distance(p1, p1) == 0.0


class TestTrackingAnalyzer:
    """Tests for TrackingAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TrackingAnalyzer(
            min_track_length=5,
            id_switch_iou_threshold=0.3,
            id_switch_distance_threshold=100.0,
            target_fps=30.0,
        )
    
    @pytest.fixture
    def simple_annotations(self):
        """Create simple COCO-format annotations with one complete track."""
        return {
            "images": [
                {"id": 0, "frame_number": 0},
                {"id": 1, "frame_number": 1},
                {"id": 2, "frame_number": 2},
                {"id": 3, "frame_number": 3},
                {"id": 4, "frame_number": 4},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [10, 10, 50, 50], "category_id": 1},
                {"id": 1, "image_id": 1, "track_id": 1, "bbox": [12, 12, 50, 50], "category_id": 1},
                {"id": 2, "image_id": 2, "track_id": 1, "bbox": [14, 14, 50, 50], "category_id": 1},
                {"id": 3, "image_id": 3, "track_id": 1, "bbox": [16, 16, 50, 50], "category_id": 1},
                {"id": 4, "image_id": 4, "track_id": 1, "bbox": [18, 18, 50, 50], "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
    
    def test_analyze_returns_result(self, analyzer, simple_annotations):
        """Analyze returns EvaluationResult."""
        result = analyzer.analyze(simple_annotations, video_name="test")
        
        assert result.video_name == "test"
        assert result.tracks.total_tracks == 1
        assert result.performance.total_frames == 5
    
    def test_fragmentation_complete_track(self, analyzer, simple_annotations):
        """Complete track has 100% coverage."""
        result = analyzer.analyze(simple_annotations)
        
        assert result.fragmentation.total_tracks == 1
        assert result.fragmentation.fragmented_tracks == 0
        assert result.fragmentation.avg_coverage_ratio == 1.0
        assert result.fragmentation.total_gaps == 0
    
    def test_fragmentation_with_gap(self, analyzer):
        """Track with gap detected correctly."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(10)],
            "annotations": [
                # Track 1: frames 0, 1, 2, 5, 6, 7, 8, 9 (gap at 3-4)
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 1, "image_id": 1, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 2, "image_id": 2, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                # Gap at frames 3, 4
                {"id": 3, "image_id": 5, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 4, "image_id": 6, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 5, "image_id": 7, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 6, "image_id": 8, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 7, "image_id": 9, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        assert result.fragmentation.total_gaps == 1
        assert result.fragmentation.avg_coverage_ratio == 0.8  # 8/10
        assert result.fragmentation.fragmented_tracks == 0  # 80% is not fragmented
    
    def test_fragmentation_short_track(self, analyzer):
        """Short tracks counted correctly."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(10)],
            "annotations": [
                # Track 1: only 3 frames (short)
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 1, "image_id": 1, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 2, "image_id": 2, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                # Track 2: 7 frames (not short)
                {"id": 3, "image_id": 3, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 4, "image_id": 4, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 5, "image_id": 5, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 6, "image_id": 6, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 7, "image_id": 7, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 8, "image_id": 8, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
                {"id": 9, "image_id": 9, "track_id": 2, "bbox": [50, 50, 10, 10], "category_id": 1},
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        assert result.fragmentation.total_tracks == 2
        assert result.fragmentation.short_tracks == 1
    
    def test_id_switch_detection(self, analyzer):
        """ID switches detected when tracks end and nearby tracks begin."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(6)],
            "annotations": [
                # Track 1: frames 0, 1, 2 (then disappears)
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [100, 100, 50, 50], "category_id": 1},
                {"id": 1, "image_id": 1, "track_id": 1, "bbox": [102, 100, 50, 50], "category_id": 1},
                {"id": 2, "image_id": 2, "track_id": 1, "bbox": [104, 100, 50, 50], "category_id": 1},
                # Track 2: frames 3, 4, 5 (appears nearby - likely same object!)
                {"id": 3, "image_id": 3, "track_id": 2, "bbox": [106, 100, 50, 50], "category_id": 1},
                {"id": 4, "image_id": 4, "track_id": 2, "bbox": [108, 100, 50, 50], "category_id": 1},
                {"id": 5, "image_id": 5, "track_id": 2, "bbox": [110, 100, 50, 50], "category_id": 1},
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        # Should detect potential ID switch from track 1 -> 2
        assert result.id_switches.total_switches >= 1
    
    def test_id_switch_no_detection_distant(self, analyzer):
        """No ID switch when new track is far from disappeared track."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(6)],
            "annotations": [
                # Track 1: frames 0, 1, 2
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [0, 0, 50, 50], "category_id": 1},
                {"id": 1, "image_id": 1, "track_id": 1, "bbox": [0, 0, 50, 50], "category_id": 1},
                {"id": 2, "image_id": 2, "track_id": 1, "bbox": [0, 0, 50, 50], "category_id": 1},
                # Track 2: frames 3, 4, 5 (far away - different object)
                {"id": 3, "image_id": 3, "track_id": 2, "bbox": [500, 500, 50, 50], "category_id": 1},
                {"id": 4, "image_id": 4, "track_id": 2, "bbox": [500, 500, 50, 50], "category_id": 1},
                {"id": 5, "image_id": 5, "track_id": 2, "bbox": [500, 500, 50, 50], "category_id": 1},
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        # Should not detect ID switch (tracks too far apart)
        assert result.id_switches.total_switches == 0
    
    def test_track_metrics_concurrent(self, analyzer):
        """Max concurrent tracks computed correctly."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(5)],
            "annotations": [
                # Frame 0: 2 tracks
                {"id": 0, "image_id": 0, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 1, "image_id": 0, "track_id": 2, "bbox": [50, 0, 10, 10], "category_id": 1},
                # Frame 1: 3 tracks
                {"id": 2, "image_id": 1, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 3, "image_id": 1, "track_id": 2, "bbox": [50, 0, 10, 10], "category_id": 1},
                {"id": 4, "image_id": 1, "track_id": 3, "bbox": [100, 0, 10, 10], "category_id": 1},
                # Frame 2: 2 tracks
                {"id": 5, "image_id": 2, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
                {"id": 6, "image_id": 2, "track_id": 3, "bbox": [100, 0, 10, 10], "category_id": 1},
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        assert result.tracks.total_tracks == 3
        assert result.tracks.max_concurrent_tracks == 3
    
    def test_track_duration_histogram(self, analyzer):
        """Track duration histogram computed correctly."""
        annotations = {
            "images": [{"id": i, "frame_number": i} for i in range(30)],
            "annotations": [
                # Track 1: 3 frames (bucket "1-5")
                *[{"id": i, "image_id": i, "track_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1} for i in range(3)],
                # Track 2: 15 frames (bucket "6-20")
                *[{"id": 10+i, "image_id": 5+i, "track_id": 2, "bbox": [50, 0, 10, 10], "category_id": 1} for i in range(15)],
            ],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        assert result.tracks.duration_histogram["1-5"] == 1
        assert result.tracks.duration_histogram["6-20"] == 1
    
    def test_continuity_score_perfect(self, analyzer, simple_annotations):
        """Perfect continuity gives high score."""
        result = analyzer.analyze(simple_annotations)
        
        # Single complete track should score well
        assert result.continuity_score >= 75.0
    
    def test_stability_score_no_switches(self, analyzer, simple_annotations):
        """No ID switches gives perfect stability."""
        result = analyzer.analyze(simple_annotations)
        
        assert result.stability_score == 100.0
    
    def test_empty_annotations(self, analyzer):
        """Handle empty annotations gracefully."""
        annotations = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        
        result = analyzer.analyze(annotations)
        
        assert result.tracks.total_tracks == 0
        assert result.fragmentation.total_tracks == 0
    
    def test_config_passed_through(self, analyzer, simple_annotations):
        """Configuration dict stored in result."""
        config = {"detector": {"device": "cuda"}, "tracker": {"max_age": 10}}
        result = analyzer.analyze(simple_annotations, config=config)
        
        assert result.config_used == config


class TestTimingContext:
    """Tests for TimingContext."""
    
    def test_timing_recorded(self):
        """Timing context records elapsed time."""
        analyzer = TrackingAnalyzer()
        
        with TimingContext(analyzer, "total") as ctx:
            time.sleep(0.01)  # Sleep 10ms
        
        assert ctx.elapsed >= 0.01
        assert len(analyzer._frame_times) == 1
    
    def test_detection_timing(self):
        """Detection component timing recorded separately."""
        analyzer = TrackingAnalyzer()
        
        with TimingContext(analyzer, "detection"):
            time.sleep(0.01)
        
        assert len(analyzer._detection_times) == 1
        assert analyzer._detection_times[0] >= 0.01
    
    def test_tracking_timing(self):
        """Tracking component timing recorded separately."""
        analyzer = TrackingAnalyzer()
        
        with TimingContext(analyzer, "tracking"):
            time.sleep(0.01)
        
        assert len(analyzer._tracking_times) == 1
        assert analyzer._tracking_times[0] >= 0.01
    
    def test_reset_timings(self):
        """Reset clears all timing data."""
        analyzer = TrackingAnalyzer()
        analyzer._frame_times = [0.1, 0.2]
        analyzer._detection_times = [0.05]
        analyzer._tracking_times = [0.03]
        
        analyzer.reset_timings()
        
        assert len(analyzer._frame_times) == 0
        assert len(analyzer._detection_times) == 0
        assert len(analyzer._tracking_times) == 0


class TestScoring:
    """Tests for score calculations."""
    
    def test_speed_score_at_target(self):
        """Speed at target FPS gives perfect score."""
        analyzer = TrackingAnalyzer(target_fps=30.0)
        analyzer._frame_times = [1/30.0] * 30  # Exactly 30 FPS
        
        perf = analyzer._build_performance_metrics(30)
        score = analyzer._score_speed(perf)
        
        assert score == 100.0
    
    def test_speed_score_above_target(self):
        """Speed above target still gives perfect score."""
        analyzer = TrackingAnalyzer(target_fps=30.0)
        analyzer._frame_times = [1/60.0] * 60  # 60 FPS
        
        perf = analyzer._build_performance_metrics(60)
        score = analyzer._score_speed(perf)
        
        assert score == 100.0
    
    def test_speed_score_below_target(self):
        """Speed below target reduces score."""
        analyzer = TrackingAnalyzer(target_fps=30.0)
        analyzer._frame_times = [1/15.0] * 15  # 15 FPS = 50% of target
        
        perf = analyzer._build_performance_metrics(15)
        score = analyzer._score_speed(perf)
        
        assert 50 <= score <= 60  # Should be around 50-60
    
    def test_stability_score_many_switches(self):
        """Many ID switches reduces stability score."""
        analyzer = TrackingAnalyzer()
        
        # Create mock id_switches result with high switch rate
        from smart_desk_monitor.evaluation.metrics import IDSwitchMetrics
        
        id_switches = IDSwitchMetrics(
            total_switches=10,
            switches_per_100_frames=10.0,  # High rate
        )
        
        score = analyzer._score_stability(id_switches, 100)
        
        assert score < 50  # Should be low due to many switches
