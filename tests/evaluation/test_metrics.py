"""Tests for evaluation metrics data structures."""

import pytest
from datetime import datetime

from smart_desk_monitor.evaluation.metrics import (
    TrackLifecycle,
    FragmentationMetrics,
    IDSwitchEvent,
    IDSwitchMetrics,
    PerformanceMetrics,
    TrackMetrics,
    EvaluationResult,
)


class TestTrackLifecycle:
    """Tests for TrackLifecycle dataclass."""
    
    def test_duration_frames(self):
        """Duration is last - first + 1."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=10,
            last_frame=20,
            total_detections=11,
        )
        assert track.duration_frames == 11
    
    def test_coverage_ratio_complete(self):
        """Coverage is 1.0 when all frames have detections."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=9,
            total_detections=10,
        )
        assert track.coverage_ratio == 1.0
    
    def test_coverage_ratio_partial(self):
        """Coverage < 1.0 when frames are missing."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=9,
            total_detections=5,
        )
        assert track.coverage_ratio == 0.5
    
    def test_coverage_ratio_zero_duration(self):
        """Coverage is 0 for zero-duration tracks."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=5,
            last_frame=4,  # Invalid but should handle gracefully
            total_detections=0,
        )
        assert track.coverage_ratio == 0.0
    
    def test_gap_count(self):
        """Gap count returns number of gaps."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=10,
            total_detections=8,
            gaps=[(3, 4), (7, 7)],
        )
        assert track.gap_count == 2
    
    def test_total_gap_frames(self):
        """Total gap frames sums all gap durations."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=10,
            total_detections=8,
            gaps=[(3, 4), (7, 7)],  # 2 frames + 1 frame = 3
        )
        assert track.total_gap_frames == 3
    
    def test_is_fragmented_true(self):
        """Track is fragmented when coverage < 80%."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=9,
            total_detections=7,  # 70% coverage
        )
        assert track.is_fragmented is True
    
    def test_is_fragmented_false(self):
        """Track is not fragmented when coverage >= 80%."""
        track = TrackLifecycle(
            track_id=1,
            first_frame=0,
            last_frame=9,
            total_detections=8,  # 80% coverage
        )
        assert track.is_fragmented is False


class TestFragmentationMetrics:
    """Tests for FragmentationMetrics."""
    
    def test_fragmentation_rate(self):
        """Fragmentation rate is fragmented/total."""
        metrics = FragmentationMetrics(
            total_tracks=10,
            fragmented_tracks=3,
        )
        assert metrics.fragmentation_rate == 0.3
    
    def test_fragmentation_rate_zero_tracks(self):
        """Fragmentation rate is 0 with no tracks."""
        metrics = FragmentationMetrics(total_tracks=0)
        assert metrics.fragmentation_rate == 0.0
    
    def test_short_track_rate(self):
        """Short track rate is short/total."""
        metrics = FragmentationMetrics(
            total_tracks=20,
            short_tracks=4,
        )
        assert metrics.short_track_rate == 0.2


class TestIDSwitchMetrics:
    """Tests for IDSwitchMetrics."""
    
    def test_switch_rate(self):
        """Switch rate returns switches per 100 frames."""
        metrics = IDSwitchMetrics(
            switches_per_100_frames=2.5,
        )
        assert metrics.switch_rate == 2.5


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""
    
    def test_fps(self):
        """FPS is frames / time."""
        metrics = PerformanceMetrics(
            total_frames=100,
            total_time_seconds=4.0,
        )
        assert metrics.fps == 25.0
    
    def test_fps_zero_time(self):
        """FPS is 0 when time is 0."""
        metrics = PerformanceMetrics(total_frames=100, total_time_seconds=0)
        assert metrics.fps == 0.0
    
    def test_detection_fps(self):
        """Detection FPS calculated from detection time."""
        metrics = PerformanceMetrics(
            total_frames=100,
            detection_time=2.0,
        )
        assert metrics.detection_fps == 50.0
    
    def test_avg_frame_time_ms(self):
        """Average frame time in milliseconds."""
        metrics = PerformanceMetrics(
            frame_times=[0.01, 0.02, 0.03],
        )
        assert metrics.avg_frame_time_ms == 20.0  # 0.02 * 1000
    
    def test_p95_frame_time_ms(self):
        """95th percentile frame time."""
        # 100 samples, p95 should be near index 95
        frame_times = [i / 100.0 for i in range(100)]
        metrics = PerformanceMetrics(frame_times=frame_times)
        # p95 of [0.0, 0.01, ..., 0.99] is around 0.95
        assert 900 <= metrics.p95_frame_time_ms <= 960


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_overall_score_weighted(self):
        """Overall score is weighted combination."""
        result = EvaluationResult(
            video_name="test",
            continuity_score=80.0,  # * 0.4 = 32
            stability_score=70.0,   # * 0.4 = 28
            speed_score=60.0,       # * 0.2 = 12
        )
        # 32 + 28 + 12 = 72
        assert result.overall_score == 72.0
    
    def test_to_dict_structure(self):
        """to_dict returns expected structure."""
        result = EvaluationResult(
            video_name="test_video",
            continuity_score=85.0,
            stability_score=75.0,
            speed_score=90.0,
        )
        
        d = result.to_dict()
        
        assert d["video_name"] == "test_video"
        assert "timestamp" in d
        assert "scores" in d
        assert d["scores"]["continuity"] == 85.0
        assert "fragmentation" in d
        assert "id_switches" in d
        assert "performance" in d
        assert "tracks" in d
    
    def test_to_dict_rounded_values(self):
        """to_dict rounds floating point values."""
        result = EvaluationResult(
            video_name="test",
            fragmentation=FragmentationMetrics(
                avg_coverage_ratio=0.123456789,
            ),
        )
        
        d = result.to_dict()
        assert d["fragmentation"]["avg_coverage_ratio"] == 0.123
    
    def test_default_timestamp(self):
        """Timestamp is set to current time by default."""
        result = EvaluationResult(video_name="test")
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)
