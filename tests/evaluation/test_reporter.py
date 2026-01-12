"""Tests for the EvaluationReporter."""

import pytest
import json
import tempfile
from pathlib import Path
from io import StringIO

from smart_desk_monitor.evaluation.reporter import EvaluationReporter
from smart_desk_monitor.evaluation.metrics import (
    EvaluationResult,
    FragmentationMetrics,
    IDSwitchMetrics,
    PerformanceMetrics,
    TrackMetrics,
    TrackLifecycle,
    IDSwitchEvent,
)


@pytest.fixture
def sample_result():
    """Create a sample evaluation result for testing."""
    return EvaluationResult(
        video_name="test_video",
        continuity_score=85.0,
        stability_score=75.0,
        speed_score=90.0,
        fragmentation=FragmentationMetrics(
            total_tracks=10,
            fragmented_tracks=2,
            total_gaps=5,
            avg_gap_length=3.5,
            max_gap_length=8,
            avg_track_duration=25.0,
            avg_coverage_ratio=0.92,
            short_tracks=1,
            track_lifecycles=[
                TrackLifecycle(
                    track_id=1,
                    first_frame=0,
                    last_frame=30,
                    total_detections=25,
                    gaps=[(10, 12), (20, 22)],
                ),
                TrackLifecycle(
                    track_id=2,
                    first_frame=5,
                    last_frame=50,
                    total_detections=46,
                    gaps=[],
                ),
            ],
        ),
        id_switches=IDSwitchMetrics(
            total_switches=3,
            high_confidence_switches=2,
            switches_per_100_frames=1.5,
            avg_switch_distance=45.2,
            avg_switch_iou=0.35,
            switch_events=[
                IDSwitchEvent(
                    frame=15,
                    old_track_id=1,
                    new_track_id=3,
                    spatial_distance=30.0,
                    iou_overlap=0.4,
                    confidence=0.85,
                ),
            ],
        ),
        performance=PerformanceMetrics(
            total_frames=200,
            total_time_seconds=8.0,
            detection_time=5.0,
            tracking_time=2.0,
            io_time=1.0,
            frame_times=[0.04] * 200,
        ),
        tracks=TrackMetrics(
            total_tracks=10,
            active_tracks_per_frame={i: 3 for i in range(200)},
            avg_active_tracks=3.0,
            max_concurrent_tracks=5,
            duration_histogram={
                "1-5": 1,
                "6-20": 2,
                "21-50": 5,
                "51-100": 2,
                "100+": 0,
            },
        ),
    )


class TestEvaluationReporter:
    """Tests for EvaluationReporter."""
    
    def test_print_summary_outputs_text(self, sample_result):
        """print_summary produces output."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.print_summary(sample_result, file=output)
        
        text = output.getvalue()
        assert "TRACKING EVALUATION" in text
        assert "test_video" in text
        assert "SCORES" in text
        assert "85.0" in text  # continuity score
    
    def test_print_summary_contains_all_sections(self, sample_result):
        """print_summary includes all metric sections."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.print_summary(sample_result, file=output)
        
        text = output.getvalue()
        assert "TRACK STATISTICS" in text
        assert "FRAGMENTATION" in text
        assert "ID SWITCHES" in text
        assert "PERFORMANCE" in text
    
    def test_print_detailed_tracks(self, sample_result):
        """print_detailed_tracks outputs track information."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.print_detailed_tracks(sample_result, file=output)
        
        text = output.getvalue()
        assert "PROBLEMATIC TRACKS" in text
        assert "Track" in text
    
    def test_print_id_switch_events(self, sample_result):
        """print_id_switch_events outputs event details."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.print_id_switch_events(sample_result, file=output)
        
        text = output.getvalue()
        assert "ID SWITCH EVENTS" in text
        assert "Frame" in text
    
    def test_save_json(self, sample_result):
        """save_json creates valid JSON file."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.json"
            reporter.save_json(sample_result, output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert data["video_name"] == "test_video"
            assert data["scores"]["continuity"] == 85.0
            assert "fragmentation" in data
    
    def test_save_json_pretty(self, sample_result):
        """save_json with pretty=True creates formatted JSON."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.json"
            reporter.save_json(sample_result, output_path, pretty=True)
            
            with open(output_path) as f:
                content = f.read()
            
            # Pretty JSON has newlines and indentation
            assert "\n" in content
            assert "  " in content
    
    def test_save_json_compact(self, sample_result):
        """save_json with pretty=False creates compact JSON."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.json"
            reporter.save_json(sample_result, output_path, pretty=False)
            
            with open(output_path) as f:
                content = f.read()
            
            # Compact JSON is single line
            assert content.count("\n") <= 1
    
    def test_save_markdown(self, sample_result):
        """save_markdown creates valid Markdown file."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.md"
            reporter.save_markdown(sample_result, output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                content = f.read()
            
            assert "# Tracking Evaluation Report" in content
            assert "test_video" in content
            assert "## Summary Scores" in content
    
    def test_save_markdown_with_details(self, sample_result):
        """save_markdown includes problematic tracks when requested."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.md"
            reporter.save_markdown(sample_result, output_path, include_details=True)
            
            with open(output_path) as f:
                content = f.read()
            
            assert "## Problematic Tracks" in content
    
    def test_save_markdown_without_details(self, sample_result):
        """save_markdown excludes details when not requested."""
        reporter = EvaluationReporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation.md"
            reporter.save_markdown(sample_result, output_path, include_details=False)
            
            with open(output_path) as f:
                content = f.read()
            
            assert "## Problematic Tracks" not in content
    
    def test_compare_results(self, sample_result):
        """compare_results outputs comparison table."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        # Create a second result
        result2 = EvaluationResult(
            video_name="video_2",
            continuity_score=70.0,
            stability_score=80.0,
            speed_score=85.0,
            tracks=TrackMetrics(total_tracks=5),
            performance=PerformanceMetrics(total_frames=100, total_time_seconds=5.0),
        )
        
        reporter.compare_results([sample_result, result2], file=output)
        
        text = output.getvalue()
        assert "EVALUATION COMPARISON" in text
        assert "test_video" in text
        assert "video_2" in text
        assert "AVERAGE" in text
    
    def test_compare_results_single(self, sample_result):
        """compare_results works with single result."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.compare_results([sample_result], file=output)
        
        text = output.getvalue()
        assert "test_video" in text
        # No AVERAGE line for single result
    
    def test_compare_results_empty(self):
        """compare_results handles empty list."""
        reporter = EvaluationReporter(use_colors=False)
        output = StringIO()
        
        reporter.compare_results([], file=output)
        
        text = output.getvalue()
        assert "No results to compare" in text
    
    def test_colors_disabled_in_non_tty(self):
        """Colors are disabled when stdout is not a TTY."""
        # StringIO is not a TTY
        reporter = EvaluationReporter(use_colors=True)
        
        # The reporter should detect non-TTY and disable colors
        # This is tested indirectly through the output not containing ANSI codes
        output = StringIO()
        
        result = EvaluationResult(
            video_name="test",
            continuity_score=50.0,
        )
        reporter.print_summary(result, file=output)
        
        text = output.getvalue()
        # Should not contain ANSI escape codes when writing to StringIO
        assert "\033[" not in text or not reporter.use_colors


class TestScoreColors:
    """Tests for score coloring logic."""
    
    def test_good_score_color(self):
        """Scores >= 80 get green color."""
        reporter = EvaluationReporter()
        assert reporter._score_color(80.0) == "green"
        assert reporter._score_color(100.0) == "green"
    
    def test_fair_score_color(self):
        """Scores 60-79 get yellow color."""
        reporter = EvaluationReporter()
        assert reporter._score_color(60.0) == "yellow"
        assert reporter._score_color(79.9) == "yellow"
    
    def test_poor_score_color(self):
        """Scores < 60 get red color."""
        reporter = EvaluationReporter()
        assert reporter._score_color(59.9) == "red"
        assert reporter._score_color(0.0) == "red"
