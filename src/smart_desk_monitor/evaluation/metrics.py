"""
Core metric data structures for tracking evaluation.

Provides typed containers for all evaluation metrics computed
without ground truth annotations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TrackLifecycle:
    """Statistics for a single track's lifecycle."""
    
    track_id: int
    first_frame: int
    last_frame: int
    total_detections: int
    gaps: List[tuple[int, int]] = field(default_factory=list)  # (start_frame, end_frame)
    
    @property
    def duration_frames(self) -> int:
        """Total frames from first to last detection."""
        return self.last_frame - self.first_frame + 1
    
    @property
    def coverage_ratio(self) -> float:
        """Ratio of frames with detections to total duration."""
        if self.duration_frames == 0:
            return 0.0
        return self.total_detections / self.duration_frames
    
    @property
    def gap_count(self) -> int:
        """Number of gaps (missing detections) in track."""
        return len(self.gaps)
    
    @property
    def total_gap_frames(self) -> int:
        """Total frames missing within track lifetime."""
        return sum(end - start + 1 for start, end in self.gaps)
    
    @property
    def is_fragmented(self) -> bool:
        """Track is considered fragmented if coverage < 80%."""
        return self.coverage_ratio < 0.8


@dataclass
class FragmentationMetrics:
    """Metrics measuring track fragmentation and continuity issues."""
    
    total_tracks: int = 0
    fragmented_tracks: int = 0
    total_gaps: int = 0
    avg_gap_length: float = 0.0
    max_gap_length: int = 0
    avg_track_duration: float = 0.0
    avg_coverage_ratio: float = 0.0
    short_tracks: int = 0  # Tracks < 5 frames
    
    # Per-track details
    track_lifecycles: List[TrackLifecycle] = field(default_factory=list)
    
    @property
    def fragmentation_rate(self) -> float:
        """Percentage of tracks that are fragmented."""
        if self.total_tracks == 0:
            return 0.0
        return self.fragmented_tracks / self.total_tracks
    
    @property
    def short_track_rate(self) -> float:
        """Percentage of tracks that are very short (likely false positives)."""
        if self.total_tracks == 0:
            return 0.0
        return self.short_tracks / self.total_tracks


@dataclass
class IDSwitchEvent:
    """Record of a potential ID switch event."""
    
    frame: int
    old_track_id: int
    new_track_id: int
    spatial_distance: float  # Pixel distance between bounding box centers
    iou_overlap: float  # IoU between the two bounding boxes
    confidence: float  # Confidence that this is a true ID switch (0-1)


@dataclass
class IDSwitchMetrics:
    """Metrics for tracking ID switch/reassignment issues."""
    
    total_switches: int = 0
    high_confidence_switches: int = 0  # Confidence > 0.7
    switches_per_100_frames: float = 0.0
    
    # Spatial analysis
    avg_switch_distance: float = 0.0
    avg_switch_iou: float = 0.0
    
    # Events
    switch_events: List[IDSwitchEvent] = field(default_factory=list)
    
    @property
    def switch_rate(self) -> float:
        """Normalized switch rate (switches per 100 frames)."""
        return self.switches_per_100_frames


@dataclass
class PerformanceMetrics:
    """Processing speed and resource utilization metrics."""
    
    total_frames: int = 0
    total_time_seconds: float = 0.0
    
    # Per-component timing (seconds)
    detection_time: float = 0.0
    tracking_time: float = 0.0
    io_time: float = 0.0
    
    # Frame-level statistics
    frame_times: List[float] = field(default_factory=list)
    
    @property
    def fps(self) -> float:
        """Average frames per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_frames / self.total_time_seconds
    
    @property
    def detection_fps(self) -> float:
        """Frames per second for detection alone."""
        if self.detection_time == 0:
            return 0.0
        return self.total_frames / self.detection_time
    
    @property
    def tracking_fps(self) -> float:
        """Frames per second for tracking alone."""
        if self.tracking_time == 0:
            return 0.0
        return self.total_frames / self.tracking_time
    
    @property
    def avg_frame_time_ms(self) -> float:
        """Average time per frame in milliseconds."""
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times) * 1000
    
    @property
    def p95_frame_time_ms(self) -> float:
        """95th percentile frame time in milliseconds."""
        if not self.frame_times:
            return 0.0
        sorted_times = sorted(self.frame_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000


@dataclass
class TrackMetrics:
    """Aggregate track-level metrics."""
    
    total_tracks: int = 0
    active_tracks_per_frame: Dict[int, int] = field(default_factory=dict)
    avg_active_tracks: float = 0.0
    max_concurrent_tracks: int = 0
    
    # Track duration distribution
    duration_histogram: Dict[str, int] = field(default_factory=dict)
    # Keys: "1-5", "6-20", "21-50", "51-100", "100+"
    
    # Category breakdown
    tracks_by_category: Dict[int, int] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a video or session."""
    
    video_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core metrics
    fragmentation: FragmentationMetrics = field(default_factory=FragmentationMetrics)
    id_switches: IDSwitchMetrics = field(default_factory=IDSwitchMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    tracks: TrackMetrics = field(default_factory=TrackMetrics)
    
    # Summary scores (0-100, higher is better)
    continuity_score: float = 0.0  # Based on fragmentation
    stability_score: float = 0.0   # Based on ID switches
    speed_score: float = 0.0       # Based on FPS vs target
    
    # Metadata
    config_used: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Weighted combination of component scores."""
        return (
            self.continuity_score * 0.4 +
            self.stability_score * 0.4 +
            self.speed_score * 0.2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_name": self.video_name,
            "timestamp": self.timestamp.isoformat(),
            "scores": {
                "overall": round(self.overall_score, 2),
                "continuity": round(self.continuity_score, 2),
                "stability": round(self.stability_score, 2),
                "speed": round(self.speed_score, 2),
            },
            "fragmentation": {
                "total_tracks": self.fragmentation.total_tracks,
                "fragmented_tracks": self.fragmentation.fragmented_tracks,
                "fragmentation_rate": round(self.fragmentation.fragmentation_rate, 3),
                "avg_coverage_ratio": round(self.fragmentation.avg_coverage_ratio, 3),
                "total_gaps": self.fragmentation.total_gaps,
                "avg_gap_length": round(self.fragmentation.avg_gap_length, 2),
                "short_track_rate": round(self.fragmentation.short_track_rate, 3),
            },
            "id_switches": {
                "total_switches": self.id_switches.total_switches,
                "high_confidence_switches": self.id_switches.high_confidence_switches,
                "switches_per_100_frames": round(self.id_switches.switches_per_100_frames, 2),
                "avg_switch_distance": round(self.id_switches.avg_switch_distance, 2),
            },
            "performance": {
                "total_frames": self.performance.total_frames,
                "fps": round(self.performance.fps, 2),
                "avg_frame_time_ms": round(self.performance.avg_frame_time_ms, 2),
                "p95_frame_time_ms": round(self.performance.p95_frame_time_ms, 2),
                "detection_fps": round(self.performance.detection_fps, 2),
                "tracking_fps": round(self.performance.tracking_fps, 2),
            },
            "tracks": {
                "total_tracks": self.tracks.total_tracks,
                "avg_active_tracks": round(self.tracks.avg_active_tracks, 2),
                "max_concurrent_tracks": self.tracks.max_concurrent_tracks,
                "duration_histogram": self.tracks.duration_histogram,
            },
        }
