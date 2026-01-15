"""
Tracking analyzer for computing evaluation metrics.

Analyzes tracking results to compute fragmentation, ID switches,
and other quality metrics without requiring ground truth.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np

from .metrics import (
    TrackLifecycle,
    FragmentationMetrics,
    IDSwitchEvent,
    IDSwitchMetrics,
    PerformanceMetrics,
    TrackMetrics,
    EvaluationResult,
)


logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        box1: [x, y, w, h] format
        box2: [x, y, w, h] format
    
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def box_center(box: List[float]) -> Tuple[float, float]:
    """Get center point of bounding box [x, y, w, h]."""
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class TrackingAnalyzer:
    """
    Analyzes tracking results to compute quality metrics.
    
    Works with COCO-format annotations extended with track_id fields.
    Does not require ground truth data.
    """
    
    def __init__(
        self,
        min_track_length: int = 5,
        id_switch_iou_threshold: float = 0.3,
        id_switch_distance_threshold: float = 100.0,
        target_fps: float = 30.0,
    ):
        """
        Initialize the analyzer.
        
        Args:
            min_track_length: Tracks shorter than this are flagged as short
            id_switch_iou_threshold: Min IoU to consider as potential ID switch
            id_switch_distance_threshold: Max distance (pixels) for ID switch
            target_fps: Target FPS for speed scoring
        """
        self.min_track_length = min_track_length
        self.id_switch_iou_threshold = id_switch_iou_threshold
        self.id_switch_distance_threshold = id_switch_distance_threshold
        self.target_fps = target_fps
        
        # Timing accumulator
        self._frame_times: List[float] = []
        self._detection_times: List[float] = []
        self._tracking_times: List[float] = []
    
    def analyze(
        self,
        annotations: Dict[str, Any],
        video_name: str = "unknown",
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Analyze tracking results and compute all metrics.
        
        Args:
            annotations: COCO-format annotations dict with images, annotations, categories
            video_name: Name of the video being analyzed
            config: Pipeline configuration used (for logging)
        
        Returns:
            EvaluationResult with all computed metrics
        """
        logger.info(f"Analyzing tracking results for {video_name}")
        
        # Build frame-indexed structure
        frame_annotations = self._build_frame_index(annotations)
        total_frames = len(annotations.get("images", []))
        
        # Compute each metric category
        fragmentation = self._compute_fragmentation(frame_annotations, total_frames)
        id_switches = self._compute_id_switches(frame_annotations)
        performance = self._build_performance_metrics(total_frames)
        tracks = self._compute_track_metrics(frame_annotations, total_frames)
        
        # Compute summary scores
        continuity_score = self._score_continuity(fragmentation)
        stability_score = self._score_stability(id_switches, total_frames)
        speed_score = self._score_speed(performance)
        
        result = EvaluationResult(
            video_name=video_name,
            fragmentation=fragmentation,
            id_switches=id_switches,
            performance=performance,
            tracks=tracks,
            continuity_score=continuity_score,
            stability_score=stability_score,
            speed_score=speed_score,
            config_used=config or {},
        )
        
        logger.info(
            f"Analysis complete: overall_score={result.overall_score:.1f}, "
            f"tracks={tracks.total_tracks}, fps={performance.fps:.1f}"
        )
        
        return result
    
    def record_frame_time(
        self,
        total_time: float,
        detection_time: Optional[float] = None,
        tracking_time: Optional[float] = None,
    ) -> None:
        """
        Record timing for a single frame.
        
        Args:
            total_time: Total frame processing time in seconds
            detection_time: Detection component time in seconds
            tracking_time: Tracking component time in seconds
        """
        self._frame_times.append(total_time)
        if detection_time is not None:
            self._detection_times.append(detection_time)
        if tracking_time is not None:
            self._tracking_times.append(tracking_time)
    
    def reset_timings(self) -> None:
        """Clear accumulated timing data."""
        self._frame_times.clear()
        self._detection_times.clear()
        self._tracking_times.clear()
    
    def _build_frame_index(
        self, annotations: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Build a frame-indexed structure from COCO annotations."""
        frame_annotations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        # Map image_id to frame index
        image_id_to_frame = {}
        for img in annotations.get("images", []):
            # Frame number from filename or id
            frame_num = img.get("frame_number", img["id"])
            image_id_to_frame[img["id"]] = frame_num
        
        # Index annotations by frame
        for ann in annotations.get("annotations", []):
            frame = image_id_to_frame.get(ann["image_id"], ann["image_id"])
            frame_annotations[frame].append(ann)
        
        return dict(frame_annotations)
    
    def _compute_fragmentation(
        self, frame_annotations: Dict[int, List[Dict]], total_frames: int
    ) -> FragmentationMetrics:
        """Compute track fragmentation metrics."""
        # Build track histories
        track_frames: Dict[int, List[int]] = defaultdict(list)
        
        for frame, anns in frame_annotations.items():
            for ann in anns:
                track_id = ann.get("track_id")
                if track_id is not None:
                    track_frames[track_id].append(frame)
        
        # Analyze each track
        lifecycles: List[TrackLifecycle] = []
        total_gaps = 0
        all_gap_lengths: List[int] = []
        
        for track_id, frames in track_frames.items():
            if not frames:
                continue
            
            frames = sorted(frames)
            first_frame = frames[0]
            last_frame = frames[-1]
            total_detections = len(frames)
            
            # Find gaps
            gaps: List[Tuple[int, int]] = []
            frames_set = set(frames)
            
            gap_start = None
            for f in range(first_frame, last_frame + 1):
                if f not in frames_set:
                    if gap_start is None:
                        gap_start = f
                else:
                    if gap_start is not None:
                        gaps.append((gap_start, f - 1))
                        gap_length = f - gap_start
                        all_gap_lengths.append(gap_length)
                        total_gaps += 1
                        gap_start = None
            
            # Handle trailing gap
            if gap_start is not None:
                gaps.append((gap_start, last_frame))
                all_gap_lengths.append(last_frame - gap_start + 1)
                total_gaps += 1
            
            lifecycle = TrackLifecycle(
                track_id=track_id,
                first_frame=first_frame,
                last_frame=last_frame,
                total_detections=total_detections,
                gaps=gaps,
            )
            lifecycles.append(lifecycle)
        
        # Aggregate metrics
        total_tracks = len(lifecycles)
        fragmented_tracks = sum(1 for lc in lifecycles if lc.is_fragmented)
        short_tracks = sum(1 for lc in lifecycles if lc.duration_frames < self.min_track_length)
        
        avg_gap_length = np.mean(all_gap_lengths) if all_gap_lengths else 0.0
        max_gap_length = max(all_gap_lengths) if all_gap_lengths else 0
        
        avg_duration = np.mean([lc.duration_frames for lc in lifecycles]) if lifecycles else 0.0
        avg_coverage = np.mean([lc.coverage_ratio for lc in lifecycles]) if lifecycles else 0.0
        
        return FragmentationMetrics(
            total_tracks=total_tracks,
            fragmented_tracks=fragmented_tracks,
            total_gaps=total_gaps,
            avg_gap_length=float(avg_gap_length),
            max_gap_length=int(max_gap_length),
            avg_track_duration=float(avg_duration),
            avg_coverage_ratio=float(avg_coverage),
            short_tracks=short_tracks,
            track_lifecycles=lifecycles,
        )
    
    def _compute_id_switches(
        self, frame_annotations: Dict[int, List[Dict]]
    ) -> IDSwitchMetrics:
        """
        Detect potential ID switches using spatial/temporal heuristics.
        
        An ID switch is detected when:
        1. A track disappears
        2. A new track appears nearby in a subsequent frame
        3. The spatial overlap (IoU) or proximity suggests same object
        """
        events: List[IDSwitchEvent] = []
        
        # Get sorted frames
        sorted_frames = sorted(frame_annotations.keys())
        if len(sorted_frames) < 2:
            return IDSwitchMetrics()
        
        # Track active tracks and their last known positions
        track_last_seen: Dict[int, Dict] = {}  # track_id -> last annotation
        
        for i, frame in enumerate(sorted_frames):
            current_anns = frame_annotations[frame]
            current_track_ids = {ann.get("track_id") for ann in current_anns if ann.get("track_id") is not None}
            
            # Find disappeared tracks (active in previous frames, not in current)
            disappeared = set(track_last_seen.keys()) - current_track_ids
            
            # Find new tracks (in current, not in previous active set)
            previous_track_ids = set(track_last_seen.keys())
            new_tracks = current_track_ids - previous_track_ids
            
            # Check for potential switches: disappeared track + new track nearby
            for old_id in disappeared:
                old_ann = track_last_seen[old_id]
                old_box = old_ann.get("bbox", [0, 0, 0, 0])
                old_center = box_center(old_box)
                
                for new_id in new_tracks:
                    # Find the annotation for the new track
                    new_ann = next(
                        (a for a in current_anns if a.get("track_id") == new_id),
                        None
                    )
                    if new_ann is None:
                        continue
                    
                    new_box = new_ann.get("bbox", [0, 0, 0, 0])
                    new_center = box_center(new_box)
                    
                    # Compute spatial metrics
                    distance = euclidean_distance(old_center, new_center)
                    iou = compute_iou(old_box, new_box)
                    
                    # Check if this looks like an ID switch
                    is_nearby = distance < self.id_switch_distance_threshold
                    has_overlap = iou > self.id_switch_iou_threshold
                    
                    if is_nearby or has_overlap:
                        # Compute confidence based on evidence
                        confidence = 0.0
                        if has_overlap:
                            confidence += 0.5 + (iou * 0.3)  # Up to 0.8 from IoU
                        if is_nearby:
                            # Higher confidence for closer switches
                            proximity_score = 1 - (distance / self.id_switch_distance_threshold)
                            confidence += proximity_score * 0.4
                        
                        confidence = min(confidence, 1.0)
                        
                        event = IDSwitchEvent(
                            frame=frame,
                            old_track_id=old_id,
                            new_track_id=new_id,
                            spatial_distance=distance,
                            iou_overlap=iou,
                            confidence=confidence,
                        )
                        events.append(event)
            
            # Update last seen for current tracks
            for ann in current_anns:
                track_id = ann.get("track_id")
                if track_id is not None:
                    track_last_seen[track_id] = ann
            
            # Remove disappeared tracks from tracking
            for old_id in disappeared:
                del track_last_seen[old_id]
        
        # Aggregate metrics
        total_switches = len(events)
        high_conf = sum(1 for e in events if e.confidence > 0.7)
        
        total_frames = len(sorted_frames)
        switches_per_100 = (total_switches / total_frames * 100) if total_frames > 0 else 0.0
        
        avg_distance = np.mean([e.spatial_distance for e in events]) if events else 0.0
        avg_iou = np.mean([e.iou_overlap for e in events]) if events else 0.0
        
        return IDSwitchMetrics(
            total_switches=total_switches,
            high_confidence_switches=high_conf,
            switches_per_100_frames=float(switches_per_100),
            avg_switch_distance=float(avg_distance),
            avg_switch_iou=float(avg_iou),
            switch_events=events,
        )
    
    def _build_performance_metrics(self, total_frames: int) -> PerformanceMetrics:
        """Build performance metrics from accumulated timing data."""
        total_time = sum(self._frame_times) if self._frame_times else 0.0
        detection_time = sum(self._detection_times) if self._detection_times else 0.0
        tracking_time = sum(self._tracking_times) if self._tracking_times else 0.0
        io_time = total_time - detection_time - tracking_time
        
        return PerformanceMetrics(
            total_frames=total_frames,
            total_time_seconds=total_time,
            detection_time=detection_time,
            tracking_time=tracking_time,
            io_time=max(0, io_time),
            frame_times=list(self._frame_times),
        )
    
    def _compute_track_metrics(
        self, frame_annotations: Dict[int, List[Dict]], total_frames: int
    ) -> TrackMetrics:
        """Compute aggregate track-level metrics."""
        # Count tracks per frame
        active_per_frame: Dict[int, int] = {}
        all_track_ids: set = set()
        track_durations: Dict[int, int] = defaultdict(int)
        tracks_by_category: Dict[int, int] = defaultdict(int)
        
        for frame, anns in frame_annotations.items():
            track_ids_in_frame = set()
            for ann in anns:
                track_id = ann.get("track_id")
                if track_id is not None:
                    track_ids_in_frame.add(track_id)
                    all_track_ids.add(track_id)
                    track_durations[track_id] += 1
                    
                    cat_id = ann.get("category_id", 0)
                    if track_id not in tracks_by_category:
                        tracks_by_category[cat_id] = tracks_by_category.get(cat_id, 0) + 1
            
            active_per_frame[frame] = len(track_ids_in_frame)
        
        # Build duration histogram
        histogram: Dict[str, int] = {
            "1-5": 0,
            "6-20": 0,
            "21-50": 0,
            "51-100": 0,
            "100+": 0,
        }
        
        for duration in track_durations.values():
            if duration <= 5:
                histogram["1-5"] += 1
            elif duration <= 20:
                histogram["6-20"] += 1
            elif duration <= 50:
                histogram["21-50"] += 1
            elif duration <= 100:
                histogram["51-100"] += 1
            else:
                histogram["100+"] += 1
        
        # Compute averages
        counts = list(active_per_frame.values())
        avg_active = np.mean(counts) if counts else 0.0
        max_concurrent = max(counts) if counts else 0
        
        return TrackMetrics(
            total_tracks=len(all_track_ids),
            active_tracks_per_frame=active_per_frame,
            avg_active_tracks=float(avg_active),
            max_concurrent_tracks=max_concurrent,
            duration_histogram=histogram,
            tracks_by_category=dict(tracks_by_category),
        )
    
    def _score_continuity(self, fragmentation: FragmentationMetrics) -> float:
        """
        Score track continuity (0-100, higher is better).
        
        Based on:
        - Coverage ratio (how complete tracks are)
        - Fragmentation rate (% of broken tracks)
        - Short track rate (% of likely false positives)
        """
        if fragmentation.total_tracks == 0:
            return 0.0
        
        # Weight different factors
        coverage_component = fragmentation.avg_coverage_ratio * 40
        fragmentation_component = (1 - fragmentation.fragmentation_rate) * 35
        short_track_component = (1 - fragmentation.short_track_rate) * 25
        
        return min(100.0, coverage_component + fragmentation_component + short_track_component)
    
    def _score_stability(self, id_switches: IDSwitchMetrics, total_frames: int) -> float:
        """
        Score tracking stability (0-100, higher is better).
        
        Based on ID switch frequency - fewer switches = better tracking.
        """
        if total_frames == 0:
            return 0.0
        
        # Ideal: 0 switches per 100 frames
        # Acceptable: up to 2 switches per 100 frames
        # Poor: 5+ switches per 100 frames
        
        switches_per_100 = id_switches.switches_per_100_frames
        
        if switches_per_100 <= 0.5:
            return 100.0
        elif switches_per_100 <= 2.0:
            # Linear decay from 100 to 70
            return 100.0 - (switches_per_100 - 0.5) * 20
        elif switches_per_100 <= 5.0:
            # Linear decay from 70 to 30
            return 70.0 - (switches_per_100 - 2.0) * (40.0 / 3.0)
        else:
            # Below 30 for high switch rates
            return max(0.0, 30.0 - (switches_per_100 - 5.0) * 3)
    
    def _score_speed(self, performance: PerformanceMetrics) -> float:
        """
        Score processing speed (0-100, higher is better).
        
        Based on FPS relative to target.
        """
        if performance.fps <= 0:
            return 0.0
        
        # Score based on ratio to target FPS
        ratio = performance.fps / self.target_fps
        
        if ratio >= 1.0:
            return 100.0
        elif ratio >= 0.5:
            # Linear from 50 to 100
            return 50.0 + (ratio - 0.5) * 100
        else:
            # Below 50 for very slow processing
            return ratio * 100


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, analyzer: TrackingAnalyzer, component: str = "total"):
        """
        Initialize timing context.
        
        Args:
            analyzer: TrackingAnalyzer to record times to
            component: One of "total", "detection", "tracking"
        """
        self.analyzer = analyzer
        self.component = component
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "TimingContext":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        
        if self.component == "total":
            self.analyzer.record_frame_time(self.elapsed)
        elif self.component == "detection":
            self.analyzer._detection_times.append(self.elapsed)
        elif self.component == "tracking":
            self.analyzer._tracking_times.append(self.elapsed)
