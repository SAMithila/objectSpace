"""
Pipeline integration for the evaluation framework.

Provides a decorator and context manager to automatically
record timing metrics during pipeline execution.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

from .analyzer import TrackingAnalyzer, TimingContext
from .metrics import EvaluationResult
from .reporter import EvaluationReporter


logger = logging.getLogger(__name__)

# Global analyzer instance for convenience
_global_analyzer: Optional[TrackingAnalyzer] = None


def get_analyzer() -> TrackingAnalyzer:
    """Get or create the global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = TrackingAnalyzer()
    return _global_analyzer


def reset_analyzer() -> None:
    """Reset the global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is not None:
        _global_analyzer.reset_timings()


def configure_analyzer(
    min_track_length: int = 5,
    id_switch_iou_threshold: float = 0.3,
    id_switch_distance_threshold: float = 100.0,
    target_fps: float = 30.0,
) -> TrackingAnalyzer:
    """
    Configure and return a new global analyzer.
    
    Args:
        min_track_length: Minimum frames for non-short track
        id_switch_iou_threshold: IoU threshold for ID switch detection
        id_switch_distance_threshold: Distance threshold for ID switch
        target_fps: Target FPS for speed scoring
    
    Returns:
        Configured TrackingAnalyzer instance
    """
    global _global_analyzer
    _global_analyzer = TrackingAnalyzer(
        min_track_length=min_track_length,
        id_switch_iou_threshold=id_switch_iou_threshold,
        id_switch_distance_threshold=id_switch_distance_threshold,
        target_fps=target_fps,
    )
    return _global_analyzer


@contextmanager
def time_frame(component: str = "total"):
    """
    Context manager to time a frame processing component.
    
    Args:
        component: One of "total", "detection", "tracking"
    
    Usage:
        with time_frame("detection"):
            detections = detector.detect(frame)
        
        with time_frame("tracking"):
            tracks = tracker.update(detections)
    """
    analyzer = get_analyzer()
    with TimingContext(analyzer, component) as ctx:
        yield ctx


# Type variable for decorated function
F = TypeVar("F", bound=Callable[..., Any])


def timed(component: str = "total") -> Callable[[F], F]:
    """
    Decorator to time a function call.
    
    Args:
        component: One of "total", "detection", "tracking"
    
    Usage:
        @timed("detection")
        def detect_objects(frame):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with time_frame(component):
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


class EvaluatedPipeline:
    """
    Wrapper that adds evaluation to an existing pipeline.
    
    Usage:
        from smart_desk_monitor import DetectionTrackingPipeline
        from smart_desk_monitor.evaluation import EvaluatedPipeline
        
        pipeline = DetectionTrackingPipeline()
        evaluated = EvaluatedPipeline(pipeline)
        
        results, evaluation = evaluated.process_video("video.mp4")
    """
    
    def __init__(
        self,
        pipeline: Any,
        analyzer: Optional[TrackingAnalyzer] = None,
        reporter: Optional[EvaluationReporter] = None,
    ):
        """
        Initialize with an existing pipeline.
        
        Args:
            pipeline: The detection/tracking pipeline to wrap
            analyzer: Optional custom analyzer (uses global if not provided)
            reporter: Optional custom reporter
        """
        self.pipeline = pipeline
        self.analyzer = analyzer or get_analyzer()
        self.reporter = reporter or EvaluationReporter()
    
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> tuple[Dict[str, Any], EvaluationResult]:
        """
        Process video with evaluation.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            **kwargs: Additional arguments for pipeline
        
        Returns:
            Tuple of (pipeline_results, evaluation_result)
        """
        # Reset timing for this video
        self.analyzer.reset_timings()
        
        # Run pipeline
        # Note: The pipeline should use time_frame() internally
        # or we can wrap the process method
        results = self.pipeline.process_video(video_path, output_dir=output_dir, **kwargs)
        
        # Get annotations from results
        annotations = results.get("annotations", {})
        video_name = results.get("video_name", video_path)
        
        # Get config if available
        config = {}
        if hasattr(self.pipeline, "config"):
            config = self.pipeline.config.__dict__ if hasattr(self.pipeline.config, "__dict__") else {}
        
        # Run evaluation
        evaluation = self.analyzer.analyze(
            annotations,
            video_name=video_name,
            config=config,
        )
        
        return results, evaluation
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> tuple[list[Dict[str, Any]], list[EvaluationResult]]:
        """
        Process directory with evaluation.
        
        Args:
            input_dir: Input directory with videos
            output_dir: Output directory for results
            **kwargs: Additional arguments for pipeline
        
        Returns:
            Tuple of (list of pipeline_results, list of evaluation_results)
        """
        all_results = []
        all_evaluations = []
        
        # Get video files
        from pathlib import Path
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = [
            f for f in Path(input_dir).iterdir()
            if f.suffix.lower() in video_extensions
        ]
        
        for video_path in video_files:
            results, evaluation = self.process_video(
                str(video_path),
                output_dir=output_dir,
                **kwargs,
            )
            all_results.append(results)
            all_evaluations.append(evaluation)
        
        return all_results, all_evaluations
    
    def print_summary(self, evaluation: EvaluationResult) -> None:
        """Print evaluation summary to console."""
        self.reporter.print_summary(evaluation)
    
    def save_evaluation(
        self,
        evaluation: EvaluationResult,
        output_dir: str,
        formats: list[str] = ["json", "markdown"],
    ) -> None:
        """
        Save evaluation in specified formats.
        
        Args:
            evaluation: Evaluation results
            output_dir: Output directory
            formats: List of formats ("json", "markdown")
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        
        if "json" in formats:
            self.reporter.save_json(
                evaluation,
                output_dir / f"{evaluation.video_name}_evaluation.json",
            )
        
        if "markdown" in formats:
            self.reporter.save_markdown(
                evaluation,
                output_dir / f"{evaluation.video_name}_evaluation.md",
            )


def quick_evaluate(
    annotations_path: str,
    output_dir: Optional[str] = None,
    print_summary: bool = True,
) -> EvaluationResult:
    """
    Quick evaluation of existing COCO annotations.
    
    Args:
        annotations_path: Path to COCO JSON file
        output_dir: Optional output directory for reports
        print_summary: Whether to print summary to console
    
    Returns:
        EvaluationResult
    """
    import json
    from pathlib import Path
    
    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    # Get video name from path
    video_name = Path(annotations_path).stem.replace("_annotations", "")
    
    # Analyze
    analyzer = get_analyzer()
    result = analyzer.analyze(annotations, video_name=video_name)
    
    # Report
    reporter = EvaluationReporter()
    
    if print_summary:
        reporter.print_summary(result)
    
    if output_dir:
        output_dir = Path(output_dir)
        reporter.save_json(result, output_dir / f"{video_name}_evaluation.json")
        reporter.save_markdown(result, output_dir / f"{video_name}_evaluation.md")
    
    return result
