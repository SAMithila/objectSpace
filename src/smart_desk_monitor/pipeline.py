"""
Main pipeline for object detection and tracking.

This module orchestrates the complete processing pipeline,
from video input through detection, tracking, and output generation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, get_default_config
from .detection import MaskRCNNDetector, DetectionResult
from .tracking import SORTTracker, TrackingResult
from .io import (
    VideoReader,
    COCOExporter,
    TrackingVisualizer,
    save_frames,
    find_videos,
)

# Evaluation framework imports
from .evaluation import (
    TrackingAnalyzer,
    EvaluationReporter,
    EvaluationResult,
    time_frame,
)

logger = logging.getLogger(__name__)


class DetectionTrackingPipeline:
    """
    End-to-end pipeline for video object detection and tracking.

    This class orchestrates the complete workflow:
    1. Video frame extraction
    2. Object detection on each frame
    3. Multi-object tracking across frames
    4. Result export (COCO JSON, visualizations)
    5. Optional evaluation metrics

    Args:
        config: Pipeline configuration. Uses defaults if None.
        enable_evaluation: Whether to collect evaluation metrics (default: True)

    Example:
        >>> pipeline = DetectionTrackingPipeline()
        >>> results = pipeline.process_video("input.mp4", output_dir="output/")
        >>> 
        >>> # With evaluation
        >>> results, evaluation = pipeline.process_video_with_evaluation("input.mp4")
        >>> print(f"Tracking score: {evaluation.overall_score:.1f}/100")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        enable_evaluation: bool = True,
    ):
        """Initialize the pipeline with configuration."""
        self._config = config or get_default_config()
        self._setup_logging()

        # Initialize components (lazy loading)
        self._detector: Optional[MaskRCNNDetector] = None
        self._tracker: Optional[SORTTracker] = None
        self._visualizer: Optional[TrackingVisualizer] = None

        # Evaluation components
        self._enable_evaluation = enable_evaluation
        self._analyzer: Optional[TrackingAnalyzer] = None
        self._reporter: Optional[EvaluationReporter] = None

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                *(
                    [logging.FileHandler(self._config.log_file)]
                    if self._config.log_file else []
                )
            ]
        )

    @property
    def config(self) -> PipelineConfig:
        """Get the pipeline configuration."""
        return self._config

    @property
    def detector(self) -> MaskRCNNDetector:
        """Get or create the detector (lazy initialization)."""
        if self._detector is None:
            logger.info("Initializing detector...")
            self._detector = MaskRCNNDetector(self._config.detector)
        return self._detector

    @property
    def tracker(self) -> SORTTracker:
        """Get or create the tracker (lazy initialization)."""
        if self._tracker is None:
            logger.info("Initializing tracker...")
            self._tracker = SORTTracker(self._config.tracker)
        return self._tracker

    @property
    def visualizer(self) -> TrackingVisualizer:
        """Get or create the visualizer."""
        if self._visualizer is None:
            self._visualizer = TrackingVisualizer(self._config.output)
        return self._visualizer

    @property
    def analyzer(self) -> TrackingAnalyzer:
        """Get or create the evaluation analyzer."""
        if self._analyzer is None:
            self._analyzer = TrackingAnalyzer(
                min_track_length=5,
                id_switch_iou_threshold=0.3,
                id_switch_distance_threshold=100.0,
                target_fps=30.0,
            )
        return self._analyzer

    @property
    def reporter(self) -> EvaluationReporter:
        """Get or create the evaluation reporter."""
        if self._reporter is None:
            self._reporter = EvaluationReporter(use_colors=True)
        return self._reporter

    def _detect_frames(
        self,
        frames: List[np.ndarray],
        show_progress: bool = True
    ) -> List[DetectionResult]:
        """Run detection on all frames with timing."""
        results = []
        iterator = tqdm(frames, desc="Detecting") if show_progress else frames

        for frame in iterator:
            # Time the detection if evaluation is enabled
            if self._enable_evaluation:
                with time_frame("detection"):
                    result = self.detector.detect(frame)
            else:
                result = self.detector.detect(frame)
            results.append(result)

        logger.info(f"Detected objects in {len(results)} frames")
        return results

    def _track_detections(
        self,
        detection_results: List[DetectionResult],
        show_progress: bool = True
    ) -> List[TrackingResult]:
        """Run tracking on detection results with timing."""
        # Reset tracker for new video
        self.tracker.reset()

        tracking_results = []
        iterator = enumerate(detection_results)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Tracking")

        for frame_idx, det_result in iterator:
            boxes, class_ids, _ = det_result.to_numpy()

            # Time the tracking if evaluation is enabled
            if self._enable_evaluation:
                with time_frame("tracking"):
                    track_result = self.tracker.update(
                        boxes=boxes,
                        class_ids=class_ids,
                        frame_idx=frame_idx
                    )
            else:
                track_result = self.tracker.update(
                    boxes=boxes,
                    class_ids=class_ids,
                    frame_idx=frame_idx
                )
            tracking_results.append(track_result)

        # Log tracking statistics
        track_counts = self.tracker.get_track_count()
        total_tracks = sum(track_counts.values())
        logger.info(f"Tracking complete: {total_tracks} active tracks")

        return tracking_results

    def _save_results(
        self,
        frames: List[np.ndarray],
        tracking_results: List[TrackingResult],
        output_dir: Path,
        video_name: str
    ) -> Tuple[Path, Dict[str, Any]]:
        """Save all results to disk and return COCO annotations."""
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Save original frames
        frame_paths = save_frames(
            frames,
            video_output_dir / "frames",
            prefix="frame",
            is_rgb=True
        )

        # Save tracked visualizations
        if self._config.output.save_tracked_frames:
            tracked_dir = video_output_dir / "tracked"
            tracked_dir.mkdir(exist_ok=True)

            for i, (frame, result) in enumerate(zip(frames, tracking_results)):
                vis_frame = self.visualizer.draw_tracks(frame, result)
                self.visualizer.save_frame(
                    vis_frame,
                    tracked_dir / f"tracked_{i:04d}.jpg",
                    is_rgb=True
                )

        # Build and save COCO JSON
        annotations = {}
        if self._config.output.save_coco_json:
            exporter = COCOExporter(include_track_ids=True)

            h, w = frames[0].shape[:2]
            for i, (frame_path, result) in enumerate(zip(frame_paths, tracking_results)):
                exporter.add_tracking_result(
                    result=result,
                    image_id=i,
                    file_name=frame_path.name,
                    height=h,
                    width=w
                )

            json_path = output_dir / f"{video_name}_annotations.json"
            exporter.save(json_path)
            
            # Get annotations dict for evaluation
            annotations = exporter.to_dict()

        logger.info(f"Results saved to {video_output_dir}")
        return video_output_dir, annotations

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        n_frames: Optional[int] = None,
        show_progress: bool = True
    ) -> List[TrackingResult]:
        """
        Process a single video through the complete pipeline.

        Args:
            video_path: Path to input video
            output_dir: Directory for outputs. Uses config default if None.
            n_frames: Number of frames to sample. Uses config default if None.
            show_progress: Whether to show progress bars

        Returns:
            List of TrackingResult for each processed frame
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir or self._config.output.output_dir)
        n_frames = n_frames or self._config.video.max_frames

        logger.info(f"Processing video: {video_path}")

        # Reset evaluation timing for new video
        if self._enable_evaluation:
            self.analyzer.reset_timings()

        # Step 1: Extract frames
        with VideoReader(video_path, self._config.video) as reader:
            logger.info(f"Video info: {reader.metadata}")
            frames = reader.sample_frames(n_frames)

        # Step 2: Run detection
        detection_results = self._detect_frames(frames, show_progress)

        # Step 3: Run tracking
        tracking_results = self._track_detections(
            detection_results, show_progress)

        # Step 4: Save results
        video_name = video_path.stem
        self._save_results(frames, tracking_results, output_dir, video_name)

        return tracking_results

    def process_video_with_evaluation(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        n_frames: Optional[int] = None,
        show_progress: bool = True,
        print_summary: bool = True,
        save_evaluation: bool = True,
    ) -> Tuple[List[TrackingResult], EvaluationResult]:
        """
        Process a single video and return evaluation metrics.

        This is the recommended method when you want both tracking results
        and quality metrics.

        Args:
            video_path: Path to input video
            output_dir: Directory for outputs. Uses config default if None.
            n_frames: Number of frames to sample. Uses config default if None.
            show_progress: Whether to show progress bars
            print_summary: Whether to print evaluation summary to console
            save_evaluation: Whether to save evaluation reports to disk

        Returns:
            Tuple of (TrackingResults, EvaluationResult)
        
        Example:
            >>> pipeline = DetectionTrackingPipeline()
            >>> results, evaluation = pipeline.process_video_with_evaluation("video.mp4")
            >>> print(f"Overall Score: {evaluation.overall_score:.1f}/100")
            >>> print(f"ID Switches: {evaluation.id_switches.total_switches}")
            >>> print(f"FPS: {evaluation.performance.fps:.1f}")
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir or self._config.output.output_dir)
        n_frames = n_frames or self._config.video.max_frames

        logger.info(f"Processing video with evaluation: {video_path}")

        # Reset evaluation timing for new video
        self.analyzer.reset_timings()

        # Step 1: Extract frames
        with VideoReader(video_path, self._config.video) as reader:
            logger.info(f"Video info: {reader.metadata}")
            frames = reader.sample_frames(n_frames)

        # Step 2: Run detection (with timing)
        detection_results = self._detect_frames(frames, show_progress)

        # Step 3: Run tracking (with timing)
        tracking_results = self._track_detections(
            detection_results, show_progress)

        # Step 4: Save results and get annotations
        video_name = video_path.stem
        _, annotations = self._save_results(
            frames, tracking_results, output_dir, video_name
        )

        # Step 5: Run evaluation
        evaluation = self.analyzer.analyze(
            annotations=annotations,
            video_name=video_name,
            config=self._config_to_dict(),
        )

        # Step 6: Report results
        if print_summary:
            self.reporter.print_summary(evaluation)

        if save_evaluation:
            # Save JSON report
            self.reporter.save_json(
                evaluation,
                output_dir / f"{video_name}_evaluation.json"
            )
            # Save Markdown report
            self.reporter.save_markdown(
                evaluation,
                output_dir / f"{video_name}_evaluation.md",
                include_details=True
            )

        return tracking_results, evaluation

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> dict:
        """
        Process all videos in a directory.

        Args:
            input_dir: Directory containing videos
            output_dir: Directory for outputs
            exclude_patterns: Filename patterns to exclude
            show_progress: Whether to show progress bars

        Returns:
            Dictionary mapping video names to their TrackingResults
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir or self._config.output.output_dir)

        videos = find_videos(
            input_dir,
            self._config.video,
            exclude_patterns=exclude_patterns
        )

        if not videos:
            logger.warning(f"No videos found in {input_dir}")
            return {}

        results = {}
        for i, video_path in enumerate(videos, 1):
            logger.info(f"\n{'='*60}")
            logger.info(
                f"Processing video {i}/{len(videos)}: {video_path.name}")
            logger.info(f"{'='*60}")

            try:
                tracking_results = self.process_video(
                    video_path,
                    output_dir=output_dir,
                    show_progress=show_progress
                )
                results[video_path.stem] = tracking_results

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                continue

        logger.info(
            f"\nCompleted processing {len(results)}/{len(videos)} videos")
        return results

    def process_directory_with_evaluation(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        exclude_patterns: Optional[List[str]] = None,
        show_progress: bool = True,
        print_summary: bool = True,
        save_evaluation: bool = True,
    ) -> Tuple[Dict[str, List[TrackingResult]], Dict[str, EvaluationResult]]:
        """
        Process all videos in a directory with evaluation.

        Args:
            input_dir: Directory containing videos
            output_dir: Directory for outputs
            exclude_patterns: Filename patterns to exclude
            show_progress: Whether to show progress bars
            print_summary: Whether to print evaluation summary for each video
            save_evaluation: Whether to save evaluation reports

        Returns:
            Tuple of (tracking_results_dict, evaluation_results_dict)
        
        Example:
            >>> pipeline = DetectionTrackingPipeline()
            >>> results, evaluations = pipeline.process_directory_with_evaluation("videos/")
            >>> 
            >>> # Compare all videos
            >>> pipeline.reporter.compare_results(list(evaluations.values()))
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir or self._config.output.output_dir)

        videos = find_videos(
            input_dir,
            self._config.video,
            exclude_patterns=exclude_patterns
        )

        if not videos:
            logger.warning(f"No videos found in {input_dir}")
            return {}, {}

        results = {}
        evaluations = {}
        
        for i, video_path in enumerate(videos, 1):
            logger.info(f"\n{'='*60}")
            logger.info(
                f"Processing video {i}/{len(videos)}: {video_path.name}")
            logger.info(f"{'='*60}")

            try:
                tracking_results, evaluation = self.process_video_with_evaluation(
                    video_path,
                    output_dir=output_dir,
                    show_progress=show_progress,
                    print_summary=print_summary,
                    save_evaluation=save_evaluation,
                )
                results[video_path.stem] = tracking_results
                evaluations[video_path.stem] = evaluation

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                continue

        # Print comparison of all videos
        if print_summary and len(evaluations) > 1:
            logger.info(f"\n{'='*60}")
            logger.info("OVERALL COMPARISON")
            logger.info(f"{'='*60}\n")
            self.reporter.compare_results(list(evaluations.values()))

        logger.info(
            f"\nCompleted processing {len(results)}/{len(videos)} videos")
        return results, evaluations

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for evaluation storage."""
        return {
            "detector": {
                "device": str(self._config.detector.device),
                "default_confidence": self._config.detector.default_confidence,
            },
            "tracker": {
                "max_age": self._config.tracker.max_age,
                "min_hits": self._config.tracker.min_hits,
                "iou_threshold": self._config.tracker.iou_threshold,
            },
            "video": {
                "max_frames": self._config.video.max_frames,
            },
        }


def run_pipeline(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    exclude_patterns: Optional[List[str]] = None,
    show_progress: bool = True,
    with_evaluation: bool = False,
) -> dict:
    """
    Convenience function to run the pipeline.

    Args:
        input_path: Video file or directory of videos
        output_dir: Output directory
        config: Pipeline configuration
        exclude_patterns: Filename patterns to exclude (only for directories)
        show_progress: Whether to show progress bars
        with_evaluation: Whether to include evaluation metrics

    Returns:
        Dictionary of results (includes evaluation if with_evaluation=True)
    """
    pipeline = DetectionTrackingPipeline(config, enable_evaluation=with_evaluation)
    input_path = Path(input_path)

    if input_path.is_file():
        if with_evaluation:
            results, evaluation = pipeline.process_video_with_evaluation(
                input_path,
                output_dir,
                show_progress=show_progress
            )
            return {input_path.stem: {"tracking": results, "evaluation": evaluation}}
        else:
            results = pipeline.process_video(
                input_path,
                output_dir,
                show_progress=show_progress
            )
            return {input_path.stem: results}
    else:
        if with_evaluation:
            results, evaluations = pipeline.process_directory_with_evaluation(
                input_path,
                output_dir,
                exclude_patterns=exclude_patterns,
                show_progress=show_progress
            )
            return {
                name: {"tracking": results[name], "evaluation": evaluations[name]}
                for name in results
            }
        else:
            return pipeline.process_directory(
                input_path,
                output_dir,
                exclude_patterns=exclude_patterns,
                show_progress=show_progress
            )


# Convenience function for quick evaluation of existing results
def evaluate_annotations(
    annotations_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    print_summary: bool = True,
) -> EvaluationResult:
    """
    Evaluate tracking quality from existing COCO annotations.
    
    Use this to analyze results from a previous pipeline run.
    
    Args:
        annotations_path: Path to COCO JSON file with track_id fields
        output_dir: Optional directory for evaluation reports
        print_summary: Whether to print summary to console
    
    Returns:
        EvaluationResult with all metrics
    
    Example:
        >>> from smart_desk_monitor.pipeline import evaluate_annotations
        >>> result = evaluate_annotations("output/video_annotations.json")
        >>> print(f"Score: {result.overall_score:.1f}/100")
    """
    import json
    
    annotations_path = Path(annotations_path)
    
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    video_name = annotations_path.stem.replace("_annotations", "")
    
    analyzer = TrackingAnalyzer()
    result = analyzer.analyze(annotations, video_name=video_name)
    
    reporter = EvaluationReporter()
    
    if print_summary:
        reporter.print_summary(result)
    
    if output_dir:
        output_dir = Path(output_dir)
        reporter.save_json(result, output_dir / f"{video_name}_evaluation.json")
        reporter.save_markdown(result, output_dir / f"{video_name}_evaluation.md")
    
    return result