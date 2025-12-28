"""
Main pipeline for object detection and tracking.

This module orchestrates the complete processing pipeline,
from video input through detection, tracking, and output generation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

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

logger = logging.getLogger(__name__)


class DetectionTrackingPipeline:
    """
    End-to-end pipeline for video object detection and tracking.

    This class orchestrates the complete workflow:
    1. Video frame extraction
    2. Object detection on each frame
    3. Multi-object tracking across frames
    4. Result export (COCO JSON, visualizations)

    Args:
        config: Pipeline configuration. Uses defaults if None.

    Example:
        >>> pipeline = DetectionTrackingPipeline()
        >>> results = pipeline.process_video("input.mp4", output_dir="output/")
        >>> 
        >>> # Or process multiple videos
        >>> pipeline.process_directory("videos/", output_dir="output/")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline with configuration."""
        self._config = config or get_default_config()
        self._setup_logging()

        # Initialize components (lazy loading)
        self._detector: Optional[MaskRCNNDetector] = None
        self._tracker: Optional[SORTTracker] = None
        self._visualizer: Optional[TrackingVisualizer] = None

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

    def _detect_frames(
        self,
        frames: List[np.ndarray],
        show_progress: bool = True
    ) -> List[DetectionResult]:
        """Run detection on all frames."""
        results = []
        iterator = tqdm(frames, desc="Detecting") if show_progress else frames

        for frame in iterator:
            result = self.detector.detect(frame)
            results.append(result)

        logger.info(f"Detected objects in {len(results)} frames")
        return results

    def _track_detections(
        self,
        detection_results: List[DetectionResult],
        show_progress: bool = True
    ) -> List[TrackingResult]:
        """Run tracking on detection results."""
        # Reset tracker for new video
        self.tracker.reset()

        tracking_results = []
        iterator = enumerate(detection_results)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Tracking")

        for frame_idx, det_result in iterator:
            boxes, class_ids, _ = det_result.to_numpy()

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
    ) -> Path:
        """Save all results to disk."""
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

        # Save COCO JSON
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

        logger.info(f"Results saved to {video_output_dir}")
        return video_output_dir

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


def run_pipeline(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> dict:
    """
    Convenience function to run the pipeline.

    Args:
        input_path: Video file or directory of videos
        output_dir: Output directory
        config: Pipeline configuration
        **kwargs: Additional arguments passed to process methods

    Returns:
        Dictionary of results
    """
    pipeline = DetectionTrackingPipeline(config)
    input_path = Path(input_path)

    if input_path.is_file():
        results = pipeline.process_video(input_path, output_dir, **kwargs)
        return {input_path.stem: results}
    else:
        return pipeline.process_directory(input_path, output_dir, **kwargs)
