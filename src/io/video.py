"""
Video input/output utilities.

This module provides video reading, frame extraction, and sampling
functionality for processing video files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def __str__(self) -> str:
        return (
            f"Video({self.path.name}: {self.width}x{self.height}, "
            f"{self.fps:.2f}fps, {self.frame_count} frames, "
            f"{self.duration_seconds:.2f}s)"
        )


class VideoReader:
    """
    Video file reader with frame sampling capabilities.

    This class provides efficient video reading with support for
    uniform frame sampling and resizing.

    Args:
        path: Path to video file
        config: Video configuration for target resolution and sampling

    Example:
        >>> reader = VideoReader("video.mp4")
        >>> print(reader.metadata)
        >>> frames = reader.sample_frames(n_frames=100)
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[VideoConfig] = None
    ):
        self._path = Path(path)
        self._config = config or VideoConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

        self._validate_path()
        self._load_metadata()

    def _validate_path(self) -> None:
        """Validate video file exists and has supported format."""
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        suffix = self._path.suffix.lower()
        if suffix not in self._config.supported_formats:
            raise ValueError(
                f"Unsupported video format: {suffix}. "
                f"Supported: {self._config.supported_formats}"
            )

    def _load_metadata(self) -> None:
        """Load video metadata."""
        cap = cv2.VideoCapture(str(self._path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self._metadata = VideoMetadata(
                path=self._path,
                width=width,
                height=height,
                fps=fps if fps > 0 else 30.0,
                frame_count=frame_count,
                duration_seconds=frame_count / max(fps, 1.0)
            )

            logger.info(f"Loaded video: {self._metadata}")

        finally:
            cap.release()

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        if self._metadata is None:
            raise RuntimeError("Video metadata not loaded")
        return self._metadata

    def _open(self) -> cv2.VideoCapture:
        """Open video capture if not already open."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self._path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self._path}")
        return self._cap

    def _close(self) -> None:
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        target_size = (self._config.target_width, self._config.target_height)
        return cv2.resize(frame, target_size)

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a specific frame from the video.

        Args:
            frame_idx: Zero-based frame index

        Returns:
            RGB frame as numpy array, or None if read failed
        """
        cap = self._open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            return None

        frame = self._resize_frame(frame)
        return self._to_rgb(frame)

    def sample_frames(
        self,
        n_frames: Optional[int] = None,
        return_indices: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
        """
        Sample frames uniformly from the video.

        Args:
            n_frames: Number of frames to sample. Uses config default if None.
            return_indices: If True, also return the frame indices

        Returns:
            List of RGB frames, optionally with frame indices
        """
        n_frames = n_frames or self._config.max_frames
        total_frames = self.metadata.frame_count

        # Calculate sampling step
        step = max(1, total_frames // n_frames)

        frames = []
        indices = []

        cap = self._open()

        try:
            for i in range(0, total_frames, step):
                if len(frames) >= n_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame {i}")
                    continue

                frame = self._resize_frame(frame)
                frame = self._to_rgb(frame)
                frames.append(frame)
                indices.append(i)

            logger.info(f"Sampled {len(frames)} frames from {self._path.name}")

        finally:
            self._close()

        if return_indices:
            return frames, indices
        return frames

    def iterate_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate over video frames.

        Args:
            start: Starting frame index
            end: Ending frame index (exclusive). None = end of video.
            step: Frame step size

        Yields:
            Tuples of (frame_index, rgb_frame)
        """
        end = end or self.metadata.frame_count
        cap = self._open()

        try:
            for i in range(start, end, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    break

                frame = self._resize_frame(frame)
                frame = self._to_rgb(frame)
                yield i, frame

        finally:
            self._close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self._close()

    def __del__(self) -> None:
        self._close()


def find_videos(
    directory: Union[str, Path],
    config: Optional[VideoConfig] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[Path]:
    """
    Find all supported video files in a directory.

    Args:
        directory: Directory to search
        config: Video config for supported formats
        exclude_patterns: Substrings to exclude from filenames

    Returns:
        List of video file paths
    """
    directory = Path(directory)
    config = config or VideoConfig()
    exclude_patterns = exclude_patterns or []

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    videos = []
    for fmt in config.supported_formats:
        for path in directory.glob(f"*{fmt}"):
            # Check exclusion patterns
            if any(pattern in path.name for pattern in exclude_patterns):
                continue
            videos.append(path)

    videos.sort()
    logger.info(f"Found {len(videos)} videos in {directory}")

    return videos
