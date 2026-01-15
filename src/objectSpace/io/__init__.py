"""
I/O module for Smart Desk Monitor.

This module provides video reading, frame export, and result
serialization functionality.

Example:
    >>> from smart_desk_monitor.io import VideoReader, COCOExporter
    >>> reader = VideoReader("video.mp4")
    >>> frames = reader.sample_frames(100)
"""

from .video import (
    VideoReader,
    VideoMetadata,
    find_videos,
)
from .export import (
    COCOExporter,
    COCOImage,
    COCOAnnotation,
    COCOCategory,
    TrackingVisualizer,
    save_frames,
)

__all__ = [
    # Video I/O
    "VideoReader",
    "VideoMetadata",
    "find_videos",
    # Export
    "COCOExporter",
    "COCOImage",
    "COCOAnnotation",
    "COCOCategory",
    "TrackingVisualizer",
    "save_frames",
]
