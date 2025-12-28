"""
Export utilities for detection and tracking results.

This module provides functionality to export results in various formats
including COCO JSON, and to generate visualization images.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..config import COCO_CLASSES, OutputConfig
from ..tracking import TrackingResult

logger = logging.getLogger(__name__)


@dataclass
class COCOImage:
    """COCO format image entry."""
    id: int
    file_name: str
    height: int
    width: int


@dataclass
class COCOAnnotation:
    """COCO format annotation entry."""
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    area: float
    iscrowd: int = 0
    track_id: Optional[int] = None


@dataclass
class COCOCategory:
    """COCO format category entry."""
    id: int
    name: str
    supercategory: str = "object"


class COCOExporter:
    """
    Export tracking results to COCO JSON format.

    This exporter creates COCO-compatible JSON files that can be
    imported into annotation tools like CVAT or used for evaluation.

    Example:
        >>> exporter = COCOExporter()
        >>> exporter.add_image(0, "frame_0000.jpg", 1080, 1920)
        >>> exporter.add_annotation(0, 0, 1, [100, 100, 50, 80], track_id=0)
        >>> exporter.save("annotations.json")
    """

    def __init__(self, include_track_ids: bool = True):
        """
        Initialize the COCO exporter.

        Args:
            include_track_ids: Whether to include track_id in annotations
        """
        self._include_track_ids = include_track_ids
        self._images: List[COCOImage] = []
        self._annotations: List[COCOAnnotation] = []
        self._annotation_id = 1

        # Build categories from COCO classes (exclude background)
        self._categories = [
            COCOCategory(id=class_id, name=name)
            for class_id, name in COCO_CLASSES.items()
            if class_id > 0
        ]

    def add_image(
        self,
        image_id: int,
        file_name: str,
        height: int,
        width: int
    ) -> None:
        """Add an image entry."""
        self._images.append(COCOImage(
            id=image_id,
            file_name=file_name,
            height=height,
            width=width
        ))

    def add_annotation(
        self,
        annotation_id: Optional[int],
        image_id: int,
        category_id: int,
        bbox: Union[List[float], np.ndarray],
        track_id: Optional[int] = None
    ) -> int:
        """
        Add an annotation entry.

        Args:
            annotation_id: Unique annotation ID. Auto-generated if None.
            image_id: ID of the image this annotation belongs to
            category_id: COCO category ID
            bbox: Bounding box as [x1, y1, x2, y2] (will be converted to COCO format)
            track_id: Optional tracking ID

        Returns:
            The annotation ID used
        """
        if annotation_id is None:
            annotation_id = self._annotation_id
            self._annotation_id += 1

        # Convert [x1, y1, x2, y2] to [x, y, width, height]
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        x1, y1, x2, y2 = bbox
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]
        area = coco_bbox[2] * coco_bbox[3]

        annotation = COCOAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            bbox=coco_bbox,
            area=area,
            track_id=track_id if self._include_track_ids else None
        )

        self._annotations.append(annotation)
        return annotation_id

    def add_tracking_result(
        self,
        result: TrackingResult,
        image_id: int,
        file_name: str,
        height: int,
        width: int
    ) -> None:
        """
        Add all tracks from a TrackingResult.

        Args:
            result: TrackingResult from the tracker
            image_id: ID for this image
            file_name: Image filename
            height: Image height
            width: Image width
        """
        self.add_image(image_id, file_name, height, width)

        for class_id, track_id, bbox in result.get_all_tracks():
            self.add_annotation(
                annotation_id=None,
                image_id=image_id,
                category_id=class_id,
                bbox=bbox,
                track_id=track_id
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to COCO JSON dictionary."""
        result = {
            "images": [asdict(img) for img in self._images],
            "annotations": [],
            "categories": [asdict(cat) for cat in self._categories]
        }

        for ann in self._annotations:
            ann_dict = asdict(ann)
            if not self._include_track_ids:
                del ann_dict["track_id"]
            elif ann_dict["track_id"] is None:
                del ann_dict["track_id"]
            result["annotations"].append(ann_dict)

        return result

    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        """
        Save to JSON file.

        Args:
            path: Output file path
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

        logger.info(
            f"Saved COCO JSON: {path} "
            f"({len(self._images)} images, {len(self._annotations)} annotations)"
        )

    def reset(self) -> None:
        """Clear all images and annotations."""
        self._images.clear()
        self._annotations.clear()
        self._annotation_id = 1


class TrackingVisualizer:
    """
    Visualize tracking results on video frames.

    This class draws bounding boxes and track IDs on frames,
    with consistent colors per class.

    Args:
        config: Output configuration for visualization parameters
        seed: Random seed for color generation (for reproducibility)
    """

    def __init__(
        self,
        config: Optional[OutputConfig] = None,
        seed: int = 42
    ):
        self._config = config or OutputConfig()
        self._rng = np.random.RandomState(seed)
        self._class_colors: Dict[int, Tuple[int, int, int]] = {}

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a class."""
        if class_id not in self._class_colors:
            color = tuple(self._rng.randint(0, 255, 3).tolist())
            self._class_colors[class_id] = color
        return self._class_colors[class_id]

    def draw_tracks(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        copy: bool = True
    ) -> np.ndarray:
        """
        Draw tracking results on a frame.

        Args:
            frame: RGB or BGR image
            result: TrackingResult from tracker
            copy: If True, work on a copy of the frame

        Returns:
            Frame with visualizations drawn
        """
        if copy:
            frame = frame.copy()

        thickness = self._config.visualization_line_thickness
        font_scale = self._config.visualization_font_scale
        font = cv2.FONT_HERSHEY_SIMPLEX

        for class_id, track_id, bbox in result.get_all_tracks():
            color = self._get_color(class_id)
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            label = f"{class_name}_{track_id}"

            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

        return frame

    def save_frame(
        self,
        frame: np.ndarray,
        path: Union[str, Path],
        is_rgb: bool = True
    ) -> None:
        """
        Save a frame to disk.

        Args:
            frame: Image to save
            path: Output path
            is_rgb: If True, convert from RGB to BGR before saving
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), frame)


def save_frames(
    frames: List[np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "frame",
    is_rgb: bool = True
) -> List[Path]:
    """
    Save multiple frames to disk.

    Args:
        frames: List of images
        output_dir: Output directory
        prefix: Filename prefix
        is_rgb: If True, convert from RGB to BGR

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, frame in enumerate(frames):
        path = output_dir / f"{prefix}_{i:04d}.jpg"

        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), frame)
        paths.append(path)

    logger.info(f"Saved {len(paths)} frames to {output_dir}")
    return paths
