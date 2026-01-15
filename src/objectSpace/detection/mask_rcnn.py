"""
Mask R-CNN detector implementation using torchvision.

This module provides a concrete implementation of the BaseDetector
interface using the pre-trained Mask R-CNN model from torchvision.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

from .base import BaseDetector, Detection, DetectionResult
from ..config import DetectorConfig

logger = logging.getLogger(__name__)


class MaskRCNNDetector(BaseDetector):
    """
    Object detector using Mask R-CNN with ResNet-50 FPN backbone.

    This detector provides instance segmentation with bounding boxes,
    class labels, confidence scores, and optional segmentation masks.

    Example:
        >>> config = DetectorConfig(device="cuda")
        >>> detector = MaskRCNNDetector(config)
        >>> result = detector.detect(rgb_frame)
        >>> for det in result:
        ...     print(f"Found {det.class_id} at {det.bbox}")
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the Mask R-CNN detector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self._config = config or DetectorConfig()
        self._device = self._resolve_device()
        self._model = self._load_model()

        logger.info(
            f"Initialized MaskRCNNDetector on {self._device} "
            f"(default_conf={self._config.default_confidence})"
        )

    def _resolve_device(self) -> torch.device:
        """Resolve the device to use based on configuration and availability."""
        if self._config.device == "auto":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self._config.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")

        return device

    def _load_model(self) -> torch.nn.Module:
        """Load and configure the Mask R-CNN model."""
        logger.debug("Loading Mask R-CNN model...")

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model.eval()
        model.to(self._device)

        logger.debug("Model loaded successfully")
        return model

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a frame for model input.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            Tensor ready for model inference
        """
        # Convert to tensor and normalize to [0, 1]
        tensor = F.to_tensor(frame)
        return tensor.unsqueeze(0).to(self._device)

    def _postprocess(
        self,
        output: dict,
        frame_shape: tuple
    ) -> DetectionResult:
        """
        Convert model output to DetectionResult.

        Args:
            output: Raw model output dictionary
            frame_shape: Original frame shape (H, W, C)

        Returns:
            DetectionResult with filtered detections
        """
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        masks = output.get('masks')

        if masks is not None:
            masks = masks.cpu().numpy()

        detections = []

        for i in range(len(boxes)):
            class_id = int(labels[i])
            confidence = float(scores[i])
            threshold = self._config.get_threshold(class_id)

            if confidence < threshold:
                continue

            detection = Detection(
                bbox=boxes[i].astype(np.float32),
                class_id=class_id,
                confidence=confidence,
                mask=masks[i, 0] if masks is not None else None
            )
            detections.append(detection)

        return DetectionResult(detections=detections, frame_shape=frame_shape)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: RGB image as numpy array, shape (H, W, 3)

        Returns:
            DetectionResult containing all detections above threshold
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image (H, W, 3), got shape {frame.shape}")

        input_tensor = self._preprocess(frame)

        with torch.no_grad():
            outputs = self._model(input_tensor)

        result = self._postprocess(outputs[0], frame.shape)

        # Clean up GPU memory
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Run detection on a batch of frames.

        For memory efficiency, processes frames one at a time.
        Override in subclass for true batch processing if needed.

        Args:
            frames: List of RGB images

        Returns:
            List of DetectionResult, one per frame
        """
        results = []
        for frame in frames:
            result = self.detect(frame)
            results.append(result)
        return results

    @property
    def device(self) -> str:
        """Return the device being used."""
        return str(self._device)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "maskrcnn_resnet50_fpn"

    def __repr__(self) -> str:
        return f"MaskRCNNDetector(device={self._device}, model={self.model_name})"
