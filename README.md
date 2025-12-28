# Smart Desk Monitor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality object detection and tracking pipeline for desk/workspace monitoring. Built with Mask R-CNN for detection and SORT (Simple Online and Realtime Tracking) algorithm with Kalman filtering for robust multi-object tracking.

## Features

- **Object Detection**: Pre-trained Mask R-CNN with per-class confidence thresholds
- **Multi-Object Tracking**: SORT algorithm with 8D Kalman filtering
- **Modular Architecture**: Clean separation of concerns for easy extension
- **Multiple Output Formats**: COCO JSON annotations, visualization frames
- **CLI & API**: Use from command line or integrate into Python code
- **Type Safety**: Full type hints throughout the codebase

## Architecture

```
smart_desk_monitor/
├── src/smart_desk_monitor/
│   ├── config.py           # Centralized configuration
│   ├── detection/          # Object detection module
│   │   ├── base.py         # Abstract detector interface
│   │   └── mask_rcnn.py    # Mask R-CNN implementation
│   ├── tracking/           # Multi-object tracking
│   │   ├── kalman.py       # Kalman filter for motion prediction
│   │   ├── association.py  # IoU & Hungarian matching
│   │   └── sort_tracker.py # SORT algorithm implementation
│   ├── io/                 # Input/Output utilities
│   │   ├── video.py        # Video reading & frame extraction
│   │   └── export.py       # COCO JSON & visualization export
│   ├── pipeline.py         # Main orchestration
│   └── cli.py              # Command-line interface
├── tests/                  # Unit & integration tests
├── configs/                # Configuration files
└── pyproject.toml          # Package configuration
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-desk-monitor.git
cd smart-desk-monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

## Quick Start

### Command Line

```bash
# Process a single video
smart-desk-monitor process video.mp4 -o output/

# Process all videos in a directory
smart-desk-monitor process videos/ -o output/

# Use custom config
smart-desk-monitor process video.mp4 -c configs/custom.yaml -o output/

# Generate default config file
smart-desk-monitor config --generate my_config.yaml
```

### Python API

```python
from smart_desk_monitor import DetectionTrackingPipeline, PipelineConfig

# Use default configuration
pipeline = DetectionTrackingPipeline()

# Process a video
results = pipeline.process_video("video.mp4", output_dir="output/")

# Or process multiple videos
results = pipeline.process_directory("videos/", output_dir="output/")
```

### Custom Configuration

```python
from smart_desk_monitor import PipelineConfig, DetectorConfig, TrackerConfig

config = PipelineConfig(
    detector=DetectorConfig(
        device="cuda",
        default_confidence=0.4,
    ),
    tracker=TrackerConfig(
        max_age=10,
        iou_threshold=0.25,
    ),
)

pipeline = DetectionTrackingPipeline(config)
```

## Configuration

See `configs/default.yaml` for all available options:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| detector | device | auto | Device for inference (auto/cuda/cpu) |
| detector | default_confidence | 0.3 | Default detection threshold |
| tracker | max_age | 8 | Frames to keep track without detection |
| tracker | iou_threshold | 0.3 | Minimum IoU for association |
| video | max_frames | 100 | Frames to sample per video |

## Output Format

### COCO JSON

Annotations are exported in COCO format with tracking extensions:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 100, 50, 80],
      "area": 4000,
      "track_id": 0
    }
  ],
  "categories": [...]
}
```

### Visualization

Tracked frames with bounding boxes and IDs are saved to `output/video_name/tracked/`.

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Run all checks (lint + type-check + test)
make check

# Format code
make format
```

## Project Structure Rationale

This project follows software engineering best practices:

1. **Separation of Concerns**: Detection, tracking, and I/O are independent modules
2. **Dependency Injection**: Components receive configuration, not global state
3. **Abstract Interfaces**: `BaseDetector` allows swapping detection backends
4. **Type Safety**: Full type hints enable IDE support and catch errors early
5. **Configuration as Code**: YAML configs with typed dataclasses
6. **CLI + API**: Both programmatic and command-line interfaces

## Extending the Pipeline

### Adding a New Detector

```python
from smart_desk_monitor.detection import BaseDetector, DetectionResult

class YOLODetector(BaseDetector):
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # Your implementation
        pass
```

### Adding Custom Metrics

See Phase 2 (Evaluation Framework) for adding MOTA/MOTP metrics.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [SORT Paper](https://arxiv.org/abs/1602.00763): Bewley et al., "Simple Online and Realtime Tracking"
- [Mask R-CNN](https://arxiv.org/abs/1703.06870): He et al., "Mask R-CNN"
- [torchvision](https://pytorch.org/vision/): Pre-trained models
