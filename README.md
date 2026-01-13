# Smart Desk Monitor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality object detection and tracking pipeline for desk/workspace monitoring. Built with Mask R-CNN for detection and SORT (Simple Online and Realtime Tracking) algorithm with Kalman filtering for robust multi-object tracking.

## Features

- **Object Detection**: Pre-trained Mask R-CNN with per-class confidence thresholds
- **Multi-Object Tracking**: SORT algorithm with 8D Kalman filtering
- **Self-Supervised Evaluation**: Track quality metrics without ground truth
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
│   ├── evaluation/         # Quality assessment framework
│   │   ├── metrics.py      # Metric dataclasses (FragmentationMetrics, etc.)
│   │   ├── analyzer.py     # TrackingAnalyzer for computing metrics
│   │   ├── reporter.py     # Report generation (console, JSON, Markdown)
│   │   ├── integration.py  # Pipeline integration utilities
│   │   └── cli.py          # Evaluation CLI commands
│   ├── io/                 # Input/Output utilities
│   │   ├── video.py        # Video reading & frame extraction
│   │   └── export.py       # COCO JSON & visualization export
│   ├── pipeline.py         # Main orchestration
│   └── cli.py              # Command-line interface
├── tests/                  
│   ├── evaluation/         # Evaluation framework tests
│   │   ├── test_metrics.py
│   │   ├── test_analyzer.py
│   │   └── test_reporter.py
│   ├── test_config.py
│   ├── test_detection.py
│   └── test_tracking.py
├── configs/                
│   ├── default.yaml        # Default configuration
│   └── tuned.yaml          # Optimized parameters
└── pyproject.toml          
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/SAMithila/smart-desk-monitor.git
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

### Process with Evaluation

```python
from smart_desk_monitor import DetectionTrackingPipeline

pipeline = DetectionTrackingPipeline()

# Process video AND get quality metrics
results, evaluation = pipeline.process_video_with_evaluation(
    "video.mp4",
    output_dir="output/"
)

# Access metrics
print(f"Overall Score: {evaluation.overall_score:.1f}/100")
print(f"ID Switches: {evaluation.id_switches.total_switches}")
print(f"Track Continuity: {evaluation.continuity_score:.1f}%")
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

---

## Evaluation Framework

The evaluation framework provides self-supervised metrics to assess tracking quality **without requiring ground truth annotations**. This is valuable for real-world scenarios where labeled data is unavailable.

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Continuity Score** | Track completeness - penalizes gaps and fragmentation |
| **Stability Score** | ID consistency - penalizes ID switches |
| **Speed Score** | Processing performance relative to target FPS |
| **Overall Score** | Weighted combination (40% continuity, 40% stability, 20% speed) |

### Detailed Metrics

- **Track Fragmentation**: Coverage ratio, gap count, average gap length
- **ID Switches**: Detected using spatial proximity and IoU overlap heuristics
- **Track Lifecycle**: Duration distribution, short track detection
- **Performance**: FPS, frame time percentiles (P50, P95)

### Usage

#### Evaluate Existing Annotations

```python
from smart_desk_monitor.pipeline import evaluate_annotations

# Analyze tracking quality from previous run
result = evaluate_annotations("output/video_annotations.json")

print(f"Overall Score: {result.overall_score:.1f}/100")
print(f"Fragmented Tracks: {result.fragmentation.fragmented_tracks}")
print(f"ID Switches: {result.id_switches.total_switches}")
```

#### Compare Multiple Videos

```python
from smart_desk_monitor.evaluation import TrackingAnalyzer, EvaluationReporter

analyzer = TrackingAnalyzer()
reporter = EvaluationReporter()

results = [
    analyzer.analyze(annotations1, video_name="video1"),
    analyzer.analyze(annotations2, video_name="video2"),
]

# Print comparison table
reporter.compare_results(results)
```

#### CLI Evaluation

```bash
# Evaluate single annotation file
python run_evaluation.py

# Compare all videos
python compare_videos.py

# Generate summary report
python generate_report.py
```

### Evaluation Results

Performance across test videos:

| Video | Overall | Continuity | Stability | Tracks | ID Switches | Fragmented |
|-------|---------|------------|-----------|--------|-------------|------------|
| task3.1_video1 | 36.8 | 66.5 | 25.4 | 23 | 6 | 13 |
| task3.1_video2 | 44.4 | 67.8 | 43.3 | 11 | 3 | 6 |
| task3.1_video4 | 78.4 | 95.9 | 100.0 | 8 | 0 | 1 |
| **Average** | **53.2** | **76.7** | **56.3** | **42** | **9** | **20** |

### Key Findings

1. **Tracker scales with scene complexity**: 
   - Simple scenes (8 tracks) → 100% stability
   - Complex scenes (23 tracks) → 25% stability

2. **Primary bottleneck identified**: ID association in crowded scenes
   - IoU-based matching struggles when objects are close together
   - Recommended improvement: Add appearance features (Deep SORT)

3. **Parameter sensitivity**:
   - `max_age=8` causes premature track deletion (avg gap = 8.9 frames)
   - `iou_threshold=0.3` rejects valid matches (avg switch IoU = 0.36)

### Recommended Parameter Tuning

Based on evaluation results, `configs/tuned.yaml` provides optimized settings:

```yaml
tracker:
  max_age: 15          # Increased from 8 (handles longer occlusions)
  iou_threshold: 0.2   # Reduced from 0.3 (fewer false ID switches)
```

---

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

1. **Separation of Concerns**: Detection, tracking, evaluation, and I/O are independent modules
2. **Dependency Injection**: Components receive configuration, not global state
3. **Abstract Interfaces**: `BaseDetector` allows swapping detection backends
4. **Type Safety**: Full type hints enable IDE support and catch errors early
5. **Configuration as Code**: YAML configs with typed dataclasses
6. **CLI + API**: Both programmatic and command-line interfaces
7. **Self-Supervised Evaluation**: Quality metrics without ground truth dependency

## Extending the Pipeline

### Adding a New Detector

```python
from smart_desk_monitor.detection import BaseDetector, DetectionResult

class YOLODetector(BaseDetector):
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # Your implementation
        pass
```

### Adding Custom Evaluation Metrics

```python
from smart_desk_monitor.evaluation import TrackingAnalyzer

class CustomAnalyzer(TrackingAnalyzer):
    def _compute_custom_metric(self, frame_annotations):
        # Your custom metric logic
        pass
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [SORT Paper](https://arxiv.org/abs/1602.00763): Bewley et al., "Simple Online and Realtime Tracking"
- [Mask R-CNN](https://arxiv.org/abs/1703.06870): He et al., "Mask R-CNN"
- [torchvision](https://pytorch.org/vision/): Pre-trained models
