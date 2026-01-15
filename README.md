# ObjectSpace

> 🎯 A production-quality object detection & tracking pipeline for workspace monitoring — demonstrating real-world ML engineering with self-supervised evaluation metrics.

![Demo](assets/demo.gif)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🚀 Why This Project?

Smart workspace monitoring enables:
- **Productivity analytics** — Track object interactions over time
- **Ergonomics research** — Monitor desk setup and posture indicators  
- **Automated inventory** — Detect and track items on workspaces

This project demonstrates **end-to-end ML pipeline engineering**: from raw video to tracked objects with quality metrics — all without requiring ground truth annotations.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | Pre-trained Mask R-CNN with configurable confidence thresholds |
| **Multi-Object Tracking** | SORT algorithm with 8D Kalman filtering |
| **Self-Supervised Evaluation** | Quality metrics without ground truth |
| **Modular Architecture** | Clean separation of detection, tracking, I/O, and evaluation |
| **Multiple Outputs** | COCO JSON annotations, visualization frames, evaluation reports |
| **CLI + Python API** | Flexible usage for scripts or integration |

---

## 📊 Evaluation Results

The built-in evaluation framework measures tracking quality **without ground truth**:

| Video | Overall | Continuity | Stability | Tracks | ID Switches |
|-------|---------|------------|-----------|--------|-------------|
| video1 (complex) | 36.8 | 66.5 | 25.4 | 23 | 6 |
| video2 (medium) | 44.4 | 67.8 | 43.3 | 11 | 3 |
| video4 (simple) | 78.4 | 95.9 | 100.0 | 8 | 0 |
| **Average** | **53.2** | **76.7** | **56.3** | - | - |

### Key Findings

- ✅ **100% stability** on simple scenes (≤8 concurrent tracks)
- ⚠️ **Stability degrades** with scene complexity (IoU-based matching limitation)
- 🔧 **Identified bottleneck**: ID association in crowded scenes → recommends Deep SORT

---

## 🏗️ Architecture

```
objectSpace/
├── src/objectSpace/
│   ├── detection/          # Mask R-CNN object detection
│   ├── tracking/           # SORT with Kalman filtering
│   ├── evaluation/         # Self-supervised quality metrics
│   ├── io/                 # Video I/O and COCO export
│   ├── pipeline.py         # Main orchestration
│   └── config.py           # Typed configuration
├── tests/                  # Unit & integration tests
├── configs/                # YAML configurations
└── assets/                 # Demo media
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/SAMithila/objectSpace.git
cd objectSpace
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Process a Video

```python
from objectSpace import DetectionTrackingPipeline

pipeline = DetectionTrackingPipeline()
results = pipeline.process_video("video.mp4", output_dir="output/")
```

### Process with Evaluation

```python
# Get tracking results + quality metrics
results, evaluation = pipeline.process_video_with_evaluation("video.mp4")

print(f"Overall Score: {evaluation.overall_score:.1f}/100")
print(f"ID Switches: {evaluation.id_switches.total_switches}")
```

### CLI Usage

```bash
# Process single video
objectSpace process video.mp4 -o output/

# Process directory
objectSpace process videos/ -o output/
```

---

## 📈 Evaluation Framework

The evaluation module computes tracking quality **without ground truth annotations**:

### Metrics

| Metric | What It Measures |
|--------|------------------|
| **Continuity Score** | Track completeness (gaps, fragmentation) |
| **Stability Score** | ID consistency (fewer switches = better) |
| **Speed Score** | Processing FPS vs target |
| **Overall Score** | Weighted combination |

### Usage

```python
from objectSpace.pipeline import evaluate_annotations

# Evaluate existing tracking results
result = evaluate_annotations("output/video_annotations.json")

print(f"Fragmented tracks: {result.fragmentation.fragmented_tracks}")
print(f"ID switches: {result.id_switches.total_switches}")
print(f"Avg coverage: {result.fragmentation.avg_coverage_ratio:.1%}")
```

### Compare Videos

```bash
python compare_videos.py
```

Output:
```
EVALUATION COMPARISON
================================================================================
Video                      Overall    Cont.    Stab.    Speed  Tracks
--------------------------------------------------------------------------------
task3.1_video1                36.8     66.5     25.4      0.0      23
task3.1_video2                44.4     67.8     43.3      0.0      11
task3.1_video4                78.4     95.9    100.0      0.0       8
--------------------------------------------------------------------------------
AVERAGE                       53.2     76.7     56.3      0.0      42
```

---

## ⚙️ Configuration

Default settings in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector.device` | auto | CPU/CUDA selection |
| `detector.default_confidence` | 0.3 | Detection threshold |
| `tracker.max_age` | 8 | Frames to keep lost tracks |
| `tracker.iou_threshold` | 0.3 | Minimum IoU for matching |

### Tuned Configuration

Based on evaluation results, `configs/tuned.yaml` improves performance:

```yaml
tracker:
  max_age: 15          # Handles longer occlusions
  iou_threshold: 0.2   # Fewer false ID switches
```

---

## 🧪 Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# All checks
make check
```

---

## 📁 Output Format

### COCO JSON with Tracking

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 100, 50, 80],
      "track_id": 0
    }
  ]
}
```

### Evaluation Reports

- `*_evaluation.json` — Machine-readable metrics
- `*_evaluation.md` — Human-readable report
- `EVALUATION_SUMMARY.md` — Cross-video comparison

---

## 🛠️ Extending

### Add New Detector

```python
from objectSpace.detection import BaseDetector

class YOLODetector(BaseDetector):
    def detect(self, frame):
        # Your implementation
        pass
```

### Add Custom Metrics

```python
from objectSpace.evaluation import TrackingAnalyzer

class CustomAnalyzer(TrackingAnalyzer):
    def compute_custom_metric(self, annotations):
        # Your metric logic
        pass
```

---

## 📚 Technical Highlights

This project demonstrates:

1. **Modular Design** — Separate concerns for detection, tracking, evaluation
2. **Type Safety** — Full type hints with dataclasses
3. **Configuration Management** — YAML configs with typed validation
4. **Self-Supervised ML** — Quality metrics without labeled data
5. **Production Patterns** — Logging, error handling, CLI interface

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [SORT](https://arxiv.org/abs/1602.00763) — Bewley et al.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870) — He et al.
- [torchvision](https://pytorch.org/vision/) — Pre-trained models# CI trigger test
