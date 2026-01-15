"""
Evaluation framework for multi-object tracking without ground truth.

This module provides self-supervised metrics to assess tracking quality:
- Track fragmentation analysis
- ID switch detection
- Processing speed benchmarking
- Track lifecycle statistics

These metrics help identify tracking issues without requiring labeled data.
"""

from .metrics import (
    TrackMetrics,
    FragmentationMetrics,
    PerformanceMetrics,
    EvaluationResult,
    TrackLifecycle,
    IDSwitchEvent,
    IDSwitchMetrics,
)
from .analyzer import TrackingAnalyzer, TimingContext
from .reporter import EvaluationReporter
from .integration import (
    get_analyzer,
    reset_analyzer,
    configure_analyzer,
    time_frame,
    timed,
    EvaluatedPipeline,
    quick_evaluate,
)

__all__ = [
    # Metrics
    "TrackMetrics",
    "FragmentationMetrics",
    "PerformanceMetrics",
    "EvaluationResult",
    "TrackLifecycle",
    "IDSwitchEvent",
    "IDSwitchMetrics",
    # Analyzer
    "TrackingAnalyzer",
    "TimingContext",
    # Reporter
    "EvaluationReporter",
    # Integration
    "get_analyzer",
    "reset_analyzer",
    "configure_analyzer",
    "time_frame",
    "timed",
    "EvaluatedPipeline",
    "quick_evaluate",
]
