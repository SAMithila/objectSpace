"""
CLI commands for the evaluation framework.

Provides commands to:
- Analyze existing tracking results (COCO JSON)
- Run evaluation during pipeline processing
- Generate reports in various formats
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from .analyzer import TrackingAnalyzer
from .reporter import EvaluationReporter
from .metrics import EvaluationResult


logger = logging.getLogger(__name__)


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add evaluate subcommand to CLI."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate tracking results from COCO JSON annotations",
        description="Analyze tracking quality without ground truth",
    )
    
    parser.add_argument(
        "annotations",
        type=Path,
        help="Path to COCO JSON annotations file with track_id fields",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for evaluation reports",
    )
    
    parser.add_argument(
        "--format",
        choices=["console", "json", "markdown", "all"],
        default="console",
        help="Output format (default: console)",
    )
    
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Video name for report (default: from filename)",
    )
    
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=5,
        help="Minimum frames for a track to not be 'short' (default: 5)",
    )
    
    parser.add_argument(
        "--id-switch-iou",
        type=float,
        default=0.3,
        help="IoU threshold for ID switch detection (default: 0.3)",
    )
    
    parser.add_argument(
        "--id-switch-distance",
        type=float,
        default=100.0,
        help="Distance threshold (px) for ID switch detection (default: 100)",
    )
    
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target FPS for speed scoring (default: 30)",
    )
    
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include detailed track-by-track analysis",
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )
    
    parser.set_defaults(func=run_evaluate)


def run_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluation command."""
    # Load annotations
    annotations_path = args.annotations
    if not annotations_path.exists():
        logger.error(f"Annotations file not found: {annotations_path}")
        return 1
    
    try:
        with open(annotations_path) as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in annotations file: {e}")
        return 1
    
    # Determine video name
    video_name = args.video_name or annotations_path.stem
    
    # Create analyzer
    analyzer = TrackingAnalyzer(
        min_track_length=args.min_track_length,
        id_switch_iou_threshold=args.id_switch_iou,
        id_switch_distance_threshold=args.id_switch_distance,
        target_fps=args.target_fps,
    )
    
    # Run analysis
    logger.info(f"Analyzing {annotations_path}...")
    result = analyzer.analyze(annotations, video_name=video_name)
    
    # Create reporter
    reporter = EvaluationReporter(use_colors=not args.no_color)
    
    # Output results
    output_dir = args.output or annotations_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ("console", "all"):
        reporter.print_summary(result)
        if args.details:
            reporter.print_detailed_tracks(result)
            reporter.print_id_switch_events(result)
    
    if args.format in ("json", "all"):
        json_path = output_dir / f"{video_name}_evaluation.json"
        reporter.save_json(result, json_path)
        print(f"Saved JSON report: {json_path}")
    
    if args.format in ("markdown", "all"):
        md_path = output_dir / f"{video_name}_evaluation.md"
        reporter.save_markdown(result, md_path, include_details=args.details)
        print(f"Saved Markdown report: {md_path}")
    
    return 0


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add compare subcommand to CLI."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare evaluation results across multiple videos",
        description="Generate comparison table for multiple tracking runs",
    )
    
    parser.add_argument(
        "json_files",
        type=Path,
        nargs="+",
        help="Paths to evaluation JSON files",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file for comparison (default: stdout)",
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )
    
    parser.set_defaults(func=run_compare)


def run_compare(args: argparse.Namespace) -> int:
    """Execute comparison command."""
    results = []
    
    for json_path in args.json_files:
        if not json_path.exists():
            logger.warning(f"File not found, skipping: {json_path}")
            continue
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Reconstruct minimal EvaluationResult from JSON
            from .metrics import (
                EvaluationResult, FragmentationMetrics, IDSwitchMetrics,
                PerformanceMetrics, TrackMetrics
            )
            
            result = EvaluationResult(
                video_name=data.get("video_name", json_path.stem),
                continuity_score=data.get("scores", {}).get("continuity", 0),
                stability_score=data.get("scores", {}).get("stability", 0),
                speed_score=data.get("scores", {}).get("speed", 0),
                fragmentation=FragmentationMetrics(
                    total_tracks=data.get("fragmentation", {}).get("total_tracks", 0),
                    fragmented_tracks=data.get("fragmentation", {}).get("fragmented_tracks", 0),
                ),
                id_switches=IDSwitchMetrics(
                    total_switches=data.get("id_switches", {}).get("total_switches", 0),
                ),
                performance=PerformanceMetrics(
                    total_frames=data.get("performance", {}).get("total_frames", 0),
                    total_time_seconds=data.get("performance", {}).get("total_frames", 0) / 
                                       max(data.get("performance", {}).get("fps", 1), 0.1),
                ),
                tracks=TrackMetrics(
                    total_tracks=data.get("tracks", {}).get("total_tracks", 0),
                ),
            )
            results.append(result)
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading {json_path}: {e}")
            continue
    
    if not results:
        logger.error("No valid evaluation results to compare")
        return 1
    
    reporter = EvaluationReporter(use_colors=not args.no_color)
    
    if args.output:
        with open(args.output, "w") as f:
            reporter.compare_results(results, file=f)
        print(f"Saved comparison to {args.output}")
    else:
        reporter.compare_results(results)
    
    return 0


def setup_evaluation_cli(parser: argparse.ArgumentParser) -> None:
    """
    Set up evaluation CLI commands.
    
    Call this from your main CLI setup to add evaluation commands.
    
    Usage:
        from smart_desk_monitor.evaluation.cli import setup_evaluation_cli
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        setup_evaluation_cli(subparsers)
    """
    subparsers = parser.add_subparsers(dest="command")
    add_evaluate_parser(subparsers)
    add_compare_parser(subparsers)
