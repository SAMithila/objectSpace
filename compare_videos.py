#!/usr/bin/env python
"""Compare tracking evaluation across multiple videos."""

from pathlib import Path
from smart_desk_monitor.evaluation import (
    TrackingAnalyzer,
    EvaluationReporter,
)
import json


def evaluate_video(annotations_path: Path) -> dict:
    """Evaluate a single video's annotations."""
    with open(annotations_path) as f:
        annotations = json.load(f)

    video_name = annotations_path.stem.replace("_annotations", "")

    analyzer = TrackingAnalyzer()
    result = analyzer.analyze(annotations, video_name=video_name)

    return result


def main():
    # Find all annotation files
    output_dir = Path("output")
    annotation_files = list(output_dir.glob("*_annotations.json"))

    if not annotation_files:
        print("No annotation files found in output/")
        print("Run the pipeline first on your videos.")
        return

    print(f"Found {len(annotation_files)} annotation file(s)\n")

    # Evaluate each video
    results = []
    for ann_file in sorted(annotation_files):
        print(f"Evaluating: {ann_file.name}")
        result = evaluate_video(ann_file)
        results.append(result)
        print(f"  → Overall: {result.overall_score:.1f}/100\n")

    # Print comparison table
    reporter = EvaluationReporter(use_colors=True)

    print("\n" + "="*80)
    print("COMPARISON ACROSS ALL VIDEOS")
    print("="*80 + "\n")

    reporter.compare_results(results)

    # Summary statistics
    if len(results) > 1:
        avg_overall = sum(r.overall_score for r in results) / len(results)
        avg_continuity = sum(
            r.continuity_score for r in results) / len(results)
        avg_stability = sum(r.stability_score for r in results) / len(results)

        print(f"\n{'='*50}")
        print("AVERAGE ACROSS ALL VIDEOS:")
        print(f"{'='*50}")
        print(f"  Overall Score:    {avg_overall:.1f}/100")
        print(f"  Continuity:       {avg_continuity:.1f}/100")
        print(f"  Stability:        {avg_stability:.1f}/100")


if __name__ == "__main__":
    main()
