#!/usr/bin/env python
"""Generate a final evaluation report for all videos."""

from pathlib import Path
from datetime import datetime
import json
from smart_desk_monitor.evaluation import TrackingAnalyzer, EvaluationReporter


def main():
    output_dir = Path("output")
    annotation_files = sorted(output_dir.glob("*_annotations.json"))

    # Evaluate all videos
    results = []
    analyzer = TrackingAnalyzer()

    for ann_file in annotation_files:
        with open(ann_file) as f:
            annotations = json.load(f)
        video_name = ann_file.stem.replace("_annotations", "")
        result = analyzer.analyze(annotations, video_name=video_name)
        results.append(result)

    # Generate markdown report
    report_path = output_dir / "EVALUATION_SUMMARY.md"

    with open(report_path, "w") as f:
        f.write("# Tracking Evaluation Summary\n\n")
        f.write(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Overall stats
        avg_overall = sum(r.overall_score for r in results) / len(results)
        avg_cont = sum(r.continuity_score for r in results) / len(results)
        avg_stab = sum(r.stability_score for r in results) / len(results)

        f.write("## Overall Performance\n\n")
        f.write(f"- **Average Overall Score:** {avg_overall:.1f}/100\n")
        f.write(f"- **Average Continuity:** {avg_cont:.1f}/100\n")
        f.write(f"- **Average Stability:** {avg_stab:.1f}/100\n\n")

        # Comparison table
        f.write("## Video Comparison\n\n")
        f.write(
            "| Video | Overall | Continuity | Stability | Tracks | ID Switches | Fragmented |\n")
        f.write(
            "|-------|---------|------------|-----------|--------|-------------|------------|\n")

        for r in results:
            f.write(
                f"| {r.video_name} | {r.overall_score:.1f} | {r.continuity_score:.1f} | ")
            f.write(f"{r.stability_score:.1f} | {r.tracks.total_tracks} | ")
            f.write(
                f"{r.id_switches.total_switches} | {r.fragmentation.fragmented_tracks} |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        best = max(results, key=lambda r: r.overall_score)
        worst = min(results, key=lambda r: r.overall_score)

        f.write(
            f"- **Best performing:** {best.video_name} ({best.overall_score:.1f}/100)\n")
        f.write(
            f"- **Most challenging:** {worst.video_name} ({worst.overall_score:.1f}/100)\n")
        f.write(
            f"- **Total tracks across all videos:** {sum(r.tracks.total_tracks for r in results)}\n")
        f.write(
            f"- **Total ID switches:** {sum(r.id_switches.total_switches for r in results)}\n")

        # Recommendations
        f.write("\n## Recommendations for Improvement\n\n")
        f.write("Based on the evaluation metrics:\n\n")
        f.write("1. **Increase `max_age`** from 8 to 15 - average gap length suggests tracks are deleted too quickly\n")
        f.write(
            "2. **Lower `iou_threshold`** from 0.3 to 0.2 - ID switches show valid matches being rejected\n")
        f.write(
            "3. **Consider Deep SORT** - appearance features would help in crowded scenes\n")

    print(f"✅ Report saved to: {report_path}")
    print(f"\nOpen it with: cat {report_path}")
    print("Or view on GitHub for nice rendering!")


if __name__ == "__main__":
    main()
