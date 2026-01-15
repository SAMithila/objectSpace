#!/usr/bin/env python
"""Evaluate existing tracking results (no reprocessing needed)."""

from objectSpace.pipeline import evaluate_annotations


def main():
    # Evaluate your existing annotations - instant, no GPU/memory needed!
    result = evaluate_annotations(
        "output/task3.1_video1_annotations.json",
        output_dir="output/",
        print_summary=True
    )

    # Print key metrics
    print(f"\n{'='*50}")
    print(f"KEY METRICS FOR YOUR PORTFOLIO:")
    print(f"{'='*50}")
    print(f"Overall Score:    {result.overall_score:.1f}/100")
    print(f"Track Continuity: {result.continuity_score:.1f}/100")
    print(f"ID Stability:     {result.stability_score:.1f}/100")
    print(f"")
    print(f"Total Tracks:     {result.tracks.total_tracks}")
    print(f"ID Switches:      {result.id_switches.total_switches}")
    print(f"Fragmented Tracks: {result.fragmentation.fragmented_tracks}")


if __name__ == "__main__":
    main()
