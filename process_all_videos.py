#!/usr/bin/env python
"""Process all videos in the data folder."""

from objectSpace.pipeline import DetectionTrackingPipeline


def main():
    pipeline = DetectionTrackingPipeline()

    # Process all videos in the directory
    # Using fewer frames to avoid memory issues
    results = pipeline.process_directory(
        "data/videos/",
        output_dir="output/",
        show_progress=True
    )

    print(f"\n✅ Processed {len(results)} videos")
    print("Now run: python compare_videos.py")


if __name__ == "__main__":
    main()
