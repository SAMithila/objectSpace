#!/usr/bin/env python
"""Process a single video with reduced memory usage."""

import sys
from objectSpace.pipeline import DetectionTrackingPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_one_video.py <video_name>")
        print("Example: python process_one_video.py task3.1_video2")
        return

    video_name = sys.argv[1]
    video_path = f"data/videos/{video_name}.mp4"

    print(f"Processing: {video_path}")
    print("Using only 30 frames to save memory...\n")

    pipeline = DetectionTrackingPipeline()

    results = pipeline.process_video(
        video_path,
        output_dir="output/",
        n_frames=30,  # Reduced from 100 to save memory
        show_progress=True
    )

    print(f"\n✅ Done! Output saved to output/{video_name}/")
    print(f"Annotations: output/{video_name}_annotations.json")


if __name__ == "__main__":
    main()
