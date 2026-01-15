"""
Command-line interface for ObjectSpace.

Usage:
    ObjectSpace process video.mp4 --output output/
    ObjectSpace process videos/ --output output/
    ObjectSpace config --show
    ObjectSpace config --generate config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import PipelineConfig, get_default_config
from .pipeline import run_pipeline


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="objectSpace",
        description="Object detection and tracking for desk monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single video:
    objectSpace process video.mp4 -o output/

  Process all videos in a directory:
    objectSpace process videos/ -o output/

  Use a custom config file:
    objectSpace process video.mp4 -c config.yaml -o output/

  Generate a default config file:
    objectSpace config --generate my_config.yaml
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process video(s) for detection and tracking",
    )
    process_parser.add_argument(
        "input",
        type=Path,
        help="Input video file or directory",
    )
    process_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    process_parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config YAML file",
    )
    process_parser.add_argument(
        "-n", "--num-frames",
        type=int,
        help="Number of frames to process per video",
    )
    process_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    process_parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Filename patterns to exclude",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration utilities",
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--show",
        action="store_true",
        help="Show current default configuration",
    )
    config_group.add_argument(
        "--generate",
        type=Path,
        metavar="FILE",
        help="Generate a default config file",
    )

    return parser


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_process(args: argparse.Namespace) -> int:
    """Handle the process command."""
    # Load config
    if args.config:
        if not args.config.exists():
            print(
                f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = get_default_config()

    # Override config with CLI args
    config.output.output_dir = args.output
    if args.num_frames:
        config.video.max_frames = args.num_frames

    # Validate input
    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        return 1

    # Run pipeline
    try:
        results = run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            config=config,
            exclude_patterns=args.exclude,
            show_progress=not args.no_progress,
        )

        print(f"\nProcessing complete!")
        print(f"Processed {len(results)} video(s)")
        print(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Handle the config command."""
    config = get_default_config()

    if args.show:
        import yaml
        # Convert to dict and print
        config_dict = {
            'detector': {
                'model_name': config.detector.model_name,
                'device': config.detector.device,
                'default_confidence': config.detector.default_confidence,
            },
            'tracker': {
                'max_age': config.tracker.max_age,
                'min_hits': config.tracker.min_hits,
                'iou_threshold': config.tracker.iou_threshold,
            },
            'video': {
                'target_width': config.video.target_width,
                'target_height': config.video.target_height,
                'max_frames': config.video.max_frames,
            },
            'output': {
                'save_tracked_frames': config.output.save_tracked_frames,
                'save_coco_json': config.output.save_coco_json,
            },
        }
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        return 0

    if args.generate:
        config.to_yaml(args.generate)
        print(f"Generated config file: {args.generate}")
        return 0

    return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)

    if args.command == "process":
        return cmd_process(args)
    elif args.command == "config":
        return cmd_config(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
