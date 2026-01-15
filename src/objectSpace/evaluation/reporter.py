"""
Evaluation reporter for generating human-readable and machine-readable reports.

Supports:
- Console output with colored formatting
- JSON export for programmatic analysis
- Markdown reports for documentation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO
import sys

from .metrics import EvaluationResult, TrackLifecycle


logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in various formats."""
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize reporter.
        
        Args:
            use_colors: Whether to use ANSI colors in console output
        """
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _score_color(self, score: float) -> str:
        """Get color for a score value."""
        if score >= 80:
            return "green"
        elif score >= 60:
            return "yellow"
        else:
            return "red"
    
    def print_summary(
        self,
        result: EvaluationResult,
        file: TextIO = sys.stdout,
    ) -> None:
        """
        Print a summary of evaluation results to console.
        
        Args:
            result: Evaluation results to summarize
            file: Output file (default: stdout)
        """
        print(file=file)
        print(self._color("=" * 60, "bold"), file=file)
        print(self._color(f"  TRACKING EVALUATION: {result.video_name}", "bold"), file=file)
        print(self._color("=" * 60, "bold"), file=file)
        print(file=file)
        
        # Overall scores
        print(self._color("SCORES", "cyan"), file=file)
        print("-" * 40, file=file)
        
        overall = result.overall_score
        print(
            f"  Overall:    {self._color(f'{overall:5.1f}', self._score_color(overall))}/100",
            file=file,
        )
        
        cont = result.continuity_score
        print(
            f"  Continuity: {self._color(f'{cont:5.1f}', self._score_color(cont))}/100  "
            f"(track completeness)",
            file=file,
        )
        
        stab = result.stability_score
        print(
            f"  Stability:  {self._color(f'{stab:5.1f}', self._score_color(stab))}/100  "
            f"(ID consistency)",
            file=file,
        )
        
        speed = result.speed_score
        print(
            f"  Speed:      {self._color(f'{speed:5.1f}', self._score_color(speed))}/100  "
            f"(processing rate)",
            file=file,
        )
        print(file=file)
        
        # Track statistics
        print(self._color("TRACK STATISTICS", "cyan"), file=file)
        print("-" * 40, file=file)
        print(f"  Total tracks:        {result.tracks.total_tracks}", file=file)
        print(f"  Avg active/frame:    {result.tracks.avg_active_tracks:.1f}", file=file)
        print(f"  Max concurrent:      {result.tracks.max_concurrent_tracks}", file=file)
        
        # Duration histogram
        print(f"  Duration distribution:", file=file)
        for bucket, count in result.tracks.duration_histogram.items():
            bar = "█" * min(count, 20)
            print(f"    {bucket:>8} frames: {count:3} {bar}", file=file)
        print(file=file)
        
        # Fragmentation
        frag = result.fragmentation
        print(self._color("FRAGMENTATION", "cyan"), file=file)
        print("-" * 40, file=file)
        print(f"  Fragmented tracks:   {frag.fragmented_tracks}/{frag.total_tracks} "
              f"({frag.fragmentation_rate:.1%})", file=file)
        print(f"  Avg coverage ratio:  {frag.avg_coverage_ratio:.2%}", file=file)
        print(f"  Total gaps:          {frag.total_gaps}", file=file)
        print(f"  Avg gap length:      {frag.avg_gap_length:.1f} frames", file=file)
        print(f"  Max gap length:      {frag.max_gap_length} frames", file=file)
        print(f"  Short tracks (<5f):  {frag.short_tracks} ({frag.short_track_rate:.1%})", file=file)
        print(file=file)
        
        # ID switches
        ids = result.id_switches
        print(self._color("ID SWITCHES", "cyan"), file=file)
        print("-" * 40, file=file)
        print(f"  Total detected:      {ids.total_switches}", file=file)
        print(f"  High confidence:     {ids.high_confidence_switches}", file=file)
        print(f"  Rate (per 100 frames): {ids.switches_per_100_frames:.2f}", file=file)
        
        if ids.switch_events:
            print(f"  Avg switch distance: {ids.avg_switch_distance:.1f}px", file=file)
            print(f"  Avg switch IoU:      {ids.avg_switch_iou:.3f}", file=file)
        print(file=file)
        
        # Performance
        perf = result.performance
        print(self._color("PERFORMANCE", "cyan"), file=file)
        print("-" * 40, file=file)
        print(f"  Frames processed:    {perf.total_frames}", file=file)
        print(f"  Total time:          {perf.total_time_seconds:.2f}s", file=file)
        print(f"  Average FPS:         {perf.fps:.1f}", file=file)
        print(f"  Avg frame time:      {perf.avg_frame_time_ms:.1f}ms", file=file)
        print(f"  P95 frame time:      {perf.p95_frame_time_ms:.1f}ms", file=file)
        
        if perf.detection_time > 0:
            print(f"  Detection time:      {perf.detection_time:.2f}s ({perf.detection_fps:.1f} FPS)", file=file)
        if perf.tracking_time > 0:
            print(f"  Tracking time:       {perf.tracking_time:.2f}s ({perf.tracking_fps:.1f} FPS)", file=file)
        
        print(file=file)
        print(self._color("=" * 60, "bold"), file=file)
        print(file=file)
    
    def print_detailed_tracks(
        self,
        result: EvaluationResult,
        top_n: int = 10,
        file: TextIO = sys.stdout,
    ) -> None:
        """
        Print detailed track-by-track analysis.
        
        Args:
            result: Evaluation results
            top_n: Number of problematic tracks to show
            file: Output file
        """
        lifecycles = result.fragmentation.track_lifecycles
        
        if not lifecycles:
            print("No tracks to analyze.", file=file)
            return
        
        print(self._color("PROBLEMATIC TRACKS (by fragmentation)", "cyan"), file=file)
        print("-" * 60, file=file)
        
        # Sort by coverage ratio (worst first)
        sorted_tracks = sorted(lifecycles, key=lambda t: t.coverage_ratio)[:top_n]
        
        for track in sorted_tracks:
            status = "⚠️" if track.is_fragmented else "✓"
            print(
                f"  {status} Track {track.track_id:4d}: "
                f"frames {track.first_frame}-{track.last_frame} "
                f"({track.duration_frames} total), "
                f"coverage: {track.coverage_ratio:.1%}, "
                f"gaps: {track.gap_count}",
                file=file,
            )
            
            if track.gaps and track.gap_count <= 5:
                gap_str = ", ".join(f"{s}-{e}" for s, e in track.gaps[:5])
                print(f"       Gap frames: {gap_str}", file=file)
        
        print(file=file)
    
    def print_id_switch_events(
        self,
        result: EvaluationResult,
        top_n: int = 10,
        file: TextIO = sys.stdout,
    ) -> None:
        """
        Print detailed ID switch events.
        
        Args:
            result: Evaluation results
            top_n: Number of events to show
            file: Output file
        """
        events = result.id_switches.switch_events
        
        if not events:
            print("No ID switch events detected.", file=file)
            return
        
        print(self._color("ID SWITCH EVENTS (by confidence)", "cyan"), file=file)
        print("-" * 60, file=file)
        
        # Sort by confidence
        sorted_events = sorted(events, key=lambda e: e.confidence, reverse=True)[:top_n]
        
        for event in sorted_events:
            conf_color = "red" if event.confidence > 0.7 else "yellow"
            print(
                f"  Frame {event.frame:5d}: "
                f"Track {event.old_track_id} → {event.new_track_id}, "
                f"dist: {event.spatial_distance:.1f}px, "
                f"IoU: {event.iou_overlap:.3f}, "
                f"conf: {self._color(f'{event.confidence:.2f}', conf_color)}",
                file=file,
            )
        
        print(file=file)
    
    def save_json(
        self,
        result: EvaluationResult,
        output_path: Path,
        pretty: bool = True,
    ) -> None:
        """
        Save evaluation results as JSON.
        
        Args:
            result: Results to save
            output_path: Output file path
            pretty: Whether to format JSON nicely
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = result.to_dict()
        
        with open(output_path, "w") as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
        
        logger.info(f"Saved evaluation JSON to {output_path}")
    
    def save_markdown(
        self,
        result: EvaluationResult,
        output_path: Path,
        include_details: bool = True,
    ) -> None:
        """
        Save evaluation report as Markdown.
        
        Args:
            result: Results to save
            output_path: Output file path
            include_details: Whether to include per-track details
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(f"# Tracking Evaluation Report\n\n")
            f.write(f"**Video:** {result.video_name}  \n")
            f.write(f"**Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            # Score summary
            f.write("## Summary Scores\n\n")
            f.write("| Metric | Score | Rating |\n")
            f.write("|--------|-------|--------|\n")
            
            def rating(score):
                if score >= 80:
                    return "🟢 Good"
                elif score >= 60:
                    return "🟡 Fair"
                else:
                    return "🔴 Needs Work"
            
            f.write(f"| **Overall** | {result.overall_score:.1f} | {rating(result.overall_score)} |\n")
            f.write(f"| Continuity | {result.continuity_score:.1f} | {rating(result.continuity_score)} |\n")
            f.write(f"| Stability | {result.stability_score:.1f} | {rating(result.stability_score)} |\n")
            f.write(f"| Speed | {result.speed_score:.1f} | {rating(result.speed_score)} |\n\n")
            
            # Track stats
            f.write("## Track Statistics\n\n")
            f.write(f"- **Total tracks:** {result.tracks.total_tracks}\n")
            f.write(f"- **Avg active per frame:** {result.tracks.avg_active_tracks:.1f}\n")
            f.write(f"- **Max concurrent:** {result.tracks.max_concurrent_tracks}\n\n")
            
            f.write("### Track Duration Distribution\n\n")
            f.write("| Duration | Count |\n")
            f.write("|----------|-------|\n")
            for bucket, count in result.tracks.duration_histogram.items():
                f.write(f"| {bucket} frames | {count} |\n")
            f.write("\n")
            
            # Fragmentation
            frag = result.fragmentation
            f.write("## Fragmentation Analysis\n\n")
            f.write(f"- **Fragmented tracks:** {frag.fragmented_tracks}/{frag.total_tracks} "
                    f"({frag.fragmentation_rate:.1%})\n")
            f.write(f"- **Avg coverage ratio:** {frag.avg_coverage_ratio:.2%}\n")
            f.write(f"- **Total gaps:** {frag.total_gaps}\n")
            f.write(f"- **Avg gap length:** {frag.avg_gap_length:.1f} frames\n")
            f.write(f"- **Short tracks (<5f):** {frag.short_tracks}\n\n")
            
            # ID switches
            ids = result.id_switches
            f.write("## ID Switch Analysis\n\n")
            f.write(f"- **Total switches:** {ids.total_switches}\n")
            f.write(f"- **High confidence:** {ids.high_confidence_switches}\n")
            f.write(f"- **Rate:** {ids.switches_per_100_frames:.2f} per 100 frames\n\n")
            
            # Performance
            perf = result.performance
            f.write("## Performance\n\n")
            f.write(f"- **Frames:** {perf.total_frames}\n")
            f.write(f"- **Total time:** {perf.total_time_seconds:.2f}s\n")
            f.write(f"- **FPS:** {perf.fps:.1f}\n")
            f.write(f"- **Avg frame time:** {perf.avg_frame_time_ms:.1f}ms\n")
            f.write(f"- **P95 frame time:** {perf.p95_frame_time_ms:.1f}ms\n\n")
            
            # Detailed tracks
            if include_details and frag.track_lifecycles:
                f.write("## Problematic Tracks\n\n")
                
                worst_tracks = sorted(
                    frag.track_lifecycles,
                    key=lambda t: t.coverage_ratio
                )[:10]
                
                f.write("| Track ID | Frames | Duration | Coverage | Gaps |\n")
                f.write("|----------|--------|----------|----------|------|\n")
                
                for track in worst_tracks:
                    status = "⚠️" if track.is_fragmented else "✓"
                    f.write(
                        f"| {status} {track.track_id} | "
                        f"{track.first_frame}-{track.last_frame} | "
                        f"{track.duration_frames} | "
                        f"{track.coverage_ratio:.1%} | "
                        f"{track.gap_count} |\n"
                    )
                f.write("\n")
            
            # Config
            if result.config_used:
                f.write("## Configuration Used\n\n")
                f.write("```yaml\n")
                for key, value in result.config_used.items():
                    f.write(f"{key}: {value}\n")
                f.write("```\n")
        
        logger.info(f"Saved evaluation report to {output_path}")
    
    def compare_results(
        self,
        results: List[EvaluationResult],
        file: TextIO = sys.stdout,
    ) -> None:
        """
        Print comparison of multiple evaluation results.
        
        Args:
            results: List of results to compare
            file: Output file
        """
        if not results:
            print("No results to compare.", file=file)
            return
        
        print(self._color("EVALUATION COMPARISON", "bold"), file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        # Header
        print(f"{'Video':<25} {'Overall':>8} {'Cont.':>8} {'Stab.':>8} "
              f"{'Speed':>8} {'Tracks':>7} {'FPS':>6}", file=file)
        print("-" * 80, file=file)
        
        for result in results:
            name = result.video_name[:24]
            print(
                f"{name:<25} "
                f"{result.overall_score:>8.1f} "
                f"{result.continuity_score:>8.1f} "
                f"{result.stability_score:>8.1f} "
                f"{result.speed_score:>8.1f} "
                f"{result.tracks.total_tracks:>7} "
                f"{result.performance.fps:>6.1f}",
                file=file,
            )
        
        print("-" * 80, file=file)
        
        # Averages
        if len(results) > 1:
            avg_overall = sum(r.overall_score for r in results) / len(results)
            avg_cont = sum(r.continuity_score for r in results) / len(results)
            avg_stab = sum(r.stability_score for r in results) / len(results)
            avg_speed = sum(r.speed_score for r in results) / len(results)
            total_tracks = sum(r.tracks.total_tracks for r in results)
            avg_fps = sum(r.performance.fps for r in results) / len(results)
            
            print(
                f"{'AVERAGE':<25} "
                f"{avg_overall:>8.1f} "
                f"{avg_cont:>8.1f} "
                f"{avg_stab:>8.1f} "
                f"{avg_speed:>8.1f} "
                f"{total_tracks:>7} "
                f"{avg_fps:>6.1f}",
                file=file,
            )
        
        print(file=file)
