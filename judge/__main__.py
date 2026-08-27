"""CLI entry for Judge.

Usage:
    python -m judge                 # launch GUI
    python -m judge gui
    python -m judge analyze PATH... [--output reports/out] [--format all]
    python -m judge demo [--duration 20]
    python -m judge version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_gui(_args: argparse.Namespace) -> int:
    from judge.gui.main import main as gui_main
    gui_main()
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from judge import __version__
    print(f"judge {__version__}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from judge.utils.demo import generate_demo_dataset
    from judge.core.session import AnalysisSession
    from judge.reporting.generator import generate_report

    out_dir = Path(args.output_dir)
    print(f"Generating synthetic demo dataset ({args.duration:.0f}s)...")
    files = generate_demo_dataset(out_dir, duration_s=args.duration)
    print("Dataset:", files)

    session = AnalysisSession(
        sensitivity=args.sensitivity,
        min_event_duration=args.min_duration,
        cross_modal_window=args.window,
    )
    session.add_file(files["video"])
    session.add_file(files["audio"])
    session.add_file(files["sensor"])

    print("Running detectors + fusion...")
    result = session.run()
    print(f"Detected {len(result.events)} candidate events.")
    for ev in sorted(result.events, key=lambda e: -e.score)[:8]:
        print(f"  [{ev.modality.value}] {ev.pretty_time()} score={ev.score:.2f}  {ev.description[:72]}")

    report_dir = Path(args.report)
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(result, report_dir, format=args.format)
    print(f"Report: {report}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from judge.core.session import AnalysisSession
    from judge.reporting.generator import generate_report

    session = AnalysisSession(
        sensitivity=args.sensitivity,
        min_event_duration=args.min_duration,
        cross_modal_window=args.window,
    )
    added = 0
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            added += session.add_directory(path, recursive=not args.no_recursive)
        else:
            if session.add_file(path):
                added += 1
    if added == 0:
        print("No supported files were added.", file=sys.stderr)
        return 1

    print(f"Analyzing {len(session.files)} file(s)...")
    result = session.run()
    print(f"Detected {len(result.events)} candidate events.")
    for ev in sorted(result.events, key=lambda e: -e.score)[: args.top]:
        print(f"  [{ev.modality.value}] {ev.pretty_time()} score={ev.score:.2f}  {ev.description[:72]}")

    if args.output:
        out = generate_report(result, Path(args.output), format=args.format)
        print(f"Report: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge",
        description="JUDGE — Joint Unconventional Data & Geophysical Examination",
    )
    sub = parser.add_subparsers(dest="command")

    gui_p = sub.add_parser("gui", help="Launch the desktop GUI")
    gui_p.set_defaults(func=_cmd_gui)

    ver_p = sub.add_parser("version", help="Print the installed version")
    ver_p.set_defaults(func=_cmd_version)

    demo_p = sub.add_parser("demo", help="Generate synthetic data and run the full pipeline")
    demo_p.add_argument("--duration", type=float, default=20.0, help="Synthetic clip length in seconds")
    demo_p.add_argument("--output-dir", default="demo_data", help="Directory for generated demo files")
    demo_p.add_argument("--report", default="reports/demo_report", help="Report path without required suffix")
    demo_p.add_argument("--format", choices=["pdf", "markdown", "json", "all"], default="all")
    demo_p.add_argument("--sensitivity", type=float, default=0.50)
    demo_p.add_argument("--min-duration", type=float, default=0.06)
    demo_p.add_argument("--window", type=float, default=1.8)
    demo_p.set_defaults(func=_cmd_demo)

    an_p = sub.add_parser("analyze", help="Run headless analysis on files or directories")
    an_p.add_argument("paths", nargs="+", help="Files or directories to analyze")
    an_p.add_argument("--output", "-o", help="Report output path (suffix added from --format)")
    an_p.add_argument("--format", choices=["pdf", "markdown", "json", "all"], default="all")
    an_p.add_argument("--sensitivity", type=float, default=0.50)
    an_p.add_argument("--min-duration", type=float, default=0.06)
    an_p.add_argument("--window", type=float, default=1.8)
    an_p.add_argument("--top", type=int, default=12, help="How many ranked events to print")
    an_p.add_argument("--no-recursive", action="store_true", help="Do not recurse into directories")
    an_p.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _cmd_gui(argparse.Namespace())
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
