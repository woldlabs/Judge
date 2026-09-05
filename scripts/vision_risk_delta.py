#!/usr/bin/env python3
"""Map PR touched paths → existing VISION.md themes (links only; no invented north-star text)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Set, Tuple

MARKER = "<!-- judge-vision-risk-delta -->"
VISION_BLOB = "https://github.com/woldlabs/Judge/blob/main/VISION.md"

# (path prefixes or exact files, theme heading, anchor, label candidates that may exist)
RULES: List[Tuple[Tuple[str, ...], str, str, Tuple[str, ...]]] = [
    (
        ("VISION.md", "docs/VISION_PROCESS.md", "docs/TRIAGE.md"),
        "Product purpose / process docs",
        "product-purpose",
        ("needs-vision-review", "docs"),
    ),
    (
        ("docs/", "README.md"),
        "Success metrics (docs / operator path)",
        "success-metrics",
        ("docs",),
    ),
    (
        (
            "judge/core/detectors/",
            "judge/core/fusion.py",
            "judge/core/models.py",
            "judge/core/session.py",
            "judge/core/ingestion.py",
        ),
        "Architecture boundaries",
        "architecture-boundaries-change-only-with-review",
        ("needs-vision-review",),
    ),
    (
        ("judge/reporting/",),
        "Architecture boundaries + Relation to Rift (report/event schema)",
        "relation-to-rift",
        ("needs-vision-review",),
    ),
    (
        ("judge/__main__.py", "judge/gui/", "run_judge.py", "tests/test_cli.py"),
        "Success metrics (CLI / headless)",
        "success-metrics",
        (),
    ),
    (
        ("tests/", ".github/workflows/", "requirements", "pyproject.toml", "scripts/"),
        "Success metrics (CI / pytest)",
        "success-metrics",
        (),
    ),
    (
        ("judge/",),
        "Product purpose",
        "product-purpose",
        (),
    ),
]


def _match_rule(path: str) -> Tuple[str, str, Tuple[str, ...]] | None:
    norm = path.replace("\\", "/").removeprefix("./")
    for prefixes, theme, anchor, labels in RULES:
        for p in prefixes:
            if p.endswith("/") and norm.startswith(p):
                return theme, anchor, labels
            if not p.endswith("/") and (norm == p or norm.startswith(p)):
                # requirements*.txt style
                if p == "requirements" and norm.startswith("requirements"):
                    return theme, anchor, labels
                if p != "requirements" and (norm == p or norm.startswith(p + "/")):
                    return theme, anchor, labels
                if norm == p:
                    return theme, anchor, labels
    return None


def map_paths(paths: List[str]) -> Dict[str, object]:
    """First matching rule wins per path; themes aggregated; labels de-duped."""
    theme_to_paths: Dict[str, List[str]] = {}
    theme_to_anchor: Dict[str, str] = {}
    labels: Set[str] = set()
    unmatched: List[str] = []

    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        hit = _match_rule(path)
        if not hit:
            unmatched.append(path)
            continue
        theme, anchor, labs = hit
        theme_to_paths.setdefault(theme, []).append(path)
        theme_to_anchor[theme] = anchor
        labels.update(labs)

    # Prefer more specific themes already collected; drop generic Product purpose
    # if a more specific theme also matched the same path set only via fallback —
    # RULES order already prefers specific prefixes before judge/.

    rows = []
    for theme, plist in theme_to_paths.items():
        anchor = theme_to_anchor[theme]
        sample = ", ".join(f"`{p}`" for p in plist[:5])
        if len(plist) > 5:
            sample += f" (+{len(plist) - 5} more)"
        link = f"{VISION_BLOB}#{anchor}"
        rows.append({"theme": theme, "link": link, "paths_sample": sample, "paths": plist})

    return {
        "marker": MARKER,
        "themes": rows,
        "labels": sorted(labels),
        "unmatched": unmatched,
        "noop": len(rows) == 0,
    }


def render_comment(mapped: Dict[str, object]) -> str:
    if mapped.get("noop"):
        return (
            f"{MARKER}\n"
            "## VISION risk-delta (informational)\n\n"
            "No mapped VISION themes for touched paths (no-op).\n\n"
            "Comments/labels only — no approve/merge from this workflow.\n"
            "See [`docs/TRIAGE.md`](../blob/main/docs/TRIAGE.md).\n"
        )
    lines = [
        MARKER,
        "## VISION risk-delta (informational)",
        "",
        "Touched paths map to these existing [VISION.md](" + VISION_BLOB + ") themes "
        "(quote/link only — **no new north-star text**):",
        "",
        "| Paths (sample) | VISION theme |",
        "|----------------|--------------|",
    ]
    for row in mapped["themes"]:  # type: ignore[index]
        lines.append(
            f"| {row['paths_sample']} | [{row['theme']}]({row['link']}) |"
        )
    lines.extend(
        [
            "",
            "**Reminder:** comments and labels only — this workflow does **not** approve, "
            "request-changes, or merge. Triage rubric: [`docs/TRIAGE.md`](https://github.com/woldlabs/Judge/blob/main/docs/TRIAGE.md).",
            "",
            "Non-goals / counsel / HOIC blur are out of scope for this comment.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Touched paths (or pass --paths-file)")
    parser.add_argument("--paths-file", default="", help="File with one path per line")
    parser.add_argument("--json", action="store_true", help="Emit JSON mapping to stdout")
    parser.add_argument(
        "--comment-out",
        default="",
        help="Write markdown comment body to this path",
    )
    args = parser.parse_args(argv)

    paths: List[str] = list(args.paths)
    if args.paths_file:
        with open(args.paths_file, encoding="utf-8") as handle:
            paths.extend(line.strip() for line in handle if line.strip())

    mapped = map_paths(paths)
    if args.json:
        print(json.dumps(mapped, indent=2))
    if args.comment_out:
        Path = __import__("pathlib").Path
        Path(args.comment_out).write_text(render_comment(mapped), encoding="utf-8")
    elif not args.json:
        print(render_comment(mapped), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
