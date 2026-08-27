"""
Cross-modal fusion and event ranking.

Correlates events across modalities within a configurable temporal window
and produces composite evidence scores. All original events are preserved;
coincident detections receive a score boost and cross-modal tags.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from judge.core.models import AnomalyEvent


def fuse_events(
    events: List[AnomalyEvent],
    window_seconds: float = 1.5,
) -> List[AnomalyEvent]:
    """
    Boost scores for temporally coincident events across modalities.

    Events are never dropped: investigators keep the full catalog, with
    cross-modal support reflected in score, tags, and description.
    """
    if not events:
        return []

    window_seconds = max(0.0, float(window_seconds))
    ordered = sorted(events, key=lambda e: e.start_time)
    n = len(ordered)
    fused: List[AnomalyEvent] = []
    left = 0

    for i, ev in enumerate(ordered):
        while left < n and ordered[left].start_time < ev.start_time - window_seconds:
            left += 1
        right = i
        while right + 1 < n and ordered[right + 1].start_time <= ev.start_time + window_seconds:
            right += 1

        other_mods = {
            other.modality
            for other in ordered[left : right + 1]
            if other is not ev and other.modality != ev.modality
        }

        if other_mods:
            boost = 0.35 * len(other_mods) * ev.score
            fused_score = min(99.0, ev.score + boost)
            tags = list(dict.fromkeys(
                list(ev.tags) + ["cross-modal"] + [m.value for m in sorted(other_mods, key=lambda m: m.value)]
            ))
            features = dict(ev.features)
            features["supporting_modalities"] = float(len(other_mods) + 1)
            coinc = ", ".join(sorted(m.value for m in other_mods))
            desc = ev.description
            marker = f"[coincident with {coinc}"
            if marker not in desc:
                desc = f"{desc} [coincident with {coinc} within {window_seconds:.1f}s]"
            fused.append(
                replace(
                    ev,
                    score=fused_score,
                    peak_score=min(99.0, max(ev.peak_score, fused_score)),
                    tags=tags,
                    features=features,
                    description=desc,
                )
            )
        else:
            tags = list(ev.tags)
            if "single" not in tags:
                tags = tags + ["single"]
            desc = ev.description
            if "(single-modality)" not in desc:
                desc = desc + " (single-modality)"
            fused.append(replace(ev, tags=tags, description=desc))

    fused.sort(key=lambda e: (-e.score, e.start_time))
    return fused
