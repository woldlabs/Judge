
"""
Cross-modal fusion and event ranking.

Correlates events across modalities within a configurable temporal window
and produces composite evidence scores.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from collections import defaultdict

from judge.core.models import AnomalyEvent, Modality


def fuse_events(
    events: List[AnomalyEvent],
    window_seconds: float = 1.5,
) -> List[AnomalyEvent]:
    """
    Group near-coincident events and boost scores for cross-modal support.
    Returns a new list of (possibly merged/boosted) events.
    """
    if not events:
        return []

    # Group by approximate time buckets
    buckets: Dict[int, List[AnomalyEvent]] = defaultdict(list)
    bucket_size = max(0.05, window_seconds / 3)

    for ev in events:
        bucket = int(ev.start_time / bucket_size)
        buckets[bucket].append(ev)

    fused: List[AnomalyEvent] = []
    seen = set()

    for bkt_events in buckets.values():
        if not bkt_events:
            continue

        # Find clusters within window
        bkt_events.sort(key=lambda e: e.start_time)
        cluster: List[AnomalyEvent] = [bkt_events[0]]

        for ev in bkt_events[1:]:
            if ev.start_time - cluster[-1].start_time <= window_seconds:
                cluster.append(ev)
            else:
                fused.append(_merge_cluster(cluster, window_seconds))
                cluster = [ev]
        if cluster:
            fused.append(_merge_cluster(cluster, window_seconds))

    # Dedup by id
    final = []
    for ev in fused:
        if ev.event_id not in seen:
            seen.add(ev.event_id)
            final.append(ev)

    final.sort(key=lambda e: (-e.score, e.start_time))
    return final


def _merge_cluster(cluster: List[AnomalyEvent], window: float) -> AnomalyEvent:
    if len(cluster) == 1:
        ev = cluster[0]
        # Light boost for interesting single-modality events
        boosted = AnomalyEvent(
            event_id=ev.event_id,
            modality=ev.modality,
            start_time=ev.start_time,
            duration=ev.duration,
            score=min(99.0, ev.score * 1.0),
            peak_score=min(99.0, ev.peak_score),
            features=dict(ev.features),
            description=ev.description + " (single-modality)",
            file_path=ev.file_path,
            frame_start=ev.frame_start,
            frame_end=ev.frame_end,
            channel=ev.channel,
            tags=ev.tags + ["single"],
            shape_description=ev.shape_description,
            geometry=dict(ev.geometry) if ev.geometry else None,
        )
        return boosted

    # True cross-modal cluster
    start = min(e.start_time for e in cluster)
    end = max(e.end_time for e in cluster)
    duration = max(end - start, max(e.duration for e in cluster))

    # Score fusion: max + small additive for each supporting modality
    base = max(e.score for e in cluster)
    modalities = {e.modality for e in cluster}
    boost = 0.35 * (len(modalities) - 1) * base
    fused_score = min(99.0, base + boost)

    peak = max(e.peak_score for e in cluster)
    desc_parts = [e.description for e in cluster]
    desc = "CROSS-MODAL: " + " | ".join(desc_parts[:3])

    tags = ["cross-modal"] + [m.value for m in modalities]

    # Use the highest scoring event as primary carrier
    primary = max(cluster, key=lambda e: e.score)
    return AnomalyEvent(
        event_id=primary.event_id,
        modality=primary.modality,
        start_time=start,
        duration=duration,
        score=fused_score,
        peak_score=peak,
        features={**primary.features, "supporting_modalities": len(modalities)},
        description=desc,
        file_path=primary.file_path,
        frame_start=primary.frame_start,
        frame_end=primary.frame_end,
        channel=primary.channel,
        tags=list(set(primary.tags + tags)),
        shape_description=primary.shape_description,
        geometry=dict(primary.geometry) if primary.geometry else None,
    )
