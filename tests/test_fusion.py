"""Cross-modal fusion tests."""

from judge.core.fusion import fuse_events
from judge.core.models import AnomalyEvent, Modality


def _ev(event_id, modality, start, score, tags=None):
    return AnomalyEvent(
        event_id=event_id,
        modality=modality,
        start_time=start,
        duration=0.2,
        score=score,
        peak_score=score,
        features={"mad_composite": score},
        description=f"{modality.value} event",
        file_path=f"{modality.value}.bin",
        tags=list(tags or []),
    )


def test_fuse_empty():
    assert fuse_events([]) == []


def test_fuse_preserves_all_events_and_boosts_cross_modal():
    video = _ev("v1", Modality.VIDEO, 10.0, 8.0)
    audio = _ev("a1", Modality.AUDIO, 10.4, 6.0)
    later = _ev("s1", Modality.SENSOR, 40.0, 5.0)

    fused = fuse_events([video, audio, later], window_seconds=1.5)
    assert len(fused) == 3
    ids = {e.event_id for e in fused}
    assert ids == {"v1", "a1", "s1"}

    by_id = {e.event_id: e for e in fused}
    assert by_id["v1"].score > 8.0
    assert by_id["a1"].score > 6.0
    assert "cross-modal" in by_id["v1"].tags
    assert "audio" in by_id["v1"].tags
    assert "single" in by_id["s1"].tags
    assert by_id["s1"].score == 5.0


def test_fuse_does_not_merge_across_bucket_boundaries():
    """Events near a former bucket edge must still fuse if within the window."""
    a = _ev("a", Modality.AUDIO, 2.99, 7.0)
    v = _ev("v", Modality.VIDEO, 3.01, 7.0)
    fused = fuse_events([a, v], window_seconds=1.5)
    assert len(fused) == 2
    assert all("cross-modal" in e.tags for e in fused)


def test_fuse_single_modality_keeps_score():
    e1 = _ev("v1", Modality.VIDEO, 1.0, 9.0)
    e2 = _ev("v2", Modality.VIDEO, 1.1, 4.0)
    fused = fuse_events([e1, e2], window_seconds=1.5)
    assert len(fused) == 2
    by_id = {e.event_id: e for e in fused}
    assert by_id["v1"].score == 9.0
    assert "single" in by_id["v1"].tags
