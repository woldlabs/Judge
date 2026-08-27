"""Model serialization tests."""

from pathlib import Path

from judge.core.models import AnomalyEvent, AnalysisResult, Modality


def test_pretty_time():
    ev = AnomalyEvent(
        event_id="x",
        modality=Modality.AUDIO,
        start_time=3661.042,
        duration=0.1,
        score=1.0,
        peak_score=1.0,
        features={},
        description="t",
        file_path="a.wav",
    )
    assert ev.pretty_time() == "01:01:01.042"
    assert ev.end_time == 3661.142


def test_json_roundtrip(tmp_path):
    ev = AnomalyEvent(
        event_id="abc123",
        modality=Modality.VIDEO,
        start_time=12.4,
        duration=0.5,
        score=7.2,
        peak_score=9.1,
        features={"mean_flow": 1.2, "supporting_modalities": 2},
        description="kinematic",
        file_path="clip.mp4",
        frame_start=10,
        frame_end=20,
        tags=["cross-modal", "audio"],
        shape_description="blob",
        geometry={"bbox": [1, 2, 3, 4], "area": 12.0},
    )
    result = AnalysisResult(
        session_id="sess1",
        files_processed=["clip.mp4"],
        events=[ev],
        modality_stats={"video": {"event_count": 1, "max_score": 7.2}},
        parameters={"sensitivity": 0.5},
        duration_seconds=42.0,
        notes=["ok"],
    )
    path = tmp_path / "result.json"
    result.save_json(str(path))
    loaded = AnalysisResult.from_json(str(path))
    assert loaded.session_id == "sess1"
    assert len(loaded.events) == 1
    assert loaded.events[0].modality == Modality.VIDEO
    assert loaded.events[0].geometry["bbox"] == [1, 2, 3, 4]
    assert loaded.events[0].features["supporting_modalities"] == 2
    assert loaded.notes == ["ok"]


def test_from_json_ignores_unknown_fields(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"session_id": "s", "events": [{"event_id": "e", "modality": "sensor",'
        ' "start_time": 1, "duration": 0.2, "score": 3, "peak_score": 3,'
        ' "features": {}, "description": "x", "file_path": "m.csv", "legacy_field": 1}],'
        ' "mystery": true}',
        encoding="utf-8",
    )
    loaded = AnalysisResult.from_json(str(path))
    assert loaded.session_id == "s"
    assert loaded.events[0].modality == Modality.SENSOR
