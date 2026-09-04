"""Evidence pack export tests (no live network, no HOIC)."""

import json
import zipfile
from pathlib import Path

from judge.core.models import AnalysisResult, AnomalyEvent, Modality
from judge.reporting.evidence_pack import (
    PACK_SCHEMA_VERSION,
    export_evidence_pack,
    write_shapes_catalog,
)
from judge.__main__ import build_parser, main


def _sample_result() -> AnalysisResult:
    ev = AnomalyEvent(
        event_id="abc123",
        modality=Modality.VIDEO,
        start_time=12.4,
        duration=0.5,
        score=7.2,
        peak_score=9.1,
        features={"mean_flow": 1.2},
        description="kinematic transient",
        file_path="missing_clip.mp4",
        frame_start=10,
        frame_end=20,
        tags=["cross-modal"],
        shape_description="blob",
        geometry={"bbox": [1, 2, 3, 4], "area": 12.0, "aspect_ratio": 0.75, "centroid": [2.5, 4.0]},
    )
    audio = AnomalyEvent(
        event_id="aud9",
        modality=Modality.AUDIO,
        start_time=12.5,
        duration=0.2,
        score=5.0,
        peak_score=5.5,
        features={},
        description="onset",
        file_path="missing.wav",
        tags=["cross-modal"],
    )
    return AnalysisResult(
        session_id="sess-pack",
        files_processed=["missing_clip.mp4", "missing.wav"],
        events=[ev, audio],
        modality_stats={"video": {"event_count": 1, "max_score": 7.2}},
        parameters={"sensitivity": 0.5},
        duration_seconds=42.0,
        notes=["fixture"],
    )


def test_write_shapes_catalog(tmp_path):
    result = _sample_result()
    path = tmp_path / "shapes.csv"
    n = write_shapes_catalog(result.events, path)
    assert n == 1
    text = path.read_text(encoding="utf-8")
    assert "event_id" in text
    assert "abc123" in text
    assert "blob" in text


def test_export_pack_zip_and_rift_schema(tmp_path):
    result = _sample_result()
    out = tmp_path / "field_pack.zip"
    pack = export_evidence_pack(
        result,
        out,
        default_lat=37.77,
        default_lon=-122.42,
        include_clips=True,
        as_zip=True,
    )
    assert pack.zip_path is not None and pack.zip_path.is_file()
    assert pack.report_json.is_file()
    assert pack.shapes_csv.is_file()
    assert pack.readme.is_file()
    assert pack.manifest_json.is_file()
    assert pack.clips_exported == 0
    assert any("missing source" in s for s in pack.clips_skipped)

    data = json.loads(pack.report_json.read_text(encoding="utf-8"))
    # Rift is_judge_report / import_judge_report expectations
    assert data["session_id"] == "sess-pack"
    assert isinstance(data["events"], list) and data["events"]
    assert data["events"][0]["modality"] == "video"
    assert "file_path" in data["events"][0]
    assert "event_id" in data["events"][0]

    manifest = json.loads(pack.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema"] == "judge.evidence_pack"
    assert manifest["schema_version"] == PACK_SCHEMA_VERSION
    assert manifest["rift_import_file"] == "report.json"
    assert manifest["default_lat"] == 37.77
    assert manifest["local_only"] is True

    readme = pack.readme.read_text(encoding="utf-8")
    assert "37.77" in readme
    assert "report.json" in readme
    assert "Local-only" in readme

    with zipfile.ZipFile(pack.zip_path) as zf:
        names = set(zf.namelist())
    assert "report.json" in names
    assert "shapes.csv" in names
    assert "README.md" in names
    assert "pack_manifest.json" in names


def test_export_pack_dir_no_clips(tmp_path):
    result = _sample_result()
    out = tmp_path / "pack_dir"
    pack = export_evidence_pack(result, out, include_clips=False, as_zip=False)
    assert pack.zip_path is None
    assert pack.pack_dir.is_dir()
    assert pack.clips_exported == 0
    assert pack.clips_skipped == []


def test_cli_evidence_pack_parser():
    parser = build_parser()
    args = parser.parse_args(
        ["evidence-pack", "reports/demo.json", "-o", "out/pack.zip", "--lat", "40.7", "--lon", "-74.0", "--no-clips"]
    )
    assert args.command == "evidence-pack"
    assert args.report == "reports/demo.json"
    assert args.output == "out/pack.zip"
    assert args.lat == 40.7
    assert args.lon == -74.0
    assert args.no_clips is True


def test_cli_evidence_pack_runs(tmp_path, capsys):
    result = _sample_result()
    report = tmp_path / "result.json"
    result.save_json(str(report))
    out = tmp_path / "cli_pack.zip"
    rc = main(
        [
            "evidence-pack",
            str(report),
            "-o",
            str(out),
            "--lat",
            "1.5",
            "--lon",
            "2.5",
            "--no-clips",
        ]
    )
    assert rc == 0
    assert out.is_file()
    printed = capsys.readouterr().out
    assert "Evidence pack" in printed or "pack" in printed.lower()
