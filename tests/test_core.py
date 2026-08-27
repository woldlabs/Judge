"""Core smoke tests for Judge."""

from pathlib import Path
import tempfile

from judge.utils.demo import generate_demo_dataset
from judge.core.session import AnalysisSession
from judge.reporting.generator import generate_report


def test_demo_generation_and_run():
    with tempfile.TemporaryDirectory() as tmp:
        files = generate_demo_dataset(tmp, duration_s=18)
        sess = AnalysisSession(sensitivity=0.55, min_event_duration=0.03)
        sess.add_file(files["video"])
        sess.add_file(files["audio"])
        sess.add_file(files["sensor"])

        result = sess.run()
        assert result is not None
        assert isinstance(result.events, list)
        assert len(result.events) >= 1
        assert result.files_processed
        assert "sensitivity" in result.parameters

        report_path = Path(tmp) / "test_report"
        out = generate_report(result, report_path, format="all")
        assert out is not None
        assert Path(str(report_path) + ".pdf").exists() or out.with_suffix(".pdf").exists()
        assert Path(str(report_path) + ".md").exists() or out.with_suffix(".md").exists()
        assert Path(str(report_path) + ".json").exists() or out.with_suffix(".json").exists()


def test_session_skips_missing_files():
    sess = AnalysisSession()
    assert sess.add_file("this_file_does_not_exist.mp4") is False
    result = sess.run()
    assert result.events == []
    assert result.notes


def test_add_directory_and_clear(tmp_path):
    files = generate_demo_dataset(tmp_path, duration_s=8)
    sess = AnalysisSession()
    added = sess.add_directory(tmp_path, recursive=False)
    assert added >= 3
    names = {p.name for p in sess.files}
    assert Path(files["video"]).name in names
    sess.clear()
    assert sess.files == []
