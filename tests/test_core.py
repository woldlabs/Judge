"""Basic smoke tests for Judge core."""

from pathlib import Path
import tempfile
import pytest

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
        # With injected anomalies we should find at least a couple
        assert len(result.events) >= 1

        # Report generation should succeed
        report_path = Path(tmp) / "test_report"
        out = generate_report(result, report_path, format="all")
        assert out is not None or Path(str(report_path) + ".pdf").exists() or Path(str(report_path) + ".md").exists()
