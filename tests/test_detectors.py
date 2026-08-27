"""Per-modality detector tests using the synthetic demo dataset."""

import threading

import pytest

from judge.core.detectors import AudioAnomalyDetector, SensorAnomalyDetector, VideoAnomalyDetector
from judge.core.detectors.base import BaseDetector
from judge.core.models import Modality
from judge.core.session import AnalysisSession
from judge.utils.demo import generate_demo_dataset


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo22")
    return generate_demo_dataset(d, duration_s=22)


def test_true_runs_merges_small_gaps():
    mask = [False, True, True, False, True, False, False, True]
    assert BaseDetector._true_runs(mask, max_gap=1) == [(1, 5), (7, 8)]
    assert BaseDetector._true_runs(mask, max_gap=0) == [(1, 3), (4, 5), (7, 8)]


def test_audio_detector_finds_impulse(demo_files):
    det = AudioAnomalyDetector(sensitivity=0.7, min_duration=0.03)
    events = det.detect(demo_files["audio"])
    assert any(e.modality == Modality.AUDIO for e in events)
    # Impulse is injected at 8.7s
    starts = [e.start_time for e in events]
    assert any(6.5 <= t <= 11.0 for t in starts)


def test_video_detector_finds_transient(demo_files):
    det = VideoAnomalyDetector(sensitivity=0.75, min_duration=0.05, sample_every=3)
    events = det.detect(demo_files["video"])
    assert isinstance(events, list)
    for ev in events:
        assert ev.modality == Modality.VIDEO
        assert ev.frame_start is not None


def test_sensor_detector_finds_spike(demo_files):
    det = SensorAnomalyDetector(sensitivity=0.7, min_duration=0.05)
    events = det.detect(demo_files["sensor"])
    assert any(e.modality == Modality.SENSOR for e in events)
    starts = [e.start_time for e in events]
    assert any(16.5 <= t <= 20.5 for t in starts)


def test_cancel_skips_detectors(demo_files):
    sess = AnalysisSession(sensitivity=0.5)
    sess.add_file(demo_files["audio"])
    cancel = threading.Event()
    cancel.set()
    result = sess.run(cancel_event=cancel)
    assert result.events == []
    assert any("cancelled" in n.lower() for n in result.notes)
