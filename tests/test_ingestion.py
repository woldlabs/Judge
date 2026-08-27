"""Ingestion and metadata tests."""

from pathlib import Path

from judge.core.ingestion import DataIngestor
from judge.utils.demo import generate_demo_dataset


def test_detect_modality():
    assert DataIngestor.detect_modality(Path("a.mp4")) == "video"
    assert DataIngestor.detect_modality(Path("a.WAV")) == "audio"
    assert DataIngestor.detect_modality(Path("a.csv")) == "sensor"
    assert DataIngestor.detect_modality(Path("a.txt")) is None


def test_validate_file(tmp_path):
    missing = tmp_path / "nope.mp4"
    ok, err = DataIngestor.validate_file(missing)
    assert ok is False
    assert "does not exist" in err

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    ok, err = DataIngestor.validate_file(empty)
    assert ok is False
    assert "empty" in err.lower()


def test_sensor_and_audio_metadata(tmp_path):
    files = generate_demo_dataset(tmp_path, duration_s=6)
    audio_meta = DataIngestor.extract_metadata(Path(files["audio"]))
    assert audio_meta.modality == "audio"
    assert audio_meta.duration > 5
    assert audio_meta.sample_rate == 44100

    sensor_meta = DataIngestor.extract_metadata(Path(files["sensor"]))
    assert sensor_meta.modality == "sensor"
    assert sensor_meta.channels >= 3
    assert sensor_meta.duration > 5


def test_list_files_from_directory(tmp_path):
    generate_demo_dataset(tmp_path, duration_s=4)
    files = DataIngestor.list_files_from_directory(tmp_path)
    suffixes = {p.suffix.lower() for p in files}
    assert ".mp4" in suffixes
    assert ".wav" in suffixes
    assert ".csv" in suffixes
