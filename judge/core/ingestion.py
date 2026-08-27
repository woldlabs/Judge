"""
Ingestion layer for Judge.

Handles discovery, validation, and lightweight metadata extraction
for video, audio, and sensor files. Designed to be memory-efficient
for very large source files.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SUPPORTED_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma"}
SUPPORTED_SENSOR = {".csv", ".json"}


@dataclass
class MediaMetadata:
    """Lightweight metadata extracted from a media file."""
    path: Path
    modality: str
    duration: float
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0


class DataIngestor:
    """Central ingestion service. Stateless and reusable."""

    @staticmethod
    def detect_modality(path: Path) -> Optional[str]:
        suffix = Path(path).suffix.lower()
        if suffix in SUPPORTED_VIDEO:
            return "video"
        if suffix in SUPPORTED_AUDIO:
            return "audio"
        if suffix in SUPPORTED_SENSOR:
            return "sensor"
        return None

    @staticmethod
    def validate_file(path: Path) -> Tuple[bool, Optional[str]]:
        path = Path(path)
        if not path.exists():
            return False, "File does not exist"
        if path.stat().st_size == 0:
            return False, "File is empty"
        modality = DataIngestor.detect_modality(path)
        if modality is None:
            return False, f"Unsupported file type: {path.suffix}"
        return True, None

    @staticmethod
    def extract_metadata(path: Path) -> MediaMetadata:
        """Extract minimal metadata without loading full media into memory."""
        path = Path(path)
        modality = DataIngestor.detect_modality(path)
        if modality is None:
            raise ValueError(f"Unsupported modality for {path}")

        if modality == "video":
            return DataIngestor._video_metadata(path)
        elif modality == "audio":
            return DataIngestor._audio_metadata(path)
        else:
            return DataIngestor._sensor_metadata(path)

    @staticmethod
    def _video_metadata(path: Path) -> MediaMetadata:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        # Fallback duration estimation for some containers
        if duration <= 0:
            # Read a few frames to guess
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, _ = cap.read()
            if ok:
                # Very rough fallback
                duration = max(1.0, frame_count / 30.0) if frame_count else 60.0

        cap.release()

        return MediaMetadata(
            path=path,
            modality="video",
            duration=duration,
            fps=fps,
            frame_count=frame_count if frame_count > 0 else None,
            width=width,
            height=height,
        )

    @staticmethod
    def _audio_metadata(path: Path) -> MediaMetadata:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return MediaMetadata(
                path=path,
                modality="audio",
                duration=float(info.duration),
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
            )
        except Exception:
            logger.info("soundfile could not read %s; falling back to librosa", path)
            import librosa
            duration = float(librosa.get_duration(path=str(path)))
            _y, sr = librosa.load(str(path), sr=None, mono=False, duration=0.05)
            channels = int(_y.shape[0]) if getattr(_y, "ndim", 1) > 1 else 1
            return MediaMetadata(
                path=path,
                modality="audio",
                duration=duration,
                sample_rate=int(sr),
                channels=channels,
                extra={"decoder": "librosa"},
            )

    @staticmethod
    def _sensor_metadata(path: Path) -> MediaMetadata:
        duration = 0.0
        channels = 0
        sample_rate = None

        if path.suffix.lower() == ".csv":
            # Peek without full load
            import pandas as pd
            df = pd.read_csv(path, nrows=5000)
            # Try to find time column
            time_col = None
            for c in df.columns:
                if str(c).lower() in {"t", "time", "timestamp", "seconds", "sec", "ts"}:
                    time_col = c
                    break
            if time_col is not None and len(df) > 1:
                t = pd.to_numeric(df[time_col], errors="coerce").dropna()
                if len(t) > 1:
                    duration = float(t.iloc[-1] - t.iloc[0])
                tail_duration = DataIngestor._csv_tail_duration(path, time_col, t.iloc[0] if len(t) else None)
                if tail_duration is not None:
                    duration = tail_duration
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            channels = len(numeric_cols)
        else:
            # JSON: expect list of records or {"data": [...]}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                records = data["data"]
            elif isinstance(data, list):
                records = data
            else:
                records = []
            if records and isinstance(records[0], dict):
                channels = len([k for k in records[0] if isinstance(records[0][k], (int, float))])
                # crude duration
                if "t" in records[0] or "time" in records[0]:
                    times = [r.get("t", r.get("time", 0)) for r in records if isinstance(r.get("t", r.get("time")), (int, float))]
                    if len(times) > 1:
                        duration = float(max(times) - min(times))

        return MediaMetadata(
            path=path,
            modality="sensor",
            duration=duration,
            channels=channels if channels else None,
            sample_rate=sample_rate,
            extra={"inferred": True},
        )

    @staticmethod
    def _csv_tail_duration(path: Path, time_col: str, t0) -> Optional[float]:
        """Estimate full duration from the last rows when the file is larger than the peek."""
        try:
            import pandas as pd
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < 2048:
                    return None
                f.seek(max(0, size - 8192))
                tail = f.read().decode("utf-8", errors="ignore")
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            if not lines:
                return None
            header = None
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            if not header:
                return None
            from io import StringIO
            blob = header + "\n".join(lines[-8:])
            df_tail = pd.read_csv(StringIO(blob))
            if time_col not in df_tail.columns:
                return None
            t_tail = pd.to_numeric(df_tail[time_col], errors="coerce").dropna()
            if t0 is None or len(t_tail) == 0:
                return None
            return float(t_tail.iloc[-1] - float(t0))
        except Exception:
            return None

    @staticmethod
    def list_files_from_directory(directory: Path, recursive: bool = True) -> List[Path]:
        exts = SUPPORTED_VIDEO | SUPPORTED_AUDIO | SUPPORTED_SENSOR
        if recursive:
            files = [p for p in directory.rglob("*") if p.suffix.lower() in exts]
        else:
            files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(files)
