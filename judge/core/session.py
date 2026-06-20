"""
AnalysisSession orchestrates ingestion, detection, fusion, and result assembly.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import uuid
import logging
import threading

from tqdm import tqdm

from judge.core.ingestion import DataIngestor, MediaMetadata
from judge.core.models import AnalysisResult, AnomalyEvent, Modality
from judge.core.detectors import (
    VideoAnomalyDetector,
    AudioAnomalyDetector,
    SensorAnomalyDetector,
)
from judge.core.fusion import fuse_events

logger = logging.getLogger(__name__)


class AnalysisSession:
    """
    High level entry point.

    Example:
        session = AnalysisSession(sensitivity=0.65)
        session.add_file("drone_night.mp4")
        session.add_file("mic_omni.wav")
        results = session.run()
    """

    def __init__(
        self,
        sensitivity: float = 0.50,
        min_event_duration: float = 0.06,
        cross_modal_window: float = 1.8,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.sensitivity = sensitivity
        self.min_event_duration = min_event_duration
        self.cross_modal_window = cross_modal_window
        self.progress_callback = progress_callback or (lambda msg, pct: None)

        self.files: List[Path] = []
        self.metadata: Dict[str, MediaMetadata] = {}
        self.detectors = {
            Modality.VIDEO: VideoAnomalyDetector(
                sensitivity=sensitivity, min_duration=min_event_duration
            ),
            Modality.AUDIO: AudioAnomalyDetector(
                sensitivity=sensitivity, min_duration=min_event_duration
            ),
            Modality.SENSOR: SensorAnomalyDetector(
                sensitivity=sensitivity, min_duration=min_event_duration
            ),
        }

    def add_file(self, path: str | Path) -> bool:
        p = Path(path)
        ok, err = DataIngestor.validate_file(p)
        if not ok:
            logger.warning("Skipping %s: %s", p, err)
            return False
        if p not in self.files:
            self.files.append(p)
        return True

    def add_directory(self, directory: str | Path, recursive: bool = True) -> int:
        files = DataIngestor.list_files_from_directory(Path(directory), recursive=recursive)
        added = 0
        for f in files:
            if self.add_file(f):
                added += 1
        return added

    def clear(self) -> None:
        self.files.clear()
        self.metadata.clear()

    def _report_progress(self, message: str, pct: float) -> None:
        self.progress_callback(message, max(0.0, min(100.0, pct)))

    def _check_and_pause(self, pause_event):
        if pause_event and pause_event.is_set():
            while pause_event.is_set():
                import time
                time.sleep(0.1)

    def run(self, cancel_event: Optional["threading.Event"] = None, pause_event: Optional["threading.Event"] = None) -> AnalysisResult:
        session_id = str(uuid.uuid4())[:12]
        self.metadata.clear()
        all_events: List[AnomalyEvent] = []
        files_processed: List[str] = []
        modality_stats: Dict[str, Dict[str, Any]] = {}

        total = len(self.files)
        if total == 0:
            return AnalysisResult(session_id=session_id)

        self._report_progress("Loading file metadata...", 2)

        for idx, fpath in enumerate(self.files):
            try:
                meta = DataIngestor.extract_metadata(fpath)
                self.metadata[str(fpath)] = meta
                files_processed.append(str(fpath))
                self._report_progress(f"Loaded {fpath.name}", 5 + (idx / total) * 10)
            except Exception as e:
                logger.exception("Metadata extraction failed for %s", fpath)
                self._report_progress(f"Failed: {fpath.name}", 5 + (idx / total) * 10)

        if cancel_event and cancel_event.is_set():
            self._report_progress("Analysis cancelled", 18)
            return AnalysisResult(session_id=session_id, files_processed=files_processed, events=[])

        self._check_and_pause(pause_event)

        self._report_progress("Running modality detectors...", 18)

        detector_progress = 18
        for idx, fpath in enumerate(files_processed):
            meta = self.metadata.get(fpath)
            modality_str = meta.modality if meta else DataIngestor.detect_modality(Path(fpath))
            if modality_str is None:
                continue

            mod = Modality(modality_str)
            detector = self.detectors.get(mod)
            if detector is None:
                continue

            pct_base = detector_progress + (idx / max(1, len(files_processed))) * 55
            self._report_progress(f"Analyzing {Path(fpath).name} ({mod.value})", pct_base)

            def _sub_progress(msg: str, loc: float):
                # Update main bar a bit + provide live activity message
                adj_pct = min(73.0, pct_base + min(50.0, loc * 50.0))
                self._report_progress(f"Analyzing {Path(fpath).name}: {msg}", adj_pct)

            try:
                evs = detector.detect(
                    Path(fpath),
                    metadata=meta.__dict__ if meta else None,
                    progress_callback=_sub_progress,
                    cancel_event=cancel_event,
                    pause_event=pause_event
                )
                all_events.extend(evs)
                self._report_progress(
                    f"Found {len(evs)} events in {Path(fpath).name}", pct_base + 52
                )
            except Exception as e:
                logger.exception("Detector failed on %s", fpath)
                self._report_progress(f"Error on {Path(fpath).name}", pct_base)

            if cancel_event and cancel_event.is_set():
                self._report_progress("Analysis stopped by user", pct_base + 52)
                break

            self._check_and_pause(pause_event)

        self._report_progress("Performing cross-modal fusion...", 78)

        fused = fuse_events(all_events, window_seconds=self.cross_modal_window)

        # Simple modality stats
        for mod in Modality:
            mod_events = [e for e in fused if e.modality == mod]
            if mod_events:
                modality_stats[mod.value] = {
                    "event_count": len(mod_events),
                    "max_score": max(e.score for e in mod_events),
                    "total_anomalous_duration": sum(e.duration for e in mod_events),
                }

        self._report_progress("Assembling final result...", 95)

        result = AnalysisResult(
            session_id=session_id,
            files_processed=files_processed,
            events=fused,
            modality_stats=modality_stats,
            parameters={
                "sensitivity": self.sensitivity,
                "min_event_duration": self.min_event_duration,
                "cross_modal_window": self.cross_modal_window,
            },
            duration_seconds=max((m.duration for m in self.metadata.values()), default=0.0),
        )

        self._report_progress("Analysis complete.", 100)
        return result
