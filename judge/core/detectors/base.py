"""
Base classes for anomaly detectors in Judge.

All detectors must implement a consistent interface returning
well-scored AnomalyEvent objects with rich feature attribution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import threading
import time
import uuid

from judge.core.models import AnomalyEvent, Modality


class BaseDetector(ABC):
    """Abstract base for all modality-specific anomaly detectors."""

    name: str = "base"
    modality: Modality = Modality.VIDEO

    def __init__(self, sensitivity: float = 0.7, min_duration: float = 0.05, **kwargs: Any):
        """
        sensitivity: 0.0 (very strict) to 1.0 (permissive)
        min_duration: minimum duration (seconds) to consider an event valid
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.min_duration = max(0.01, min_duration)
        self.params: Dict[str, Any] = dict(kwargs)
        self._progress_cb: Callable[[str, float], None] = kwargs.get("progress_callback") or (lambda msg, p: None)

    def _report_progress(self, message: str, local_pct: float = 0.0):
        """Report sub-task progress (0.0-1.0) for live status updates."""
        try:
            self._progress_cb(str(message)[:100], max(0.0, min(1.0, float(local_pct))))
        except Exception:
            pass

    def _check_pause(self, pause_event):
        if pause_event and pause_event.is_set():
            while pause_event.is_set():
                time.sleep(0.1)

    @abstractmethod
    def detect(self, file_path: Path, metadata: Optional[Dict] = None, progress_callback: Optional[Callable[[str, float], None]] = None, cancel_event: Optional["threading.Event"] = None, pause_event: Optional["threading.Event"] = None) -> List[AnomalyEvent]:
        """Run detection and return list of events sorted by time."""
        ...

    def _make_event(
        self,
        start_time: float,
        duration: float,
        score: float,
        peak_score: float,
        features: Dict[str, Any],
        description: str,
        file_path: Path,
        **extra,
    ) -> AnomalyEvent:
        score = float(min(99.0, max(0.0, score)))
        peak_score = float(min(99.0, max(0.0, peak_score)))
        return AnomalyEvent(
            event_id=str(uuid.uuid4())[:8],
            modality=self.modality,
            start_time=float(start_time),
            duration=float(duration),
            score=score,
            peak_score=peak_score,
            features=features,
            description=description,
            file_path=str(file_path),
            **extra,
        )

    def _threshold(self, base: float) -> float:
        """Map sensitivity into a decision threshold (higher sensitivity = lower threshold)."""
        # sensitivity 0.0 -> strict (high thresh), 1.0 -> loose (low thresh)
        return base * (1.6 - 1.1 * self.sensitivity)

    @staticmethod
    def _true_runs(mask, max_gap: int = 0):
        """Return [start, end) index pairs of True runs, merging gaps of at most max_gap samples."""
        n = len(mask)
        runs = []
        i = 0
        max_gap = max(0, int(max_gap))
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            end = i + 1
            i += 1
            while i < n:
                if mask[i]:
                    end = i + 1
                    i += 1
                    continue
                gap = 0
                j = i
                while j < n and not mask[j] and gap < max_gap:
                    j += 1
                    gap += 1
                if j < n and mask[j] and gap <= max_gap:
                    i = j
                    continue
                break
            runs.append((start, end))
            i = max(i, end)
        return runs
