"""
Base classes for anomaly detectors in Judge.

All detectors must implement a consistent interface returning
well-scored AnomalyEvent objects with rich feature attribution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

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

    @abstractmethod
    def detect(self, file_path: Path, metadata: Optional[Dict] = None) -> List[AnomalyEvent]:
        """Run detection and return list of events sorted by time."""
        ...

    def _make_event(
        self,
        start_time: float,
        duration: float,
        score: float,
        peak_score: float,
        features: Dict[str, float],
        description: str,
        file_path: Path,
        **extra,
    ) -> AnomalyEvent:
        import uuid
        return AnomalyEvent(
            event_id=str(uuid.uuid4())[:8],
            modality=self.modality,
            start_time=float(start_time),
            duration=float(duration),
            score=float(score),
            peak_score=float(peak_score),
            features=features,
            description=description,
            file_path=str(file_path),
            **extra,
        )

    def _threshold(self, base: float) -> float:
        """Map sensitivity into a decision threshold (higher sensitivity = lower threshold)."""
        # sensitivity 0.0 -> strict (high thresh), 1.0 -> loose (low thresh)
        return base * (1.6 - 1.1 * self.sensitivity)
