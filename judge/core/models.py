"""
Core data models for Judge framework.

Defines immutable, serializable representations of detected events,
modalities, and full analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timezone


class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SENSOR = "sensor"


@dataclass(frozen=True)
class AnomalyEvent:
    """Single detected anomalous event with quantitative attribution.
    For video events, shape_description and geometry capture detected object
    shapes / bounding regions (e.g. bbox, area, aspect) for forensic analysis.
    """
    event_id: str
    modality: Modality
    start_time: float  # seconds from file/session origin
    duration: float    # seconds
    score: float       # composite anomaly score (higher = more anomalous)
    peak_score: float  # instantaneous peak within the event window
    features: Dict[str, Any]  # raw feature values that contributed
    description: str   # technical, human-readable explanation
    file_path: str
    frame_start: Optional[int] = None  # for video
    frame_end: Optional[int] = None
    channel: Optional[str] = None      # for multi-channel sensor/audio
    tags: List[str] = field(default_factory=list)
    shape_description: Optional[str] = None  # human description of detected object shape
    geometry: Optional[Dict[str, Any]] = None  # e.g. {"bbox": [x,y,w,h], "area":.., "aspect_ratio":.., "centroid": [cx,cy]}

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value
        return d

    def pretty_time(self) -> str:
        """Return HH:MM:SS.mmm representation of start time."""
        h = int(self.start_time // 3600)
        m = int((self.start_time % 3600) // 60)
        s = self.start_time % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


@dataclass
class AnalysisResult:
    """Complete results for a single analysis run."""
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    files_processed: List[str] = field(default_factory=list)
    events: List[AnomalyEvent] = field(default_factory=list)
    modality_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "files_processed": self.files_processed,
            "events": [e.to_dict() for e in self.events],
            "modality_stats": self.modality_stats,
            "parameters": self.parameters,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "AnalysisResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        event_field_names = {f.name for f in fields(AnomalyEvent)}
        events = []
        for ed in data.get("events", []):
            payload = {k: v for k, v in ed.items() if k in event_field_names}
            if "modality" in payload:
                payload["modality"] = Modality(payload["modality"])
            events.append(AnomalyEvent(**payload))
        result_field_names = {f.name for f in fields(cls)}
        result_payload = {k: v for k, v in data.items() if k in result_field_names}
        result_payload["events"] = events
        return cls(**result_payload)
