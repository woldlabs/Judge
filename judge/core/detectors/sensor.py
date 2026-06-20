"""
Sensor / time-series anomaly detector.

Multivariate robust statistical detection suitable for magnetometers,
accelerometers, environmental sensors, etc.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from judge.core.models import Modality, AnomalyEvent
from judge.core.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class SensorAnomalyDetector(BaseDetector):
    name = "sensor"
    modality = Modality.SENSOR

    def __init__(
        self,
        sensitivity: float = 0.7,
        min_duration: float = 0.1,
        contamination: float = 0.02,
        **kwargs,
    ):
        super().__init__(sensitivity=sensitivity, min_duration=min_duration, **kwargs)
        self.contamination = contamination

    def detect(self, file_path: Path, metadata: Optional[Dict] = None, progress_callback: Optional[Callable[[str, float], None]] = None) -> List[AnomalyEvent]:
        if progress_callback:
            self._progress_cb = progress_callback
            progress_callback("loading sensor data", 0.1)

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    import json
                    data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    df = pd.DataFrame(data["data"])
                else:
                    df = pd.DataFrame(data)
        except Exception as e:
            logger.error("Failed to load sensor data %s: %s", file_path, e)
            return []

        if progress_callback:
            progress_callback("multivariate outlier detection", 0.4)

        # Identify time column
        time_col = None
        for c in df.columns:
            lc = str(c).lower()
            if lc in {"t", "time", "timestamp", "sec", "seconds", "ts"}:
                time_col = c
                break

        if time_col is None:
            # fabricate a time axis
            time_col = "__time__"
            df[time_col] = np.arange(len(df)) * 0.01  # assume 100 Hz default

        t = pd.to_numeric(df[time_col], errors="coerce").values
        numeric = df.select_dtypes(include=[np.number]).drop(columns=[time_col], errors="ignore")

        if numeric.shape[1] == 0 or len(numeric) < 10:
            return []

        X = numeric.values.astype(float)
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust scaling per column
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) + 1e-6
        Xz = (X - med) / mad

        # Isolation Forest for multivariate outliers
        n_samples = Xz.shape[0]
        iso = IsolationForest(
            contamination=min(max(self.contamination, 0.001), 0.1),
            random_state=42,
            n_estimators=100,
        )
        iso.fit(Xz)
        scores = -iso.decision_function(Xz)  # higher = more anomalous
        scores = np.clip(scores, 0, 15.0)

        # Also compute rolling MAD on the L2 norm of standardized values
        l2 = np.linalg.norm(Xz, axis=1)
        win = max(5, int(len(l2) * 0.005))
        rolling_med = pd.Series(l2).rolling(win, center=True, min_periods=1).median().values
        rolling_mad = pd.Series(np.abs(l2 - rolling_med)).rolling(win, center=True, min_periods=1).median().values + 1e-6
        robust_dev = np.abs(l2 - rolling_med) / rolling_mad
        robust_dev = np.clip(robust_dev, 0, 20.0)

        composite = 0.6 * scores + 0.4 * robust_dev
        composite = np.clip(composite, 0, 25.0)

        thresh = self._threshold(4.2)
        mask = composite > thresh

        # Convert time
        if not np.all(np.isfinite(t)):
            t = np.linspace(0, metadata.get("duration", 60.0) if metadata else 60.0, len(t))

        events = self._group_sensor_events(t, composite, mask, file_path, numeric.columns.tolist())

        for ev in events:
            # attach channel-wise peak deviations for the window
            mask_idx = (t >= ev.start_time) & (t <= ev.end_time)
            if np.any(mask_idx):
                for ci, col in enumerate(numeric.columns):
                    vals = Xz[mask_idx, ci]
                    if len(vals):
                        ev.features[f"peak_z_{col}"] = float(np.max(np.abs(vals)))
        self._report_progress("sensor analysis complete", 1.0)
        return events

    def _group_sensor_events(
        self,
        t: np.ndarray,
        scores: np.ndarray,
        mask: np.ndarray,
        file_path: Path,
        channel_names: List[str],
    ) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        n = len(t)
        i = 0
        min_len = max(3, int(self.min_duration / (np.median(np.diff(t)) if len(t) > 1 else 0.01)))

        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1

            if (j - i) < min_len:
                i = j
                continue

            start_t = float(t[i])
            end_t = float(t[j - 1])
            dur = max(0.0, end_t - start_t)
            if dur < self.min_duration:
                i = j
                continue

            seg = scores[i:j]
            peak = float(np.max(seg))
            mean_s = float(np.mean(seg))

            desc = (
                f"Multivariate sensor excursion (IsolationForest + rolling MAD). "
                f"peak_score={peak:.2f}, duration={dur:.3f}s. "
                f"Channels: {", ".join(channel_names[:4])}{'...' if len(channel_names) > 4 else ''}"
            )

            ev = self._make_event(
                start_time=start_t,
                duration=dur,
                score=mean_s,
                peak_score=peak,
                features={"mad_composite": mean_s, "n_channels": len(channel_names)},
                description=desc,
                file_path=file_path,
                channel=None,
            )
            events.append(ev)
            i = j
        return events
