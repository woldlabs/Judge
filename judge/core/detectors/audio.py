"""
Audio anomaly detector.

Focuses on impulsive, spectrally unusual, or statistically extreme
acoustic transients using time-frequency features.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging

import numpy as np
import librosa
import soundfile as sf
import threading

from judge.core.models import Modality, AnomalyEvent
from judge.core.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class AudioAnomalyDetector(BaseDetector):
    name = "audio"
    modality = Modality.AUDIO

    def __init__(
        self,
        sensitivity: float = 0.7,
        min_duration: float = 0.03,
        n_fft: int = 2048,
        hop_length: int = 512,
        **kwargs,
    ):
        super().__init__(sensitivity=sensitivity, min_duration=min_duration, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def detect(self, file_path: Path, metadata: Optional[Dict] = None, progress_callback: Optional[Callable[[str, float], None]] = None, cancel_event: Optional["threading.Event"] = None, pause_event: Optional["threading.Event"] = None) -> List[AnomalyEvent]:
        if progress_callback:
            self._progress_cb = progress_callback
            progress_callback("loading audio waveform", 0.05)

        if cancel_event and cancel_event.is_set():
            self._report_progress("cancelled", 0.0)
            return []

        self._check_pause(pause_event)

        try:
            y, sr = librosa.load(str(file_path), sr=None, mono=True)
        except Exception as e:
            logger.error("Failed to load audio %s: %s", file_path, e)
            return []

        if len(y) < sr * 0.1:
            return []

        if progress_callback:
            progress_callback("computing spectral features", 0.25)

        # Compute key features
        # RMS energy envelope
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)[0]

        # Spectral flux (sudden timbre change)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        # pad to match
        spectral_flux = np.concatenate([[spectral_flux[0]], spectral_flux])

        # Spectral centroid deviation
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]

        # Onset strength (good for percussive / impulsive)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Align lengths
        min_len = min(len(rms), len(spectral_flux), len(cent), len(onset_env))
        rms = rms[:min_len]
        spectral_flux = spectral_flux[:min_len]
        cent = cent[:min_len]
        onset_env = onset_env[:min_len]

        times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=self.hop_length)

        if progress_callback:
            progress_callback("robust scoring + event grouping", 0.7)

        # Normalize robustly (median + MAD)
        def robust_z(x: np.ndarray) -> np.ndarray:
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-6
            z = (x - med) / mad
            return np.clip(z, -25.0, 25.0)

        z_rms = robust_z(rms)
        z_flux = robust_z(spectral_flux)
        z_cent = robust_z(cent)
        z_onset = robust_z(onset_env)

        # Composite score tuned for transients
        composite = 0.35 * np.abs(z_rms) + 0.30 * np.abs(z_flux) + 0.15 * np.abs(z_cent) + 0.20 * np.abs(z_onset)
        composite = np.clip(composite, 0, 25.0)

        thresh = self._threshold(6.0)
        mask = composite > thresh

        events = self._group_audio_events(times, composite, mask, sr, file_path)

        # Enrich features
        for ev in events:
            mask_idx = (times >= ev.start_time) & (times <= ev.end_time)
            if np.any(mask_idx):
                ev.features.update({
                    "rms_peak": float(np.max(rms[mask_idx])),
                    "flux_peak": float(np.max(spectral_flux[mask_idx])),
                    "onset_peak": float(np.max(onset_env[mask_idx])),
                    "centroid_var": float(np.var(cent[mask_idx])),
                })
        self._report_progress("audio analysis complete", 1.0)
        return events

    def _group_audio_events(
        self,
        times: np.ndarray,
        scores: np.ndarray,
        mask: np.ndarray,
        sr: int,
        file_path: Path,
    ) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        n = len(times)
        i = 0
        min_samples = max(2, int(self.min_duration / (times[1] - times[0]) if len(times) > 1 else 3))

        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1

            if (j - i) < min_samples:
                i = j
                continue

            start_t = float(times[i])
            end_t = float(times[j - 1])
            dur = end_t - start_t
            if dur < self.min_duration:
                i = j
                continue

            seg = scores[i:j]
            peak = float(np.max(seg))
            mean_s = float(np.mean(seg))

            desc = (
                f"Acoustic transient: composite MAD {mean_s:.2f}. "
                f"Duration {dur*1000:.0f} ms. High spectral flux / energy deviation."
            )

            ev = self._make_event(
                start_time=start_t,
                duration=dur,
                score=mean_s,
                peak_score=peak,
                features={"mad_composite": mean_s, "duration_ms": dur * 1000},
                description=desc,
                file_path=file_path,
            )
            events.append(ev)
            i = j
        return events
