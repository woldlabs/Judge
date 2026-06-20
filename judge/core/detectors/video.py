
"""
Video anomaly detector.

Focuses on physically interpretable, transient kinematic and photometric
signatures using dense optical flow and frame-level statistics.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import logging

import cv2
import numpy as np

from judge.core.models import Modality, AnomalyEvent
from judge.core.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class VideoAnomalyDetector(BaseDetector):
    name = "video"
    modality = Modality.VIDEO

    def __init__(
        self,
        sensitivity: float = 0.7,
        min_duration: float = 0.08,
        sample_every: int = 2,
        flow_block_size: int = 16,
        **kwargs,
    ):
        super().__init__(sensitivity=sensitivity, min_duration=min_duration, **kwargs)
        self.sample_every = max(1, sample_every)
        self.flow_block_size = flow_block_size

    def detect(self, file_path: Path, metadata: Optional[Dict] = None, progress_callback: Optional[Callable[[str, float], None]] = None) -> List[AnomalyEvent]:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            logger.error("Failed to open video %s", file_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if progress_callback:
            self._progress_cb = progress_callback

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return []

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        times: List[float] = []
        flow_mags: List[float] = []
        delta_intensities: List[float] = []
        edge_densities: List[float] = []
        frame_idx = 0
        last_report = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Always compute successive-frame diffs for accurate flow (sampling only downsamples the series)
            diff = cv2.absdiff(gray, prev_gray)
            delta_int = float(np.mean(diff))

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(np.mean(mag))

            # Edge density (Sobel) - fraction of strong edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = float(np.mean(np.hypot(sobelx, sobely) > 60))

            if frame_idx % self.sample_every == 0:
                times.append(frame_idx / fps)
                flow_mags.append(flow_mag)
                delta_intensities.append(delta_int)
                edge_densities.append(edge_density)

            prev_gray = gray

            # Live sub-progress so large videos don't look frozen
            if frame_idx - last_report >= 80:
                local = (frame_idx / total_frames) if total_frames > 10 else (frame_idx / max(1000, frame_idx + 200))
                self._report_progress(f"optical flow + features (frame {frame_idx})", local)
                last_report = frame_idx

        cap.release()

        self._report_progress("video feature extraction complete", 1.0)

        if len(times) < 3:
            return []

        # Convert to arrays
        t = np.array(times)
        fmag = np.array(flow_mags)
        dint = np.array(delta_intensities)
        edens = np.array(edge_densities)

        # Robust baseline using median + MAD, with floor to avoid explosion on near-constant signals
        def mad_scores(x: np.ndarray) -> np.ndarray:
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-6
            z = np.abs(x - med) / mad
            return np.clip(z, 0, 25.0)  # cap to prevent extreme outliers dominating

        s_flow = mad_scores(fmag)
        s_int = mad_scores(dint)
        s_edge = mad_scores(edens)

        composite = 0.5 * s_flow + 0.3 * s_int + 0.2 * s_edge

        # Higher threshold for better specificity (fewer spurious events)
        thresh = self._threshold(8.0)
        candidate_mask = composite > thresh

        # Group into contiguous events (respecting sampling stride)
        events = self._group_events(t, composite, candidate_mask, fps, file_path)

        # Attach rich features to each event
        for ev in events:
            mask = (t >= ev.start_time) & (t <= ev.end_time)
            if np.any(mask):
                ev.features.update({
                    "mean_flow": float(np.mean(fmag[mask])),
                    "peak_flow": float(np.max(fmag[mask])),
                    "mean_intensity_delta": float(np.mean(dint[mask])),
                    "peak_intensity_delta": float(np.max(dint[mask])),
                    "mean_edge_density": float(np.mean(edens[mask])),
                })
        return events

    def _group_events(
        self,
        t: np.ndarray,
        scores: np.ndarray,
        mask: np.ndarray,
        fps: float,
        file_path: Path,
    ) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        n = len(t)
        i = 0
        min_len_samples = max(2, int(self.min_duration * fps / max(1, self.sample_every)))

        while i < n:
            if not mask[i]:
                i += 1
                continue

            j = i
            while j < n and mask[j]:
                j += 1

            # Expand slightly for context
            start_idx = max(0, i - 1)
            end_idx = min(n, j + 1)

            if (end_idx - start_idx) < min_len_samples:
                i = j
                continue

            start_t = float(t[start_idx])
            end_t = float(t[end_idx - 1])
            duration = end_t - start_t
            if duration < self.min_duration:
                i = j
                continue

            seg_scores = scores[start_idx:end_idx]
            peak = float(np.max(seg_scores))
            mean_score = float(np.mean(seg_scores))

            # Build description
            desc = (
                f"Kinematic transient: peak MAD-score {peak:.2f} "
                f"(flow+intensity+edge). Duration {duration*1000:.0f} ms."
            )

            ev = self._make_event(
                start_time=start_t,
                duration=duration,
                score=mean_score,
                peak_score=peak,
                features={
                    "mad_composite": float(mean_score),
                    "duration_ms": duration * 1000,
                },
                description=desc,
                file_path=file_path,
                frame_start=int(start_t * fps),
                frame_end=int(end_t * fps),
            )
            events.append(ev)
            i = j
        return events
