"""
Video anomaly detector.

Focuses on physically interpretable, transient kinematic and photometric
signatures using dense optical flow and frame-level statistics.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
import logging

import cv2
import numpy as np
import time
import threading
from dataclasses import replace

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
        sample_every: int = 3,
        flow_block_size: int = 16,
        extract_shapes: bool = True,
        analysis_scale: float = 0.5,
        **kwargs,
    ):
        super().__init__(sensitivity=sensitivity, min_duration=min_duration, **kwargs)
        self.sample_every = max(1, sample_every)
        self.flow_block_size = flow_block_size
        self.extract_shapes = extract_shapes
        self.analysis_scale = max(0.1, min(1.0, float(analysis_scale)))  # downsample factor for heavy CV ops; 0.5=4x speedup typically

    def detect(self, file_path: Path, metadata: Optional[Dict] = None, progress_callback: Optional[Callable[[str, float], None]] = None, cancel_event: Optional[threading.Event] = None, pause_event: Optional[threading.Event] = None) -> List[AnomalyEvent]:
        file_path = Path(file_path)
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

        # Read as downsampled early for speed (cvt + flow on smaller is much faster)
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(full_gray, (0, 0), fx=self.analysis_scale, fy=self.analysis_scale, interpolation=cv2.INTER_AREA) if self.analysis_scale < 1.0 else full_gray

        times: List[float] = []
        flow_mags: List[float] = []
        delta_intensities: List[float] = []
        edge_densities: List[float] = []
        frame_idx = 0
        last_report_time = time.time()

        while True:
            if cancel_event and cancel_event.is_set():
                self._report_progress("cancelled by user", 0.0)
                break

            self._check_pause(pause_event)

            frame_idx += 1
            # Skip decode+flow on frames we will not sample. Optical flow is then
            # computed between consecutive *sampled* frames, which is the intended
            # temporal resolution and avoids paying Farneback on every frame.
            if self.sample_every > 1 and (frame_idx % self.sample_every) != 0:
                if not cap.grab():
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(full_gray, (0, 0), fx=self.analysis_scale, fy=self.analysis_scale, interpolation=cv2.INTER_AREA) if self.analysis_scale < 1.0 else full_gray

            # Compute on (possibly downsampled) images -- 4x faster typically at 0.5 scale
            diff = cv2.absdiff(gray, prev_gray)
            delta_int = float(np.mean(diff))

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=13,
                iterations=1, poly_n=3, poly_sigma=1.1, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(np.mean(mag))

            # Edge density (Sobel) - fraction of strong edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = float(np.mean(np.hypot(sobelx, sobely) > 50))

            times.append(frame_idx / fps)
            flow_mags.append(flow_mag)
            delta_intensities.append(delta_int)
            edge_densities.append(edge_density)

            prev_gray = gray

            # Live sub-progress ~every 2 seconds so it doesn't look frozen
            now = time.time()
            if now - last_report_time >= 2.0:
                local = (frame_idx / total_frames) if total_frames > 10 else (frame_idx / max(1000, frame_idx + 200))
                self._report_progress(f"optical flow + features (frame {frame_idx})", local)
                last_report_time = now

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

        if self.extract_shapes:
            events = self._enrich_with_shapes(events, file_path, fps)

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
        min_len_samples = max(2, int(self.min_duration * fps / max(1, self.sample_every)))
        runs = self._true_runs(mask, max_gap=1)

        for i, j in runs:
            # Expand slightly for context
            start_idx = max(0, i - 1)
            end_idx = min(len(t), j + 1)

            if (end_idx - start_idx) < min_len_samples:
                continue

            start_t = float(t[start_idx])
            end_t = float(t[end_idx - 1])
            duration = end_t - start_t
            if duration < self.min_duration:
                continue

            seg_scores = scores[start_idx:end_idx]
            peak = float(np.max(seg_scores))
            mean_score = float(np.mean(seg_scores))

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
        return events

    def _enrich_with_shapes(
        self, events: List[AnomalyEvent], file_path: Path, fps: float
    ) -> List[AnomalyEvent]:
        """Re-extract representative frame shapes (bbox + description) for video events."""
        enriched: List[AnomalyEvent] = []
        for ev in events:
            if ev.modality != Modality.VIDEO:
                enriched.append(ev)
                continue
            # Pick a representative time near peak (use midpoint)
            rep_t = ev.start_time + max(0.0, min(ev.duration * 0.5, 0.2))
            shape_desc, geom = self._extract_shape_at_time(file_path, rep_t, fps)
            if shape_desc or geom:
                new_features = dict(ev.features)
                if geom:
                    new_features.update({
                        "shape_area": float(geom.get("area", 0)),
                        "shape_aspect": float(geom.get("aspect_ratio", 1.0)),
                    })
                ev = replace(
                    ev,
                    shape_description=shape_desc or ev.shape_description,
                    geometry=geom or ev.geometry,
                    features=new_features,
                )
            enriched.append(ev)
        return enriched

    def _extract_shape_at_time(
        self, file_path: Path, t: float, fps: float
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Seek to time t, analyze local motion to derive a simple object shape description + geometry."""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return None, None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_pos = max(0, min(int(t * fps), max(0, total - 2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret1, f1 = cap.read()
            ret2, f2 = cap.read()
            if not ret1 or f1 is None or not ret2 or f2 is None:
                return None, None

            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            # downscale for shape mask too (fast); we'll scale geometry back
            scale = self.analysis_scale
            if scale < 1.0:
                g1 = cv2.resize(g1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                g2 = cv2.resize(g2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(g1, g2)
            # Threshold motion
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            # Dilate to connect components
            kernel = np.ones((5, 5), np.uint8)
            motion = cv2.dilate(motion, kernel, iterations=2)
            motion = cv2.erode(motion, kernel, iterations=1)

            contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            inv = 1.0
            if scale < 1.0:
                inv = 1.0 / scale
                if area * (inv**2) < 4:  # effective full res area
                    return None, None
            elif area < 4:
                return None, None
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 1.0
            cx, cy = x + w // 2, y + h // 2

            # scale geometry back to original video resolution for correct overlays / reports
            if scale < 1.0:
                x, y, w, h = [int(v * inv) for v in (x, y, w, h)]
                area *= (inv * inv)
                cx, cy = int(cx * inv), int(cy * inv)

            # Classify rough shape
            if aspect >= 2.8:
                stype = "elongated / linear streak"
            elif aspect <= 0.36:
                stype = "vertically elongated"
            elif area < 40:
                stype = "compact point-like"
            elif area > 800:
                stype = "large extended region"
            else:
                stype = "irregular blob"

            shape_desc = f"Detected object: {stype} (area~{area:.0f}px, aspect~{aspect:.1f})"
            geom: Dict[str, Any] = {
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area),
                "aspect_ratio": round(float(aspect), 2),
                "centroid": [int(cx), int(cy)],
                "type": "motion_blob",
            }
            return shape_desc, geom
        finally:
            cap.release()
