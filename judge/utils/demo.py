"""
Demo dataset generator.

Creates synthetic but realistic video (as mp4), audio (wav), and sensor (csv)
files containing injected anomalous events. Useful for quick validation of
the Judge pipeline without external recordings.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

import cv2


def generate_demo_dataset(
    output_dir: str | Path,
    duration_s: float = 45.0,
    fps: float = 30.0,
    sr: int = 44100,
) -> dict:
    """
    Generate a small demo dataset with injected anomalies.
    Returns dict of created file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_path = out / "demo_video.mp4"
    audio_path = out / "demo_audio.wav"
    sensor_path = out / "demo_sensor.csv"

    # --- VIDEO ---
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    n_frames = int(duration_s * fps)
    rng = np.random.default_rng(42)

    for i in range(n_frames):
        t = i / fps
        # Background: slow moving gradient + mild noise
        img = np.zeros((h, w, 3), dtype=np.uint8)
        val = int(40 + 15 * np.sin(t * 0.7))
        img[:] = (val, val + 8, val + 18)

        # Add slow drifting "stars"
        for _ in range(22):
            x = int((rng.random() * w + t * 12) % w)
            y = int((rng.random() * h + t * 3) % h)
            cv2.circle(img, (x, y), 1, (210, 225, 255), -1)

        # Inject a strong kinematic transient around t=12.4s and t=31.8s
        if 12.2 < t < 12.9:
            cx = int(w * 0.4 + (t - 12.2) * 140)
            cy = int(h * 0.5 + np.sin(t * 9) * 38)
            cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), 14, (180, 195, 255), 1)
        if 31.5 < t < 32.3:
            cx = int(w * 0.65 - (t - 31.5) * 90)
            cy = int(h * 0.38 + np.cos(t * 11) * 22)
            cv2.circle(img, (cx, cy), 5, (255, 235, 210), -1)

        vw.write(img)
    vw.release()

    # --- AUDIO ---
    t_audio = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Background noise + low hum
    audio = 0.015 * rng.standard_normal(len(t_audio))
    audio += 0.03 * np.sin(2 * np.pi * 48 * t_audio)  # 48 Hz hum

    # Inject two impulsive + spectral anomalies (guarded)
    def add_impulse(a, t0, amp=0.7, freq=680):
        idx = int(t0 * sr)
        length = int(0.12 * sr)
        if idx < 0 or idx + length > len(a):
            return
        env = np.exp(-np.linspace(0, 6, length))
        tone = np.sin(2 * np.pi * freq * np.linspace(0, length / sr, length))
        a[idx:idx + length] += amp * env * tone
        tone2 = np.sin(2 * np.pi * 1850 * np.linspace(0, length / sr, length))
        a[idx:idx + length] += 0.4 * amp * env * tone2 * 0.6

    add_impulse(audio, 8.7, amp=0.55, freq=920)
    if duration_s > 20:
        add_impulse(audio, 27.15, amp=0.8, freq=420)

    # Clip safely
    audio = np.clip(audio, -0.98, 0.98).astype(np.float32)
    sf.write(str(audio_path), audio, sr)

    # --- SENSOR (magnetometer-like) ---
    n_samp = int(duration_s * 120)  # 120 Hz
    ts = np.linspace(0, duration_s, n_samp)
    rng2 = np.random.default_rng(7)
    bx = 0.8 * rng2.standard_normal(n_samp) + 12.0
    by = 0.6 * rng2.standard_normal(n_samp) - 3.4
    bz = 1.1 * rng2.standard_normal(n_samp) + 48.2
    # temperature channel
    temp = 21.5 + 0.4 * np.sin(ts * 0.04) + 0.1 * rng2.standard_normal(n_samp)

    # Inject anomalies (guarded for short demo clips)
    # 1. Fast vector spike ~18.3s
    spike = (ts > 18.1) & (ts < 18.55)
    if spike.any():
        bx[spike] += 28 * np.exp(-((ts[spike] - 18.3) ** 2) / 0.012)
        by[spike] -= 11 * np.exp(-((ts[spike] - 18.3) ** 2) / 0.012)

    # 2. Slow non-stationary drift in z (only when duration allows)
    if duration_s > 30:
        drift = (ts > 35.5) & (ts < 37.8)
        if drift.any():
            bz[drift] += 6.5 * np.sin((ts[drift] - 35.5) * 2.8) * (1 + 0.6 * rng2.random(len(ts[drift])))

    import pandas as pd
    df = pd.DataFrame({
        "t": ts,
        "Bx": bx.astype(np.float32),
        "By": by.astype(np.float32),
        "Bz": bz.astype(np.float32),
        "temp_C": temp.astype(np.float32),
    })
    df.to_csv(sensor_path, index=False)

    return {
        "video": str(video_path),
        "audio": str(audio_path),
        "sensor": str(sensor_path),
        "dir": str(out),
    }


if __name__ == "__main__":
    # Allow quick generation from command line
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_data"
    files = generate_demo_dataset(out)
    print("Generated demo dataset:")
    for k, v in files.items():
        print(f"  {k}: {v}")
