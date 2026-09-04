"""
Judge → Rift evidence pack export.

Builds a local-only folder or zip shaped for Rift's existing Judge import:
report JSON (map pins) + shapes CSV + optional evidence clips + README
with site coordinate placeholders.

No network calls. No HOIC coupling. Packs may contain sensitive field media.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from judge.core.models import AnalysisResult, AnomalyEvent, Modality

logger = logging.getLogger(__name__)

PACK_SCHEMA_VERSION = 1
DEFAULT_CLIP_CAP = 30


@dataclass
class EvidencePackResult:
    """Paths produced by :func:`export_evidence_pack`."""

    pack_dir: Path
    report_json: Path
    shapes_csv: Path
    readme: Path
    manifest_json: Path
    clips_dir: Path
    zip_path: Optional[Path] = None
    clips_exported: int = 0
    clips_skipped: List[str] = field(default_factory=list)


def write_shapes_catalog(events: Sequence[AnomalyEvent], path: Path) -> int:
    """Write the same shapes CSV schema as the GUI Export Shapes action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    video_events = [e for e in events if e.modality == Modality.VIDEO]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "time",
                "score",
                "duration_ms",
                "shape_description",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "area",
                "aspect_ratio",
                "centroid_x",
                "centroid_y",
                "file",
            ]
        )
        for ev in sorted(video_events, key=lambda e: -e.score):
            geom = ev.geometry or {}
            bbox = geom.get("bbox", [None] * 4) if isinstance(geom, dict) else [None] * 4
            if not isinstance(bbox, (list, tuple)):
                bbox = [None] * 4
            bbox = list(bbox) + [None] * (4 - len(bbox))
            cx = cy = None
            if isinstance(geom, dict):
                c = geom.get("centroid", [None, None])
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    cx, cy = c[0], c[1]
            writer.writerow(
                [
                    ev.event_id,
                    ev.pretty_time(),
                    round(ev.score, 3),
                    round(ev.duration * 1000, 1),
                    ev.shape_description or "",
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    geom.get("area") if isinstance(geom, dict) else None,
                    geom.get("aspect_ratio") if isinstance(geom, dict) else None,
                    cx,
                    cy,
                    Path(ev.file_path).name,
                ]
            )
    return len(video_events)


def export_event_clip(ev: AnomalyEvent, outdir: Path) -> Optional[Path]:
    """
    Export a short window around one event (video/audio/sensor).

    Returns the output path, or None if the source file is missing.
    Raises on hard decode/write failures after a source was found.
    """
    p = Path(ev.file_path)
    if not p.is_file():
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    start = max(0.0, ev.start_time - 0.6)
    dur = min(ev.duration + 1.8, 8.0)

    if ev.modality == Modality.VIDEO:
        import cv2

        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = outdir / f"{ev.event_id}_{ev.modality.value}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        target_frames = int(dur * fps)
        bbox = None
        if ev.geometry and isinstance(ev.geometry, dict):
            bb = ev.geometry.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                bbox = [int(v) for v in bb]
        for _ in range(target_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
                label = (ev.shape_description or "anomaly")[:30]
                cv2.putText(
                    frame,
                    label,
                    (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )
            writer.write(frame)
        writer.release()
        cap.release()
        return out_path

    if ev.modality == Modality.AUDIO:
        import soundfile as sf

        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        si = int(start * sr)
        ei = min(len(y), int((start + dur) * sr))
        out_path = outdir / f"{ev.event_id}_{ev.modality.value}.wav"
        sf.write(str(out_path), y[si:ei], sr)
        return out_path

    # sensor
    import pandas as pd

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.read_csv(p)
    time_col = None
    for c in df.columns:
        if str(c).lower() in {"t", "time", "timestamp", "sec", "seconds", "ts"}:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    mask = (pd.to_numeric(df[time_col], errors="coerce") >= start) & (
        pd.to_numeric(df[time_col], errors="coerce") <= start + dur
    )
    out_path = outdir / f"{ev.event_id}_{ev.modality.value}.csv"
    df[mask].to_csv(out_path, index=False)
    return out_path


def _readme_text(
    *,
    session_id: str,
    event_count: int,
    default_lat: Optional[float],
    default_lon: Optional[float],
    clips_exported: int,
    clips_skipped: Sequence[str],
) -> str:
    lat_s = "REPLACE_ME" if default_lat is None else f"{default_lat}"
    lon_s = "REPLACE_ME" if default_lon is None else f"{default_lon}"
    skip_block = ""
    if clips_skipped:
        skip_block = "\n".join(f"- `{s}`" for s in clips_skipped[:40])
        if len(clips_skipped) > 40:
            skip_block += f"\n- … and {len(clips_skipped) - 40} more"
    else:
        skip_block = "_None_"

    return f"""# Judge → Rift evidence pack

**Local-only export.** This folder/zip may contain sensitive field media.
Do not upload to shared drives or chat without an authorization review.

## Session
- Judge `session_id`: `{session_id}`
- Events in report: {event_count}
- Evidence clips written: {clips_exported}

## Site coordinates (placeholders for Rift pin placement)
Edit these before import if you did not pass `--lat` / `--lon`:

- `default_lat`: `{lat_s}`
- `default_lon`: `{lon_s}`

Rift's Judge bridge uses these defaults when the report has no geo fields
(`rift/integrations/judge.py` → `import_judge_report`).

## Import into Rift
1. Unzip if needed.
2. In Rift, **Upload** `report.json` (or POST `/api/upload` / `/api/import_judge`).
3. Rift auto-detects a Judge AnalysisResult and places `judge` pins.
4. `shapes.csv` and `clips/` are investigator archive — map pins come from `report.json`.

## Contents
- `report.json` — Judge AnalysisResult (Rift import target)
- `shapes.csv` — video shape / geometry catalog
- `clips/` — short evidence windows (when source media was available)
- `pack_manifest.json` — pack metadata
- `README.md` — this file

## Clips skipped (missing source or export failure)
{skip_block}

## Privacy
Default is local filesystem only. No network calls are made by this exporter.
"""


def export_evidence_pack(
    result: AnalysisResult,
    output: str | Path,
    *,
    default_lat: Optional[float] = None,
    default_lon: Optional[float] = None,
    include_clips: bool = True,
    clip_cap: int = DEFAULT_CLIP_CAP,
    as_zip: bool = True,
) -> EvidencePackResult:
    """
    Write a Rift-ready evidence pack next to ``output``.

    If ``as_zip`` is True (default), ``output`` should end in ``.zip`` (or
    ``.zip`` is appended). The working folder is created beside the zip and
    left in place for inspection; the zip is the shareable artifact.

    If ``as_zip`` is False, ``output`` is treated as a directory path.
    """
    output = Path(output)
    if as_zip:
        zip_path = output if output.suffix.lower() == ".zip" else output.with_suffix(".zip")
        pack_dir = zip_path.with_suffix("")  # foo.zip -> foo/
        if pack_dir.suffix:  # edge: foo.bar.zip already handled
            pack_dir = Path(str(zip_path)[:-4])
    else:
        zip_path = None
        pack_dir = output

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = pack_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    report_json = pack_dir / "report.json"
    result.save_json(str(report_json))

    shapes_csv = pack_dir / "shapes.csv"
    write_shapes_catalog(result.events, shapes_csv)

    clips_exported = 0
    clips_skipped: List[str] = []
    if include_clips:
        for ev in sorted(result.events, key=lambda e: -e.score)[: max(0, clip_cap)]:
            try:
                out = export_event_clip(ev, clips_dir)
                if out is None:
                    clips_skipped.append(f"{ev.event_id}: missing source {ev.file_path}")
                else:
                    clips_exported += 1
            except Exception as exc:  # noqa: BLE001 — pack should continue
                logger.warning("Clip export failed for %s: %s", ev.event_id, exc)
                clips_skipped.append(f"{ev.event_id}: {exc}")

    readme = pack_dir / "README.md"
    readme.write_text(
        _readme_text(
            session_id=result.session_id,
            event_count=len(result.events),
            default_lat=default_lat,
            default_lon=default_lon,
            clips_exported=clips_exported,
            clips_skipped=clips_skipped,
        ),
        encoding="utf-8",
    )

    manifest: Dict[str, Any] = {
        "schema": "judge.evidence_pack",
        "schema_version": PACK_SCHEMA_VERSION,
        "session_id": result.session_id,
        "timestamp": result.timestamp,
        "event_count": len(result.events),
        "rift_import_file": "report.json",
        "default_lat": default_lat,
        "default_lon": default_lon,
        "clips_exported": clips_exported,
        "clips_skipped": list(clips_skipped),
        "local_only": True,
        "privacy_note": "May contain sensitive field media; default local-only export.",
    }
    manifest_json = pack_dir / "pack_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    final_zip: Optional[Path] = None
    if as_zip and zip_path is not None:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(pack_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(pack_dir)))
        final_zip = zip_path

    return EvidencePackResult(
        pack_dir=pack_dir,
        report_json=report_json,
        shapes_csv=shapes_csv,
        readme=readme,
        manifest_json=manifest_json,
        clips_dir=clips_dir,
        zip_path=final_zip,
        clips_exported=clips_exported,
        clips_skipped=clips_skipped,
    )
