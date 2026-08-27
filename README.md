# Judge

**JUDGE** (Joint Unconventional Data & Geophysical Examination) is a multimodal anomaly detection framework for identifying and characterizing statistically unusual or physically unconventional events in video, audio, and sensor datasets.

Designed for researchers, field investigators, and data scientists working with observational recordings where transient, non-stationary, or cross-modal signatures may indicate rare phenomena.

**Current version:** 0.2.0

## Core Capabilities

- **Multimodal Ingestion**: Native support for video (MP4, MOV, AVI, MKV, M4V), audio (WAV, FLAC, MP3, OGG, M4A, WMA), and multivariate time-series sensor data (CSV/JSON). Compressed audio falls back to librosa when soundfile cannot read the container.
- **Signal-Processing-First Detectors**:
  - Video: dense optical flow, frame-to-frame luminance change, and Sobel edge-density transients, with optional motion-blob shape capture (bounding box, area, aspect, centroid).
  - Audio: RMS energy envelope, short-time spectral flux, spectral centroid deviation, and onset-strength outliers.
  - Sensor: Isolation Forest multivariate outliers plus rolling MAD on the standardized L2 trajectory.
- **Cross-Modal Fusion**: Events that co-occur across modalities within a configurable time window keep their own catalog entries and receive a score boost plus `cross-modal` tags. Supporting detections are no longer dropped.
- **High-Precision Event Localization**: Sub-second (and frame-accurate for video) timestamping with quantitative metrics and human-readable technical descriptions.
- **Object Shape Capture**: For video anomalies, extracts bounding boxes, area, aspect ratio, and a rough shape class (point-like / blob / streak). Shapes are saved in JSON, reports, and an exportable catalog CSV. Annotated clips overlay the boxes.
- **Interactive Analysis GUI**: Dark-themed desktop interface with a ranked event catalog, timeline scatter, video frame preview, resizable slidedeck, Pause/Resume, Stop, and a background Hail Mary parameter sweep.
- **CLI**: `python -m judge` launches the GUI. `python -m judge analyze`, `demo`, and `version` support headless workflows.
- **Automated Reporting**: PDF + Markdown + structured JSON with statistical summaries, ranked events, shape geometry, coincidence counts, and a reproducibility footer (package versions and git commit when available).
- **Evidence Extraction**: Export short video/audio clips and sensor windows centered on detections. Video clips include shape overlays when geometry is present.
- **Reproducible & Extensible**: Deterministic processing with seeded RNG where applicable. Detectors implement `BaseDetector`.

## Target Use Cases

- Analysis of field recordings for Unidentified Aerial Phenomena (UAP) signatures (kinematic transients, luminosity excursions).
- Acoustic and environmental monitoring for unexplained transient events.
- Quality assurance on large instrumentation datasets (sensor glitches vs. genuine signals).
- Post-processing of long-duration multi-instrument experiments.

## Technical Approach

Judge prioritizes interpretable, physics-adjacent signal features over opaque deep learning. Detectors produce explainable metrics (robust MAD z-scores, flow magnitude, spectral flux, Isolation Forest scores). Optional classical ML (scikit-learn IsolationForest) provides a secondary ranking on sensor channels.

The system is deliberately conservative: high specificity is favored. Users control global sensitivity, minimum event duration, and the cross-modal coincidence window.

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (recommended for compressed video/audio containers; optional but improves compatibility)

### Installation

```bash
git clone https://github.com/woldlabs/Judge.git
cd Judge

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Development extras (pytest, ruff): `pip install -r requirements-dev.txt`

### Launch GUI

```bash
python -m judge
# or
python run_judge.py
# or
python -m judge.gui.main
```

### Headless analysis

```bash
python -m judge analyze observation_night_01.mp4 mic_array_01.wav magnetometer_log.csv -o reports/observation_night_01
python -m judge demo --duration 20
python -m judge version
```

```python
from judge.core.session import AnalysisSession
from judge.reporting.generator import generate_report

session = AnalysisSession(sensitivity=0.50, min_event_duration=0.06, cross_modal_window=1.8)
session.add_file("observation_night_01.mp4")
session.add_file("mic_array_01.wav")
session.add_file("magnetometer_log.csv")

results = session.run()
generate_report(results, output_path="reports/observation_night_01", format="pdf")
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a longer walkthrough.

## GUI Overview

1. **Data Panel** — Add individual files or entire directories. Files are typed and validated (duration, sample rate, channels extracted).
2. **Configuration** — Global sensitivity, minimum event duration, and cross-modal window, with live slider values.
3. **Analysis Execution** — Progress, live activity, ETA, Pause/Resume on the Run button, and Stop (finishes the current file then cancels).
4. **Event Explorer** — Ranked catalog of candidate events. A post-run min-score filter hides weaker hits without re-running.
5. **Visualization Canvas** — Time vs. score scatter colored by modality; selecting an event highlights it.
6. **Action Bar** — Run; Export Report (PDF/MD/JSON); Export Clips (evidence slices, with shape bbox overlays on video); Export Timeline (annotated PNG); Export Shapes (CSV catalog); Stop; Hail Mary (10 preset configs, background thread); Slidedeck.
7. **Slidedeck** — Resizable window for browsing event stills. Frames are preloaded so navigation and maximize/expand stay responsive.

## Report Contents

Technical reports include:

- Session metadata and data provenance
- Per-modality event counts and peak scores
- Ranked list of anomalous events with:
  - Precise timecodes (HH:MM:SS.fff / frame number)
  - Quantitative scores and contributing features
  - Shape descriptions + geometry (bbox/area/aspect) for visual objects
  - Cross-modal coincidence tags when another modality fired in the fusion window
- Cross-modal coincidence count
- Analysis parameters
- Reproducibility footer (Judge version, library versions, git commit if available)

All events are statistical candidates only.

## Performance Notes

- Video analysis downsamples frames (`analysis_scale=0.5` by default) and skips unsampled frames with `VideoCapture.grab()` so Farneback optical flow is not paid on every frame.
- Memory efficient: streams video frames and loads audio/sensor files per detector.
- CPU-only. OpenCV CUDA is not required.

## Extending Judge

Detectors live under `judge/core/detectors/`. Implement the `BaseDetector` interface. Fusion logic and event model live in `judge/core/`. Headless entry points live in `judge/__main__.py`.

Contributions that improve detection rigor, add physically motivated features, or enhance visualization/reporting are welcome.

## Updates

### 0.2.0

- Added a command-line interface: `python -m judge` (GUI), `analyze`, `demo`, and `version`.
- Cross-modal fusion now **keeps every detection**. Coincident events are score-boosted and tagged instead of being collapsed into a single surviving event.
- Video detector skips unsampled frames instead of running optical flow on every frame, which is substantially faster on long clips at the default `sample_every=3`.
- Detectors merge brief gaps in the anomaly mask so a real transient is not split into fragments shorter than `min_event_duration` and discarded.
- Audio metadata extraction falls back to librosa when soundfile cannot read the file (typical for MP3/OGG without extra codecs), so those files are no longer skipped at ingest.
- Sensor CSV duration is estimated from the file tail, not only the first few thousand rows.
- Hail Mary runs in a background thread, can be paused/stopped, and no longer crashes on frozen event dataclasses when tagging sweep passes. Sweep results are a first-class `AnalysisResult` so reports and exports work afterwards.
- Reports now include coincidence counts and a reproducibility footer (package versions, git commit when present). JSON load ignores unknown fields so older/newer result files round-trip.
- Restored the Wold Labs logo assets used by the GUI.
- Expanded unit/integration tests (fusion, models, ingestion, detectors, CLI) and added GitHub Actions CI.

### 0.1.0

- Initial public release: multimodal detectors, GUI, shape capture, report/clip/timeline/shape exports, slidedeck, and synthetic demo pipeline.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Judge builds upon foundational techniques in statistical signal processing, robust statistics, and time-frequency analysis. Core dependencies include OpenCV, librosa, scikit-learn, NumPy/SciPy, pandas, matplotlib, customtkinter, and reportlab.

---

**Judge** — Because extraordinary claims require extraordinary data hygiene.
