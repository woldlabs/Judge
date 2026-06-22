# Judge

**JUDGE** (Joint Unconventional Data & Geophysical Examination) is a professional-grade, multimodal anomaly detection framework for the systematic identification and characterization of statistically anomalous or physically unconventional events in large video, audio, and sensor datasets.

Designed for researchers, field investigators, and data scientists working with observational recordings where transient, non-stationary, or cross-modal signatures may indicate rare phenomena.

## Core Capabilities

- **Multimodal Ingestion**: Native support for high-resolution video (MP4, MOV, AVI, MKV), audio (WAV, FLAC, MP3, OGG), and multivariate time-series sensor data (CSV/JSON).
- **Signal-Processing-First Detectors**:
  - Video: dense optical flow statistics, frame-to-frame luminance/gradient transients, structural similarity deviations, motion entropy analysis.
  - Audio: short-time spectral flux, envelope kurtosis, onset strength outliers, band-limited energy excursions, and phase discontinuity detection.
  - Sensor: robust multivariate outlier detection (Isolation Forest + rolling MAD), CUSUM-style change-point detection, and cross-channel decorrelation events.
- **Cross-Modal Fusion**: Temporal correlation engine that surfaces co-occurring or causally plausible events across modalities with composite anomaly scores.
- **High-Precision Event Localization**: Sub-second (and frame-accurate for video) timestamping with quantitative metrics and human-readable technical descriptions.
- **Object Shape Capture** (new): For video anomalies, automatically extracts bounding boxes, area, aspect ratio and classifies rough shape (point-like / blob / streak) of motion regions. Shapes saved in JSON, reports, and exportable catalog CSV. Annotated clips include overlaid shapes.
- **Interactive Analysis GUI**: Dark-themed professional desktop interface featuring synchronized timeline visualization, per-modality spectrograms/heatmaps, clickable anomaly catalog, and live parameter tuning.
- **Automated Reporting**: Generates comprehensive, publication-quality technical reports (PDF + Markdown + structured JSON) containing statistical summaries, annotated figures, candidate classifications, and exportable evidence packages.
- **Evidence Extraction**: One-click export of short video/audio clips and sensor windows centered on detected events for downstream forensic or ML analysis. Video clips now include overlaid shape bounding boxes when available. Dedicated "Export Shapes" produces CSV catalog of object geometries for mission data correlation.
- **Reproducible & Extensible**: Deterministic processing with seeded RNG where applicable. Clean plugin architecture for additional detectors.

## Target Use Cases

- Analysis of field recordings for Unidentified Aerial Phenomena (UAP) signatures (kinematic transients, multi-spectral luminosity excursions).
- Acoustic and environmental monitoring for unexplained transient events.
- Quality assurance on large instrumentation datasets (detecting sensor glitches vs. genuine signals).
- Post-processing of long-duration multi-instrument experiments.

## Technical Approach

Judge prioritizes interpretable, physics-adjacent signal features over opaque deep learning. All detectors produce explainable metrics (e.g., "z = 7.4 on 120–180 Hz band power, 42 ms duration"). Optional classical ML components (scikit-learn IsolationForest) provide secondary ranking. A lightweight heuristic layer maps quantitative signatures to candidate categories (e.g., "high-velocity translational transient", "impulsive broadband acoustic event", "non-stationary EM-like sensor coupling").

The system is deliberately conservative: high specificity is favored. Users control global sensitivity and per-detector thresholds.

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (recommended for robust video/audio container handling; optional but improves compatibility)

### Installation

```bash
git clone https://github.com/woldlabs/Judge.git
cd Judge

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Launch GUI

```bash
python -m judge.gui.main
```

Or on Windows:

```powershell
python -m judge.gui.main
```

### Headless / Scripted Usage (example)

```python
from judge.core.session import AnalysisSession
from judge.core.detectors import VideoAnomalyDetector, AudioAnomalyDetector
from judge.reporting.generator import generate_report

session = AnalysisSession()
session.add_file("observation_night_01.mp4")
session.add_file("mic_array_01.wav")
session.add_file("magnetometer_log.csv")

results = session.run()
generate_report(results, output_path="reports/observation_night_01.pdf", format="pdf")
```

## GUI Overview

1. **Data Panel** — Add individual files or entire directories. Files are automatically typed and validated (duration, sample rate, channels extracted).
2. **Configuration** — Global sensitivity, minimum event duration, cross-modal window, and detector-specific tunables.
3. **Analysis Execution** — Real-time progress with per-file logging and cancellation support.
4. **Event Explorer** — Filterable, sortable table of all candidate events. Selection updates synchronized visualizations.
5. **Visualization Canvas** — Overview timeline + modality-specific plots (waveform + spectrogram for audio, frame-difference / flow magnitude trace for video, multi-trace sensor).
6. **Action Bar** — Export full report, export selected events (CSV/JSON), batch-export evidence clips.

## Report Contents

Technical reports include:

- Session metadata and data provenance
- Per-modality statistical baselines
- Ranked list of anomalous events with:
  - Precise timecodes (HH:MM:SS.fff / frame number)
  - Quantitative scores and contributing features
  - Shape descriptions + geometry (bbox/area/aspect) for visual objects (video)
  - Candidate phenomenological classification with confidence rationale
  - Embedded thumbnail / waveform excerpt
- Cross-modal coincidence matrix
- Aggregate metrics (total anomalous duration, peak deviation, etc.)
- Reproducibility footer (library versions, parameters, git commit if available)

## Performance Notes

- Designed for multi-hour recordings. Video analysis downsamples intelligently while preserving transient detection.
- Memory efficient: streams frames/audio blocks where possible.
- GPU optional (OpenCV CUDA paths auto-detected when available; CPU fallback always works).

## Extending Judge

Detectors live under `judge/core/detectors/`. Implement the `BaseDetector` interface. Fusion logic and event model live in `judge/core/`.

Contributions that improve detection rigor, add new physically motivated features, or enhance visualization/reporting are welcome.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Judge builds upon foundational techniques in statistical signal processing, robust statistics, and time-frequency analysis. Core dependencies include OpenCV, librosa, scikit-learn, NumPy/SciPy, pandas, matplotlib, customtkinter, and reportlab.

---

**Judge** — Because extraordinary claims require extraordinary data hygiene.
