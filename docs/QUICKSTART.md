# Quickstart Guide

## 1. Installation

```bash
git clone https://github.com/woldlabs/Judge.git
cd Judge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For tests and linting: `pip install -r requirements-dev.txt`.

Optional but recommended on Windows/macOS:
- Install FFmpeg (https://ffmpeg.org) and ensure `ffmpeg` is on PATH. Needed for some compressed audio/video containers.

## 2. Launch the GUI

```bash
python -m judge
# or
python run_judge.py
# or
python -m judge.gui.main
```

## 3. Quick Validation with Synthetic Data

```bash
python -m judge demo
# or
python examples/run_demo.py
```

This will:
- Generate synthetic multi-modal data containing injected transients
- Run the full detection + fusion pipeline
- Emit a technical PDF + Markdown + JSON report under `reports/`

## 4. Headless analysis of your own files

```bash
python -m judge analyze observation.mp4 mic.wav mag.csv -o reports/night_01 --format all
python -m judge analyze /data/field_kit --sensitivity 0.45 --min-duration 0.05
```

## 5. Using Your Own Data in the GUI

- Click **+ Add Files** or **+ Add Folder**
- Supported: MP4/MOV/AVI/MKV/M4V (video), WAV/FLAC/MP3/OGG/M4A (audio), CSV/JSON (sensor)
- Tune **Sensitivity**, **Min duration**, and **Cross-modal window**
- Click **RUN ANALYSIS** (Pause/Resume on the same button; Stop cancels after the current file)
- Browse ranked events in the Events tab
- Click any event for full technical attribution in the right panel
- Use **Export Report** for PDF + Markdown + JSON
- Use **Export Clips** to obtain short annotated evidence segments around detections
- Use **Export Shapes** for a CSV catalog of video object geometries
- **Hail Mary** re-runs analysis across 10 preset configs in a background thread

## 6. Headless / Batch Scripting

```python
from judge.core.session import AnalysisSession
from judge.reporting.generator import generate_report
from pathlib import Path

sess = AnalysisSession(sensitivity=0.50)
sess.add_directory("/data/night_observation_03")
res = sess.run()
generate_report(res, Path("out/report"), format="all")
```

## Tips for High-Quality Results

- Start with sensitivity ~0.45–0.60 (lower = higher specificity)
- Use the shortest sensible min-duration for your phenomena of interest
- Cross-modal window of 1–3 seconds works well for most physical coupling
- Run on the original highest quality files when possible
- Always review raw clips + original sensor streams; statistical candidates are never proof
