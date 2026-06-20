# Quickstart Guide

## 1. Installation

```bash
git clone https://github.com/woldlabs/Judge.git
cd Judge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional but recommended on Windows/macOS:
- Install FFmpeg (https://ffmpeg.org) and ensure `ffmpeg` is on PATH.

## 2. Launch the GUI

```bash
python -m judge.gui.main
# or
python run_judge.py
```

## 3. Quick Validation with Synthetic Data

```bash
python examples/run_demo.py
```

This will:
- Generate 45 s of synthetic multi-modal data containing several injected transients
- Run the full detection + fusion pipeline
- Emit a technical PDF + Markdown + JSON report under `reports/`

## 4. Using Your Own Data

- Click **+ Add Files** or **+ Add Folder**
- Supported: MP4/MOV/AVI/MKV (video), WAV/FLAC/MP3/OGG (audio), CSV/JSON (sensor)
- Tune **Sensitivity**, **Min duration**, and **Cross-modal window**
- Click **RUN ANALYSIS**
- Browse ranked events in the Events tab
- Click any event for full technical attribution in the right panel
- Use **Export Report** for a publication-grade PDF
- Use **Export Clips** to obtain short annotated evidence segments around every detection

## 5. Headless / Batch Scripting

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
