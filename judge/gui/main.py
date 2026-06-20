"""
Professional GUI entry point for Judge.

Dark modern interface built on customtkinter + embedded matplotlib.
"""

from __future__ import annotations
import sys
import threading
import logging
from pathlib import Path
from typing import Optional, List
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
from PIL import Image

from judge.core.session import AnalysisSession
from judge.core.models import AnalysisResult, AnomalyEvent
from judge.reporting.generator import generate_report

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("judge.gui")


class JudgeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JUDGE — Multimodal Anomaly Detection")
        self.geometry("1180x760")
        self.minsize(980, 620)

        self.session: Optional[AnalysisSession] = None
        self.result: Optional[AnalysisResult] = None
        self.events: List[AnomalyEvent] = []
        self.selected_event: Optional[AnomalyEvent] = None
        self._file_paths: List[str] = []
        self._analysis_start_time = None
        self.cancel_event = None

        self._build_ui()
        self._log("JUDGE ready. Add files and run analysis.")

    def _build_ui(self):
        # Top bar
        top = ctk.CTkFrame(self, height=58, corner_radius=0)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

        # Wold Labs logo (with b64 fallback so it can live in the repo)
        try:
            logo_path = Path(__file__).parent.parent.parent / "assets" / "woldlabs_logo.jpg"
            if not logo_path.exists():
                logo_path = Path("assets/woldlabs_logo.jpg")
            if logo_path.exists():
                pil_logo = Image.open(str(logo_path))
            else:
                b64p = Path("assets/woldlabs_logo.b64")
                if b64p.exists():
                    import base64
                    from io import BytesIO
                    with open(str(b64p), "r") as f:
                        raw = base64.b64decode(f.read())
                    pil_logo = Image.open(BytesIO(raw))
                else:
                    raise FileNotFoundError("no logo")
            target_h = 38
            ratio = target_h / pil_logo.height
            new_size = (int(pil_logo.width * ratio), target_h)
            pil_logo = pil_logo.resize(new_size, Image.LANCZOS)
            self.logo_ctk = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=new_size)
            logo_lbl = ctk.CTkLabel(top, image=self.logo_ctk, text="")
            logo_lbl.pack(side="left", padx=(8, 2), pady=6)
        except Exception:
            pass  # logo optional

        title = ctk.CTkLabel(top, text="JUDGE", font=ctk.CTkFont(size=26, weight="bold"), text_color="#e0e0ff")
        title.pack(side="left", padx=(18, 4), pady=10)

        subtitle = ctk.CTkLabel(top, text="Joint Unconventional Data & Geophysical Examination", font=ctk.CTkFont(size=12))
        subtitle.pack(side="left", padx=(2, 12), pady=14)

        # Sensitivity
        self.sens_var = ctk.DoubleVar(value=0.50)
        sens_frame = ctk.CTkFrame(top, fg_color="transparent")
        sens_frame.pack(side="left", padx=20)
        ctk.CTkLabel(sens_frame, text="Sensitivity", font=ctk.CTkFont(size=10)).pack()
        self.sens_slider = ctk.CTkSlider(sens_frame, from_=0.1, to=0.95, variable=self.sens_var, width=140)
        self.sens_slider.pack()
        self.sens_label = ctk.CTkLabel(sens_frame, text="0.50", font=ctk.CTkFont(size=10))
        self.sens_label.pack()
        self.sens_slider.configure(command=lambda v: self.sens_label.configure(text=f"{float(v):.2f}"))

        # Action buttons
        btn_frame = ctk.CTkFrame(top, fg_color="transparent")
        btn_frame.pack(side="right", padx=18)

        self.run_btn = ctk.CTkButton(btn_frame, text="▶ RUN ANALYSIS", width=148, height=36,
                                     fg_color="#3a7ca5", hover_color="#2f6a8f",
                                     command=self._on_run_analysis)
        self.run_btn.pack(side="left", padx=6)

        self.report_btn = ctk.CTkButton(btn_frame, text="Export Report", width=118, height=36,
                                        command=self._on_export_report, state="disabled")
        self.report_btn.pack(side="left", padx=4)

        self.export_clips_btn = ctk.CTkButton(btn_frame, text="Export Clips", width=108, height=36,
                                              command=self._on_export_clips, state="disabled")
        self.export_clips_btn.pack(side="left", padx=4)

        self.timeline_btn = ctk.CTkButton(btn_frame, text="Export Timeline", width=118, height=36,
                                          command=self._on_export_timeline, state="disabled")
        self.timeline_btn.pack(side="left", padx=4)

        self.stop_btn = ctk.CTkButton(btn_frame, text="⏹ Stop", width=70, height=36,
                                      fg_color="#c0392b", hover_color="#e74c3c",
                                      command=self._on_stop_analysis, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.slidedeck_btn = ctk.CTkButton(btn_frame, text="Slidedeck", width=85, height=36,
                                           command=self._open_slidedeck, state="disabled")
        self.slidedeck_btn.pack(side="left", padx=4)

        # Main area
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 6))

        # Left sidebar - files
        left = ctk.CTkFrame(main, width=260, corner_radius=8)
        left.pack(side="left", fill="y", padx=(0, 6), pady=4)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="DATA SOURCES", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 4), anchor="w")

        self.files_container = ctk.CTkScrollableFrame(left, width=238, height=120)
        self.files_container.pack(padx=10, pady=2, fill="x")

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.pack(padx=10, pady=6, fill="x")
        ctk.CTkButton(btns, text="+ Add Files", command=self._on_add_files, width=108).pack(side="left")
        ctk.CTkButton(btns, text="+ Add Folder", command=self._on_add_folder, width=108).pack(side="left", padx=4)
        ctk.CTkButton(left, text="Remove File", fg_color="#5c2a2a", hover_color="#3f1f1f",
                      command=self._on_remove_file, width=105).pack(side="left", padx=2)
        ctk.CTkButton(left, text="Clear All", fg_color="#5c2a2a", hover_color="#3f1f1f",
                      command=self._on_clear_files, width=105).pack(side="left", padx=2)

        # Config
        cfg = ctk.CTkFrame(left, fg_color="#22262f")
        cfg.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(cfg, text="CONFIGURATION", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=8, pady=4)
        self.min_dur_var = ctk.DoubleVar(value=0.06)
        ctk.CTkLabel(cfg, text="Min event duration (s)", font=ctk.CTkFont(size=9)).pack(anchor="w", padx=8)
        ctk.CTkSlider(cfg, from_=0.02, to=0.8, variable=self.min_dur_var, width=210).pack(padx=8, pady=2)

        self.cross_win_var = ctk.DoubleVar(value=1.8)
        ctk.CTkLabel(cfg, text="Cross-modal window (s)", font=ctk.CTkFont(size=9)).pack(anchor="w", padx=8, pady=(6,0))
        ctk.CTkSlider(cfg, from_=0.2, to=6.0, variable=self.cross_win_var, width=210).pack(padx=8, pady=2)

        # Post-run min score filter (useful for exploring results on large files without re-running)
        filter_frame = ctk.CTkFrame(left, fg_color="#22262f")
        filter_frame.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkLabel(filter_frame, text="Min score filter (post-run)", font=ctk.CTkFont(size=9)).pack(anchor="w", padx=8)
        self.min_score_var = ctk.DoubleVar(value=0.0)
        min_slider = ctk.CTkSlider(filter_frame, from_=0.0, to=25.0, variable=self.min_score_var, width=210,
                                   command=lambda v: self._apply_min_score_filter())
        min_slider.pack(padx=8, pady=2)
        self.min_score_label = ctk.CTkLabel(filter_frame, text="0.0", font=ctk.CTkFont(size=9))
        self.min_score_label.pack()

        # Center - visualizations + events
        center = ctk.CTkFrame(main, corner_radius=8)
        center.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Notebook header
        tabs = ctk.CTkTabview(center, height=420)
        tabs.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_events = tabs.add("Events")
        self.tab_timeline = tabs.add("Timeline")

        # Events tab
        self.event_scroll = ctk.CTkScrollableFrame(self.tab_events, label_text="Candidate Events (sorted by score)")
        self.event_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # Timeline tab
        self.fig = Figure(figsize=(8, 4.2), facecolor="#1a1c23")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1a1c23")
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self.ax.tick_params(colors="#aaa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_timeline)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Right detail panel
        right = ctk.CTkFrame(main, width=290, corner_radius=8)
        right.pack(side="right", fill="y", padx=(4, 0), pady=4)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="EVENT DETAILS", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 4), anchor="w")

        self.detail_box = ctk.CTkTextbox(right, height=260, font=ctk.CTkFont(size=11), wrap="word")
        self.detail_box.pack(fill="x", padx=10, pady=4)

        self.detail_box.insert("1.0", "Select an event from the list to view technical attribution and features.")
        self.detail_box.configure(state="disabled")

        # Small Image Preview Window for video events and detections
        self.preview_frame = ctk.CTkFrame(right, fg_color="#1f222a", corner_radius=6)
        self.preview_frame.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(self.preview_frame, text="Image Preview", font=ctk.CTkFont(size=10, weight="bold")).pack(pady=(4, 2))
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Select video event to preview", width=170, height=95, fg_color="#2a2d36")
        self.preview_label.pack(pady=4)
        self._preview_img = None

        # Bottom status bar
        bottom = ctk.CTkFrame(self, height=84, corner_radius=0)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.progress = ctk.CTkProgressBar(bottom, width=580, height=16)
        self.progress.pack(side="left", padx=16, pady=(8, 2))
        self.progress.set(0)

        # Bigger % + main message
        self.pct_label = ctk.CTkLabel(bottom, text="0%", font=ctk.CTkFont(size=15, weight="bold"), width=50)
        self.pct_label.pack(side="left", padx=(0, 4))

        self.status_label = ctk.CTkLabel(bottom, text="Idle", font=ctk.CTkFont(size=13))
        self.status_label.pack(side="left", padx=4)

        # Short live activity area (prevents "frozen" feeling on large files)
        self.activity_label = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=10), text_color="#aaa")
        self.activity_label.pack(side="left", padx=8)

        # Time remaining estimate
        self.eta_label = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=10), text_color="#888")
        self.eta_label.pack(side="left", padx=4)

        self.log_box = ctk.CTkTextbox(bottom, height=58, width=420, font=ctk.CTkFont(family="Consolas", size=9))
        self.log_box.pack(side="right", padx=10, pady=6, fill="x", expand=True)
        self.log_box.configure(state="disabled")

    def _log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{self._now()}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        logger.info(msg)

    def _now(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def _update_file_list(self):
        for widget in self.files_container.winfo_children():
            widget.destroy()
        paths = self._file_paths or ([str(p) for p in getattr(self.session, 'files', [])] if self.session else [])
        for fp in paths:
            item_frame = ctk.CTkFrame(self.files_container, fg_color="transparent")
            item_frame.pack(fill="x", pady=1)
            name = Path(fp).name
            lbl = ctk.CTkLabel(item_frame, text=f"• {name}", font=ctk.CTkFont(size=9), anchor="w")
            lbl.pack(side="left", padx=4, fill="x", expand=True)
            rem_btn = ctk.CTkButton(item_frame, text="✕", width=18, height=18, fg_color="#c0392b", hover_color="#e74c3c",
                                    command=lambda p=fp: self._remove_specific_file(p))
            rem_btn.pack(side="right", padx=2)

    def _on_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select video, audio or sensor files",
            filetypes=[
                ("All supported", "*.mp4 *.mov *.avi *.mkv *.wav *.flac *.mp3 *.ogg *.csv *.json"),
                ("Video", "*.mp4 *.mov *.avi *.mkv"),
                ("Audio", "*.wav *.flac *.mp3 *.ogg"),
                ("Sensor", "*.csv *.json"),
                ("All files", "*.*"),
            ]
        )
        if not paths:
            return
        if self.session is None:
            self.session = AnalysisSession(sensitivity=self.sens_var.get())
        added = 0
        for p in paths:
            if self.session.add_file(p):
                added += 1
        self._file_paths = [str(p) for p in self.session.files]
        self._update_file_list()
        self._log(f"Added {added} files")

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing data files")
        if not folder:
            return
        if self.session is None:
            self.session = AnalysisSession(sensitivity=self.sens_var.get())
        count = self.session.add_directory(folder)
        self._file_paths = [str(p) for p in self.session.files]
        self._update_file_list()
        self._log(f"Added {count} files from folder")

    def _on_remove_file(self):
        if not self._file_paths:
            return
        path_str = self._file_paths[-1]
        self._remove_specific_file(path_str)

    def _remove_specific_file(self, path_str):
        if path_str in self._file_paths:
            self._file_paths.remove(path_str)
        if self.session:
            try:
                self.session.files = [p for p in self.session.files if str(p) != path_str]
            except:
                pass
        self._update_file_list()
        self._log(f"Removed {Path(path_str).name}")

    def _on_clear_files(self):
        if self.session:
            self.session.clear()
        self._file_paths = []
        self._update_file_list()
        self.result = None
        self.events = []
        self._clear_events_ui()
        self._clear_plot()
        self.report_btn.configure(state="disabled")
        self.export_clips_btn.configure(state="disabled")
        self.timeline_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.slidedeck_btn.configure(state="disabled")
        self._update_activity("")
        self.eta_label.configure(text="")
        self.status_label.configure(text="Idle")
        self.pct_label.configure(text="0%")
        self._analysis_start_time = None
        self.cancel_event = None
        if hasattr(self, 'min_score_var'):
            self.min_score_var.set(0.0)
            self.min_score_label.configure(text="0.0")
        self._log("Cleared session")

    def _clear_events_ui(self):
        for child in self.event_scroll.winfo_children():
            child.destroy()

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#1a1c23")
        self.ax.set_title("No analysis results yet", color="#777")
        self.canvas.draw()

    def _on_run_analysis(self):
        if not self._file_paths:
            messagebox.showwarning("No data", "Please add files or a folder first.")
            return

        # Rebuild fresh session with current UI params (so sliders take effect)
        self.session = AnalysisSession(
            sensitivity=self.sens_var.get(),
            min_event_duration=self.min_dur_var.get(),
            cross_modal_window=self.cross_win_var.get(),
            progress_callback=self._progress_cb,
        )
        for p in self._file_paths:
            self.session.add_file(p)

        self.cancel_event = threading.Event()
        self.run_btn.configure(state="disabled", text="ANALYZING...")
        self.stop_btn.configure(state="normal")
        self.slidedeck_btn.configure(state="disabled")
        self.progress.set(0.05)
        self.pct_label.configure(text="5%")
        self._log("Starting analysis...")
        self._update_activity("loading metadata & preparing detectors")
        self._analysis_start_time = time.time()
        self.eta_label.configure(text="Estimating...")

        def worker():
            try:
                res = self.session.run(cancel_event=self.cancel_event)
                if getattr(self, 'cancel_event', None) and self.cancel_event.is_set():
                    self.after(0, lambda: self._analysis_cancelled(res))
                else:
                    self.after(0, lambda: self._analysis_finished(res))
            except Exception as ex:
                logger.exception("Analysis failed")
                self.after(0, lambda: self._analysis_error(str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop_analysis(self):
        if self.cancel_event:
            self.cancel_event.set()
            self._update_activity("stop requested - finishing current step")
            self._log("Stop requested. Will finish current file and cancel.")
            self.stop_btn.configure(state="disabled")

    def _progress_cb(self, msg: str, pct: float):
        self.after(0, lambda: self._update_progress(msg, pct))
        # Populate short live activity area from detailed sub-messages
        if ": " in msg and not msg.startswith("Loaded") and not msg.startswith("Found"):
            activity = msg.split(": ", 1)[-1]
            self.after(0, lambda a=activity: self._update_activity(a))

    def _update_progress(self, msg: str, pct: float):
        self.progress.set(pct / 100.0)
        self.pct_label.configure(text=f"{int(pct)}%")
        self.status_label.configure(text=msg)
        self._update_eta(pct)

    def _update_activity(self, text: str):
        # Short live status so user sees it's working (e.g. "optical flow (frame 1234)")
        self.after(0, lambda: self.activity_label.configure(text=text))

    def _update_eta(self, pct: float):
        if not self._analysis_start_time or pct < 1.5:
            self.eta_label.configure(text="")
            return
        try:
            elapsed = time.time() - self._analysis_start_time
            if elapsed > 1 and pct > 0:
                remaining = elapsed * (100 - pct) / pct
                if remaining > 0:
                    mins, secs = divmod(int(remaining), 60)
                    self.eta_label.configure(text=f"~{mins}m {secs}s left")
                    return
            self.eta_label.configure(text="")
        except Exception:
            self.eta_label.configure(text="")

    def _analysis_finished(self, result: AnalysisResult):
        self.result = result
        self.all_events = list(result.events)
        self.events = self.all_events
        self._file_paths = list(result.files_processed)

        self.run_btn.configure(state="normal", text="▶ RUN ANALYSIS")
        self.stop_btn.configure(state="disabled")
        self.progress.set(1.0)
        self.pct_label.configure(text="100%")
        self.status_label.configure(text="Analysis complete")
        self._update_activity("")
        self.eta_label.configure(text="")
        self._analysis_start_time = None
        self.cancel_event = None
        if hasattr(self, 'min_score_var'):
            self.min_score_var.set(0.0)
            self.min_score_label.configure(text="0.0")
        self._log(f"Analysis complete. {len(self.events)} candidate events.")

        self._render_event_list()
        self._render_overview_plot()
        self.report_btn.configure(state="normal")
        self.export_clips_btn.configure(state="normal" if self.events else "disabled")
        self.timeline_btn.configure(state="normal" if self.events else "disabled")
        self.slidedeck_btn.configure(state="normal" if self.events else "disabled")

    def _analysis_cancelled(self, result: AnalysisResult):
        self.result = result
        self.all_events = list(getattr(result, 'events', []) or [])
        self.events = self.all_events
        self.run_btn.configure(state="normal", text="▶ RUN ANALYSIS")
        self.stop_btn.configure(state="disabled")
        self.progress.set(0)
        self.pct_label.configure(text="")
        self.status_label.configure(text="Cancelled (partial results shown)")
        self._update_activity("")
        self.eta_label.configure(text="")
        self._analysis_start_time = None
        self.cancel_event = None
        self._log(f"Analysis cancelled by user. {len(self.events)} events captured so far.")

        if self.events:
            self._render_event_list()
            self._render_overview_plot()
            self.report_btn.configure(state="normal")
            self.export_clips_btn.configure(state="normal" if self.events else "disabled")
            self.timeline_btn.configure(state="normal" if self.events else "disabled")
            self.slidedeck_btn.configure(state="normal" if self.events else "disabled")

    def _analysis_error(self, err: str):
        self.run_btn.configure(state="normal", text="▶ RUN ANALYSIS")
        self.stop_btn.configure(state="disabled")
        self.progress.set(0)
        self.pct_label.configure(text="0%")
        self.status_label.configure(text="Error")
        self._update_activity("")
        self.eta_label.configure(text="")
        self._analysis_start_time = None
        self.cancel_event = None
        messagebox.showerror("Analysis Error", err)
        self._log(f"ERROR: {err}")

    def _render_event_list(self):
        self._clear_events_ui()
        if not self.events:
            lbl = ctk.CTkLabel(self.event_scroll, text="No events detected at current sensitivity.")
            lbl.pack(padx=8, pady=12)
            return

        for ev in sorted(self.events, key=lambda e: -e.score):
            card = ctk.CTkFrame(self.event_scroll, fg_color="#252932", corner_radius=6)
            card.pack(fill="x", padx=4, pady=3)

            header = f"{ev.modality.value.upper()}  •  {ev.pretty_time()}  +{ev.duration*1000:.0f}ms   score={ev.score:.2f}"
            ctk.CTkLabel(card, text=header, font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#dfe3ff").pack(anchor="w", padx=8, pady=(5, 0))

            short = ev.description[:140] + ("..." if len(ev.description) > 140 else "")
            ctk.CTkLabel(card, text=short, font=ctk.CTkFont(size=10), wraplength=640,
                         justify="left").pack(anchor="w", padx=8, pady=(0, 4))

            card.bind("<Button-1>", lambda e, event=ev: self._select_event(event))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, event=ev: self._select_event(event))

    def _select_event(self, ev: AnomalyEvent):
        self.selected_event = ev
        self.detail_box.configure(state="normal")
        self.detail_box.delete("1.0", "end")

        info = (
            f"ID: {ev.event_id}\n"
            f"Modality: {ev.modality.value}\n"
            f"Time: {ev.pretty_time()}  (+{ev.duration*1000:.1f} ms)\n"
            f"Score: {ev.score:.3f}   Peak: {ev.peak_score:.3f}\n"
            f"File: {Path(ev.file_path).name}\n\n"
            f"{ev.description}\n\n"
        )
        if ev.features:
            info += "Features:\n"
            for k, v in ev.features.items():
                if isinstance(v, float):
                    info += f"  {k}: {v:.4g}\n"
                else:
                    info += f"  {k}: {v}\n"
        if ev.tags:
            info += f"\nTags: {', '.join(ev.tags)}"

        self.detail_box.insert("1.0", info)
        self.detail_box.configure(state="disabled")

        # Highlight in plot if possible
        self._highlight_event_on_plot(ev)

        # Update small preview for video events / detections
        self._load_event_preview(ev)

    def _load_event_preview(self, ev: AnomalyEvent):
        self.preview_label.configure(image=None, text="Loading preview...")
        self._preview_img = None
        if ev.modality.value != "video":
            self.preview_label.configure(text=f"No image preview for {ev.modality.value}")
            return
        try:
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(str(ev.file_path))
            if not cap.isOpened():
                self.preview_label.configure(text="Cannot open video")
                cap.release()
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_pos = int(ev.start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pos))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                self.preview_label.configure(text="Frame not found")
                return
            # resize to small preview size
            h, w = frame.shape[:2]
            target_w = 160
            scale = target_w / float(w)
            new_h = int(h * scale)
            if new_h > 95:
                scale = 95 / float(new_h)
                target_w = int(w * scale)
                new_h = 95
            frame = cv2.resize(frame, (target_w, new_h))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, new_h))
            self.preview_label.configure(image=ctk_img, text="")
            self._preview_img = ctk_img
        except Exception as e:
            self.preview_label.configure(text=f"Preview err")

    def _open_slidedeck(self):
        """Slidedeck view of events with image previews for video detections."""
        if not self.events:
            return
        win = ctk.CTkToplevel(self)
        win.title("Event Slidedeck")
        win.geometry("620x520")
        win.resizable(False, False)

        self._slide_index = 0
        self._slide_events = sorted(self.events, key=lambda e: e.start_time)

        self.slide_info = ctk.CTkLabel(win, text="", font=ctk.CTkFont(size=11), wraplength=580, justify="left")
        self.slide_info.pack(pady=8, padx=10)

        self.slide_preview = ctk.CTkLabel(win, text="", width=320, height=180, fg_color="#2a2d36")
        self.slide_preview.pack(pady=6)
        self._slide_img = None

        nav_frame = ctk.CTkFrame(win)
        nav_frame.pack(pady=10)

        ctk.CTkButton(nav_frame, text="◀ Prev", width=80, command=self._prev_slide).pack(side="left", padx=5)
        self.slide_pos_label = ctk.CTkLabel(nav_frame, text="1 / 1", font=ctk.CTkFont(size=11))
        self.slide_pos_label.pack(side="left", padx=10)
        ctk.CTkButton(nav_frame, text="Next ▶", width=80, command=self._next_slide).pack(side="left", padx=5)

        ctk.CTkButton(nav_frame, text="Close", width=80, command=win.destroy).pack(side="left", padx=20)

        self._update_slide()

    def _update_slide(self):
        if not self._slide_events:
            return
        ev = self._slide_events[self._slide_index]
        n = len(self._slide_events)
        self.slide_pos_label.configure(text=f"{self._slide_index + 1} / {n}")
        info_text = f"[{ev.modality.value.upper()}]  {ev.pretty_time()}  +{ev.duration*1000:.0f}ms   score={ev.score:.2f}\n{ev.description}\nFile: {Path(ev.file_path).name}"
        self.slide_info.configure(text=info_text)

        self.slide_preview.configure(image=None, text="Loading frame...")
        self._slide_img = None

        if ev.modality.value == "video":
            try:
                import cv2
                from PIL import Image
                cap = cv2.VideoCapture(str(ev.file_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                pos = int(ev.start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.resize(frame, (320, 180))
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ctkimg = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 180))
                    self.slide_preview.configure(image=ctkimg, text="")
                    self._slide_img = ctkimg
                else:
                    self.slide_preview.configure(text="Frame unavailable")
            except Exception as e:
                self.slide_preview.configure(text=f"Error: {str(e)[:50]}")
        else:
            self.slide_preview.configure(text=f"No frame preview for {ev.modality.value}")

    def _prev_slide(self):
        if self._slide_index > 0:
            self._slide_index -= 1
            self._update_slide()

    def _next_slide(self):
        if self._slide_index < len(self._slide_events) - 1:
            self._slide_index += 1
            self._update_slide()

    def _render_overview_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#1a1c23")
        if not self.events:
            self.ax.text(0.5, 0.5, "No events", ha="center", va="center", color="#555")
            self.canvas.draw()
            return

        # Simple scatter of score vs time colored by modality
        times = [e.start_time for e in self.events]
        scores = [e.score for e in self.events]
        mods = [e.modality.value for e in self.events]

        color_map = {"video": "#ff6b6b", "audio": "#4ecdc4", "sensor": "#ffe66d"}
        c = [color_map.get(m, "#aaa") for m in mods]

        self.ax.scatter(times, scores, c=c, s=28, alpha=0.85, edgecolors="none")
        self.ax.set_xlabel("Time (s)", color="#ccc")
        self.ax.set_ylabel("Anomaly Score", color="#ccc")
        self.ax.set_title("Anomaly Events — Time vs Score", color="#ddd", fontsize=10)
        self.ax.grid(True, alpha=0.15)

        # Add legend
        for m, col in color_map.items():
            self.ax.scatter([], [], c=col, label=m, s=30)
        self.ax.legend(loc="upper right", fontsize=8, facecolor="#222")

        self.fig.tight_layout()
        self.canvas.draw()

    def _highlight_event_on_plot(self, ev: AnomalyEvent):
        self._render_overview_plot()
        # overlay marker
        self.ax.axvline(ev.start_time, color="#ffdd57", linewidth=1.5, alpha=0.9, linestyle="--")
        self.ax.scatter([ev.start_time], [ev.score], s=120, facecolors="none", edgecolors="#ffdd57", linewidths=2)
        self.canvas.draw()

    def _on_export_report(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("Markdown", "*.md"), ("JSON", "*.json")],
            title="Export technical report",
        )
        if not path:
            return
        try:
            out = generate_report(self.result, Path(path), format="all")
            self._log(f"Report exported: {out}")
            messagebox.showinfo("Report Generated", f"Report written to:\n{out}")
        except Exception as e:
            messagebox.showerror("Report Error", str(e))

    def _on_export_clips(self):
        if not self.result or not self.events:
            return
        folder = filedialog.askdirectory(title="Choose folder for exported evidence clips")
        if not folder:
            return
        outdir = Path(folder) / f"judge_clips_{self.result.session_id}"
        outdir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for ev in self.events[:30]:  # cap for practicality
            try:
                self._export_single_clip(ev, outdir)
                exported += 1
            except Exception as ex:
                logger.warning("Clip export failed for %s: %s", ev.event_id, ex)
        self._log(f"Exported {exported} evidence clips to {outdir}")
        messagebox.showinfo("Clips Exported", f"Exported {exported} clips to\n{outdir}")

    def _export_single_clip(self, ev: AnomalyEvent, outdir: Path):
        """Export a short window around the event (video or audio)."""
        import soundfile as sf
        p = Path(ev.file_path)
        start = max(0.0, ev.start_time - 0.6)
        dur = min(ev.duration + 1.8, 8.0)  # reasonable clip length

        if ev.modality.value == "video":
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
            for _ in range(target_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            cap.release()

        elif ev.modality.value == "audio":
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            si = int(start * sr)
            ei = min(len(y), int((start + dur) * sr))
            clip = y[si:ei]
            out_path = outdir / f"{ev.event_id}_{ev.modality.value}.wav"
            sf.write(str(out_path), clip, sr)

        else:
            # sensor: export slice of CSV
            import pandas as pd
            df = pd.read_csv(p)
            time_col = None
            for c in df.columns:
                if str(c).lower() in {"t", "time", "timestamp"}:
                    time_col = c
                    break
            if time_col is None:
                time_col = df.columns[0]
            mask = (pd.to_numeric(df[time_col], errors="coerce") >= start) & \
                   (pd.to_numeric(df[time_col], errors="coerce") <= start + dur)
            out_df = df[mask]
            out_path = outdir / f"{ev.event_id}_{ev.modality.value}.csv"
            out_df.to_csv(out_path, index=False)

    def _on_export_timeline(self):
        """New feature: export a clean annotated timeline PNG (very useful for logs, papers, quick overviews)."""
        if not self.result or not self.events:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Export annotated timeline image",
        )
        if not path:
            return
        try:
            self._export_timeline_image(Path(path))
            self._log(f"Timeline image exported: {path}")
            messagebox.showinfo("Timeline Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Timeline Error", str(e))

    def _export_timeline_image(self, out_path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        events = sorted(self.events, key=lambda e: e.start_time)
        if not events:
            return

        times = [e.start_time for e in events]
        scores = [e.score for e in events]
        mods = [e.modality.value for e in events]

        colors_map = {"video": "#e63946", "audio": "#457b9d", "sensor": "#2a9d8f"}
        c = [colors_map.get(m, "#555555") for m in mods]

        fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="white")
        ax.set_facecolor("#fafafa")

        # Vertical lines + markers for visibility
        for t, s, col in zip(times, scores, c):
            ax.axvline(t, color=col, alpha=0.25, linewidth=1.0, zorder=1)
        ax.scatter(times, scores, c=c, s=55, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Time (seconds from start)", fontsize=10)
        ax.set_ylabel("Anomaly Score", fontsize=10)
        ax.set_title(f"Judge Timeline — {len(events)} candidate events  |  Session {self.result.session_id}", fontsize=11, pad=8)

        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_ylim(0, max(max(scores) * 1.15, 5))

        # Legend
        for m, col in colors_map.items():
            ax.scatter([], [], c=col, s=40, label=m, alpha=0.9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        plt.tight_layout()
        fig.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _apply_min_score_filter(self):
        if not hasattr(self, "all_events") or not self.all_events:
            return
        min_s = self.min_score_var.get()
        self.min_score_label.configure(text=f"{min_s:.1f}")
        self.events = [e for e in self.all_events if e.score >= min_s]
        self._render_event_list()


def main():
    app = JudgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
