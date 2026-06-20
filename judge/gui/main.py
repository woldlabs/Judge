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

        self._build_ui()
        self._log("JUDGE ready. Add files and run analysis.")

    def _build_ui(self):
        # Top bar
        top = ctk.CTkFrame(self, height=58, corner_radius=0)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

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

        # Main area
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 6))

        # Left sidebar - files
        left = ctk.CTkFrame(main, width=260, corner_radius=8)
        left.pack(side="left", fill="y", padx=(0, 6), pady=4)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="DATA SOURCES", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 4), anchor="w")

        self.file_listbox = ctk.CTkTextbox(left, height=220, width=238, font=ctk.CTkFont(size=10))
        self.file_listbox.pack(padx=10, pady=4, fill="x")
        self.file_listbox.configure(state="disabled")

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.pack(padx=10, pady=6, fill="x")
        ctk.CTkButton(btns, text="+ Add Files", command=self._on_add_files, width=108).pack(side="left")
        ctk.CTkButton(btns, text="+ Add Folder", command=self._on_add_folder, width=108).pack(side="left", padx=4)
        ctk.CTkButton(left, text="Clear All", fg_color="#5c2a2a", hover_color="#3f1f1f",
                      command=self._on_clear_files, width=220).pack(padx=10, pady=2)

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

        self.detail_box = ctk.CTkTextbox(right, height=260, font=ctk.CTkFont(size=10))
        self.detail_box.pack(fill="x", padx=10, pady=4)

        self.detail_box.insert("1.0", "Select an event from the list to view technical attribution and features.")
        self.detail_box.configure(state="disabled")

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

        self.status_label = ctk.CTkLabel(bottom, text="Idle", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=4)

        # Short live activity area (prevents "frozen" feeling on large files)
        self.activity_label = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=9), text_color="#aaa")
        self.activity_label.pack(side="left", padx=8)

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
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", "end")
        paths = self._file_paths or ([str(p) for p in self.session.files] if self.session else [])
        for fp in paths:
            name = Path(fp).name
            self.file_listbox.insert("end", f"• {name}\n")
        self.file_listbox.configure(state="disabled")

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
        self._update_activity("")
        self.status_label.configure(text="Idle")
        self.pct_label.configure(text="0%")
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

        self.run_btn.configure(state="disabled", text="ANALYZING...")
        self.progress.set(0.05)
        self.pct_label.configure(text="5%")
        self._log("Starting analysis...")
        self._update_activity("loading metadata & preparing detectors")

        def worker():
            try:
                res = self.session.run()
                self.after(0, lambda: self._analysis_finished(res))
            except Exception as ex:
                logger.exception("Analysis failed")
                self.after(0, lambda: self._analysis_error(str(ex)))

        threading.Thread(target=worker, daemon=True).start()

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

    def _update_activity(self, text: str):
        # Short live status so user sees it's working (e.g. "optical flow (frame 1234)")
        self.after(0, lambda: self.activity_label.configure(text=text))

    def _analysis_finished(self, result: AnalysisResult):
        self.result = result
        self.events = result.events
        self._file_paths = list(result.files_processed)

        self.run_btn.configure(state="normal", text="▶ RUN ANALYSIS")
        self.progress.set(1.0)
        self.pct_label.configure(text="100%")
        self.status_label.configure(text="Analysis complete")
        self._update_activity("")
        self._log(f"Analysis complete. {len(self.events)} candidate events.")

        self._render_event_list()
        self._render_overview_plot()
        self.report_btn.configure(state="normal")
        self.export_clips_btn.configure(state="normal" if self.events else "disabled")
        self.timeline_btn.configure(state="normal" if self.events else "disabled")

    def _analysis_error(self, err: str):
        self.run_btn.configure(state="normal", text="▶ RUN ANALYSIS")
        self.progress.set(0)
        self.pct_label.configure(text="0%")
        self.status_label.configure(text="Error")
        self._update_activity("")
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
            ctk.CTkLabel(card, text=header, font=ctk.CTkFont(size=10, weight="bold"),
                         text_color="#dfe3ff").pack(anchor="w", padx=8, pady=(5, 0))

            short = ev.description[:140] + ("..." if len(ev.description) > 140 else "")
            ctk.CTkLabel(card, text=short, font=ctk.CTkFont(size=9), wraplength=640,
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
            info += f"\nTags: {", ".join(ev.tags)}"

        self.detail_box.insert("1.0", info)
        self.detail_box.configure(state="disabled")

        # Highlight in plot if possible
        self._highlight_event_on_plot(ev)

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


def main():
    app = JudgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
