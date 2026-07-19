"""
app.py
------
Desktop GUI (Tkinter) for the Video Forensics Object Tracking Tool.

Workflow:
  1. Pick a video -> preview loads with Play/Pause + a seek slider.
  2. Play (or scrub) the video to the point where the object of interest
     appears, click Pause (or just drag on the frame - it auto-pauses).
  3. Drag a box around ONE specific object instance on that paused frame
     -> only that object gets tracked & reported, even if other objects
     of the same class are in the video. Skip this step to track every
     instance of the selected class instead.
  4. Pick the object class from the dropdown.
  5. Start -> frames + PDF + CSV report get generated in the output folder.
"""

import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.report_generator import ForensicReportGenerator
from core.tracker_engine import TrackerEngine
from core.video_enhancer import VideoEnhancer, describe_options, options_are_default
from core.face_matcher import FaceMatcher

PREVIEW_W, PREVIEW_H = 480, 270  # fixed preview canvas size (16:9)


def _resource_path(relative_path):
    """
    Resolve a bundled resource (e.g. the app icon) whether the app is
    running from source (python main.py) or as a PyInstaller-built exe.
    PyInstaller unpacks --add-data files into a temp folder pointed to
    by sys._MEIPASS at runtime.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, relative_path)


def _set_app_icon(root):
    """
    Set the window/taskbar icon to our custom Smart CCTV Investigator icon.
    Safe no-op if the icon file is missing so the app never crashes over this.
    """
    ico_path = _resource_path(os.path.join("assets", "app_icon.ico"))
    try:
        if sys.platform.startswith("win") and os.path.exists(ico_path):
            root.iconbitmap(default=ico_path)
        else:
            png_path = _resource_path(os.path.join("assets", "app_icon.png"))
            if os.path.exists(png_path):
                icon_img = tk.PhotoImage(file=png_path)
                root.iconphoto(True, icon_img)
                root._icon_img_ref = icon_img  # keep a reference alive
    except Exception:
        pass  # icon is cosmetic only - never block the app from starting


class VideoForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart CCTV Investigator")
        _set_app_icon(self.root)
        self.root.geometry("980x720")
        self.root.minsize(700, 480)
        self.root.resizable(True, True)

        self.video_path = None
        self.engine = None
        self.output_dir = None

        # Preview / playback state
        self.preview_cap = None            # cv2.VideoCapture used for scrubbing/playback
        self.preview_total_frames = 1
        self.preview_fps = 25.0
        self.current_frame_num = 1         # 1-based, matches engine's frame numbering
        self.is_playing = False
        self._suppress_seek_callback = False
        self.preview_frame_bgr = None      # currently displayed frame (numpy, BGR)
        self.tk_preview_image = None       # keep reference alive
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Selection state
        self.drag_start = None
        self.rect_id = None
        self.selection_bbox = None         # [x1,y1,x2,y2] in ORIGINAL frame coords
        self.selection_frame_idx = None    # which frame number the selection was made on

        self._build_ui()
        self._load_model_async()

    # ---------- UI ----------
    def _build_scrollable_area(self):
        """
        Wraps all the app's content in a scrollable Canvas so every section
        (including Face Search and the Start/Open Output buttons at the
        bottom) stays reachable even on small screens or small windows,
        instead of being cut off below the visible area.
        Returns a plain tk.Frame - every existing widget is built inside it
        exactly as before; only how that frame is scrolled/displayed changes.
        """
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = tk.Frame(canvas, padx=20, pady=10)
        window_id = canvas.create_window((0, 0), window=main, anchor="nw")

        def _on_main_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        main.bind("<Configure>", _on_main_configure)

        def _on_canvas_configure(event):
            # Keep the inner frame exactly as wide as the visible canvas so
            # existing fill="x" widgets behave exactly as before.
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
            canvas.yview_scroll(int(delta), "units")

        def _bind_mousewheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows/macOS
            canvas.bind_all("<Button-4>", _on_mousewheel)        # Linux scroll up
            canvas.bind_all("<Button-5>", _on_mousewheel)        # Linux scroll down

        def _unbind_mousewheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        return main

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1a1a2e", height=70)
        header.pack(fill="x")
        tk.Label(header, text="SMART CCTV INVESTIGATOR", bg="#1a1a2e", fg="white",
                  font=("Segoe UI", 16, "bold")).pack(pady=(12, 0))
        tk.Label(header, text="Object Detection & Tracking for Investigation Use",
                  bg="#1a1a2e", fg="#aaaaaa", font=("Segoe UI", 9)).pack()

        main = self._build_scrollable_area()

        case_frame = tk.LabelFrame(main, text="Case Details", padx=10, pady=8)
        case_frame.pack(fill="x", pady=3)
        tk.Label(case_frame, text="Case Reference:").grid(row=0, column=0, sticky="w")
        self.case_ref_entry = tk.Entry(case_frame, width=28)
        self.case_ref_entry.grid(row=0, column=1, padx=5)
        tk.Label(case_frame, text="Officer Name:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.officer_entry = tk.Entry(case_frame, width=22)
        self.officer_entry.grid(row=0, column=3, padx=5)

        video_frame = tk.LabelFrame(main, text="Video Evidence", padx=10, pady=8)
        video_frame.pack(fill="x", pady=3)
        self.video_label = tk.Label(video_frame, text="No video selected", fg="grey", anchor="w")
        self.video_label.pack(side="left", fill="x", expand=True)
        tk.Button(video_frame, text="Browse Video...", command=self.select_video).pack(side="right")

        # --- Video Enhancement (optional pre-processing, runs before detection) ---
        enhance_frame = tk.LabelFrame(main, text="Video Enhancement - optional pre-processing (runs before detection)",
                                       padx=10, pady=8)
        enhance_frame.pack(fill="x", pady=3)

        row1 = tk.Frame(enhance_frame)
        row1.pack(fill="x")
        self.denoise_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row1, text="Denoise", variable=self.denoise_var).pack(side="left")
        self.sharpen_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row1, text="Sharpen / Deblur", variable=self.sharpen_var).pack(side="left", padx=(10, 0))
        self.auto_contrast_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row1, text="Auto Brightness/Contrast (CLAHE)",
                        variable=self.auto_contrast_var).pack(side="left", padx=(10, 0))
        tk.Label(row1, text="Rotate:").pack(side="left", padx=(15, 0))
        self.rotate_var = tk.StringVar(value="0\u00b0")
        ttk.Combobox(row1, textvariable=self.rotate_var, values=["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"],
                     state="readonly", width=5).pack(side="left", padx=5)

        row2 = tk.Frame(enhance_frame)
        row2.pack(fill="x", pady=(6, 0))
        tk.Label(row2, text="Brightness:").pack(side="left")
        self.brightness_var = tk.IntVar(value=0)
        tk.Scale(row2, from_=-100, to=100, orient="horizontal", variable=self.brightness_var,
                 length=140).pack(side="left", padx=(5, 20))
        tk.Label(row2, text="Contrast:").pack(side="left")
        self.contrast_var = tk.DoubleVar(value=1.0)
        tk.Scale(row2, from_=0.5, to=2.0, resolution=0.05, orient="horizontal",
                  variable=self.contrast_var, length=140).pack(side="left", padx=5)
        tk.Label(enhance_frame, text="Leave everything at default to skip this step entirely - the original "
                                      "video is analysed unchanged. When enabled, a separate enhanced copy is "
                                      "created and analysed; the original file is never modified.",
                 fg="#555", font=("Segoe UI", 8, "italic"), wraplength=880, justify="left").pack(anchor="w", pady=(6, 0))

        # --- Preview + playback + selection ---
        preview_frame = tk.LabelFrame(main, text="Play video & select the object to track (optional)",
                                       padx=10, pady=8)
        preview_frame.pack(fill="x", pady=3)

        left_col = tk.Frame(preview_frame)
        left_col.pack(side="left")

        self.canvas = tk.Canvas(left_col, width=PREVIEW_W, height=PREVIEW_H,
                                 bg="#111111", highlightthickness=1, highlightbackground="#888")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        playback_row = tk.Frame(left_col)
        playback_row.pack(fill="x", pady=(6, 0))
        self.play_btn = tk.Button(playback_row, text="\u25b6 Play", width=8, command=self.toggle_play)
        self.play_btn.pack(side="left")
        self.frame_counter_label = tk.Label(playback_row, text="Frame - / -", fg="#333")
        self.frame_counter_label.pack(side="left", padx=10)

        self.seek_var = tk.DoubleVar(value=1)
        self.seek_scale = tk.Scale(left_col, from_=1, to=100, orient="horizontal",
                                    variable=self.seek_var, showvalue=False,
                                    length=PREVIEW_W, command=self._on_seek)
        self.seek_scale.pack(fill="x", pady=(2, 0))

        right_col = tk.Frame(preview_frame, padx=15)
        right_col.pack(side="left", fill="y", anchor="n")
        tk.Label(right_col, text="Play / scrub to the moment your object\n"
                                  "appears, then drag a box around it to\n"
                                  "track only that ONE object.\n\n"
                                  "Leave it unselected to track every\n"
                                  "object of the chosen class instead.",
                 justify="left", fg="#333").pack(anchor="w", pady=(0, 8))
        self.selection_status = tk.Label(right_col, text="No object selected",
                                          fg="grey", font=("Segoe UI", 9, "bold"),
                                          wraplength=220, justify="left")
        self.selection_status.pack(anchor="w")
        tk.Button(right_col, text="Clear Selection", command=self.clear_selection).pack(anchor="w", pady=6)

        class_frame = tk.LabelFrame(main, text="Target Object Class", padx=10, pady=8)
        class_frame.pack(fill="x", pady=3)
        tk.Label(class_frame, text="Select object class:").pack(side="left")
        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(class_frame, textvariable=self.class_var,
                                            state="disabled", width=28)
        self.class_dropdown.pack(side="left", padx=10)
        tk.Label(class_frame, text="Confidence:").pack(side="left", padx=(20, 0))
        self.conf_var = tk.DoubleVar(value=0.4)
        tk.Scale(class_frame, from_=0.1, to=0.9, resolution=0.05, orient="horizontal",
                 variable=self.conf_var, length=140).pack(side="left")
        tk.Label(main, text="Tip: choose 'person' to also auto-extract a zoomed face image per "
                             "tracked individual; choose car / motorcycle / bus / truck to also "
                             "auto-read number plates. Both are added to the PDF report.",
                 fg="#555", font=("Segoe UI", 8, "italic")).pack(anchor="w", pady=(0, 4))

        # --- Face Search & Duplicate Face Removal (optional, only relevant when class = person) ---
        face_tools_frame = tk.LabelFrame(main, text="Face Search & Duplicate Face Removal - optional "
                                                     "(only used when tracking 'person')", padx=10, pady=8)
        face_tools_frame.pack(fill="x", pady=3)
        ref_row = tk.Frame(face_tools_frame)
        ref_row.pack(fill="x")
        tk.Button(ref_row, text="Browse Reference Photo...",
                  command=self.browse_reference_face).pack(side="left")
        self.reference_face_path = None
        self.reference_face_label = tk.Label(ref_row, text="No reference photo selected", fg="grey")
        self.reference_face_label.pack(side="left", padx=10)
        tk.Button(ref_row, text="Clear", command=self.clear_reference_face).pack(side="left")
        self.dedupe_faces_var = tk.BooleanVar(value=False)
        tk.Checkbutton(face_tools_frame, text="Enable Duplicate Face Removal (group Track IDs that are "
                                               "likely the same person)",
                        variable=self.dedupe_faces_var).pack(anchor="w", pady=(6, 0))
        tk.Label(face_tools_frame, text="Face Search compares tracked people's faces to your reference "
                                          "photo; Duplicate Removal groups Track IDs likely to be the "
                                          "same person. Both are probabilistic LEADS to verify manually, "
                                          "not confirmed identification. Leave both off to skip - report "
                                          "renders exactly as before.",
                 fg="#555", font=("Segoe UI", 8, "italic"), wraplength=880, justify="left").pack(anchor="w", pady=(4, 0))

        control_frame = tk.Frame(main)
        control_frame.pack(fill="x", pady=6)
        self.start_btn = tk.Button(control_frame, text="Start Tracking & Generate Report",
                                    bg="#0f3460", fg="white", font=("Segoe UI", 10, "bold"),
                                    command=self.start_processing, state="disabled")
        self.start_btn.pack(side="left")
        self.open_output_btn = tk.Button(control_frame, text="Open Output Folder",
                                          command=self.open_output_folder, state="disabled")
        self.open_output_btn.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(main, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=3)

        log_frame = tk.LabelFrame(main, text="Processing Log", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, pady=3)
        self.log_text = tk.Text(log_frame, height=8, state="disabled",
                                 bg="#0f0f1a", fg="#00ff88", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

    # ---------- logging ----------
    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def _update_progress(self, current, total):
        pct = int((current / total) * 100) if total else 0
        self.root.after(0, lambda: self.progress.configure(value=pct))

    # ---------- model loading ----------
    def _load_model_async(self):
        self._log("Loading detection model (yolov8n.pt)... first run may download weights.")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.engine = TrackerEngine(log_callback=self._log,
                                         progress_callback=self._update_progress)
            classes = sorted(self.engine.get_class_list())
            self.root.after(0, lambda: self._on_model_loaded(classes))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"ERROR loading model: {e}"))

    def _on_model_loaded(self, classes):
        self.class_dropdown["values"] = classes
        self.class_dropdown["state"] = "readonly"
        if classes:
            self.class_dropdown.current(0)
        self._log("Model loaded. Ready.")
        self._update_start_btn_state()

    # ---------- video selection & preview ----------
    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select video evidence",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if not path:
            return
        self.video_path = path
        self.video_label.config(text=os.path.basename(path), fg="black")
        self.clear_selection()
        self._open_preview_video()
        self._update_start_btn_state()

    def _open_preview_video(self):
        if self.preview_cap is not None:
            self.preview_cap.release()
        self.preview_cap = cv2.VideoCapture(self.video_path)
        if not self.preview_cap.isOpened():
            self._log("ERROR: could not open this video for preview.")
            self.preview_cap = None
            return

        self.preview_total_frames = max(1, int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.preview_fps = self.preview_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.is_playing = False
        self.play_btn.config(text="\u25b6 Play")

        self._suppress_seek_callback = True
        self.seek_scale.config(from_=1, to=self.preview_total_frames)
        self.seek_var.set(1)
        self._suppress_seek_callback = False

        self._show_frame_at(1)

    def _show_frame_at(self, frame_num):
        """Seeks to an exact frame number and displays it (used by the
        seek slider and when first opening a video)."""
        if self.preview_cap is None:
            return
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
        ret, frame = self.preview_cap.read()
        if not ret:
            return
        self.current_frame_num = frame_num
        self._display_frame(frame)

    def _display_frame(self, frame_bgr):
        """Renders a frame on the canvas + updates counter/slider/selection box."""
        self.preview_frame_bgr = frame_bgr
        self._render_preview(frame_bgr)
        self.frame_counter_label.config(
            text=f"Frame {self.current_frame_num} / {self.preview_total_frames}")

        self._suppress_seek_callback = True
        self.seek_var.set(self.current_frame_num)
        self._suppress_seek_callback = False

        if self.selection_bbox is not None and self.selection_frame_idx == self.current_frame_num:
            self._draw_selection_rect_from_bbox()
        elif self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _render_preview(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]
        self.scale = min(PREVIEW_W / orig_w, PREVIEW_H / orig_h)
        disp_w, disp_h = int(orig_w * self.scale), int(orig_h * self.scale)
        self.offset_x = (PREVIEW_W - disp_w) // 2
        self.offset_y = (PREVIEW_H - disp_h) // 2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((disp_w, disp_h))
        self.tk_preview_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("frame_img")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw",
                                  image=self.tk_preview_image, tags="frame_img")
        self.canvas.tag_lower("frame_img")

    def _draw_selection_rect_from_bbox(self):
        x1, y1, x2, y2 = self.selection_bbox
        cx1 = self.offset_x + x1 * self.scale
        cy1 = self.offset_y + y1 * self.scale
        cx2 = self.offset_x + x2 * self.scale
        cy2 = self.offset_y + y2 * self.scale
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                                      outline="#00ff88", width=2)

    def _canvas_to_frame_coords(self, cx, cy):
        orig_h, orig_w = self.preview_frame_bgr.shape[:2]
        fx = (cx - self.offset_x) / self.scale
        fy = (cy - self.offset_y) / self.scale
        fx = max(0, min(orig_w, fx))
        fy = max(0, min(orig_h, fy))
        return int(fx), int(fy)

    # ---------- playback controls ----------
    def toggle_play(self):
        if self.preview_cap is None:
            return
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="\u25b6 Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="\u23f8 Pause")
            self._play_step()

    def _play_step(self):
        if not self.is_playing or self.preview_cap is None:
            return
        ret, frame = self.preview_cap.read()
        if not ret:
            self.is_playing = False
            self.play_btn.config(text="\u25b6 Play")
            return
        self.current_frame_num += 1
        self._display_frame(frame)
        if self.current_frame_num >= self.preview_total_frames:
            self.is_playing = False
            self.play_btn.config(text="\u25b6 Play")
            return
        delay = max(20, int(1000 / self.preview_fps))
        self.root.after(delay, self._play_step)

    def _on_seek(self, value):
        if self._suppress_seek_callback or self.preview_cap is None:
            return
        self.is_playing = False
        self.play_btn.config(text="\u25b6 Play")
        frame_num = int(float(value))
        self._show_frame_at(frame_num)

    # ---------- drag-to-select ----------
    def _on_drag_start(self, event):
        if self.preview_frame_bgr is None:
            return
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="\u25b6 Play")
        self.drag_start = (event.x, event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _on_drag_move(self, event):
        if self.drag_start is None:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x0, y0 = self.drag_start
        self.rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                                      outline="#00ff88", width=2)

    def _on_drag_end(self, event):
        if self.drag_start is None or self.preview_frame_bgr is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return  # too small, ignore accidental click

        fx0, fy0 = self._canvas_to_frame_coords(min(x0, x1), min(y0, y1))
        fx1, fy1 = self._canvas_to_frame_coords(max(x0, x1), max(y0, y1))
        self.selection_bbox = [fx0, fy0, fx1, fy1]
        self.selection_frame_idx = self.current_frame_num
        self.selection_status.config(
            text=f"Object selected on frame {self.current_frame_num} "
                 f"at ({fx0},{fy0})-({fx1},{fy1})", fg="#0a7d32")
        self._log(f"Specific object selected on frame {self.current_frame_num}: {self.selection_bbox}")

    def clear_selection(self):
        self.selection_bbox = None
        self.selection_frame_idx = None
        self.drag_start = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.selection_status.config(text="No object selected", fg="grey")

    def _update_start_btn_state(self):
        if self.video_path and self.engine is not None:
            self.start_btn.config(state="normal")

    # ---------- video enhancement (optional pre-processing) ----------
    def _get_enhancement_options(self):
        return {
            "rotate": int(self.rotate_var.get().replace("\u00b0", "")),
            "denoise": self.denoise_var.get(),
            "sharpen": self.sharpen_var.get(),
            "auto_contrast": self.auto_contrast_var.get(),
            "brightness": self.brightness_var.get(),
            "contrast": self.contrast_var.get(),
        }

    # ---------- face search (optional post-processing) ----------
    def browse_reference_face(self):
        path = filedialog.askopenfilename(
            title="Select reference photo of person of interest",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if path:
            self.reference_face_path = path
            self.reference_face_label.config(text=os.path.basename(path), fg="black")

    def clear_reference_face(self):
        self.reference_face_path = None
        self.reference_face_label.config(text="No reference photo selected", fg="grey")

    # ---------- processing ----------
    def start_processing(self):
        self.is_playing = False
        self.play_btn.config(text="\u25b6 Play")
        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            target_class = self.class_var.get()
            conf = self.conf_var.get()
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            self.output_dir = os.path.join(
                os.path.expanduser("~"), "VideoForensics_Output",
                f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            if self.selection_bbox:
                mode_msg = f"specific selected object only (chosen on frame {self.selection_frame_idx})"
            else:
                mode_msg = "all objects of the class"
            self._log(f"Target class: {target_class} | Confidence: {conf} | Mode: {mode_msg}")

            # Optional pre-processing step. This is completely independent of
            # the tracking pipeline: it only decides WHICH video file path
            # gets passed to engine.process_video() below. If nothing is
            # selected in the Video Enhancement panel, this is a no-op and
            # the original video is analysed exactly as before.
            enhancement_options = self._get_enhancement_options()
            video_for_analysis = self.video_path
            if not options_are_default(enhancement_options):
                self._log("Enhancement options selected - creating enhanced copy before analysis...")
                enhancer = VideoEnhancer(log_callback=self._log, progress_callback=self._update_progress)
                video_for_analysis = enhancer.enhance_video(
                    self.video_path, self.output_dir, enhancement_options)
                self.root.after(0, lambda: self.progress.configure(value=0))  # reset bar for the tracking pass that follows

            self._log("Starting detection & tracking...")

            try:
                summary, detections_log = self.engine.process_video(
                    video_for_analysis, target_class, self.output_dir, conf_threshold=conf,
                    selection_bbox=self.selection_bbox,
                    selection_frame_idx=self.selection_frame_idx or 1)
            finally:
                # The enhanced copy (if any) was only ever needed for this
                # analysis pass - remove it now so it doesn't clutter the
                # investigator's output/evidence folder.
                VideoEnhancer.cleanup_enhanced_video(video_for_analysis, self.video_path)

            summary["video_enhancement"] = {
                "applied": not options_are_default(enhancement_options),
                "description": describe_options(enhancement_options),
            }

            # Optional post-processing: runs AFTER tracking/face-extraction is
            # already done, purely on the face images already saved to disk.
            # If neither a reference photo nor duplicate-removal is selected,
            # this whole block is skipped and summary/report are unaffected.
            reference_path = self.reference_face_path
            dedupe_enabled = self.dedupe_faces_var.get()
            if summary.get("faces_enabled") and (reference_path or dedupe_enabled):
                self._log("Running Face Search / Duplicate Face Removal...")
                matcher = FaceMatcher(log_callback=self._log)
                identified_faces = summary.get("identified_faces") or {}
                if reference_path:
                    identified_faces, match_count, engine_ok = matcher.search_reference(
                        identified_faces, reference_path)
                    summary["identified_faces"] = identified_faces
                    summary["face_search_used"] = engine_ok
                    summary["face_search_matches"] = match_count
                if dedupe_enabled:
                    groups, engine_ok2 = matcher.find_duplicates(summary.get("identified_faces") or {})
                    summary["duplicate_removal_used"] = engine_ok2
                    summary["duplicate_face_groups"] = [g for g in groups if len(g) > 1]

            self._log("Generating forensic PDF report...")
            report_path = os.path.join(self.output_dir, "forensic_report.pdf")
            case_info = {
                "case_ref": self.case_ref_entry.get() or "N/A",
                "officer": self.officer_entry.get() or "N/A",
            }
            ForensicReportGenerator(report_path).generate(summary, detections_log, case_info)
            self._export_csv(detections_log)

            self._log(f"DONE. Report saved at: {report_path}")
            self.root.after(0, lambda: self.open_output_btn.config(state="normal"))

            if summary.get("selection_match_failed"):
                self.root.after(0, lambda: messagebox.showwarning(
                    "Selection Not Matched",
                    "Your selected object could not be confidently matched in the video, "
                    "so nothing was reported (to avoid reporting the wrong object).\n\n"
                    "Try again: pick a frame where the object is clearly visible and "
                    "separated from others, and draw the box closely around just it."))
            else:
                plate_count = len(summary.get("identified_plates") or {})
                face_count = len(summary.get("identified_faces") or {})
                plate_line = f"\nPlate images saved: {plate_count}" if summary.get("plates_enabled") else ""
                face_line = f"\nFace images saved: {face_count}" if summary.get("faces_enabled") else ""
                enh = summary.get("video_enhancement") or {}
                enh_line = f"\nVideo enhancement applied: {enh['description']}" if enh.get("applied") else ""
                search_line = ""
                if summary.get("face_search_used"):
                    search_line = f"\nFace Search matches to reference photo: {summary.get('face_search_matches', 0)}"
                elif reference_path and not summary.get("face_search_used"):
                    search_line = "\nFace Search: engine unavailable (see log) - reference photo not used"
                dedupe_line = ""
                dup_groups = summary.get("duplicate_face_groups") or []
                if summary.get("duplicate_removal_used"):
                    dedupe_line = f"\nDuplicate Face groups found: {len(dup_groups)}"
                elif dedupe_enabled and not summary.get("duplicate_removal_used"):
                    dedupe_line = "\nDuplicate Face Removal: engine unavailable (see log)"
                self.root.after(0, lambda: messagebox.showinfo(
                    "Processing Complete",
                    f"Tracking complete.\n\nMode: {summary.get('tracking_mode')}\n"
                    f"Detections: {summary['total_detections']}\n"
                    f"Unique objects: {summary['unique_objects_tracked']}"
                    f"{plate_line}{face_line}{enh_line}{search_line}{dedupe_line}\n\n"
                    f"Report: {report_path}"))
        except Exception as e:
            self._log(f"ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

    def _export_csv(self, detections_log):
        csv_path = os.path.join(self.output_dir, "detections_log.csv")
        fieldnames = ["frame", "timestamp", "track_id", "confidence", "bbox",
                      "plate_text", "plate_confidence"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections_log:
                writer.writerow({k: d.get(k, "") for k in fieldnames})

    def open_output_folder(self):
        if self.output_dir and os.path.exists(self.output_dir):
            if sys.platform == "win32":
                os.startfile(self.output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.output_dir])
            else:
                subprocess.Popen(["xdg-open", self.output_dir])


def main():
    root = tk.Tk()
    VideoForensicsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
