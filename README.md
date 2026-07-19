# Smart CCTV Investigator
### Object Detection & Tracking tool for Investigation / Cyber Cell use

> 📄 **New here? See `USER_MANUAL.pdf`** for a simple, non-technical quick-start
> guide — installation, step-by-step usage, and FAQs. This README below is the
> **developer/build** guide (for whoever builds and packages the app).

A Windows desktop application that lets an investigator upload a video, pick a
target object (person, car, bag, phone, etc. from 80 common classes), and
automatically:

1. **Detects and tracks** that object across the whole video (unique ID per
   object, even if it leaves and re-enters frame).
2. **Lets you pick ONE specific instance, anywhere in the video** — play,
   pause, or scrub the preview to the exact moment your object of interest
   appears (it doesn't have to be in the first frame), then drag a box
   around it. Only that object gets tracked & reported from that point
   onward (others of the same class are ignored). Skip this step to track
   every instance of the class instead.
3. **Saves every frame** the object appears in, with a bounding box + track
   ID drawn on it, into an `evidence_frames/` folder.
4. **Generates a PDF forensic report** with case details, tracking mode
   (all-instances vs specific-object), a summary (total detections, unique
   objects tracked), a full frame-by-frame log table, and embedded evidence
   thumbnails.
5. **Automatic Number Plate Recognition (ANPR)** — when the tracked class is
   a vehicle (`car`, `motorcycle`, `bus`, `truck`), each tracked vehicle's
   plate region is automatically cropped, zoomed, and read with offline OCR.
   A zoomed plate image is saved for EVERY tracked vehicle regardless of
   whether OCR could read it (so it can be read manually), added to its own
   section in the PDF report.
6. **Face Detection** — when the tracked class is `person`, a zoomed image
   is saved for every tracked individual: a tight, auto-detected face crop
   where possible, or a zoomed head/shoulders view (always saved, even when
   precise detection isn't confident - real footage often defeats automatic
   face detection, so the report is never left empty). Every saved face
   image also goes through a dedicated denoise + sharpen pass, making
   features clearer for manual review or downstream face-recognition
   matching. Added to its own section in the PDF report. Works great
   combined with specific-instance selection (#2 above): select one
   particular person and get a zoomed, enhanced face image of just them.
7. Also exports a **CSV log** of every detection (frame #, timestamp, track
   ID, confidence, bounding box, plate reading if applicable) for import
   into other forensic tools/Excel.
8. **Video Enhancement (optional pre-processing)** — Denoise, Sharpen/Deblur,
   Auto Brightness/Contrast (CLAHE), Rotation, and manual Brightness/Contrast,
   applied to a temporary enhanced copy of the video before detection runs
   (the original file is never modified, and the temporary copy is deleted
   automatically once tracking finishes - it does NOT appear in your output
   folder). Improves results on dark, grainy, sideways, or slightly blurry
   CCTV footage; also strengthens ANPR and face detection since they run on
   the enhanced frames too. Skipped entirely (zero effect on the app) if
   left at defaults.
9. **Dedicated Plate Enhancement** — every plate crop now also gets a
   plate-focused denoise + sharpen pass (in addition to the existing zoom +
   contrast boost), tuned to make small character edges stand out more
   clearly for both auto-OCR and manual reading.
10. **Face Search (optional)** — supply a reference photo of a person of
    interest, and every tracked person's saved face image is compared
    against it; possible matches are flagged in the PDF report with a
    similarity score. A probabilistic LEAD to verify manually, not a
    confirmed identification.
11. **Duplicate Face Removal (optional)** — compares every tracked person's
    face against every other, and flags Track IDs that are likely the same
    individual (e.g. the tracker lost and re-acquired them mid-video),
    noted in the PDF report.

Both Face Search and Duplicate Face Removal use the `insightface` library and
are OPTIONAL: if left unused (or if the library isn't installed), the app
works exactly as before and the report renders exactly as before.

---

## 1. Project structure

```
video_forensics_tool/
├── main.py                    # App entry point
├── requirements.txt
├── build_exe.bat              # One-click Windows .exe builder
├── core/
│   ├── tracker_engine.py      # YOLOv8 detection + ByteTrack tracking
│   ├── plate_reader.py        # Number-plate OCR (ANPR) for vehicle classes
│   ├── face_reader.py         # Face detection & zoom for the 'person' class
│   └── report_generator.py    # PDF report builder (ReportLab)
└── gui/
    └── app.py                 # Tkinter desktop UI
```

## 2. How it works (tech stack)

- **Detection & Tracking**: YOLOv8 (`ultralytics` library) with the built-in
  **ByteTrack** tracker — assigns a persistent ID to each object instance so
  you can tell "object A" apart from "object B" even if both are the same
  class (e.g. two different people).
- **Number Plate Recognition (ANPR)**: `easyocr` (offline, no internet after
  the one-time model download). For vehicle classes, each tracked vehicle's
  box is cropped to its lower half (where plates sit), upscaled 3x, contrast
  enhanced, and read with OCR every few frames until a confident reading is
  found. This is heuristic, general-purpose OCR — **not** a model trained
  specifically to detect plates — so treat every reading as a lead to verify
  against the source frame, not a confirmed plate number (see Section 6 for
  how to make it more robust).
- **Face Detection**: OpenCV's built-in Haar Cascade face detector (ships
  with `opencv-python` — no extra download/dependency), tried on frontal
  and both profile directions. For the `person` class, each tracked
  person's head/shoulders region is ALWAYS zoomed and saved as evidence;
  if the cascade also manages to pin down a precise face box, an even
  tighter zoomed crop is used instead. This two-layer approach exists
  because Haar cascades often fail outright on small/angled/poorly-lit
  real CCTV faces — without the always-save fallback, the report would end
  up empty whenever detection failed. This is face **detection**, not face
  **recognition** — it finds and zooms into a face, but does not identify
  who it belongs to or match it against any database (see Section 6 for
  how to add that later).
- **GUI**: Tkinter (ships with Python — no extra runtime needed on Windows).
- **Report**: ReportLab generates a PDF; `csv` module exports the raw log.
- Detectable classes: the standard 80 **COCO** classes — person, bicycle,
  car, motorcycle, bus, truck, backpack, handbag, suitcase, cell phone,
  laptop, knife is *not* included by default (COCO has no weapon class) —
  see Section 6 for how to add custom classes like weapons later.

## 3. Setup (Windows)

1. Install **Python 3.10–3.12** from python.org (check "Add to PATH" during
   install).
2. Open Command Prompt in the project folder and run:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python main.py
   ```
   First run will auto-download the YOLOv8 weights (~6 MB, needs internet
   once). After that it works fully offline. The first time you track a
   **vehicle class** (car/motorcycle/bus/truck), it will also download the
   OCR models used for number-plate reading (~65 MB, needs internet once) —
   after that, plate reading also works fully offline. Likewise, the first
   time you use **Face Search or Duplicate Face Removal**, it downloads a
   face-recognition model (~100 MB, needs internet once) — after that it
   also works fully offline. If `insightface`/`onnxruntime` fail to install
   or load, only Face Search / Duplicate Removal are skipped (with a clear
   log message) — everything else in the app is unaffected.

## 4. Building a standalone `.exe`

So investigators can run it without installing Python:

1. Run `build_exe.bat` (double-click it, or run from Command Prompt).
2. It installs PyInstaller and packages everything into
   `dist\VideoForensicsTool.exe`.
3. Copy `dist\VideoForensicsTool.exe` + the auto-downloaded `yolov8n.pt`
   file to any Windows machine and run it — no Python needed there.

> Note: build the `.exe` **on a Windows machine** (PyInstaller builds for
> the OS it runs on). This sandbox is Linux, so the `.exe` itself has to be
> built on your Windows PC using the steps above — all the source code is
> ready to go.

## 5. Using the app

1. Launch the app → wait for "Model loaded. Ready." in the log.
2. Fill in **Case Reference** and **Officer Name** (goes into the PDF report
   header — optional but recommended for chain-of-custody).
3. Click **Browse Video** and select the evidence video (`.mp4/.avi/.mov/.mkv/.wmv`).
4. **(Optional) Video Enhancement**: in the "Video Enhancement" panel you can
   turn on Denoise, Sharpen/Deblur, Auto Brightness/Contrast (CLAHE), rotate
   the video, or nudge Brightness/Contrast sliders. This runs **before**
   detection, on a **separate enhanced copy** of the video (the original
   file is never modified) — useful for dark, grainy, sideways, or blurry
   CCTV footage. Leave everything at its default to skip this step entirely;
   the app then behaves exactly as before this feature was added. Note:
   "Sharpen/Deblur" is unsharp-mask sharpening (counters mild blur) — it is
   not true deblurring and won't recover detail lost to heavy motion blur.
5. A **video preview player** loads automatically in the "Object Selection"
   box, with Play/Pause and a seek bar - so you can scrub to the exact
   moment your object of interest is clearly visible.
   - **To track one specific object** (e.g. just one particular person out
     of several in frame): play/scrub to the right moment, pause, then
     click-and-drag a box around that object. The log will confirm which
     Track ID got locked, and from which frame onward it will be reported.
   - **To track every object of a class instead**, just don't draw a box -
     skip to the next step.
   - Click **Clear Selection** anytime to undo a drag and go back to
     tracking all instances.
6. Pick the **object class** to track from the dropdown (e.g. `person`,
   `car`, `backpack`).
7. **(Optional, `person` class only) Face Search / Duplicate Removal**: in
   the "Face Search & Duplicate Face Removal" panel you can browse to a
   reference photo of a person of interest (their face gets compared to
   every tracked person's saved face image) and/or turn on "Duplicate Face
   Removal" (groups Track IDs likely to be the same individual). Leave both
   off to skip - report renders exactly as before. First use downloads a
   face-recognition model (~100 MB, needs internet once).
8. Adjust the **Confidence** slider if needed (higher = fewer false
   positives, lower = catches more but noisier).
9. Click **Start Tracking & Generate Report**.
10. When done, click **Open Output Folder** — you'll find:
   - `evidence_frames/` — every frame the object appeared in, boxed & labeled
   - `plate_evidence/` — zoomed plate-region images (only when tracking a
     vehicle class and a plate was read)
   - `face_evidence/` — zoomed face images, one per tracked person (only
     when tracking the `person` class)
   - `forensic_report.pdf` — the full report, including **Vehicle Number
     Plate Identification** and/or **Face Identification** sections when
     applicable
   - `detections_log.csv` — raw data log (includes plate reading columns)

> **Note:** when you select a specific object partway through the video, the
> report covers that object from the **selected frame onward** (not frames
> before it) — pick the earliest frame where the object is clearly visible
> for a complete record.

## 6. Extending it further (roadmap ideas)

- **More reliable ANPR**: train a small YOLOv8 model specifically to detect
  plate rectangles (a labelled Indian-plates dataset), then run OCR only on
  that precise crop instead of the current heuristic lower-half crop — this
  is the single biggest accuracy improvement available.
- **Custom classes (weapons, specific bags, faces, etc.)**: train a custom
  YOLOv8 model (`yolo train data=your_dataset.yaml model=yolov8n.pt`) and
  point `TrackerEngine(model_path="your_model.pt")` to it — no other code
  changes needed.
- **Face matching / recognition**: currently the app only *detects and
  zooms into* a face (Section on Face Detection). To go further and
  actually *identify* a specific known person or match faces across
  different videos/cases, add `face_recognition` / `insightface` and
  compare face embeddings against a reference database.
- **Multi-object simultaneous tracking**: currently tracks one class at a
  time by design (keeps forensic reports focused); can be extended to track
  several classes together and generate one report per class.
- **Chain-of-custody hashing**: auto-generate SHA-256 hash of the source
  video and embed it in the report for evidentiary integrity.
- **Court-ready export**: convert report to signed/watermarked PDF.
- **Plate database lookup**: once a plate is read, auto-cross-check it
  against an internal vehicle-registration database (requires access to
  such a database/API - not included here).

## 7. Packaging as a professional Windows installer (for distribution)

This turns the app into a proper installable Windows program — a `Setup.exe`
that any user (e.g. cyber cell staff) can double-click, install, and get a
desktop shortcut, exactly like commercial software. **These steps must be run
on a Windows machine** (PyInstaller builds for the OS it runs on).

**Step 1 — Build the .exe**
```
build_exe.bat
```
This installs dependencies, then runs PyInstaller in `--onedir` mode with the
custom `assets/app_icon.ico` icon and the name `SmartCCTVInvestigator`.
Output: `dist/SmartCCTVInvestigator/SmartCCTVInvestigator.exe`

**Step 2 — Build the installer**
1. Install [Inno Setup](https://jrsoftware.org/isdl.php) (free).
2. Open `installer/SmartCCTVInvestigator.iss` in Inno Setup and click
   **Build → Compile** (or run `installer/build_installer.bat`).
3. Output: `installer/output/SmartCCTVInvestigator_Setup_v1.0.exe` — this
   single file is what you distribute.

**Step 3 — Distribute**
Upload `SmartCCTVInvestigator_Setup_v1.0.exe` to your GitHub repo's
**Releases** section (tag it e.g. `v1.0`). Share the release link — users
click it, download, and run the installer. No Python or technical setup
needed on their end.

The installer creates: a Start Menu entry, an optional Desktop shortcut, a
bundled copy of `USER_MANUAL.pdf`, and a proper Windows uninstaller entry in
"Add or Remove Programs".

---
Built with: `ultralytics` (YOLOv8 + ByteTrack), `opencv-python`, `reportlab`,
`easyocr`, `tkinter`.
