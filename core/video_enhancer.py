"""
video_enhancer.py
-------------------
Optional PRE-PROCESSING module. Runs BEFORE detection/tracking, on the raw
video file, and produces a new, enhanced video file. The rest of the
pipeline (TrackerEngine, PlateReader, FaceReader, ReportGenerator) is
completely untouched - it simply receives the path of whichever video file
it should analyse (the original, or this enhanced copy), exactly as before.

This keeps the change 100% additive / non-invasive: if no enhancement is
requested, this module is never called and the app behaves exactly as it
did before this feature was added.

All operations are classical OpenCV image processing (no extra
dependencies, no internet, no heavy AI model):

  - Rotation            - fixes sideways/upside-down CCTV/dashcam footage
  - Denoise             - reduces graininess from low-light/compressed video
  - Sharpen / Deblur     - unsharp-masking to counter mild motion/focus blur
  - Brightness/Contrast  - manual linear adjustment
  - Auto Contrast (CLAHE)- adaptive local-contrast boost, good for
                            underexposed or backlit footage

HONEST LIMITATION NOTE:
"Deblur" here is unsharp-mask sharpening, which visually counters *mild*
blur - it is NOT true blind deconvolution and will not recover detail lost
to heavy motion blur or an out-of-focus lens. Treat the enhanced video as a
"clearer to look at / easier for detection" version, not a forensically
restored original. The original, unmodified video file is never altered or
deleted - the enhanced copy is always saved separately.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np

# Sensible no-op defaults - if every option is left at its default, the
# output frame is (numerically) unchanged.
DEFAULT_OPTIONS = {
    "rotate": 0,            # 0, 90, 180, 270 (clockwise)
    "denoise": False,
    "sharpen": False,
    "auto_contrast": False,  # CLAHE
    "brightness": 0,         # -100..100, 0 = no change
    "contrast": 1.0,         # 0.5..2.0, 1.0 = no change
}


def options_are_default(options):
    """True if these options would not change anything (safe to skip the
    whole pre-processing pass and use the original video untouched)."""
    o = {**DEFAULT_OPTIONS, **(options or {})}
    return (o["rotate"] == 0 and not o["denoise"] and not o["sharpen"]
            and not o["auto_contrast"] and o["brightness"] == 0
            and abs(o["contrast"] - 1.0) < 1e-6)


def describe_options(options):
    """Human-readable summary of which enhancements are active - used in
    logs and in the PDF report."""
    o = {**DEFAULT_OPTIONS, **(options or {})}
    parts = []
    if o["rotate"]:
        parts.append(f"Rotate {o['rotate']}\u00b0")
    if o["denoise"]:
        parts.append("Denoise")
    if o["sharpen"]:
        parts.append("Sharpen/Deblur")
    if o["auto_contrast"]:
        parts.append("Auto Contrast (CLAHE)")
    if o["brightness"] != 0:
        parts.append(f"Brightness {o['brightness']:+d}")
    if abs(o["contrast"] - 1.0) > 1e-6:
        parts.append(f"Contrast x{o['contrast']:.2f}")
    return ", ".join(parts) if parts else "None"


class VideoEnhancer:
    def __init__(self, log_callback=None, progress_callback=None):
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    # ---------- single-frame operations (also usable standalone) ----------
    @staticmethod
    def rotate_frame(frame, angle):
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def denoise_frame(frame):
        # Colour-preserving denoise; h/hColor kept modest so real detail
        # (plates, faces) isn't smeared away.
        return cv2.fastNlMeansDenoisingColored(frame, None, h=7, hColor=7,
                                                templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def sharpen_frame(frame):
        # Classic unsharp mask: original + (original - blurred) * amount
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    @staticmethod
    def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
        if brightness == 0 and abs(contrast - 1.0) < 1e-6:
            return frame
        return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    @staticmethod
    def auto_contrast_frame(frame):
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    def enhance_frame(self, frame, options):
        """Applies the requested operations, in a fixed, sensible order, to
        a single BGR frame. Any option left at its default is a no-op."""
        o = {**DEFAULT_OPTIONS, **(options or {})}
        out = frame
        if o["rotate"]:
            out = self.rotate_frame(out, o["rotate"])
        if o["denoise"]:
            out = self.denoise_frame(out)
        if o["sharpen"]:
            out = self.sharpen_frame(out)
        if o["brightness"] != 0 or abs(o["contrast"] - 1.0) > 1e-6:
            out = self.adjust_brightness_contrast(out, o["brightness"], o["contrast"])
        if o["auto_contrast"]:
            out = self.auto_contrast_frame(out)
        return out

    # ---------- full-video pass ----------
    def enhance_video(self, input_path, output_dir, options):
        """
        Reads input_path frame-by-frame, applies enhance_frame(), and writes
        a new video file to a PRIVATE TEMP DIRECTORY (NOT the case's output
        folder - investigators don't need this intermediate file cluttering
        their evidence folder, only the enhanced detections/plate/face
        images that come out of analysing it). Returns the temp file path.
        Does NOT modify or delete the original input file.

        Call cleanup_enhanced_video() with the returned path once tracking
        on it has finished, to remove the temp file/folder again.

        Note: `output_dir` is accepted for backward compatibility but is no
        longer used as the enhanced file's location.
        """
        if options_are_default(options):
            # Nothing to do - caller should just use the original video.
            return input_path

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video for enhancement: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate = (options or {}).get("rotate", 0)
        out_w, out_h = (h, w) if rotate in (90, 270) else (w, h)

        temp_dir = tempfile.mkdtemp(prefix="video_forensics_enhance_")
        output_path = os.path.join(temp_dir, "enhanced_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise IOError("Could not create enhanced video file (codec/output path issue).")

        self._log(f"Enhancing video before analysis: {describe_options(options)}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                enhanced = self.enhance_frame(frame, options)
                writer.write(enhanced)
                if self.progress_callback and total:
                    self.progress_callback(frame_idx, total)
        finally:
            cap.release()
            writer.release()

        self._log(f"Enhanced video ready ({frame_idx} frames) - analysing it now "
                   "(temporary file, not saved to the output folder)...")
        return output_path

    @staticmethod
    def cleanup_enhanced_video(enhanced_path, original_path):
        """
        Deletes the temporary enhanced-video file/folder created by
        enhance_video(), once tracking on it has finished. Safe to call even
        if enhanced_path == original_path (i.e. enhancement was skipped) -
        in that case it's a no-op, since that's the user's own original file.
        """
        if not enhanced_path or enhanced_path == original_path:
            return
        try:
            temp_dir = os.path.dirname(enhanced_path)
            if os.path.isdir(temp_dir) and os.path.basename(temp_dir).startswith("video_forensics_enhance_"):
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif os.path.exists(enhanced_path):
                os.remove(enhanced_path)
        except Exception:
            pass  # cleanup failure should never crash the app
