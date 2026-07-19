"""
plate_reader.py
-----------------
Automatic Number Plate Recognition (ANPR) helper module.

For vehicle classes (car, motorcycle, bus, truck), each tracked vehicle's
bounding-box crop is passed through this module in TWO independent steps:

  1. locate_and_enhance() - ALWAYS works (pure OpenCV, no internet, no heavy
     model). Crops the region of the vehicle most likely to contain the
     plate, upscales it, boosts local contrast, and runs a dedicated
     plate-focused denoise + sharpen pass (see _dedicated_plate_sharpen).
     This image is what actually solves the cyber-cell need: even when
     nothing below can auto-read the text, an investigator can open this
     zoomed image and read the plate themselves.

  2. read_text() - best-effort OFFLINE OCR (EasyOCR) on that zoomed image,
     to auto-fill the plate text when possible. This step CAN fail (model
     not installed, blurry/angled footage, poor lighting) - that is
     expected and handled gracefully; it never blocks step 1's image from
     being produced and saved.

IMPORTANT / HONEST LIMITATION NOTE:
This is a general-purpose OCR reading a heuristically-cropped region, NOT a
model specifically trained to detect and read number plates. Treat every
auto-read text as an investigative LEAD to verify manually against the
zoomed image, not as a confirmed/court-ready plate number. For production
forensic use, a plate detector trained specifically on Indian plates would
give far more reliable crops before OCR - see README "Future Improvements".
"""

import re

import cv2

# Loose Indian plate pattern: 2 letters (state), 1-2 digits (RTO code),
# 0-3 letters (series), 3-4 digits (number). Used only as a confidence
# hint, not a strict filter.
_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")

MIN_READ_CONFIDENCE = 0.25


class PlateReader:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._reader = None  # lazy-loaded so importing this module is cheap
        self._load_failed = False

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def _ensure_loaded(self):
        if self._reader is not None or self._load_failed:
            return
        try:
            self._log("Loading number-plate OCR engine (first run downloads "
                       "~65MB of OCR models, needs internet once)...")
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._log("Number-plate OCR engine ready.")
        except Exception as e:
            self._load_failed = True
            self._log(f"NOTE: automatic plate-TEXT reading is unavailable ({e}). "
                       "Zoomed plate IMAGES will still be saved for every tracked "
                       "vehicle so they can be read manually.")

    def is_available(self):
        self._ensure_loaded()
        return self._reader is not None

    def locate_and_enhance(self, vehicle_crop_bgr):
        """
        ALWAYS works (pure OpenCV, no OCR / no internet needed). Crops the
        lower ~55% of the vehicle box (where the plate usually sits),
        upscales it, boosts local contrast, and applies a dedicated
        plate-focused sharpening pass (mild denoise + unsharp mask tuned for
        small text edges). Returns the zoomed image, or None only if the
        input crop itself is empty/invalid (e.g. box at the very edge of
        frame).
        """
        if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
            return None

        h, w = vehicle_crop_bgr.shape[:2]
        lower = vehicle_crop_bgr[int(h * 0.45):h, :]
        if lower.size == 0:
            lower = vehicle_crop_bgr

        lower = cv2.resize(lower, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)

        try:
            lab = cv2.cvtColor(lower, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lower = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            pass  # contrast boost is a nice-to-have, never fatal

        lower = self._dedicated_plate_sharpen(lower)
        return lower

    @staticmethod
    def _dedicated_plate_sharpen(plate_img_bgr):
        """
        Plate-specific enhancement pass: a light bilateral denoise (removes
        compression/low-light grain WITHOUT blurring hard character edges,
        unlike a normal Gaussian blur) followed by an unsharp mask tuned to
        make small plate characters stand out more clearly. Never fatal -
        falls back to the un-sharpened image if anything goes wrong.
        """
        try:
            denoised = cv2.bilateralFilter(plate_img_bgr, d=5, sigmaColor=50, sigmaSpace=50)
            blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2)
            sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
            return sharpened
        except Exception:
            return plate_img_bgr

    def read_text(self, zoomed_region_bgr):
        """
        Best-effort OCR on an already-zoomed plate region (output of
        locate_and_enhance). Returns None if the OCR engine is unavailable
        or nothing confident was read - callers should keep/show the image
        from locate_and_enhance() regardless of this result.

        On success, returns:
            {
              "text": cleaned alphanumeric guess, e.g. "UP32AB1234",
              "raw_text": exactly what OCR returned,
              "confidence": 0.0-1.0,
              "matches_format": True/False - loose sanity check only,
            }
        """
        if zoomed_region_bgr is None or zoomed_region_bgr.size == 0:
            return None
        self._ensure_loaded()
        if self._reader is None:
            return None

        try:
            results = self._reader.readtext(zoomed_region_bgr)
        except Exception:
            return None
        if not results:
            return None

        best = max(results, key=lambda r: r[2])
        raw_text = best[1]
        conf = float(best[2])
        if conf < MIN_READ_CONFIDENCE:
            return None

        cleaned = self._clean_text(raw_text)
        if not cleaned:
            return None

        return {
            "text": cleaned,
            "raw_text": raw_text,
            "confidence": round(conf, 3),
            "matches_format": bool(_PLATE_REGEX.match(cleaned)),
        }

    @staticmethod
    def _clean_text(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())
