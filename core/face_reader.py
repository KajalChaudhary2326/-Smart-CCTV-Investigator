"""
face_reader.py
-----------------
Face-detection helper module, used only for the 'person' class.

Mirrors the same lesson learned from the number-plate module: automatic
detection (Haar Cascade here) can fail on real CCTV-quality footage - small,
angled, poorly-lit, or side-on faces are genuinely hard for it. If we only
saved an image on a successful detection, the report would end up EMPTY
whenever detection failed, even though a person was clearly tracked. So this
module works in two layers:

  1. locate_head_region() - ALWAYS works (pure OpenCV crop + upscale +
     contrast enhance + dedicated denoise/sharpen pass, no detection
     needed). Crops the top ~40% of the person's bounding box (head/
     shoulders area) and zooms into it. This guarantees there is always
     something clear in the report to look at, even when nothing below
     can find a precise face box.

  2. detect_face_box() - best-effort Haar Cascade face detection (tries
     frontal, then profile, then mirrored-profile to catch both facing
     directions) on that head region, for a tighter, more zoomed-in crop
     when it succeeds (also sharpened). This can fail - that's expected
     and handled gracefully by falling back to the head-region image from
     step 1.

IMPORTANT / HONEST LIMITATION NOTE:
This is face DETECTION (finds/zooms into a face-ish region), NOT face
RECOGNITION - it does not know WHO the face belongs to and cannot match it
against a database of known people. See README "Future Improvements" for
how to add real face identity-matching later (e.g. face_recognition /
insightface libraries).
"""

import cv2

_frontal_cascade = None
_profile_cascade = None


def _get_frontal_cascade():
    global _frontal_cascade
    if _frontal_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _frontal_cascade = cv2.CascadeClassifier(path)
    return _frontal_cascade


def _get_profile_cascade():
    global _profile_cascade
    if _profile_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        _profile_cascade = cv2.CascadeClassifier(path)
    return _profile_cascade


class FaceReader:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def locate_head_region(self, person_crop_bgr):
        """
        ALWAYS works (pure OpenCV, no detection model needed). Crops the top
        ~40% of the person's bounding box (where the head/shoulders are for
        a standing/walking person), upscales it 2x, and boosts contrast.
        Returns None only if the input crop itself is empty/invalid.
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None

        h, w = person_crop_bgr.shape[:2]
        head = person_crop_bgr[0:int(h * 0.4), :]
        if head.size == 0:
            head = person_crop_bgr

        head = cv2.resize(head, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            lab = cv2.cvtColor(head, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            head = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            pass  # contrast boost is a nice-to-have, never fatal

        return self._dedicated_face_sharpen(head)

    def detect_face_box(self, head_region_bgr):
        """
        Best-effort Haar Cascade face detection on an already-zoomed head
        region (output of locate_head_region). Tries frontal faces first,
        then side profiles facing either direction. Returns a tightly
        cropped, further-zoomed image of just the face, or None if nothing
        was confidently found - callers should fall back to the
        locate_head_region() image in that case, not treat it as an error.

        Returns: {"image": <zoomed BGR crop>, "face_area": int} or None.
        """
        if head_region_bgr is None or head_region_bgr.size == 0:
            return None

        try:
            gray = cv2.cvtColor(head_region_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            face_box = self._detect_in_gray(gray)
        except Exception:
            return None

        if face_box is None:
            return None

        fx, fy, fw, fh = face_box
        pad_x, pad_y = int(fw * 0.3), int(fh * 0.3)
        h, w = head_region_bgr.shape[:2]
        x1, y1 = max(0, fx - pad_x), max(0, fy - pad_y)
        x2, y2 = min(w, fx + fw + pad_x), min(h, fy + fh + pad_y)
        face_crop = head_region_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        zoomed = cv2.resize(face_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        zoomed = self._dedicated_face_sharpen(zoomed)
        return {"image": zoomed, "face_area": int(fw * fh)}

    @staticmethod
    def _dedicated_face_sharpen(face_img_bgr):
        """
        Face-focused enhancement pass: a light bilateral denoise (removes
        compression/low-light grain without smearing away facial features,
        unlike a normal Gaussian blur) followed by an unsharp mask that makes
        facial edges/features stand out more clearly - useful both for a
        human investigator and for any downstream face-recognition matching
        (see face_matcher.py). Never fatal - falls back to the un-sharpened
        image if anything goes wrong.
        """
        try:
            denoised = cv2.bilateralFilter(face_img_bgr, d=5, sigmaColor=50, sigmaSpace=50)
            blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2)
            return cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        except Exception:
            return face_img_bgr

    @staticmethod
    def _detect_in_gray(gray):
        """Try frontal face, then profile facing left, then profile facing
        right (via a horizontal flip, since the cascade is only trained on
        one direction). Returns (x, y, w, h) of the largest match, or None."""
        frontal = _get_frontal_cascade().detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(18, 18))
        if len(frontal) > 0:
            return tuple(max(frontal, key=lambda f: f[2] * f[3]))

        profile = _get_profile_cascade().detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(18, 18))
        if len(profile) > 0:
            return tuple(max(profile, key=lambda f: f[2] * f[3]))

        flipped = cv2.flip(gray, 1)
        profile_flipped = _get_profile_cascade().detectMultiScale(
            flipped, scaleFactor=1.05, minNeighbors=4, minSize=(18, 18))
        if len(profile_flipped) > 0:
            fx, fy, fw, fh = max(profile_flipped, key=lambda f: f[2] * f[3])
            gw = gray.shape[1]
            fx_original = gw - fx - fw  # translate back to un-flipped coords
            return (fx_original, fy, fw, fh)

        return None
