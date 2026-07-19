"""
face_matcher.py
-----------------
Optional Face Search + Duplicate Face Removal helper.

Adds two POST-PROCESSING capabilities on top of the face-region images that
TrackerEngine + FaceReader already save (core/face_reader.py,
face_evidence/*.jpg). Nothing here runs during, or changes, the frame-by-
frame tracking loop - it only looks at the face images already saved AFTER
tracking finishes, exactly like an investigator reviewing the folder by eye.
This keeps tracker_engine.py completely untouched.

  - Face Search: given a reference photo of a person of interest, compares
    it against every tracked person's saved face image and reports a
    similarity score for each - "does any tracked person look like this?"
  - Duplicate Face Removal: compares every tracked person's face image
    against every other one, and groups together Track IDs that are
    likely the SAME individual (e.g. the tracker briefly lost and
    re-acquired them, creating two different Track IDs for one person).

Uses the `insightface` library (ArcFace-based embeddings), which is more
Windows-friendly to install than `face_recognition`/dlib. Like
plate_reader.py and face_reader.py, this degrades gracefully if the
library/model isn't installed or a face can't be embedded: Face Search /
Duplicate Removal are then simply skipped with a clear log message, and
everything else in the app (tracking, ANPR, basic face detection, reports)
is completely unaffected.

IMPORTANT / HONEST LIMITATION NOTE:
Face-embedding similarity is a probabilistic match, not a certain
identification - lighting, angle, image quality, and low-resolution CCTV
crops all reduce accuracy. Treat every match/duplicate grouping as an
investigative LEAD to verify manually against the images, never as a
confirmed identity.
"""

import os

import cv2
import numpy as np

DEFAULT_MATCH_THRESHOLD = 0.45      # cosine similarity vs a reference photo
DEFAULT_DUPLICATE_THRESHOLD = 0.55  # slightly higher bar for merging two Track IDs


class FaceMatcher:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._app = None
        self._load_failed = False

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def _ensure_loaded(self):
        if self._app is not None or self._load_failed:
            return
        try:
            self._log("Loading face-recognition engine for Face Search / Duplicate "
                       "Removal (first run downloads ~100MB of models, needs internet once)...")
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(320, 320))
            self._log("Face-recognition engine ready.")
        except Exception as e:
            self._load_failed = True
            self._log(f"NOTE: Face Search / Duplicate Removal are unavailable ({e}). "
                       "Basic face detection, tracking, ANPR, and reports are unaffected.")

    def is_available(self):
        self._ensure_loaded()
        return self._app is not None

    def get_embedding(self, image_path_or_bgr):
        """Returns a normalized embedding vector for the largest face found
        in the given image (a path, or a BGR numpy array), or None if
        unavailable / no face could be embedded."""
        self._ensure_loaded()
        if self._app is None:
            return None

        img = image_path_or_bgr
        if isinstance(img, str):
            if not os.path.exists(img):
                return None
            img = cv2.imread(img)
        if img is None or img.size == 0:
            return None

        try:
            faces = self._app.get(img)
        except Exception:
            return None
        if not faces:
            return None

        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding

    @staticmethod
    def similarity(emb_a, emb_b):
        """Cosine similarity between two embeddings (higher = more alike).
        Returns 0.0 if either embedding is missing."""
        if emb_a is None or emb_b is None:
            return 0.0
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---------- high level operations used by the app ----------
    def search_reference(self, identified_faces, reference_image_path,
                          threshold=DEFAULT_MATCH_THRESHOLD):
        """
        identified_faces: dict {track_id: {..., "image_path": ...}}, as
        produced by TrackerEngine's face_evidence pipeline. Not modified in
        place - a new dict (with 'reference_match'/'reference_match_score'
        added to each entry) is returned.

        Returns (updated_identified_faces, match_count, engine_available).
        If engine_available is False, the original dict is returned
        unchanged and match_count is 0 - callers should treat this the same
        as "feature not used", not as an error.
        """
        if not self.is_available():
            return identified_faces, 0, False

        ref_emb = self.get_embedding(reference_image_path)
        if ref_emb is None:
            self._log("Could not find a usable face in the reference photo - "
                       "Face Search skipped.")
            return identified_faces, 0, False

        updated = {}
        match_count = 0
        for tid, info in identified_faces.items():
            info = dict(info)
            emb = self.get_embedding(info.get("image_path"))
            score = self.similarity(ref_emb, emb) if emb is not None else 0.0
            info["reference_match_score"] = round(score, 3)
            info["reference_match"] = score >= threshold
            if info["reference_match"]:
                match_count += 1
                self._log(f"Track ID {tid}: possible match to reference photo "
                          f"(similarity {score:.2f}).")
            updated[tid] = info
        return updated, match_count, True

    def find_duplicates(self, identified_faces, threshold=DEFAULT_DUPLICATE_THRESHOLD):
        """
        Compares every tracked person's face image against every other, and
        groups Track IDs that are likely the same individual (e.g. tracker
        lost/re-acquired them mid-video).

        Returns (groups, engine_available) where groups is a list of lists
        of track IDs, e.g. [[3, 7], [5], [9, 12, 14]] - only groups with
        more than one ID are actual duplicate clusters; singletons are
        included too so callers can distinguish "checked, no duplicate"
        from "not checked".
        """
        if not self.is_available():
            return [], False

        tids = sorted(identified_faces.keys())
        embeddings = {tid: self.get_embedding(identified_faces[tid].get("image_path"))
                      for tid in tids}

        parent = {tid: tid for tid in tids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if embeddings[a] is None or embeddings[b] is None:
                    continue
                if self.similarity(embeddings[a], embeddings[b]) >= threshold:
                    union(a, b)

        groups_map = {}
        for tid in tids:
            root = find(tid)
            groups_map.setdefault(root, []).append(tid)
        groups = list(groups_map.values())

        dup_groups = [g for g in groups if len(g) > 1]
        for g in dup_groups:
            self._log(f"Duplicate Face Removal: Track IDs {g} appear to be the same person.")
        return groups, True
