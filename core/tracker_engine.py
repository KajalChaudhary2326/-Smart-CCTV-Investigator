"""
tracker_engine.py
-------------------
Core forensic engine: loads a YOLOv8 model, runs detection + multi-object
tracking (ByteTrack) on a video for ONE user-selected object class, saves an
evidence frame every time that object appears, and returns a structured
detection log used later to build the PDF report.

Supports two modes:
  1. Class-wide tracking  -> tracks every instance of the chosen class
     (e.g. every "person" in the video).
  2. Specific-instance tracking -> the user draws a box around ONE particular
     object on the first frame; that box is matched (by IoU) to a detected
     object's track ID, and only THAT object is tracked/reported from then on,
     even if other objects of the same class are also in frame.
"""

import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from core.plate_reader import PlateReader
from core.face_reader import FaceReader

# Vehicle classes (COCO) for which we attempt automatic number-plate reading.
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Class for which we attempt automatic face-region extraction.
PERSON_CLASS = "person"


class TrackerEngine:
    def __init__(self, model_path="yolov8n.pt", progress_callback=None, log_callback=None):
        """
        model_path: path/name of YOLOv8 weights. 'yolov8n.pt' auto-downloads
                    on first run (needs internet once).
        progress_callback(frame_idx, total_frames): optional, called per frame.
        log_callback(message): optional, called for status text.
        """
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # dict {id: name}
        self.plate_reader = PlateReader(log_callback=log_callback)
        self.face_reader = FaceReader(log_callback=log_callback)

    def get_class_list(self):
        return [self.class_names[i] for i in sorted(self.class_names.keys())]

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def get_preview_frame(self, video_path, frame_number=1):
        """
        Returns a specific frame of the video (BGR numpy array) so the GUI
        can display it and let the user drag-select one specific object.
        frame_number is 1-based (1 = first frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IOError(f"Could not read frame {frame_number} of this video.")
        return frame

    def process_video(self, video_path, target_class, output_dir, conf_threshold=0.4,
                       selection_bbox=None, selection_frame_idx=1):
        """
        Runs tracking for `target_class`. Saves evidence frames (with
        bounding boxes drawn) into output_dir/evidence_frames/.

        selection_bbox: optional [x1,y1,x2,y2] in ORIGINAL pixel coordinates
                         of the frame the user viewed when drawing the box.
        selection_frame_idx: which frame number (1-based) the selection box
                         was drawn on. The engine waits until it reaches this
                         frame, matches the box to a detected object there,
                         and tracks ONLY that specific object from that frame
                         onward (ignoring other objects of the same class).

        Returns: (summary_dict, detections_log_list)
        """
        target_id = self._resolve_class_id(target_class)
        if target_id is None:
            raise ValueError(f"Class '{target_class}' not found in model.")

        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "evidence_frames")
        os.makedirs(frames_dir, exist_ok=True)

        plates_enabled = target_class.lower() in VEHICLE_CLASSES
        plates_dir = os.path.join(output_dir, "plate_evidence")
        if plates_enabled:
            os.makedirs(plates_dir, exist_ok=True)
            self._log(f"'{target_class}' is a vehicle class - number-plate "
                       "recognition will run on each tracked vehicle.")

        faces_enabled = target_class.lower() == PERSON_CLASS
        faces_dir = os.path.join(output_dir, "face_evidence")
        if faces_enabled:
            os.makedirs(faces_dir, exist_ok=True)
            self._log("'person' class selected - face detection will run on "
                       "each tracked person.")

        fps, total_frames = self._probe_video(video_path)

        detections_log = []
        track_ids_seen = set()
        frame_idx = 0

        # Best plate reading seen so far, per track ID: {track_id: {...}}
        best_plates = {}
        last_ocr_attempt_frame = {}
        OCR_RETRY_EVERY_N_FRAMES = 5
        OCR_GOOD_ENOUGH_CONFIDENCE = 0.75

        # Best face image seen so far, per track ID: {track_id: {...}}
        best_faces = {}

        # How many frames after the user's selection point we keep trying to
        # match it to a tracked object, before giving up. A single-frame
        # attempt was too fragile (small timing/coordinate differences made
        # it fail and silently fall back to tracking EVERY object - wrong).
        SELECTION_MATCH_WINDOW_FRAMES = 30

        locked_track_id = None
        locked_resolved = selection_bbox is None  # no locking needed if no selection made

        results_stream = self.model.track(
            source=video_path,
            classes=[target_id],
            conf=conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
        )

        for result in results_stream:
            frame_idx += 1
            timestamp_sec = frame_idx / fps
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                if (selection_bbox is not None and not locked_resolved
                        and frame_idx >= selection_frame_idx):
                    match_id = self._match_selection(selection_bbox, boxes)
                    if match_id is not None:
                        locked_track_id = match_id
                        locked_resolved = True
                        self._log(f"Locked onto selected object (Track ID {locked_track_id}) "
                                  f"at frame {frame_idx}. Only this object will be tracked & reported.")
                    elif frame_idx >= selection_frame_idx + SELECTION_MATCH_WINDOW_FRAMES:
                        locked_resolved = True  # give up - could not confidently match
                        self._log(
                            f"Could not confidently match your selection to any tracked object "
                            f"within {SELECTION_MATCH_WINDOW_FRAMES} frames. No report will be "
                            "generated for other objects (to avoid reporting the wrong one). "
                            "Try again: pick a frame where your object is clearly visible and "
                            "separated from others, and draw the box closely around it.")
                    # else: still within the retry window - try again on the next frame

                # Skip logging until we've resolved which object to lock onto
                # (only relevant when a selection was made further into the video)
                if selection_bbox is not None and not locked_resolved:
                    if self.progress_callback and total_frames:
                        self.progress_callback(frame_idx, total_frames)
                    continue

                if selection_bbox is None:
                    relevant_boxes = list(boxes)
                elif locked_track_id is not None:
                    relevant_boxes = [b for b in boxes
                                       if b.id is not None and int(b.id[0]) == locked_track_id]
                else:
                    # A selection was made but never matched - report NOTHING
                    # rather than silently falling back to every object.
                    relevant_boxes = []

                if relevant_boxes:
                    frame_img = result.orig_img.copy()

                    for box in relevant_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        track_ids_seen.add(track_id)

                        plate_text, plate_conf = None, None
                        if plates_enabled:
                            vehicle_crop = result.orig_img[max(0, y1):y2, max(0, x1):x2]
                            box_area = max(0, x2 - x1) * max(0, y2 - y1)
                            current_best = best_plates.get(track_id)

                            # 1) ALWAYS keep a zoomed plate-region image for this vehicle,
                            #    even if OCR never manages to read it - so a human at the
                            #    cyber cell can zoom in and read it manually. We keep
                            #    replacing it whenever we see a meaningfully bigger/closer
                            #    view of the vehicle (usually = clearer plate).
                            is_clearer_view = (current_best is None
                                                or box_area > current_best.get("box_area", 0) * 1.15)
                            if is_clearer_view:
                                zoomed = self.plate_reader.locate_and_enhance(vehicle_crop)
                                if zoomed is not None:
                                    plate_img_path = os.path.join(plates_dir, f"plate_track{track_id}.jpg")
                                    cv2.imwrite(plate_img_path, zoomed)
                                    best_plates[track_id] = {
                                        "text": current_best["text"] if current_best else None,
                                        "confidence": current_best["confidence"] if current_best else None,
                                        "matches_format": current_best["matches_format"] if current_best else False,
                                        "frame": frame_idx,
                                        "image_path": plate_img_path,
                                        "box_area": box_area,
                                    }
                                    current_best = best_plates[track_id]

                            # 2) SEPARATELY, best-effort try to auto-read the text (bonus,
                            #    not required for the image to exist). Throttled so we
                            #    don't run OCR on every single frame.
                            already_confident = (current_best is not None
                                                  and current_best.get("confidence")
                                                  and current_best["confidence"] >= OCR_GOOD_ENOUGH_CONFIDENCE)
                            frames_since_last_try = frame_idx - last_ocr_attempt_frame.get(track_id, -999)
                            should_try_ocr = (not already_confident
                                               and frames_since_last_try >= OCR_RETRY_EVERY_N_FRAMES)

                            if should_try_ocr:
                                last_ocr_attempt_frame[track_id] = frame_idx
                                zoomed_for_ocr = self.plate_reader.locate_and_enhance(vehicle_crop)
                                ocr_result = (self.plate_reader.read_text(zoomed_for_ocr)
                                              if zoomed_for_ocr is not None else None)
                                if ocr_result is not None:
                                    plate_text = ocr_result["text"]
                                    plate_conf = ocr_result["confidence"]
                                    if track_id not in best_plates:
                                        # image wasn't saved yet for some reason - save now
                                        img_path = os.path.join(plates_dir, f"plate_track{track_id}.jpg")
                                        cv2.imwrite(img_path, zoomed_for_ocr)
                                        best_plates[track_id] = {"image_path": img_path, "frame": frame_idx,
                                                                  "box_area": box_area,
                                                                  "text": None, "confidence": None,
                                                                  "matches_format": False}
                                    existing = best_plates[track_id]
                                    if existing["confidence"] is None or plate_conf > existing["confidence"]:
                                        existing["text"] = plate_text
                                        existing["confidence"] = plate_conf
                                        existing["matches_format"] = ocr_result["matches_format"]
                                        self._log(f"Track ID {track_id}: possible plate reading "
                                                  f"'{plate_text}' (confidence {plate_conf:.2f}) at frame {frame_idx}.")

                        if faces_enabled:
                            person_crop = result.orig_img[max(0, y1):y2, max(0, x1):x2]
                            face_box_area = max(0, x2 - x1) * max(0, y2 - y1)
                            current_best_face = best_faces.get(track_id)

                            # Only worth re-doing if this view is meaningfully bigger/
                            # closer than our current best - keeps long videos fast.
                            worth_trying = (current_best_face is None
                                             or face_box_area > current_best_face.get("person_box_area", 0) * 1.15)
                            if worth_trying:
                                # 1) ALWAYS produce a zoomed head/shoulders image - this
                                #    guarantees the report is never empty for this person,
                                #    even if step 2 below can't pin down an exact face box
                                #    (which happens often on real CCTV-quality footage).
                                head_zoom = self.face_reader.locate_head_region(person_crop)
                                if head_zoom is not None:
                                    # 2) BONUS: try to detect a tight face box within it for
                                    #    an even more zoomed-in crop. Falls back to the head
                                    #    region above if this doesn't find anything.
                                    tight_face = self.face_reader.detect_face_box(head_zoom)
                                    image_to_save = tight_face["image"] if tight_face else head_zoom
                                    face_img_path = os.path.join(faces_dir, f"face_track{track_id}.jpg")
                                    cv2.imwrite(face_img_path, image_to_save)
                                    best_faces[track_id] = {
                                        "image_path": face_img_path,
                                        "person_box_area": face_box_area,
                                        "auto_detected": tight_face is not None,
                                        "frame": frame_idx,
                                    }
                                    if tight_face is not None:
                                        self._log(f"Track ID {track_id}: face auto-detected and "
                                                  f"zoomed (frame {frame_idx}).")

                        cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_img, f"ID:{track_id} {conf:.2f}", (x1, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                        detections_log.append({
                            "frame": frame_idx,
                            "timestamp": self._format_time(timestamp_sec),
                            "timestamp_sec": round(timestamp_sec, 2),
                            "track_id": track_id,
                            "confidence": round(conf, 3),
                            "bbox": [x1, y1, x2, y2],
                            "plate_text": plate_text,
                            "plate_confidence": plate_conf,
                        })

                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame_img)
                    detections_log[-1]["frame_image"] = frame_path

            if self.progress_callback and total_frames:
                self.progress_callback(frame_idx, total_frames)

        selection_match_failed = selection_bbox is not None and locked_track_id is None
        if selection_match_failed:
            tracking_mode = "FAILED - could not match your selection (no objects reported)"
        elif selection_bbox is not None and locked_track_id is not None:
            tracking_mode = "Specific object instance"
        else:
            tracking_mode = "All instances of class"

        summary = {
            "video_path": video_path,
            "target_class": target_class,
            "tracking_mode": tracking_mode,
            "locked_track_id": locked_track_id,
            "selection_match_failed": selection_match_failed,
            "total_frames_scanned": frame_idx,
            "fps": round(fps, 2),
            "total_detections": len(detections_log),
            "unique_objects_tracked": len(track_ids_seen),
            "track_ids": sorted(track_ids_seen),
            "frames_dir": frames_dir,
            "plates_enabled": plates_enabled,
            "identified_plates": best_plates,
            "faces_enabled": faces_enabled,
            "identified_faces": best_faces,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._log(f"Done. {len(detections_log)} detections, "
                   f"{len(track_ids_seen)} unique object(s) tracked.")
        if plates_enabled:
            plates_with_text = sum(1 for p in best_plates.values() if p.get("text"))
            self._log(f"Saved zoomed plate images for {len(best_plates)} vehicle(s); "
                       f"auto-OCR read text for {plates_with_text} of them.")
        if faces_enabled:
            auto_detected_count = sum(1 for f in best_faces.values() if f.get("auto_detected"))
            self._log(f"Saved face-region images for {len(best_faces)} of "
                       f"{len(track_ids_seen)} tracked person(s); precise face "
                       f"auto-detected for {auto_detected_count} of them (others "
                       "use a zoomed head/shoulders view for manual identification).")
        return summary, detections_log

    def get_video_info(self, video_path):
        """Returns (fps, total_frames) for a video - used by the GUI to size
        the frame-seek slider."""
        return self._probe_video(video_path)

    def _resolve_class_id(self, target_class):
        for cid, name in self.class_names.items():
            if name.lower() == target_class.lower():
                return cid
        return None

    @staticmethod
    def _match_selection(selection_bbox, boxes, min_iou=0.1):
        """Finds which detected box (by track ID) best overlaps the user's
        hand-drawn selection box, using Intersection-over-Union."""
        best_iou = 0.0
        best_id = None
        for box in boxes:
            if box.id is None:
                continue
            bxyxy = list(map(int, box.xyxy[0].tolist()))
            tid = int(box.id[0])
            iou = TrackerEngine._iou(selection_bbox, bxyxy)
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        return best_id if best_iou >= min_iou else None

    @staticmethod
    def _iou(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _probe_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, total

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"
