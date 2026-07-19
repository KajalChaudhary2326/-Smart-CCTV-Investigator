"""
report_generator.py
--------------------
Builds a professional PDF forensic report from the detection log produced
by TrackerEngine: case details, summary stats, full frame-by-frame table,
and evidence-frame thumbnails.
"""

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)


class ForensicReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="TitleCenter", parent=self.styles["Title"], alignment=1))

    def generate(self, summary, detections_log, case_info=None, max_thumbnails=15):
        doc = SimpleDocTemplate(self.output_path, pagesize=A4,
                                 topMargin=1.5 * cm, bottomMargin=1.5 * cm)
        story = []
        case_info = case_info or {}

        story.append(Paragraph("VIDEO FORENSICS &ndash; OBJECT TRACKING REPORT", self.styles["TitleCenter"]))
        story.append(Spacer(1, 14))

        case_rows = [
            ["Case Reference", case_info.get("case_ref", "N/A")],
            ["Investigating Officer", case_info.get("officer", "N/A")],
            ["Report Generated On", summary["processed_at"]],
            ["Source Video File", os.path.basename(summary["video_path"])],
        ]
        story.append(self._info_table(case_rows, [5 * cm, 11 * cm]))
        story.append(Spacer(1, 16))

        story.append(Paragraph("Detection Summary", self.styles["Heading2"]))
        summary_rows = [
            ["Target Object Class", summary["target_class"]],
            ["Tracking Mode", summary.get("tracking_mode", "All instances of class")],
            ["Total Frames Scanned", str(summary["total_frames_scanned"])],
            ["Video FPS", str(summary["fps"])],
            ["Total Detections (frame-level)", str(summary["total_detections"])],
            ["Unique Objects Tracked", str(summary["unique_objects_tracked"])],
            ["Track IDs", ", ".join(map(str, summary["track_ids"])) or "None"],
        ]
        if summary.get("locked_track_id") is not None:
            summary_rows.append(["Locked Object Track ID", str(summary["locked_track_id"])])
        video_enhancement = summary.get("video_enhancement") or {}
        if video_enhancement.get("applied"):
            summary_rows.append(["Pre-Processing Enhancement Applied", video_enhancement.get("description", "N/A")])
        story.append(self._info_table(summary_rows, [6 * cm, 10 * cm]))
        story.append(Spacer(1, 16))

        if summary.get("selection_match_failed"):
            story.append(Paragraph(
                "&#9888; SELECTION NOT MATCHED: The specific object you selected could not be "
                "confidently identified in the video, so nothing is reported below (to avoid "
                "reporting the wrong object). Please re-run with a clearer selection.",
                self.styles["Normal"]))
            story.append(Spacer(1, 12))

        identified_plates = summary.get("identified_plates") or {}
        if summary.get("plates_enabled"):
            story.append(Paragraph("Vehicle Number Plate Identification", self.styles["Heading2"]))
            story.append(Paragraph(
                "A zoomed image of each tracked vehicle's plate region is included below for "
                "manual reading. Where offline OCR could also auto-read the text, that reading "
                "is shown too - treat it as an investigative lead to verify against the image, "
                "not a confirmed, court-ready plate number.",
                self.styles["Normal"]))
            story.append(Spacer(1, 6))

            if identified_plates:
                plate_rows = [["Track ID", "Auto-Read Text", "Confidence", "Format Check", "Frame"]]
                for tid in sorted(identified_plates.keys()):
                    p = identified_plates[tid]
                    plate_rows.append([
                        str(tid),
                        p.get("text") or "(view image)",
                        f"{p['confidence']:.2f}" if p.get("confidence") is not None else "N/A",
                        ("Looks valid" if p.get("matches_format") else "Unverified") if p.get("text") else "N/A",
                        str(p.get("frame", "")),
                    ])
                story.append(self._log_table(plate_rows))
                story.append(Spacer(1, 10))

                for tid in sorted(identified_plates.keys()):
                    p = identified_plates[tid]
                    if p.get("image_path") and os.path.exists(p["image_path"]):
                        if p.get("text"):
                            caption = (f"Track ID {tid} - Auto-read: {p['text']} "
                                       f"(confidence {p['confidence']:.2f}) - zoom in to verify:")
                        else:
                            caption = (f"Track ID {tid} - Automatic reading unavailable. "
                                       "Zoom in on the image below to read the plate manually:")
                        story.append(Paragraph(caption, self.styles["Normal"]))
                        story.append(Image(p["image_path"], width=8 * cm, height=4 * cm))
                        story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(
                    "No vehicle of the selected class was tracked in this video, so no plate "
                    "images are available.",
                    self.styles["Normal"]))
            story.append(Spacer(1, 12))

        identified_faces = summary.get("identified_faces") or {}
        if summary.get("faces_enabled"):
            story.append(Paragraph("Face Identification", self.styles["Heading2"]))
            story.append(Paragraph(
                "A zoomed image is included below for each tracked person's clearest/closest "
                "appearance - a tightly cropped, auto-detected face where possible, or a "
                "zoomed head/shoulders view for manual identification when precise face "
                "detection wasn't confident. This is face DETECTION only (finds and zooms "
                "into a face region) - it does not match identity against any database.",
                self.styles["Normal"]))
            story.append(Spacer(1, 6))

            if summary.get("face_search_used"):
                match_count = summary.get("face_search_matches", 0)
                story.append(Paragraph(
                    f"<b>Face Search:</b> each tracked person's face was compared against a "
                    f"supplied reference photo. {match_count} possible match(es) found - marked "
                    f"below. This is a probabilistic similarity score, not a confirmed "
                    f"identification; verify manually.",
                    self.styles["Normal"]))
                story.append(Spacer(1, 6))

            dup_groups = [g for g in (summary.get("duplicate_face_groups") or []) if len(g) > 1]
            if summary.get("duplicate_removal_used"):
                if dup_groups:
                    dup_text = "; ".join("Track IDs " + ", ".join(str(t) for t in g) +
                                          " likely the same person" for g in dup_groups)
                    story.append(Paragraph(
                        f"<b>Duplicate Face Removal:</b> {dup_text}. (The tracker likely lost and "
                        f"re-acquired this individual mid-video, assigning a new Track ID.)",
                        self.styles["Normal"]))
                else:
                    story.append(Paragraph(
                        "<b>Duplicate Face Removal:</b> checked - no likely duplicate Track IDs found.",
                        self.styles["Normal"]))
                story.append(Spacer(1, 6))

            if identified_faces:
                for tid in sorted(identified_faces.keys()):
                    f = identified_faces[tid]
                    if f.get("image_path") and os.path.exists(f["image_path"]):
                        if f.get("auto_detected"):
                            caption = (f"Track ID {tid} - face auto-detected and zoomed, "
                                       f"from frame {f.get('frame', '')}:")
                        else:
                            caption = (f"Track ID {tid} - precise auto face-detection was "
                                       f"inconclusive; zoomed head/shoulders view from frame "
                                       f"{f.get('frame', '')} for manual identification:")
                        if f.get("reference_match"):
                            caption += (f" [POSSIBLE MATCH to reference photo - similarity "
                                        f"{f.get('reference_match_score', 0):.2f}]")
                        story.append(Paragraph(caption, self.styles["Normal"]))
                        story.append(Image(f["image_path"], width=5 * cm, height=5 * cm))
                        story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(
                    "No person of the selected class was tracked in this video, so no face "
                    "images are available.",
                    self.styles["Normal"]))
            story.append(Spacer(1, 12))

        story.append(Paragraph("Frame-by-Frame Detection Log", self.styles["Heading2"]))
        table_data = [["Frame #", "Timestamp", "Track ID", "Confidence", "BBox (x1,y1,x2,y2)"]]
        for d in detections_log:
            table_data.append([str(d["frame"]), d["timestamp"], str(d["track_id"]),
                                f"{d['confidence']:.2f}", str(d["bbox"])])
        story.append(self._log_table(table_data))
        story.append(PageBreak())

        story.append(Paragraph("Evidence Frame Thumbnails", self.styles["Heading2"]))
        story.append(Spacer(1, 8))
        thumb_entries = [d for d in detections_log if "frame_image" in d][:max_thumbnails]
        for d in thumb_entries:
            if os.path.exists(d["frame_image"]):
                caption = (f"Frame {d['frame']} | Time {d['timestamp']} | "
                           f"Track ID {d['track_id']} | Confidence {d['confidence']:.2f}")
                story.append(Paragraph(caption, self.styles["Normal"]))
                story.append(Image(d["frame_image"], width=12 * cm, height=6.75 * cm))
                story.append(Spacer(1, 10))

        if not thumb_entries:
            story.append(Paragraph("No object detections found in this video for the selected class.",
                                    self.styles["Normal"]))

        doc.build(story)
        return self.output_path

    def _info_table(self, rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ]))
        return t

    def _log_table(self, data):
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ]))
        return t
