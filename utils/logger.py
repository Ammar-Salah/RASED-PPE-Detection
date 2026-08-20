"""
Detection event logging + aggregation for the PPE dashboard.

Every processed frame writes:
  - one row PER INDIVIDUAL DETECTION to DETECTIONS_LOG_PATH
    (drives the Violations donut, PPE Overview rings, and the
    Total Detections / Compliance Rate / Violations stat cards)
  - one row PER FRAME to EVENTS_LOG_PATH
    (drives the Recent Detections list)

Both are plain CSV files sitting next to app.py, so history survives
app restarts — this is intentionally simple (no DB) so it's easy to
swap for SQLite/Postgres later without touching the rest of the app;
every function here just needs a list-of-dicts in, list-of-dicts out.
"""

import os
import csv
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTIONS_LOG_PATH = os.path.join(PROJECT_ROOT, "detection_log.csv")
EVENTS_LOG_PATH = os.path.join(PROJECT_ROOT, "events_log.csv")

DETECTIONS_HEADER = ["timestamp", "frame_id", "camera", "class", "violation", "confidence"]
EVENTS_HEADER = ["timestamp", "frame_id", "camera", "title", "ok"]

# ----------------------------------------------------------------------------
# PPE item name normalization
# ----------------------------------------------------------------------------
# Maps a raw class name (either its compliant form OR its "no-X" violation
# form) to the canonical PPE item label used by the "PPE Compliance
# Overview" rings, so "Helmet" and "No Helmet" both roll up into one
# Helmet ring. Add entries here if your data.yaml uses different wording.
_ITEM_ALIASES = {
    "helmet": "Helmet", "hardhat": "Helmet", "hard hat": "Helmet",
    "mask": "Mask", "facemask": "Mask", "face mask": "Mask",
    "vest": "Vest", "safety vest": "Vest", "safetyvest": "Vest",
    "gloves": "Gloves", "glove": "Gloves",
    "shoes": "Safety Shoes", "shoe": "Safety Shoes",
    "boots": "Safety Shoes", "boot": "Safety Shoes",
}
_ITEM_ICONS = {
    "Helmet": "⛑️", "Mask": "😷", "Vest": "🦺", "Gloves": "🧤", "Safety Shoes": "🥾",
}
_ITEM_ORDER = ["Helmet", "Mask", "Vest", "Gloves", "Safety Shoes"]


def normalize_item(class_name: str) -> str:
    """'No Helmet' -> 'Helmet', 'no-mask' -> 'Mask', 'Vest' -> 'Vest', ..."""
    c = class_name.strip().lower()
    if c.startswith("no-"):
        c = c[3:].strip()
    elif c.startswith("no ") :
        c = c[3:].strip()
    elif c.startswith("no") and c != "no":
        c = c[2:].lstrip("-").strip()
    return _ITEM_ALIASES.get(c, c.title() or class_name)


import threading
import time

_file_lock = threading.Lock()


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def _ensure_headers(path: str, header: list) -> None:
    if not os.path.exists(path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)
        except Exception:
            pass


def _safe_append_rows(path: str, header: list, rows: list, max_retries: int = 5) -> bool:
    if not rows:
        return True
    with _file_lock:
        _ensure_headers(path, header)
        for attempt in range(max_retries):
            try:
                with open(path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                return True
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.08 * (attempt + 1))
                else:
                    print(f"[Logger Warning] Permission denied appending to {path}. File may be locked by another application.")
                    return False
            except Exception as e:
                print(f"[Logger Warning] Error writing to {path}: {e}")
                return False
    return False


def log_frame(camera: str, detections: list) -> None:
    """Append one processed frame's detections + a summary event row."""
    log_frames_batch(camera, [detections])


def log_frames_batch(camera: str, frames_detections: list) -> None:
    """Batch-append multiple frames of detections + summary event rows in a single file operation."""
    if not frames_detections:
        return

    det_rows = []
    event_rows = []

    for detections in frames_detections:
        ts = datetime.now().isoformat(timespec="seconds")
        frame_id = uuid.uuid4().hex[:8]

        for d in detections:
            det_rows.append(
                [ts, frame_id, camera, d["class"], d["violation"], f"{d['confidence']:.4f}"]
            )

        violations_in_frame = [d for d in detections if d.get("violation")]
        if violations_in_frame:
            title = f"{violations_in_frame[0]['class'].title()} Detected"
            ok = False
        elif detections:
            title = "All PPE Compliant"
            ok = True
        else:
            title = "No Objects Detected"
            ok = True

        event_rows.append([ts, frame_id, camera, title, ok])

    _safe_append_rows(DETECTIONS_LOG_PATH, DETECTIONS_HEADER, det_rows)
    _safe_append_rows(EVENTS_LOG_PATH, EVENTS_HEADER, event_rows)


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def _read_csv(path: str, max_retries: int = 3) -> list:
    if not os.path.exists(path):
        return []
    with _file_lock:
        for attempt in range(max_retries):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))
                else:
                    print(f"[Logger Warning] Permission denied reading {path}. File may be locked by another application.")
                    return []
            except Exception as e:
                print(f"[Logger Warning] Error reading {path}: {e}")
                return []
    return []


def load_detections() -> list:
    return _read_csv(DETECTIONS_LOG_PATH)


def load_events() -> list:
    return _read_csv(EVENTS_LOG_PATH)


# ----------------------------------------------------------------------------
# Aggregations consumed directly by app.py's render_* functions
# ----------------------------------------------------------------------------
def _is_violation(val) -> bool:
    """Helper to check if a violation field is true (supports bool and strings like 'True', 'true', '1')."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ["true", "1", "yes"]


def compute_stats(detections: list) -> dict:
    ppe_detections = [
        d for d in detections
        if d.get("class", "").strip().lower() not in ["person", "people", "human", "worker", "workers"]
    ]
    person_detections = [
        d for d in detections
        if d.get("class", "").strip().lower() in ["person", "people", "human", "worker", "workers"]
    ]

    total = len(ppe_detections)
    violations = sum(1 for d in ppe_detections if _is_violation(d.get("violation")))
    compliant = total - violations
    compliance_rate = (compliant / total * 100) if total else 0.0

    # If model has dedicated person detections, count them; else count unique frames
    if person_detections:
        workers_scanned = len(person_detections)
        workers_delta = "Workers detected"
    else:
        workers_scanned = len({d["frame_id"] for d in detections if "frame_id" in d})
        workers_delta = "Frames scanned"

    return {
        "total_detections": f"{total:,}",
        "total_detections_delta": "Since launch",
        "compliance_rate": f"{compliance_rate:.1f}%",
        "compliance_rate_delta": "Since launch",
        "workers_scanned": f"{workers_scanned:,}",
        "workers_scanned_delta": workers_delta,
        "violations": f"{violations:,}",
        "violations_delta": "Since launch",
    }


import base64


def get_image_base64(image_path: str) -> str:
    """Read an image from disk and return as base64 data URI."""
    if not image_path:
        return None
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ["jpg", "jpeg"] else ("png" if ext == "png" else "jpeg")
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


def compute_violations_breakdown(detections: list) -> dict:
    counts = defaultdict(int)
    for d in detections:
        if _is_violation(d.get("violation")):
            counts[d["class"].title()] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def compute_hourly_trend(detections: list, hours: int = 24):
    """
    Compute hourly compliance percentage for the last N hours.
    Returns (labels, values) where labels are formatted hour strings (e.g. '12 AM', '1 AM')
    and values are compliance percentages (0-100).
    """
    by_hour = defaultdict(lambda: [0, 0])  # 'YYYY-MM-DD-HH' -> [compliant, total]

    # Process detections from CSV
    for d in detections:
        raw_class = d.get("class", "").strip().lower()
        if raw_class in ["person", "people", "human", "worker", "workers"]:
            continue
        ts = d.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            h_key = dt.strftime("%Y-%m-%d-%H")
            by_hour[h_key][1] += 1
            if not _is_violation(d.get("violation")):
                by_hour[h_key][0] += 1
        except Exception:
            pass

    # Also check SQLite database detections
    try:
        import database
        db_detections = database.get_recent_detections(limit=1000)
        for r in db_detections:
            ts = r.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    h_key = dt.strftime("%Y-%m-%d-%H")
                    by_hour[h_key][1] += 1
                    if r.get("is_compliant"):
                        by_hour[h_key][0] += 1
                except Exception:
                    pass
    except Exception:
        pass

    now = datetime.now()
    hour_list = [now - timedelta(hours=i) for i in range(hours - 1, -1, -1)]
    labels = [h.strftime("%I %p").lstrip("0") for h in hour_list]
    values = []

    for h in hour_list:
        h_key = h.strftime("%Y-%m-%d-%H")
        compliant, total = by_hour.get(h_key, [0, 0])
        if total > 0:
            values.append(round(compliant / total * 100, 1))
        else:
            values.append(100.0 if (detections or by_hour) else 0.0)

    return labels, values


def compute_trend(detections: list, hours: int = 24):
    """Compliance trend by hours."""
    return compute_hourly_trend(detections, hours=hours)


def compute_ppe_overview(detections: list) -> list:
    by_item = defaultdict(lambda: [0, 0])  # item -> [present_count, total_count]
    for d in detections:
        raw_class = d.get("class", "").strip().lower()
        if raw_class in ["person", "people", "human", "worker", "workers"]:
            continue
        item = normalize_item(d["class"])
        by_item[item][1] += 1
        if not _is_violation(d.get("violation")):
            by_item[item][0] += 1

    ordered_labels = [l for l in _ITEM_ORDER if l in by_item]
    ordered_labels += [l for l in by_item if l not in _ITEM_ORDER]  # custom class names

    items = []
    for label in ordered_labels:
        present, total = by_item[label]
        pct = (present / total * 100) if total else 0.0
        items.append(
            {"icon": _ITEM_ICONS.get(label, "🦺"), "label": label, "count": present, "total": total, "pct": pct}
        )
    return items


def compute_recent_events(n: int = 6) -> list:
    """
    Returns recent violation and detection events with worker screenshot thumbnail (base64).
    Pulls from SQLite alerts (which store violator face/body crops) and events_log.csv.
    """
    out = []
    seen_ids = set()

    # 1. Pull from alerts table (violator screenshots)
    try:
        import database
        alerts = database.get_recent_alerts(limit=n * 3)
        faces_dir = os.path.join(PROJECT_ROOT, "static", "faces")
        all_face_files = []
        if os.path.exists(faces_dir):
            all_face_files = [
                os.path.join(faces_dir, f) for f in os.listdir(faces_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            all_face_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for a in alerts:
            alert_id = a.get("id")
            if alert_id in seen_ids:
                continue
            seen_ids.add(alert_id)

            p_id = a.get("person_id", 1)
            if isinstance(p_id, bytes):
                try:
                    p_id = int.from_bytes(p_id, "little")
                except Exception:
                    p_id = 1

            v_type = str(a.get("violation_type", "Violation"))
            img_path = a.get("face_image_path")

            # Check if file exists on disk, otherwise find closest matching violator crop
            if not img_path or not os.path.exists(img_path):
                matching = [f for f in all_face_files if f"person_{p_id}_" in f]
                if matching:
                    img_path = matching[0]
                elif all_face_files:
                    # fallback to most recent screenshot
                    img_path = all_face_files[0]

            img_b64 = get_image_base64(img_path) if img_path else None

            ts_str = str(a.get("timestamp", ""))
            try:
                dt = datetime.fromisoformat(ts_str)
                formatted_time = dt.strftime("%I:%M %p").lstrip("0")
            except Exception:
                formatted_time = ts_str[:16] if ts_str else "Just now"

            out.append({
                "id": alert_id,
                "person_id": p_id,
                "title": f"Worker #{p_id} — {v_type.replace('_', ' ').title()}",
                "camera": "Live Feed",
                "time": formatted_time,
                "ok": False,
                "image_path": img_path,
                "image_b64": img_b64,
                "icon": "⚠️",
                "bg": "#3b1219",
            })
            if len(out) >= n:
                break
    except Exception as e:
        print(f"[Logger Warning] Error reading alerts: {e}")

    # 2. If we need more items, supplement with compliant events from events_log.csv
    if len(out) < n:
        events = list(reversed(load_events()))
        for e in events:
            ok = (e.get("ok") == "True" or e.get("ok") is True)
            if ok:
                ts_str = str(e.get("timestamp", ""))
                try:
                    formatted_time = datetime.fromisoformat(ts_str).strftime("%I:%M %p").lstrip("0")
                except Exception:
                    formatted_time = ts_str
                out.append({
                    "id": e.get("frame_id", ""),
                    "person_id": None,
                    "title": e.get("title", "All PPE Compliant"),
                    "camera": e.get("camera", "Camera 01"),
                    "time": formatted_time,
                    "ok": True,
                    "image_path": None,
                    "image_b64": None,
                    "icon": "🛡️",
                    "bg": "#143525",
                })
                if len(out) >= n:
                    break

    return out[:n]