import cv2
import numpy as np
import time
import threading
import os
from datetime import datetime
from ultralytics import YOLO

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration in case config.py is missing
    MODEL_PATH = "weights/best.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640
    CLASS_NAMES = {0: 'Gloves', 1: 'Helmet', 2: 'No-Glove', 3: 'No-Helmet', 4: 'No-Vest', 5: 'Person', 6: 'Vest'}
    PPE_POSITIVE = {0: 'Gloves', 1: 'Helmet', 6: 'Vest'}
    PPE_NEGATIVE = {2: 'No-Glove', 3: 'No-Helmet', 4: 'No-Vest'}
    VIOLATION_TO_PPE = {2: 'Gloves', 3: 'Helmet', 4: 'Vest'}
    PERSON_CLASS_ID = 5
    CLASS_COLORS = {
        0: (0, 255, 0), 1: (0, 255, 0), 6: (0, 255, 0),
        2: (0, 0, 255), 3: (0, 0, 255), 4: (0, 0, 255),
        5: (0, 165, 255)
    }
    FACES_DIR = "faces"
    FACE_CROP_PADDING = 20
    ALERT_COOLDOWN = 60


class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.helmet_status = 'unknown'
        self.vest_status = 'unknown'
        self.gloves_status = 'unknown'
        self.is_compliant = False
        self.last_seen = time.time()
        self.violations = []
        self.alert_cooldowns = {}
        self.face_path = None

    def to_dict(self):
        return {
            "id": self.track_id,
            "helmet": self.helmet_status,
            "vest": self.vest_status,
            "gloves": self.gloves_status,
            "is_compliant": self.is_compliant,
            "last_seen": self.last_seen,
            "violations_count": len(self.violations),
            "face_path": self.face_path
        }


class PPEDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.running = False
        self.cap = None
        self.person_states = {}  # {track_id: PersonState}
        self.frame_count = 0
        self.fps = 0
        self.current_frame = None  # latest annotated frame
        self.lock = threading.Lock()
        self._last_time = time.time()
        self._source = None  # track the source for is_video_file
        
        # Ensure faces directory exists
        os.makedirs(FACES_DIR, exist_ok=True)

    @property
    def is_video_file(self) -> bool:
        """Return True if the current source is a local video file (not webcam/RTSP)."""
        if self._source is None:
            return False
        if isinstance(self._source, int):
            return False
        s = str(self._source)
        if s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://"):
            return False
        return os.path.isfile(s)

    def start(self, source=0):
        """Start detection on given source (0=webcam, or path/url)"""
        self._source = source
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(source)
        self.running = True
        
    def stop(self):
        """Stop detection"""
        self.running = False
        if self.cap:
            self.cap.release()

    def reset(self):
        """Clear all tracking state for a fresh session."""
        self.person_states = {}
        self.frame_count = 0
        self.fps = 0
        self._last_time = time.time()
        with self.lock:
            self.current_frame = None

    def get_source_info(self) -> dict:
        """Return metadata about the current video source."""
        info = {
            "source": str(self._source) if self._source is not None else None,
            "is_video_file": self.is_video_file,
            "is_open": self.cap.isOpened() if self.cap else False,
            "frame_count": self.frame_count,
        }
        if self.cap and self.cap.isOpened():
            info["fps"] = self.cap.get(cv2.CAP_PROP_FPS) or 0
            info["total_frames"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if info["fps"] > 0 and info["total_frames"] > 0:
                info["duration_sec"] = round(info["total_frames"] / info["fps"], 1)
            else:
                info["duration_sec"] = 0
        return info
    def process_frame(self, frame=None, conf=None, imgsz=None, tracker='bytetrack.yaml', *args, **kwargs):
        """Read and process one frame with high-performance ByteTrack tracker.
        Returns (annotated_frame, detections_data) or (None, None) if no frame."""
        if frame is None:
            if not self.running or self.cap is None:
                return None, None
                
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self._last_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = self.fps * 0.85 + current_fps * 0.15  # Smooth FPS
        self._last_time = current_time
        self.frame_count += 1
            
        # Run high-performance YOLO tracking with adaptive sensitivity
        conf_to_use = conf if conf is not None else CONFIDENCE_THRESHOLD
        track_conf = min(conf_to_use, 0.20)  # Catch small objects like Gloves/No-Glove
        
        track_kwargs = {
            "persist": True,
            "tracker": tracker,
            "conf": track_conf,
            "iou": IOU_THRESHOLD,
            "verbose": False
        }
        if imgsz is not None:
            track_kwargs["imgsz"] = imgsz

        results = self.model.track(frame, **track_kwargs)
        
        annotated_frame = frame.copy()
        current_persons = []
        current_violations = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Extract track_ids, fallback to -1 if missing (can happen if tracker hasn't assigned yet)
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.full(len(boxes), -1)
                
            # Filter person detections by user confidence threshold
            person_indices = np.where((class_ids == PERSON_CLASS_ID) & (confidences >= conf_to_use))[0]
            
            # For PPE items, allow high-sensitivity detection for small hands/gloves
            item_mask = (class_ids != PERSON_CLASS_ID) & (
                ((class_ids == 0) | (class_ids == 2)) & (confidences >= min(conf_to_use, 0.20))  # Gloves / No-Glove
                | (~((class_ids == 0) | (class_ids == 2)) & (confidences >= min(conf_to_use, 0.35))) # Helmets, Vests
            )
            item_indices = np.where(item_mask)[0]
            
            # Draw all detected PPE items on frame
            for i_idx in item_indices:
                i_box = boxes[i_idx]
                i_cls = class_ids[i_idx]
                i_conf = confidences[i_idx]
                self._draw_box(annotated_frame, i_box, i_cls, i_conf)

            # Detect persons (with flexible threshold for webcam / close-up framing)
            person_indices = np.where((class_ids == PERSON_CLASS_ID) & (confidences >= min(conf_to_use, 0.25)))[0]
            
            # If no full-body Person detected but PPE items exist (e.g. close-up webcam), treat frame as Worker #1
            if len(person_indices) == 0 and len(item_indices) > 0:
                p_track_id = 1
                if p_track_id not in self.person_states:
                    self.person_states[p_track_id] = PersonState(p_track_id)
                state = self.person_states[p_track_id]
                state.last_seen = current_time

                h_status, v_status, g_status = 'unknown', 'unknown', 'unknown'
                for i_idx in item_indices:
                    i_cls = class_ids[i_idx]
                    if i_cls == 1: h_status = 'yes'
                    elif i_cls == 3: h_status = 'no'
                    elif i_cls == 6: v_status = 'yes'
                    elif i_cls == 4: v_status = 'no'
                    elif i_cls == 0: g_status = 'yes'
                    elif i_cls == 2: g_status = 'no'

                state.helmet_status = h_status
                state.vest_status = v_status
                state.gloves_status = g_status
                state.is_compliant = (h_status == 'yes' and v_status == 'yes' and g_status == 'yes')

                # Log violations for close-up worker
                for v_name, v_status_val in [("Helmet", h_status), ("Vest", v_status), ("Gloves", g_status)]:
                    if v_status_val == 'no':
                        last_alert = state.alert_cooldowns.get(v_name, 0)
                        if current_time - last_alert > ALERT_COOLDOWN:
                            state.alert_cooldowns[v_name] = current_time
                            fresh_face = self._crop_face(frame, [0, 0, frame.shape[1], frame.shape[0]], p_track_id)
                            if fresh_face:
                                state.face_path = fresh_face
                            violation_record = {
                                "person_id": p_track_id,
                                "type": f"No-{v_name}",
                                "time": datetime.now().isoformat(),
                                "face_path": state.face_path
                            }
                            state.violations.append(violation_record)
                            current_violations.append(violation_record)

                current_persons.append({
                    "id": p_track_id,
                    "helmet": h_status,
                    "vest": v_status,
                    "gloves": g_status,
                    "compliant": state.is_compliant
                })

            else:
                for p_idx in person_indices:
                    p_box = boxes[p_idx]
                    p_track_id = int(track_ids[p_idx])
                    if p_track_id == -1:
                        p_track_id = p_idx + 1  # Fallback immediate ID for unconfirmed tracker tracks
                    p_conf = float(confidences[p_idx])
                    
                    if p_track_id not in self.person_states:
                        self.person_states[p_track_id] = PersonState(p_track_id)
                    
                    state = self.person_states[p_track_id]
                    state.last_seen = current_time
                    
                    # Default status for this frame
                    h_status = 'unknown'
                    v_status = 'unknown'
                    g_status = 'unknown'
                    
                    # Check overlapping items
                    for i_idx in item_indices:
                        i_box = boxes[i_idx]
                        i_cls = class_ids[i_idx]
                        is_hand = (i_cls in (0, 2))  # Gloves or No-Glove
                        
                        if self._check_overlap(p_box, i_box, is_hand=is_hand):
                            # Update status based on class
                            if i_cls == 1: h_status = 'yes'     # Helmet
                            elif i_cls == 3: h_status = 'no'    # No-Helmet
                            
                            elif i_cls == 6: v_status = 'yes'   # Vest
                            elif i_cls == 4: v_status = 'no'    # No-Vest
                            
                            elif i_cls == 0: g_status = 'yes'   # Gloves
                            elif i_cls == 2: g_status = 'no'    # No-Glove

                    # Update the state
                    state.helmet_status = h_status
                    state.vest_status = v_status
                    state.gloves_status = g_status
                    state.is_compliant = (h_status == 'yes' and v_status == 'yes' and g_status == 'yes')
                    
                    # Check for violations
                    new_violations_this_frame = []
                    for v_name, v_status_val in [("Helmet", h_status), ("Vest", v_status), ("Gloves", g_status)]:
                        if v_status_val == 'no':
                            last_alert = state.alert_cooldowns.get(v_name, 0)
                            if current_time - last_alert > ALERT_COOLDOWN:
                                new_violations_this_frame.append(v_name)
                                state.alert_cooldowns[v_name] = current_time
                                
                                # Crop fresh face/upper body snapshot for the violation event
                                fresh_face = self._crop_face(frame, p_box, p_track_id)
                                if fresh_face:
                                    state.face_path = fresh_face
                                    
                                violation_record = {
                                    "person_id": p_track_id,
                                    "type": f"No-{v_name}",
                                    "time": datetime.now().isoformat(),
                                    "face_path": state.face_path
                                }
                                state.violations.append(violation_record)
                                current_violations.append(violation_record)

                    # Draw Person Box
                    self._draw_person_info(annotated_frame, p_box, p_track_id, state)
                    
                    # Add to return data
                    current_persons.append({
                        "id": p_track_id,
                        "helmet": h_status,
                        "vest": v_status,
                        "gloves": g_status,
                        "compliant": state.is_compliant
                    })

        # Cleanup stale tracks (not seen for > 5 seconds)
        to_delete = [tid for tid, state in self.person_states.items() if current_time - state.last_seen > 5]
        for tid in to_delete:
            del self.person_states[tid]

        # Draw Overlay Stats
        total_p = len(self.person_states)
        compliant_p = sum(1 for s in self.person_states.values() if s.is_compliant)
        violations_p = total_p - compliant_p
        rate = (compliant_p / total_p * 100) if total_p > 0 else 100.0
        
        # Collect raw detections for dashboard logger
        raw_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            for idx in range(len(boxes)):
                c_id = int(class_ids[idx])
                c_name = CLASS_NAMES.get(c_id, f"Class {c_id}")
                is_v = (c_id in PPE_NEGATIVE)
                c_conf = float(confidences[idx])
                b_coords = [float(x) for x in boxes[idx]]
                raw_detections.append({
                    "class": c_name,
                    "violation": is_v,
                    "confidence": c_conf,
                    "box": b_coords
                })

        # Prepare detection data
        detection_data = {
            "persons": current_persons,
            "violations": current_violations,
            "raw_detections": raw_detections,
            "stats": {
                "total": total_p,
                "compliant": compliant_p,
                "violations": violations_p,
                "rate": round(rate, 1),
                "fps": round(self.fps, 1)
            }
        }
            
        return annotated_frame, detection_data

    def _draw_box(self, frame, box, cls_id, conf):
        """Draw bounding box for PPE items"""
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {conf:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Text background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if sum(color)>400 else (255,255,255), 1)

    def _draw_person_info(self, frame, box, track_id, state):
        """Draw person bbox and PPE status indicators"""
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(PERSON_CLASS_ID, (0, 165, 255)) # Orange
        
        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        label = f"ID: {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw Status Icons
        status_text = ""
        status_text += "H: " + ("✅" if state.helmet_status == 'yes' else "❌" if state.helmet_status == 'no' else "❓") + " "
        status_text += "V: " + ("✅" if state.vest_status == 'yes' else "❌" if state.vest_status == 'no' else "❓") + " "
        status_text += "G: " + ("✅" if state.gloves_status == 'yes' else "❌" if state.gloves_status == 'no' else "❓")
        
        # Try to use ASCII/Unicode fallback for OpenCV text, or simple colors
        # OpenCV doesn't natively support emojis well, so we'll use colored text instead
        y_offset = max(0, y2 + 20)
        
        def put_colored_text(img, text, pos, txt_color):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3) # Outline
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2) # Text
            
        h_color = (0, 255, 0) if state.helmet_status == 'yes' else ((0, 0, 255) if state.helmet_status == 'no' else (128, 128, 128))
        v_color = (0, 255, 0) if state.vest_status == 'yes' else ((0, 0, 255) if state.vest_status == 'no' else (128, 128, 128))
        g_color = (0, 255, 0) if state.gloves_status == 'yes' else ((0, 0, 255) if state.gloves_status == 'no' else (128, 128, 128))
        
        put_colored_text(frame, "H", (x1, y_offset), h_color)
        put_colored_text(frame, "V", (x1 + 30, y_offset), v_color)
        put_colored_text(frame, "G", (x1 + 60, y_offset), g_color)

    def _draw_overlay(self, frame, total, compliant, violations, rate):
        """Draw professional top overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        stats_text = f"FPS: {self.fps:.1f} | Persons: {total} | Compliant: {compliant} | Violations: {violations} | Rate: {rate:.1f}%"
        cv2.putText(frame, stats_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1,y1,x2,y2]"""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
        return iou

    def _check_overlap(self, person_box, item_box, threshold=0.25, is_hand=False):
        """Check if item_box belongs to person_box.
        For hands/gloves, adds horizontal and vertical padding since hands often extend outside the person torso."""
        if is_hand:
            pw = person_box[2] - person_box[0]
            ph = person_box[3] - person_box[1]
            pad_x = pw * 0.25
            pad_y = ph * 0.10
            p_box = [
                person_box[0] - pad_x,
                person_box[1] - pad_y,
                person_box[2] + pad_x,
                person_box[3] + pad_y,
            ]
            check_thresh = 0.15
        else:
            p_box = person_box
            check_thresh = threshold

        xA = max(p_box[0], item_box[0])
        yA = max(p_box[1], item_box[1])
        xB = min(p_box[2], item_box[2])
        yB = min(p_box[3], item_box[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        itemArea = (item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
        
        if itemArea <= 0:
            return False
            
        ratio = interArea / float(itemArea)
        return ratio > check_thresh
        
    def _crop_face(self, frame, person_box, person_id):
        """Crop upper portion of person bbox as face/upper body snapshot, save to FACES_DIR and return absolute path"""
        try:
            x1, y1, x2, y2 = map(int, person_box)
            fh, fw = frame.shape[:2]
            
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(x1 + 1, min(fw, x2))
            y2 = max(y1 + 1, min(fh, y2))
            
            height = y2 - y1
            # For full standing person, take top 45%; for close-up/desk views, take top 70%
            ratio = 0.45 if height > 220 else 0.70
            face_y2 = int(y1 + height * ratio)
            
            pad = FACE_CROP_PADDING
            fy1 = max(0, y1 - pad)
            fy2 = min(fh, face_y2 + pad)
            fx1 = max(0, x1 - pad)
            fx2 = min(fw, x2 + pad)
            
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop is None or face_crop.size == 0:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    face_crop = frame
                
            os.makedirs(FACES_DIR, exist_ok=True)
            timestamp = int(time.time() * 1000)
            filename = f"person_{int(person_id)}_{timestamp}.jpg"
            filepath = os.path.abspath(os.path.join(FACES_DIR, filename))
            
            cv2.imwrite(filepath, face_crop)
            return filepath
        except Exception as e:
            print(f"[_crop_face error]: {e}")
            return None
        
    def get_frame_bytes(self):
        """Return current annotated frame as JPEG bytes for MJPEG stream"""
        with self.lock:
            if self.current_frame is None:
                return None
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            return buffer.tobytes() if ret else None
            
    def get_person_states(self):
        """Return current person tracking states as list of dicts"""
        return [state.to_dict() for state in self.person_states.values()]
        
    def get_live_stats(self):
        """Return real-time stats: total_persons, compliant, violations, compliance_rate, fps"""
        total = len(self.person_states)
        compliant = sum(1 for s in self.person_states.values() if s.is_compliant)
        violations = total - compliant
        rate = (compliant / total * 100) if total > 0 else 100.0
        
        return {
            "total_persons": total,
            "compliant": compliant,
            "violations": violations,
            "compliance_rate": round(rate, 1),
            "fps": round(self.fps, 1)
        }
