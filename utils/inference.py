"""
YOLO11s PPE-detection inference utilities.

Loads a trained Ultralytics YOLO model (best.pt) and runs inference on a
PIL image, returning an annotated image (PIL) plus a list of structured
detections that the Streamlit UI can render as chips / feed into stats.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Resolves to <project_root>/weights/best.pt regardless of current working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_WEIGHTS = os.path.join(_PROJECT_ROOT, "weights", "best.pt")
MODEL_PATH = _DEFAULT_WEIGHTS if os.path.exists(_DEFAULT_WEIGHTS) else os.path.join(_PROJECT_ROOT, "best.pt")

# Classes that count as a VIOLATION (missing PPE) vs COMPLIANT (PPE present).
VIOLATION_CLASSES = {
    "no helmet", "no-helmet", "nohelmet",
    "no vest", "no-vest", "novest",
    "no gloves", "no-gloves", "nogloves",
    "no glove", "no-glove", "noglove",
}

# Display colors (BGR-ish doesn't matter here, we draw with PIL in RGB hex)
COLOR_COMPLIANT = (34, 197, 94)   # green  #22c55e
COLOR_VIOLATION = (239, 68, 68)   # red    #ef4444


@st.cache_resource(show_spinner=False)
def load_model(model_path: str = MODEL_PATH):
    """Load and cache the YOLO model. Returns None if the weights file
    is missing so the UI can show a friendly warning instead of crashing."""
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def _is_violation(class_name: str) -> bool:
    c = class_name.strip().lower()
    return (
        c in VIOLATION_CLASSES
        or c.startswith("no-")
        or c.startswith("no ")
        or (c.startswith("no") and c not in ["none", "normal"])
    )


def _get_font(size: int = 15):
    from PIL import ImageFont
    font_candidates = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def run_inference(image: Image.Image, model, conf_threshold: float = 0.4):
    """
    Run YOLO inference on a PIL image.

    Returns:
        annotated_image (PIL.Image): image with boxes/labels drawn,
            styled to match the dashboard (green = compliant, red = violation).
        detections (list[dict]): [{"class": str, "confidence": float,
            "violation": bool, "box": [x1,y1,x2,y2]}, ...]
    """
    from PIL import ImageDraw, ImageFont

    # Ensure RGB in original resolution
    rgb_image = image.convert("RGB")
    img_array = np.array(rgb_image)

    # Predict with higher sensitivity to detect small items like gloves
    base_conf = min(conf_threshold, 0.20)
    results = model.predict(img_array, conf=base_conf, verbose=False)
    result = results[0]

    annotated = rgb_image.copy()
    draw = ImageDraw.Draw(annotated)

    font = _get_font(15)

    detections = []
    names = result.names  # {class_id: class_name}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            # Adaptive confidence threshold: keep gloves/no-gloves sensitive
            is_glove_cls = ("glove" in class_name.lower())
            min_req_conf = min(conf_threshold, 0.20) if is_glove_cls else conf_threshold
            if confidence < min_req_conf:
                continue

            violation = _is_violation(class_name)
            color = COLOR_VIOLATION if violation else COLOR_COMPLIANT

            # box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # label chip (background + text), matching the reference image's
            # small pill labels above each detected item
            label = f"{class_name}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            pad_x, pad_y = 8, 4
            label_bg = [
                x1,
                max(0, y1 - text_h - 2 * pad_y),
                x1 + text_w + 2 * pad_x,
                y1,
            ]
            draw.rectangle(label_bg, fill=color)
            draw.text(
                (label_bg[0] + pad_x, label_bg[1] + pad_y // 2),
                label,
                fill="white",
                font=font,
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "violation": violation,
                    "box": [x1, y1, x2, y2],
                }
            )

    return annotated, detections