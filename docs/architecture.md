# System Architecture

## Overview

RASED AI is a modular PPE (Personal Protective Equipment) detection system
built on YOLOv11s with a Streamlit dashboard for real-time monitoring.

## Components

| Module | File | Purpose |
|--------|------|---------|
| Dashboard | `app.py` | Streamlit web UI — live feed, statistics, alerts |
| Detector | `detector.py` | YOLO + ByteTrack multi-person tracking engine |
| Inference | `utils/inference.py` | Single-image inference for uploaded photos |
| Logger | `utils/logger.py` | CSV-based detection logging and aggregation |
| Database | `database.py` | SQLite persistence for detections, alerts, sessions |
| Config | `config.py` | Central configuration (thresholds, classes, colors) |
| Telegram | `telegram_bot.py` | Asynchronous violation alert bot |

## Data Flow

```
Video Source (Webcam / RTSP / File)
        │
        ▼
   PPEDetector (detector.py)
   ├── YOLO11s inference + ByteTrack tracking
   ├── Person ↔ PPE item association
   └── Violation detection
        │
        ├──▶ Logger (CSV)     → Dashboard stats & charts
        ├──▶ Database (SQLite) → Alert history & sessions
        ├──▶ Telegram Bot     → Real-time violation alerts
        └──▶ Streamlit UI     → Live annotated feed + analytics
```

## Class Mapping

| ID | Class | Type |
|----|-------|------|
| 0 | Gloves | ✅ PPE Present |
| 1 | Helmet | ✅ PPE Present |
| 2 | No-Glove | ❌ Violation |
| 3 | No-Helmet | ❌ Violation |
| 4 | No-Vest | ❌ Violation |
| 5 | Person | 👤 Person |
| 6 | Vest | ✅ PPE Present |
