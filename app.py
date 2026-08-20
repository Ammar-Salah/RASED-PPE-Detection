import os
import tempfile
import subprocess
import base64
import time as _time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2
import random
from datetime import datetime
from PIL import Image

from utils.inference import load_model, run_inference, MODEL_PATH
import importlib
from utils import logger
importlib.reload(logger)
import detector as _detector_module
importlib.reload(_detector_module)
from detector import PPEDetector
import config
import database
importlib.reload(database)
from telegram_bot import TelegramAlertBot

# Initialize database tables
database.init_db()

# Assets
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(_PROJECT_ROOT, "logo.png")
BOTTOM_PATH = os.path.join(_PROJECT_ROOT, "bottom.png")


def _image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64 data URI string."""
    if not image_path or not os.path.exists(image_path):
        return ""
    try:
        ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
        if ext == "jpg":
            ext = "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ""

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RASED — Real-time AI Safety Equipment Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL CSS (dark dashboard theme matching the reference design)
# ----------------------------------------------------------------------------
def inject_css():
    current_theme = st.session_state.get("app_theme", "dark")
    
    light_theme_css = ""
    if current_theme == "light":
        light_theme_css = """
            /* ---- Light Theme Styles ---- */
            html, body, .stApp {
                background-color: #f1f5f9 !important;
                color: #0f172a !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-right: 1px solid #e2e8f0 !important;
            }
            .sidebar-brand-title {
                color: #0f172a !important;
            }
            .sidebar-tip-card {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
            }
            .sidebar-tip-header {
                color: #0f172a !important;
            }
            .sidebar-tip-body {
                color: #64748b !important;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] label {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
                color: #334155 !important;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background: #edf2f7 !important;
                border-color: #cbd5e1 !important;
            }
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"],
            button[aria-label="Expand sidebar"],
            button[aria-label="Collapse sidebar"] {
                background-color: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            }
            [data-testid="collapsedControl"] svg,
            [data-testid="stSidebarCollapsedControl"] svg,
            button[aria-label="Expand sidebar"] svg,
            button[aria-label="Collapse sidebar"] svg {
                fill: #0f172a !important;
                color: #0f172a !important;
            }
            .stApp [data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #ffffff !important;
                border-color: #e2e8f0 !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            }
            .stat-card {
                background: #ffffff !important;
                border-color: #e2e8f0 !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            }
            .stat-value, .side-panel-title, .tg-title, .ppe-overview-title, .recent-title-row {
                color: #0f172a !important;
            }
            .stat-label, .pill-date, .status-label, .recent-camera, .recent-time, .tg-card-meta, .ppe-ring-frac {
                color: #64748b !important;
            }
            .pill, .status-pill {
                background: #ffffff !important;
                border-color: #e2e8f0 !important;
            }
            .pill-time {
                color: #0f172a !important;
            }
            .recent-item {
                border-bottom-color: #f1f5f9 !important;
            }
            .recent-thumb {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
            }
            .tg-card {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
            }
            .tg-card-detail, .tracker-table td {
                color: #334155 !important;
            }
            .view-all-btn {
                background: #f1f5f9 !important;
                color: #334155 !important;
                border-color: #cbd5e1 !important;
            }
            .ppe-ring-label {
                color: #334155 !important;
            }
            .stream-stats-bar {
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
            }
            .stream-stat-item {
                color: #475569 !important;
            }
            .stream-stat-value {
                color: #0f172a !important;
            }
            .tracker-table th {
                background: #f1f5f9 !important;
                color: #475569 !important;
                border-bottom-color: #cbd5e1 !important;
            }
            .stream-placeholder {
                background: #f8fafc !important;
                border-color: #cbd5e1 !important;
                color: #64748b !important;
            }
        """

    base_css = """
            /* ---- hide default streamlit chrome ---- */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stToolbar"] {display: none !important;}

            header[data-testid="stHeader"] {
                background: transparent !important;
                z-index: 99999 !important;
            }

            /* ---- Hide accidental collapse button inside sidebar ---- */
            [data-testid="stSidebarCollapseButton"],
            button[aria-label="Close sidebar"],
            button[aria-label="Collapse sidebar"],
            div[data-testid="stSidebarCollapseButton"] {
                display: none !important;
                visibility: hidden !important;
            }

            /* ---- Permanent Pinned Sidebar (Always Visible & Open) ---- */
            section[data-testid="stSidebar"],
            div[data-testid="stSidebar"] {
                display: block !important;
                visibility: visible !important;
                transform: none !important;
                min-width: 290px !important;
                max-width: 320px !important;
                width: 300px !important;
                background-color: #0c101d !important;
                border-right: 1px solid #1a2038 !important;
                opacity: 1 !important;
            }
            section[data-testid="stSidebar"] > div:first-child {
                padding-top: 1.2rem !important;
                padding-bottom: 1.5rem !important;
                padding-left: 0.9rem !important;
                padding-right: 0.9rem !important;
            }
            .sidebar-brand {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 0.5rem 0.2rem 1.4rem 0.2rem;
                border-bottom: 1px solid #1a2038;
                margin-bottom: 1.3rem;
            }
            .sidebar-logo-img {
                width: 96px;
                height: 96px;
                object-fit: contain;
                margin-bottom: 0.75rem;
                filter: drop-shadow(0 4px 14px rgba(0,0,0,0.4));
            }
            .sidebar-brand-title {
                font-size: 1.45rem;
                font-weight: 800;
                color: #f5f7fa;
                letter-spacing: 0.5px;
                line-height: 1.15;
            }
            .sidebar-brand-title span {
                color: #f59e0b;
            }
            .sidebar-brand-subtitle {
                font-size: 0.76rem;
                color: #8b93a7;
                margin-top: 4px;
                font-weight: 500;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] {
                display: flex;
                flex-direction: column;
                gap: 0.45rem;
                margin-bottom: 0.8rem;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] label {
                background: #11162a;
                border: 1px solid #1f2540;
                border-radius: 12px;
                padding: 0.65rem 0.95rem !important;
                color: #c3c8d6 !important;
                font-weight: 600;
                font-size: 0.88rem;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background: #18203d;
                border-color: #3b4670;
                color: #ffffff !important;
            }
            div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
            div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, #7c6ff0, #5b4fd6) !important;
                border-color: #7c6ff0 !important;
                color: #ffffff !important;
                box-shadow: 0 4px 14px rgba(124, 111, 240, 0.35);
            }
            .sidebar-bottom-wrap {
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 1.2rem;
                margin-bottom: 0.6rem;
                overflow: hidden;
                border-radius: 14px;
            }
            .sidebar-worker-img {
                width: 100%;
                height: auto;
                object-fit: cover;
                border-radius: 14px;
                display: block;
                filter: drop-shadow(0 6px 16px rgba(0,0,0,0.45));
            }
            .sidebar-tip-card {
                background: #11162a;
                border: 1px solid #1f2540;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin-top: 0.4rem;
                position: relative;
                z-index: 2;
            }
            .sidebar-tip-header {
                font-size: 0.84rem;
                font-weight: 700;
                color: #f5f7fa;
                margin-bottom: 0.3rem;
            }
            .sidebar-tip-body {
                font-size: 0.74rem;
                color: #9aa1b4;
                line-height: 1.38;
            }
            .sidebar-tip-dots {
                display: flex;
                justify-content: center;
                gap: 6px;
                margin-top: 0.7rem;
            }
            .sidebar-tip-dots .dot {
                width: 6px;
                height: 6px;
                border-radius: 999px;
                background: #2a3352;
            }
            .sidebar-tip-dots .dot.active {
                background: #7c6ff0;
                width: 16px;
                border-radius: 999px;
            }

            /* ---- base page ---- */
            html, body, .stApp {
                background-color: #0a0d16;
                color: #e5e7eb;
                font-family: 'Segoe UI', Inter, system-ui, -apple-system, sans-serif;
            }}
            .block-container {{
                padding-top: 1.6rem;
                padding-bottom: 3rem;
                max-width: 1500px;
            }}

            /* ---- header ---- */
            .ppe-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
                margin-bottom: 1.6rem;
            }}
            .ppe-header h1 {{
                font-size: 1.6rem;
                font-weight: 700;
                color: #f5f7fa;
                margin: 0 0 2px 0;
            }}
            .ppe-header p {{
                color: #8b93a7;
                margin: 0;
                font-size: 0.92rem;
            }}
            .ppe-header-right {{
                display: flex;
                align-items: center;
                gap: 0.7rem;
            }}
            .pill {{
                background: #131826;
                border: 1px solid #212739;
                border-radius: 12px;
                padding: 0.5rem 1rem;
                text-align: center;
            }}
            .pill-time {{
                font-weight: 600;
                font-size: 0.88rem;
                color: #f0f2f5;
                line-height: 1.2;
            }}
            .pill-date {{
                font-size: 0.72rem;
                color: #8b93a7;
                line-height: 1.2;
            }}
            .status-pill {{
                background: #131826;
                border: 1px solid #212739;
                border-radius: 12px;
                padding: 0.5rem 1rem;
            }}
            .status-label {
                font-size: 0.72rem;
                color: #8b93a7;
                margin-bottom: 2px;
            }
            .status-value {
                font-weight: 600;
                font-size: 0.88rem;
                color: #22c55e;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: #22c55e;
                box-shadow: 0 0 6px #22c55e;
                display: inline-block;
            }

            /* ---- stat cards ---- */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1.1rem;
                margin-bottom: 1.6rem;
            }
            .stat-card {
                background: #11162a;
                border: 1px solid #1f2540;
                border-radius: 16px;
                padding: 1.1rem 1.3rem;
                display: flex;
                align-items: flex-start;
                gap: 0.9rem;
            }
            .stat-icon {
                width: 42px;
                height: 42px;
                min-width: 42px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
            }
            .stat-icon.purple { background: linear-gradient(135deg,#8b7ef8,#6355e0); }
            .stat-icon.green  { background: linear-gradient(135deg,#34d399,#16a34a); }
            .stat-icon.blue   { background: linear-gradient(135deg,#60a5fa,#2563eb); }
            .stat-icon.red    { background: linear-gradient(135deg,#f87171,#dc2626); }

            .stat-label {
                font-size: 0.8rem;
                color: #8b93a7;
                margin-bottom: 4px;
            }
            .stat-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: #f5f7fa;
                line-height: 1.1;
            }
            .stat-delta {
                font-size: 0.76rem;
                margin-top: 4px;
            }
            .stat-delta.positive { color: #22c55e; }
            .stat-delta.negative { color: #22c55e; } /* fewer violations is also "good" (green) */

            @media (max-width: 1100px) {
                .stats-grid { grid-template-columns: repeat(2, 1fr); }
            }

            /* ---- generic bordered "card" containers (st.container(border=True)) ---- */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #11162a;
                border: 1px solid #1f2540 !important;
                border-radius: 16px !important;
            }

            /* ---- live detection panel header ---- */
            .panel-title-row {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                margin-bottom: 0.2rem;
            }
            .panel-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #f5f7fa;
                margin: 0;
            }
            .live-badge {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 0.78rem;
                color: #22c55e;
                font-weight: 600;
            }
            .live-dot {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: #22c55e;
                box-shadow: 0 0 6px #22c55e;
                animation: pulse 1.6s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.35; }
                100% { opacity: 1; }
            }

            /* ---- empty state placeholder inside the detection panel ---- */
            .detection-placeholder {
                border: 1.5px dashed #2a3050;
                border-radius: 14px;
                padding: 4.5rem 1rem;
                text-align: center;
                color: #6b7280;
                background: #0d1120;
            }
            .detection-placeholder .icon {
                font-size: 2.2rem;
                margin-bottom: 0.6rem;
            }

            /* ---- style native streamlit widgets to match dark theme ---- */
            div[data-testid="stFileUploader"] section {
                background: #0d1120;
                border: 1.5px dashed #2a3050;
                border-radius: 12px;
            }
            div[data-baseweb="select"] > div {
                background: #131826 !important;
                border-color: #212739 !important;
                border-radius: 10px !important;
            }
            .stButton button {
                background: #6355e0;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
            }
            .stButton button:hover {
                background: #574bd1;
                color: white;
            }

            /* ---- side-panel section titles (Violations / Trend / Recent) ---- */
            .side-panel-title {
                font-size: 1.02rem;
                font-weight: 700;
                color: #f5f7fa;
                margin: 0 0 0.9rem 0;
            }

            /* ---- violations donut legend ---- */
            .legend-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.6rem;
                padding: 5px 0;
                font-size: 0.84rem;
            }
            .legend-left {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #c3c8d6;
            }
            .legend-dot {
                width: 9px;
                height: 9px;
                min-width: 9px;
                border-radius: 999px;
                display: inline-block;
            }
            .legend-value {
                color: #f5f7fa;
                font-weight: 600;
            }

            /* ---- PPE compliance overview ---- */
            .ppe-overview-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #f5f7fa;
                margin-bottom: 1.1rem;
            }
            .ppe-ring-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 0.8rem;
            }
            .ppe-ring-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            .ppe-ring-label {
                font-size: 0.85rem;
                color: #c3c8d6;
                margin-bottom: 0.6rem;
                white-space: nowrap;
            }
            .ppe-ring-wrap {
                position: relative;
                width: 84px;
                height: 84px;
            }
            .ppe-ring-svg {
                transform: rotate(-90deg);
            }
            .ppe-ring-center {
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.05rem;
            }
            .ppe-ring-frac {
                margin-top: 0.55rem;
                font-size: 0.82rem;
                color: #9aa1b4;
            }
            .ppe-ring-pct {
                font-size: 0.86rem;
                font-weight: 700;
            }

            /* ---- Telegram Notifier Feed ---- */
            .tg-header-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.75rem;
            }
            .tg-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #f5f7fa;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .tg-badge {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 0.73rem;
                padding: 3px 9px;
                border-radius: 999px;
                background: #0f2438;
                color: #229ED9;
                border: 1px solid #1a3c5a;
                font-weight: 600;
            }
            .tg-badge.live {
                background: #0f2d1e;
                color: #22c55e;
                border-color: #195232;
            }
            /* ---- Uniform Bottom 3 Panels ---- */
            .bottom-panel-card {
                min-height: 380px;
                max-height: 380px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                box-sizing: border-box;
                overflow: hidden;
            }
            .ppe-overview-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding-top: 1.5rem;
            }
            .tg-feed-container {
                display: flex;
                flex-direction: column;
                gap: 0.55rem;
                max-height: 235px;
                overflow-y: auto;
                padding-right: 4px;
                margin-top: 0.4rem;
            }
            .recent-feed-container {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
                max-height: 310px;
                overflow-y: auto;
                padding-right: 4px;
            }
            .tg-card {
                background: #0d1222;
                border: 1px solid #1c243c;
                border-left: 3px solid #ef4444;
                border-radius: 10px;
                padding: 0.6rem 0.8rem;
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 0.6rem;
            }
            .tg-card-main {
                flex: 1;
                min-width: 0;
            }
            .tg-card-header {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 0.82rem;
                font-weight: 700;
                color: #f87171;
                margin-bottom: 2px;
            }
            .tg-card-detail {
                font-size: 0.78rem;
                color: #c3c8d6;
            }
            .tg-card-meta {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.7rem;
                color: #8b93a7;
                margin-top: 3px;
            }
            .tg-status-pill {
                font-size: 0.68rem;
                padding: 2px 7px;
                border-radius: 6px;
                font-weight: 600;
                white-space: nowrap;
            }
            .tg-status-pill.sent {
                background: #143525;
                color: #4ade80;
                border: 1px solid #1e5236;
            }
            .tg-status-pill.queued {
                background: #2b2512;
                color: #fbbf24;
                border: 1px solid #4a3e1c;
            }
            .tg-empty-feed {
                text-align: center;
                padding: 1.6rem 0.8rem;
                color: #6b7280;
                font-size: 0.82rem;
            }

            /* ---- recent detections ---- */
            .recent-header-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.6rem;
            }
            .view-all-btn {
                background: #131826;
                border: 1px solid #212739;
                border-radius: 10px;
                padding: 0.3rem 0.8rem;
                font-size: 0.76rem;
                color: #c3c8d6;
                font-weight: 600;
            }
            .recent-item {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 0.55rem 0.2rem;
                border-bottom: 1px solid #1a1f33;
            }
            .recent-item:last-child { border-bottom: none; }
            .recent-thumb {
                width: 44px;
                height: 44px;
                min-width: 44px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
                overflow: hidden;
                background: #141829;
                border: 1px solid #252c48;
            }
            .recent-thumb img {
                width: 44px;
                height: 44px;
                min-width: 44px;
                border-radius: 9px;
                object-fit: cover;
                display: block;
            }
            .recent-thumb.violation {
                border-color: #ef4444;
                box-shadow: 0 0 6px rgba(239, 68, 68, 0.35);
            }
            .recent-info { flex: 1; min-width: 0; }
            .recent-title-row {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 0.86rem;
                font-weight: 600;
                color: #f0f2f5;
            }
            .recent-status-dot {
                width: 7px;
                height: 7px;
                min-width: 7px;
                border-radius: 999px;
                display: inline-block;
            }
            .recent-camera {
                font-size: 0.76rem;
                color: #8b93a7;
                margin-top: 1px;
            }
            .recent-time {
                font-size: 0.76rem;
                color: #8b93a7;
                white-space: nowrap;
            }
            .recent-chevron {
                color: #4b5262;
                font-size: 0.9rem;
            }
            /* ---- live stream panel ---- */
            .stream-container {
                position: relative;
                width: 100%;
                background: #0a0d16;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid #1f2540;
            }
            .stream-container img {
                width: 100%;
                display: block;
                border-radius: 12px;
            }
            .stream-stats-bar {
                display: flex;
                gap: 1.2rem;
                padding: 0.6rem 0.8rem;
                background: #0d1120;
                border: 1px solid #1f2540;
                border-radius: 10px;
                margin-top: 0.5rem;
                flex-wrap: wrap;
            }
            .stream-stat-item {
                display: flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.82rem;
                color: #c3c8d6;
            }
            .stream-stat-value {
                font-weight: 700;
                color: #f5f7fa;
            }
            .stream-stat-value.green { color: #22c55e; }
            .stream-stat-value.red { color: #ef4444; }
            .stream-stat-value.blue { color: #60a5fa; }

            .tracker-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                margin-top: 0.5rem;
                font-size: 0.82rem;
            }
            .tracker-table th {
                background: #0d1120;
                color: #8b93a7;
                font-weight: 600;
                padding: 0.5rem 0.6rem;
                text-align: left;
                border-bottom: 1px solid #1f2540;
            }
            .tracker-table td {
                padding: 0.45rem 0.6rem;
                color: #c3c8d6;
                border-bottom: 1px solid #141930;
            }
            .tracker-id-badge {
                display: inline-block;
                background: linear-gradient(135deg, #7c6ff0, #5b4fd6);
                color: white;
                font-weight: 700;
                font-size: 0.78rem;
                padding: 2px 10px;
                border-radius: 999px;
            }
            .ppe-dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 999px;
                margin-right: 2px;
            }
            .ppe-dot.yes { background: #22c55e; box-shadow: 0 0 4px #22c55e; }
            .ppe-dot.no { background: #ef4444; box-shadow: 0 0 4px #ef4444; }
            .ppe-dot.unknown { background: #6b7280; }
            .compliance-pill {
                display: inline-block;
                font-size: 0.72rem;
                font-weight: 600;
                padding: 2px 8px;
                border-radius: 6px;
            }
            .compliance-pill.ok {
                background: #143525;
                color: #4ade80;
                border: 1px solid #1e5236;
            }
            .compliance-pill.violation {
                background: #3b1219;
                color: #f87171;
                border: 1px solid #5c1d2b;
            }
            .stream-placeholder {
                border: 1.5px dashed #2a3050;
                border-radius: 14px;
                padding: 3rem 1rem;
                text-align: center;
                color: #6b7280;
                background: #0d1120;
                min-height: 300px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            .stream-placeholder .icon {
                font-size: 2.8rem;
                margin-bottom: 0.6rem;
            }
        """

    st.markdown(
        "<style>" + base_css + light_theme_css + "</style>",
        unsafe_allow_html=True,
    )

    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            function forceOpenSidebar() {
                try {
                    const doc = window.parent.document;
                    const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.style.display = 'block';
                        sidebar.style.visibility = 'visible';
                        sidebar.style.transform = 'none';
                        sidebar.style.opacity = '1';
                        sidebar.style.marginLeft = '0';
                    }
                    const btn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"], button[aria-label="Expand sidebar"]');
                    if (btn) {
                        btn.click();
                    }
                } catch(e) {}
            }
            forceOpenSidebar();
            setTimeout(forceOpenSidebar, 150);
            setTimeout(forceOpenSidebar, 500);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# ----------------------------------------------------------------------------
# HEADER & RANDOM SAFETY GREETINGS
# ----------------------------------------------------------------------------
SAFETY_GREETINGS = [
    ("Welcome to RASED Safety Monitor 👋", "AI-Powered Real-time PPE Compliance System"),
    ("Stay Safe, Work Smart Today 🦺", "Autonomous Workplace Safety & Equipment Monitoring"),
    ("Safety First, Excellence Always ✨", "Real-time AI Safety Equipment Detection Active"),
    ("Workplace Safety Control Center 🛡️", "Continuous Multi-Worker PPE Compliance Monitoring"),
    ("Welcome to Intelligent Site Guard 🤖", "Live Hazard Prevention & Compliance Tracking"),
    ("Protecting Teams, Preventing Hazards ⚡", "High-Performance AI Safety Surveillance"),
    ("Zero-Accident Safety Protocol Live 🟢", "Real-Time Computer Vision PPE Detection"),
    ("Active Workplace Monitoring Online 👁️", "Automated Safety Equipment Verification"),
    ("Ensuring Safety with AI Precision 🎯", "Real-Time Tracking, Incident Logging & Telegram Alerts"),
    ("Welcome to RASED Operations Hub 🌐", "Smart Industrial Safety & PPE Enforcement"),
]

def render_header(user_name: str = None):
    now = datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%B %d, %Y")
    current_theme = st.session_state.get("app_theme", "dark")

    if "session_greeting" not in st.session_state:
        st.session_state["session_greeting"] = random.choice(SAFETY_GREETINGS)

    greeting_title, greeting_subtitle = st.session_state["session_greeting"]

    head_left, head_right = st.columns([2.6, 1.8], gap="small")

    with head_left:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <h1 style="font-size: 1.65rem; font-weight: 700; color: {'#0f172a' if current_theme == 'light' else '#f5f7fa'}; margin: 0 0 2px 0;">{greeting_title}</h1>
                <p style="color: {'#64748b' if current_theme == 'light' else '#8b93a7'}; margin: 0; font-size: 0.92rem;">{greeting_subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with head_right:
        h_col1, h_col2, h_col3 = st.columns([1.15, 0.85, 1.0], gap="small")
        with h_col1:
            st.markdown(
                f"""
                <div class="pill">
                    <div class="pill-time">{time_str}</div>
                    <div class="pill-date">{date_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h_col2:
            toggle_label = "🌙 Dark" if current_theme == "dark" else "☀️ Light"
            if st.button(toggle_label, key="theme_toggle_btn", use_container_width=True, help="Switch Light / Dark Theme"):
                st.session_state["app_theme"] = "light" if current_theme == "dark" else "dark"
                st.rerun()
        with h_col3:
            st.markdown(
                """
                <div class="status-pill">
                    <div class="status-label">Model Status</div>
                    <div class="status-value"><span class="status-dot"></span>Online</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# STAT CARDS
# ----------------------------------------------------------------------------
def render_stat_cards(stats: dict, placeholder=None):
    html = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon purple">📋</div>
            <div>
                <div class="stat-label">Total Detections</div>
                <div class="stat-value">{stats['total_detections']}</div>
                <div class="stat-delta positive">{stats['total_detections_delta']}</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon green">🛡️</div>
            <div>
                <div class="stat-label">Compliance Rate</div>
                <div class="stat-value">{stats['compliance_rate']}</div>
                <div class="stat-delta positive">{stats['compliance_rate_delta']}</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon blue">👥</div>
            <div>
                <div class="stat-label">Workers Scanned</div>
                <div class="stat-value">{stats['workers_scanned']}</div>
                <div class="stat-delta positive">{stats['workers_scanned_delta']}</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon red">⚠️</div>
            <div>
                <div class="stat-label">Violations</div>
                <div class="stat-value">{stats['violations']}</div>
                <div class="stat-delta negative">{stats['violations_delta']}</div>
            </div>
        </div>
    </div>
    """
    if placeholder is not None:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TELEGRAM ALERT DISPATCHER HELPER
# ----------------------------------------------------------------------------
def dispatch_violation_alert(violation_type: str, person_id: int = 1, face_image_path: str = None):
    """Logs violation alert to SQLite and optionally dispatches via Telegram bot with violator face image."""
    try:
        # If face_image_path is not specified or missing, check if a recent capture exists
        if not face_image_path or not os.path.exists(face_image_path):
            faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "faces")
            if os.path.exists(faces_dir):
                files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    face_image_path = files[0]

        alert_id = database.log_alert(person_id=person_id, violation_type=violation_type, face_image_path=face_image_path)
        token = st.session_state.get("tg_token") or getattr(config, "TELEGRAM_BOT_TOKEN", None)
        chat = st.session_state.get("tg_chat_id") or getattr(config, "TELEGRAM_CHAT_ID", None)
        enabled = st.session_state.get("tg_enabled", getattr(config, "TELEGRAM_ENABLED", False))

        if enabled and token and token != "YOUR_BOT_TOKEN_HERE" and chat and chat != "YOUR_CHAT_ID_HERE":
            bot = TelegramAlertBot(bot_token=token, chat_id=chat, enabled=True)
            sent = bot._send_telegram_alert({
                "person_id": person_id,
                "violation_type": violation_type,
                "face_image_path": face_image_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            if sent:
                database.mark_alert_sent(alert_id)
        return alert_id
    except Exception as e:
        print(f"[Telegram Alert Error]: {e}")
        return None


# ----------------------------------------------------------------------------
# LIVE DETECTION & TRACKING PANEL
# ----------------------------------------------------------------------------
def render_live_detection_panel(live_dashboard_updater=None):
    with st.container(border=True):
        header_col, control_col1, control_col2 = st.columns([3, 2, 1.2])

        with header_col:
            st.markdown(
                """
                <div class="panel-title-row">
                    <span class="panel-title">Live Detection & Tracking</span>
                    <span class="live-badge"><span class="live-dot"></span>Live</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with control_col1:
            source = st.selectbox(
                "Input source",
                [
                    "📷 Upload Image",
                    "🎬 Stream Video",
                    "📹 Live Stream",
                ],
                label_visibility="collapsed",
            )

        with control_col2:
            paused = st.toggle("⏸ Pause", value=False)

        with st.expander("⚙️ Detection & Speed Settings", expanded=False):
            conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.4, 0.05)
            speed_mode = st.select_slider(
                "Processing Mode",
                options=["🎯 Standard (Full Resolution)", "⚡ Fast (Stride 2)", "🚀 Ultra Fast (Stride 3)"],
                value="🎯 Standard (Full Resolution)",
                help="Process video and images at their original native resolution.",
            )

        imgsz = None
        if "Stride 3" in speed_mode:
            stride = 3
        elif "Stride 2" in speed_mode:
            stride = 2
        else:
            stride = 1

        # --- streaming / tracking modes ---
        if source in ("🎬 Stream Video", "📹 Live Stream"):
            render_live_stream_panel(
                source,
                conf_threshold=conf_threshold,
                imgsz=imgsz,
                stride=stride,
                live_dashboard_updater=live_dashboard_updater
            )
            return

        # --- get an input image from the chosen source ---
        input_image = None
        frame_bytes = None  # used to fingerprint the frame so we log it once, not on every rerun
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            frame_bytes = uploaded.getvalue()
            input_image = Image.open(uploaded)

        # --- run inference / show result / show placeholder ---
        if paused and "last_annotated" in st.session_state:
            st.image(st.session_state["last_annotated"], use_container_width=True)
            st.caption("⏸ Paused — showing last processed frame.")
            return

        if input_image is not None:
            model = load_model(MODEL_PATH)
            if model is None:
                st.warning(
                    f"⚠️ Couldn't find `{MODEL_PATH}`. Place your trained "
                    f"YOLO11s weights file named `best.pt` next to `app.py` "
                    f"and reload the app."
                )
                st.image(input_image, use_container_width=True)
                return

            with st.spinner("Running detection..."):
                annotated, detections = run_inference(input_image, model, conf_threshold)

            st.session_state["last_annotated"] = annotated
            st.session_state["last_detections"] = detections

            # Log this frame once.
            frame_hash = hash(frame_bytes) if frame_bytes else None
            if frame_hash is not None and st.session_state.get("_last_logged_hash") != frame_hash:
                logger.log_frame(camera="Camera 01", detections=detections)
                st.session_state["_last_logged_hash"] = frame_hash

                # For each detected violation, crop violator screenshot and dispatch alert
                faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "faces")
                os.makedirs(faces_dir, exist_ok=True)
                img_w, img_h = input_image.size

                for idx, d in enumerate(detections):
                    if d.get("violation"):
                        face_crop_path = None
                        box = d.get("box")
                        if box:
                            try:
                                pad = 25
                                bx1 = max(0, int(box[0]) - pad)
                                by1 = max(0, int(box[1]) - pad)
                                bx2 = min(img_w, int(box[2]) + pad)
                                by2 = min(img_h, int(box[3]) + pad)
                                crop_img = input_image.crop((bx1, by1, bx2, by2))
                                
                                timestamp = int(_time.time() * 1000)
                                filename = f"person_{idx + 1}_{timestamp}.jpg"
                                face_crop_path = os.path.abspath(os.path.join(faces_dir, filename))
                                crop_img.save(face_crop_path, "JPEG")
                            except Exception as e:
                                print(f"[Crop Error]: {e}")

                        dispatch_violation_alert(
                            violation_type=d["class"],
                            person_id=idx + 1,
                            face_image_path=face_crop_path,
                        )

                # Auto-update surrounding dashboard in real-time
                if live_dashboard_updater:
                    live_dashboard_updater()

            st.image(annotated, use_container_width=True)

            if detections:
                chips = "".join(
                    f"<span style='background:{'#ef4444' if d['violation'] else '#22c55e'};"
                    f"color:white;padding:3px 10px;border-radius:999px;font-size:0.75rem;"
                    f"margin:3px;display:inline-block;'>{d['class']} "
                    f"({d['confidence']*100:.0f}%)</span>"
                    for d in detections
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No objects detected above the confidence threshold.")
        elif "last_annotated" in st.session_state and st.session_state["last_annotated"] is not None:
            # Preserve old state if returning
            st.image(st.session_state["last_annotated"], use_container_width=True)
            st.caption("⏸ Showing last uploaded image detection. Upload a new image to re-run.")
        else:
            st.markdown(
                """
                <div class="detection-placeholder">
                    <div class="icon">📷</div>
                    Upload an image to run PPE detection
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# LIVE STREAMING & REAL-TIME PERSON TRACKING PANEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ppe_detector():
    """Load and cache the YOLO-based PPEDetector with ByteTrack tracking."""
    import detector as _det_mod
    importlib.reload(_det_mod)
    return _det_mod.PPEDetector()


def _build_persons_table_html(persons: list) -> str:
    """Build an HTML table showing tracked persons with persistent static IDs & PPE status."""
    if not persons:
        return (
            '<div style="text-align:center;color:#6b7280;padding:1rem;font-size:0.84rem;">'
            'No persons currently tracked — waiting for detections…</div>'
        )

    def _dot(status):
        cls = "yes" if status == "yes" else ("no" if status == "no" else "unknown")
        label = "✓" if status == "yes" else ("✗" if status == "no" else "?")
        return f'<span class="ppe-dot {cls}" title="{status}"></span>{label}'

    def _compliance(is_compliant):
        if is_compliant:
            return '<span class="compliance-pill ok">✓ Compliant</span>'
        return '<span class="compliance-pill violation">✗ Violation</span>'

    rows = []
    for p in persons:
        rows.append(
            f"<tr>"
            f'<td><span class="tracker-id-badge">Worker #{p["id"]}</span></td>'
            f'<td>{_dot(p.get("helmet", "unknown"))}</td>'
            f'<td>{_dot(p.get("vest", "unknown"))}</td>'
            f'<td>{_dot(p.get("gloves", "unknown"))}</td>'
            f'<td>{_compliance(p.get("is_compliant", False))}</td>'
            f'<td>{p.get("violations_count", 0)}</td>'
            f"</tr>"
        )

    return (
        '<table class="tracker-table">'
        "<thead><tr>"
        "<th>Person ID</th><th>Helmet</th><th>Vest</th><th>Gloves</th>"
        "<th>Compliance</th><th>Violations</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_live_stream_panel(source_mode: str, conf_threshold: float = 0.4, imgsz: int = 480, stride: int = 1, live_dashboard_updater=None):
    """Render the live video stream with real-time ByteTrack person tracking and session state saving."""
    source_value = None

    if source_mode == "🎬 Stream Video":
        video_file = st.file_uploader(
            "Upload a video for real-time person tracking",
            type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed",
            key="stream_video_uploader",
        )
        if video_file is not None:
            video_bytes = video_file.getvalue()
            video_hash = hash(video_bytes) & 0xFFFFFFFF
            tmp_dir = tempfile.gettempdir()
            in_path = os.path.join(tmp_dir, f"ppe_stream_in_{video_hash}.mp4")
            if not os.path.exists(in_path):
                with open(in_path, "wb") as f:
                    f.write(video_bytes)
            source_value = in_path
        elif st.session_state.get("_last_stream_video_path"):
            source_value = st.session_state["_last_stream_video_path"]

    elif source_mode == "📹 Live Stream":
        cam_idx = st.selectbox(
            "Select Camera Device",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"📷 Camera {x} (Default / Primary Webcam)" if x == 0 else f"📷 Camera {x}",
            key="live_cam_select",
        )
        source_value = int(cam_idx)

    # ---- Start / Stop buttons ----
    btn_col1, btn_col2 = st.columns(2)
    start_key = f"start_{source_mode}"
    
    with btn_col1:
        start_tracking = st.button(
            "▶ Start Live Tracking",
            type="primary",
            use_container_width=True,
            disabled=(source_value is None),
            key=start_key,
        )

    with btn_col2:
        stop_tracking = st.button("⏹ Stop", use_container_width=True, key=f"stop_{source_mode}")

    if stop_tracking:
        st.session_state["stream_running"] = False
        st.info("Tracking stopped. Session results saved below.")
        st.rerun()

    if start_tracking:
        st.session_state["stream_running"] = True
        st.session_state["current_source"] = source_value
        if isinstance(source_value, str) and os.path.exists(source_value):
            st.session_state["_last_stream_video_path"] = source_value

    if st.session_state.get("stream_running") and source_value is not None:
        detector = get_ppe_detector()
        if detector is None:
            st.error("Detector could not be initialized.")
            return

        detector.reset()
        
        # Open video source (use DirectShow for instant Windows webcam access)
        if isinstance(source_value, int):
            cap = cv2.VideoCapture(source_value, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source_value)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source_value)

        if not cap.isOpened():
            st.error(f"❌ Could not open camera/video source: {source_value}. Please check camera connection or permissions.")
            st.session_state["stream_running"] = False
            return

        # Placeholders for live rendering
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        table_placeholder = st.empty()

        detector.running = True
        frame_count = 0
        logged_violations = set()
        last_annotated_frame = None
        empty_frames = 0
        batch_detections = []
        try:
            while cap.isOpened() and st.session_state.get("stream_running", True):
                ret, frame = cap.read()
                if not ret or frame is None:
                    empty_frames += 1
                    if empty_frames > 40:  # 40 consecutive empty frames
                        break
                    _time.sleep(0.03)
                    continue
                empty_frames = 0

                frame_count += 1

                # Frame sampling / stride for high FPS on CPU
                if stride > 1 and (frame_count % stride != 0) and last_annotated_frame is not None:
                    # Pass previous detection frame smoothly
                    frame_placeholder.image(last_annotated_frame, use_container_width=True)
                    continue

                # Track each person & detect PPE items using fast ByteTrack
                annotated_frame, detection_data = detector.process_frame(
                    frame,
                    conf=conf_threshold,
                    imgsz=imgsz,
                    tracker='bytetrack.yaml'
                )
                if annotated_frame is None:
                    break

                # Stream live frame directly to UI
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                last_annotated_frame = rgb_frame
                frame_placeholder.image(rgb_frame, use_container_width=True)

                # Collect detections for dashboard logger & database
                if detection_data and detection_data.get("raw_detections"):
                    batch_detections.append(detection_data["raw_detections"])
                    
                    # Log tracked persons to database
                    for p in detection_data.get("persons", []):
                        try:
                            database.log_detection(
                                person_id=int(p["id"]),
                                helmet=p.get("helmet", "unknown"),
                                vest=p.get("vest", "unknown"),
                                gloves=p.get("gloves", "unknown"),
                                is_compliant=bool(p.get("compliant", False)),
                            )
                        except Exception:
                            pass

                # Dispatch any new violations detected in this frame with face photo
                if detection_data and detection_data.get("violations"):
                    for v in detection_data["violations"]:
                        v_key = (v["person_id"], v["type"], v.get("time"))
                        if v_key not in logged_violations:
                            logged_violations.add(v_key)
                            dispatch_violation_alert(
                                violation_type=v["type"],
                                person_id=v["person_id"],
                                face_image_path=v.get("face_path"),
                            )

                # Periodic flush to CSV logger & real-time dashboard update (every 15 sampled frames)
                if len(batch_detections) >= 15:
                    logger.log_frames_batch(camera="Live Stream", frames_detections=batch_detections)
                    batch_detections = []
                    if live_dashboard_updater:
                        live_dashboard_updater()

                # Update live stats & table throttled (every 3 frames) to eliminate websocket latency
                if frame_count % 3 == 0 or frame_count == 1:
                    stats = detector.get_live_stats()
                    total_p = stats.get("total_persons", 0)
                    compliant_p = stats.get("compliant", 0)
                    violations_p = stats.get("violations", 0)
                    rate = stats.get("compliance_rate", 100.0)
                    fps = stats.get("fps", 0)

                    stats_placeholder.markdown(
                        f"""
                        <div class="stream-stats-bar">
                            <div class="stream-stat-item">⚡ FPS: <span class="stream-stat-value blue">{fps:.1f}</span></div>
                            <div class="stream-stat-item">👥 Tracked Workers: <span class="stream-stat-value">{total_p}</span></div>
                            <div class="stream-stat-item">✅ Compliant: <span class="stream-stat-value green">{compliant_p}</span></div>
                            <div class="stream-stat-item">⚠️ Violations: <span class="stream-stat-value red">{violations_p}</span></div>
                            <div class="stream-stat-item">📊 Rate: <span class="stream-stat-value green">{rate:.1f}%</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    persons = detector.get_person_states()
                    table_placeholder.markdown(
                        f'<div style="margin-top:0.6rem;">'
                        f'<b>👥 Active Tracked Workers ({len(persons)})</b>'
                        f'{_build_persons_table_html(persons)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Dispatch Telegram violation alerts in real-time
                if detection_data and detection_data.get("violations"):
                    for v in detection_data["violations"]:
                        v_key = f"{v['person_id']}_{v['type']}"
                        if v_key not in logged_violations:
                            logged_violations.add(v_key)
                            dispatch_violation_alert(
                                violation_type=v["type"],
                                person_id=int(v["person_id"]),
                                face_image_path=v.get("face_path"),
                            )

                # Save state in session_state continuously
                st.session_state["last_stream_frame"] = rgb_frame
                st.session_state["last_stream_stats"] = detector.get_live_stats()
                st.session_state["last_stream_persons"] = detector.get_person_states()
                st.session_state["last_stream_source"] = source_mode
                st.session_state["last_stream_time"] = datetime.now().strftime("%I:%M:%S %p")

                _time.sleep(0.001)

        finally:
            cap.release()
            detector.stop()
            st.session_state["stream_running"] = False
            
            # Flush any remaining detections to CSV logger so dashboard metrics update immediately
            if batch_detections:
                logger.log_frames_batch(camera="Live Stream", frames_detections=batch_detections)
                batch_detections = []
            if live_dashboard_updater:
                live_dashboard_updater()

        st.success(f"✅ Finished tracking stream ({frame_count} frames processed). Dashboard updated.")
        _time.sleep(0.3)
        st.rerun()

    else:
        # Check if we have preserved state to show when returning / stopped
        has_saved_state = (
            "last_stream_frame" in st.session_state
            and st.session_state["last_stream_frame"] is not None
        )

        if has_saved_state:
            saved_time = st.session_state.get("last_stream_time", "earlier")
            saved_src = st.session_state.get("last_stream_source", "")
            
            top_col1, top_col2 = st.columns([3, 1])
            with top_col1:
                st.caption(f"💾 **Preserved Last Tracking Session** — recorded at {saved_time} ({saved_src})")
            with top_col2:
                if st.button("🔄 Clear Preserved State", key=f"clear_state_{source_mode}"):
                    st.session_state.pop("last_stream_frame", None)
                    st.session_state.pop("last_stream_stats", None)
                    st.session_state.pop("last_stream_persons", None)
                    st.rerun()

            # Display preserved frame
            st.image(st.session_state["last_stream_frame"], use_container_width=True)

            # Display preserved stats
            saved_stats = st.session_state.get("last_stream_stats", {})
            total_p = saved_stats.get("total_persons", 0)
            compliant_p = saved_stats.get("compliant", 0)
            violations_p = saved_stats.get("violations", 0)
            rate = saved_stats.get("compliance_rate", 100.0)
            fps = saved_stats.get("fps", 0)

            st.markdown(
                f"""
                <div class="stream-stats-bar">
                    <div class="stream-stat-item">⚡ Last FPS: <span class="stream-stat-value blue">{fps:.1f}</span></div>
                    <div class="stream-stat-item">👥 Tracked Workers: <span class="stream-stat-value">{total_p}</span></div>
                    <div class="stream-stat-item">✅ Compliant: <span class="stream-stat-value green">{compliant_p}</span></div>
                    <div class="stream-stat-item">⚠️ Violations: <span class="stream-stat-value red">{violations_p}</span></div>
                    <div class="stream-stat-item">📊 Rate: <span class="stream-stat-value green">{rate:.1f}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display preserved persons table
            saved_persons = st.session_state.get("last_stream_persons", [])
            with st.expander(f"👥 Tracked Workers in Session ({len(saved_persons)})", expanded=True):
                st.markdown(_build_persons_table_html(saved_persons), unsafe_allow_html=True)

        else:
            if source_mode == "🎬 Stream Video":
                icon, msg = "🎬", "Upload a video file and click <b>Start Live Tracking</b> to view real-time person tracking without waiting"
            else:
                icon, msg = "📹", "Select camera and click <b>Start Live Tracking</b> to begin real-time webcam person tracking"

            st.markdown(
                f"""
                <div class="stream-placeholder">
                    <div class="icon">{icon}</div>
                    {msg}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# VIDEO UPLOAD + FRAME-BY-FRAME PROCESSING (batch mode — kept for download)
# ----------------------------------------------------------------------------
def _reencode_for_browser(path: str) -> str:
    """cv2.VideoWriter's mp4v codec often won't preview inline in browsers
    (Chrome in particular wants H.264). If ffmpeg is on PATH, re-encode to
    H.264 so st.video() can play it; otherwise fall back to the original
    file — it's still a valid, downloadable mp4 either way."""
    h264_path = path.replace(".mp4", "_h264.mp4")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264_path],
            capture_output=True,
            timeout=180,
        )
        if result.returncode == 0 and os.path.exists(h264_path):
            return h264_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return path


def render_video_upload_and_process(conf_threshold: float):
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
        key="video_uploader",
    )

    if video_file is None:
        st.markdown(
            """
            <div class="detection-placeholder">
                <div class="icon">🎬</div>
                Upload a video to run PPE detection frame-by-frame
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # cv2.VideoCapture needs a real file path, not raw bytes — write to disk.
    video_bytes = video_file.getvalue()
    video_hash = hash(video_bytes) & 0xFFFFFFFF
    tmp_dir = tempfile.gettempdir()
    in_path = os.path.join(tmp_dir, f"ppe_video_in_{video_hash}.mp4")
    if not os.path.exists(in_path):
        with open(in_path, "wb") as f:
            f.write(video_bytes)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Couldn't open that video file — try a standard mp4/mov/avi export.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = (total_frames / fps) if fps else 0

    st.caption(f"📹 {total_frames} frames · {fps:.0f} fps · {duration_sec:.1f}s · {width}×{height}")

    col1, col2 = st.columns(2)
    with col1:
        sample_every = st.slider(
            "Process every Nth frame", 1, 15, 5,
            help="Higher = faster but fewer frames analyzed. Skipped frames pass "
                 "through to the output video unannotated.",
        )
    with col2:
        default_cap = min(300, max(total_frames, 10))
        max_frames = st.slider(
            "Max source frames to read", 10, max(total_frames, 10), default_cap,
            help="Caps runtime on long videos. Processing stops after this many "
                 "SOURCE frames are read (not all of them get analyzed — see above).",
        )

    if not st.button("▶ Process Video", type="primary", use_container_width=True):
        cap.release()
        return

    model = load_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"⚠️ Couldn't find `{MODEL_PATH}`. Place your trained YOLO11s weights "
            f"file named `best.pt` next to `app.py` and reload the app."
        )
        cap.release()
        return

    out_path = os.path.join(tmp_dir, f"ppe_video_out_{video_hash}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Guard against re-logging the same video's detections every time the
    # user tweaks a slider and re-clicks Process — only log once per upload.
    already_logged = st.session_state.get("_last_logged_video_hash") == video_hash

    progress = st.progress(0.0, text="Starting...")
    frame_idx = 0
    analyzed_count = 0
    video_detections_batch = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            annotated, detections = run_inference(pil_frame, model, conf_threshold)
            writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))

            if not already_logged:
                video_detections_batch.append(detections)

            analyzed_count += 1
        else:
            writer.write(frame_bgr)  # pass unsampled frames through unannotated

        frame_idx += 1
        progress.progress(min(frame_idx / max_frames, 1.0), text=f"Read {frame_idx}/{max_frames} frames...")

    cap.release()
    writer.release()
    progress.empty()

    if not already_logged and video_detections_batch:
        logger.log_frames_batch(camera="Camera 01 (Video)", frames_detections=video_detections_batch)
        st.session_state["_last_logged_video_hash"] = video_hash

        # Dispatch alert for each unique violation detected in video
        seen_violations = set()
        for d_list in video_detections_batch:
            for d in d_list:
                if d.get("violation") and d["class"] not in seen_violations:
                    seen_violations.add(d["class"])
                    dispatch_violation_alert(violation_type=d["class"], person_id=1)

    st.success(f"Done — analyzed {analyzed_count} of {frame_idx} frames read.")

    playable_path = _reencode_for_browser(out_path)
    st.video(playable_path)

    with open(playable_path, "rb") as f:
        st.download_button(
            "⬇ Download annotated video",
            data=f.read(),
            file_name="ppe_detection_output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


# ----------------------------------------------------------------------------
# VIOLATIONS DETECTED — donut chart + legend
# ----------------------------------------------------------------------------
def render_violations_panel(violations: dict, placeholder=None):
    """violations: dict of label -> count, e.g. {'No Helmet': 42, ...}"""
    colors = ["#8b7ef8", "#2dd4bf", "#f59e0b", "#f472b6", "#fb7185"]
    labels = list(violations.keys())
    values = list(violations.values())
    total = sum(values)

    target = placeholder.container(border=True) if placeholder is not None else st.container(border=True)
    with target:
        st.markdown('<div class="side-panel-title">Violations Detected</div>', unsafe_allow_html=True)

        if not values:
            st.markdown(
                '<div style="padding:2.4rem 0;text-align:center;color:#8b93a7;font-size:0.85rem;">'
                'No violations logged yet.<br>Run a detection in the panel on the left to populate this chart.'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        current_theme = st.session_state.get("app_theme", "dark")
        donut_border = "#ffffff" if current_theme == "light" else "#11162a"
        total_text_color = "#0f172a" if current_theme == "light" else "#f5f7fa"
        sub_text_color = "#64748b" if current_theme == "light" else "#8b93a7"

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.68,
                    marker=dict(colors=colors, line=dict(color=donut_border, width=3)),
                    textinfo="none",
                    hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                    sort=False,
                )
            ]
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b style='font-size:26px;color:{total_text_color}'>{total}</b><br>"
                    f"<span style='font-size:12px;color:{sub_text_color}'>Total</span>",
                    x=0.5, y=0.5, showarrow=False,
                )
            ],
        )

        chart_col, legend_col = st.columns([1.1, 1])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with legend_col:
            legend_html = ["<div style='margin-top:0.6rem;'>"]
            for label, value, color in zip(labels, values, colors):
                pct = (value / total * 100) if total else 0
                legend_html.append(
                    f'<div class="legend-row">'
                    f'<div class="legend-left">'
                    f'<span class="legend-dot" style="background:{color};"></span>'
                    f'{label}'
                    f'</div>'
                    f'<div class="legend-value">{value} ({pct:.1f}%)</div>'
                    f'</div>'
                )
            legend_html.append("</div>")
            st.markdown("".join(legend_html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# COMPLIANCE TREND — hourly line chart
# ----------------------------------------------------------------------------
def render_trend_panel(detections_log: list, placeholder=None):
    target = placeholder.container(border=True) if placeholder is not None else st.container(border=True)
    with target:
        title_col, range_col = st.columns([2.2, 1.3])
        with title_col:
            st.markdown('<div class="side-panel-title">Compliance Trend (Hourly)</div>', unsafe_allow_html=True)
        with range_col:
            range_val = st.selectbox(
                "Range",
                ["24 Hours", "12 Hours", "6 Hours"],
                label_visibility="collapsed",
                key="trend_hour_range",
            )
            hours_map = {"24 Hours": 24, "12 Hours": 12, "6 Hours": 6}
            hours_count = hours_map.get(range_val, 24)

        if hasattr(logger, "compute_hourly_trend"):
            hours, values = logger.compute_hourly_trend(detections_log, hours=hours_count)
        elif hasattr(logger, "compute_trend"):
            hours, values = logger.compute_trend(detections_log, hours=hours_count)
        else:
            hours, values = ["12 AM", "6 AM", "12 PM", "6 PM"], [100, 100, 100, 100]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=hours,
                    y=values,
                    mode="lines+markers",
                    line=dict(color="#7c6ff0", width=2.5, shape="spline"),
                    marker=dict(size=6, color="#7c6ff0", line=dict(color="#0a0d16", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(124,111,240,0.08)",
                    hovertemplate="Time: <b>%{x}</b><br>Compliance: <b>%{y}%</b><extra></extra>",
                )
            ]
        )
        fig.update_layout(
            height=210,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=False,
                color="#8b93a7",
                tickfont=dict(size=9),
                tickangle=-30 if hours_count > 12 else 0,
            ),
            yaxis=dict(
                range=[0, 100],
                showgrid=True,
                gridcolor="#1a1f33",
                color="#6b7280",
                tickfont=dict(size=10),
                ticksuffix="%",
            ),
            hoverlabel=dict(bgcolor="#131826", font_color="#f5f7fa", bordercolor="#212739"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------
# PPE COMPLIANCE OVERVIEW — circular progress rings
# ----------------------------------------------------------------------------
def _ring_svg(pct: float, color: str, size: int = 84, stroke: int = 8) -> str:
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - pct / 100)
    center = size / 2
    current_theme = st.session_state.get("app_theme", "dark")
    track_color = "#e2e8f0" if current_theme == "light" else "#1c2138"
    return (
        f'<svg class="ppe-ring-svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{track_color}" stroke-width="{stroke}" />'
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" />'
        f'</svg>'
    )


def render_ppe_overview(items: list, placeholder=None):
    """items: list of dicts {icon, label, count, total, pct}"""
    target = placeholder.container(border=True) if placeholder is not None else st.container(border=True)
    with target:
        if not items:
            inner_html = (
                '<div class="bottom-panel-card">'
                '<div class="ppe-overview-title">PPE Compliance Overview</div>'
                '<div class="ppe-overview-content" style="color:#8b93a7;font-size:0.85rem;">'
                'No detections logged yet.'
                '</div>'
                '</div>'
            )
            st.markdown(inner_html, unsafe_allow_html=True)
            return

        rings_html = [
            '<div class="bottom-panel-card">',
            '<div class="ppe-overview-title">PPE Compliance Overview</div>',
            '<div class="ppe-overview-content">',
            '<div class="ppe-ring-grid">'
        ]
        for item in items:
            pct = item["pct"]
            color = "#22c55e" if pct >= 85 else ("#f59e0b" if pct >= 70 else "#ef4444")
            svg = _ring_svg(pct, color)
            rings_html.append(
                f'<div class="ppe-ring-item">'
                f'<div class="ppe-ring-label">{item["icon"]} {item["label"]}</div>'
                f'<div class="ppe-ring-wrap">'
                f'{svg}'
                f'<div class="ppe-ring-center">'
                f'<span class="ppe-ring-pct" style="color:{color};">{pct:.0f}%</span>'
                f'</div>'
                f'</div>'
                f'<div class="ppe-ring-frac">{item["count"]}/{item["total"]}</div>'
                f'</div>'
            )
        rings_html.append('</div></div></div>')
        st.markdown("".join(rings_html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# RECENT DETECTIONS — list with violator screenshots
# ----------------------------------------------------------------------------
def render_recent_detections(entries: list, placeholder=None):
    """entries: list of dicts {id, title, camera, time, ok, image_b64}"""
    target = placeholder.container(border=True) if placeholder is not None else st.container(border=True)
    with target:
        header_html = (
            '<div class="recent-header-row">'
            '<div class="side-panel-title" style="margin-bottom:0;">Recent Detections</div>'
            '<div class="view-all-btn">Live Violations</div>'
            '</div>'
        )

        if not entries:
            st.markdown(
                '<div class="bottom-panel-card">'
                + header_html +
                '<div style="padding:2.5rem 0;text-align:center;color:#8b93a7;font-size:0.85rem;">'
                'No violations detected yet.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        items_html = ['<div class="bottom-panel-card">', header_html, '<div class="recent-feed-container">']
        for e in entries:
            dot_color = "#22c55e" if e.get("ok") else "#ef4444"
            img_b64 = e.get("image_b64")

            if img_b64:
                thumb_html = (
                    f'<div class="recent-thumb {"violation" if not e.get("ok") else ""}">'
                    f'<img src="{img_b64}" alt="Worker Screenshot" />'
                    f'</div>'
                )
            else:
                icon = "🛡️" if e.get("ok") else "⚠️"
                bg = "#143525" if e.get("ok") else "#3b1219"
                thumb_html = f'<div class="recent-thumb" style="background:{bg};">{icon}</div>'

            items_html.append(
                f'<div class="recent-item">'
                f'{thumb_html}'
                f'<div class="recent-info">'
                f'<div class="recent-title-row">'
                f'<span class="recent-status-dot" style="background:{dot_color};"></span>'
                f'{e["title"]}'
                f'</div>'
                f'<div class="recent-camera">{e.get("camera", "Live Feed")}</div>'
                f'</div>'
                f'<div class="recent-time">{e.get("time", "Just now")}</div>'
                f'<div class="recent-chevron">›</div>'
                f'</div>'
            )
        items_html.append('</div></div>')
        st.markdown("".join(items_html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TELEGRAM NOTIFIER FEED — live dispatch stream
# ----------------------------------------------------------------------------
def render_telegram_feed():
    with st.container(border=True):
        alerts = database.get_recent_alerts(limit=15)

        is_enabled = st.session_state.get("tg_enabled", getattr(config, "TELEGRAM_ENABLED", False))
        token = st.session_state.get("tg_token", getattr(config, "TELEGRAM_BOT_TOKEN", ""))
        chat_id = st.session_state.get("tg_chat_id", getattr(config, "TELEGRAM_CHAT_ID", ""))
        is_configured = bool(token and token != "YOUR_BOT_TOKEN_HERE" and chat_id and chat_id != "YOUR_CHAT_ID_HERE")

        badge_class = "live" if (is_enabled and is_configured) else ""
        badge_text = "🟢 Bot Active" if (is_enabled and is_configured) else ("🟡 Standby" if is_configured else "⚪ Not Configured")

        st.markdown(
            f'<div class="tg-header-row">'
            f'<div class="tg-title">✈️ Telegram Notifier Feed</div>'
            f'<span class="tg-badge {badge_class}">{badge_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        act_col1, act_col2 = st.columns([1.1, 0.9])
        with act_col1:
            if st.button("🔔 Send Test Alert", use_container_width=True, help="Dispatches a test PPE violation report"):
                test_types = ["No-Helmet", "No-Vest", "No-Glove"]
                v_type = test_types[len(alerts) % len(test_types)]
                p_id = 101 + (len(alerts) % 20)
                dispatch_violation_alert(violation_type=v_type, person_id=p_id)
                st.toast(f"🚨 Violation alert for {v_type} dispatched!", icon="✈️")
                st.rerun()

        with act_col2:
            with st.popover("⚙️ Bot Setup", use_container_width=True):
                st.caption("Telegram Bot Configuration")
                new_token = st.text_input("Bot Token", value=token if token != "YOUR_BOT_TOKEN_HERE" else "", type="password", placeholder="123456:ABC-DEF...")
                new_chat = st.text_input("Chat ID", value=chat_id if chat_id != "YOUR_CHAT_ID_HERE" else "", placeholder="-100123456789 or 987654321")
                new_enabled = st.toggle("Enable Telegram Sending", value=is_enabled)

                if st.button("Save & Test Connection", use_container_width=True, type="primary"):
                    st.session_state["tg_token"] = new_token
                    st.session_state["tg_chat_id"] = new_chat
                    st.session_state["tg_enabled"] = new_enabled
                    test_bot = TelegramAlertBot(bot_token=new_token, chat_id=new_chat, enabled=new_enabled)
                    ok, msg = test_bot.test_connection()
                    if ok:
                        st.success(f"Connected: {msg}")
                    else:
                        st.warning(f"Status: {msg}")
                    st.rerun()

        if not alerts:
            st.markdown(
                '<div class="tg-feed-container" style="display:flex;align-items:center;justify-content:center;min-height:220px;">'
                '<div class="tg-empty-feed">'
                '<div style="font-size:1.6rem;margin-bottom:0.3rem;">✈️</div>'
                'No violation reports sent yet.<br>Click <b>Send Test Alert</b> above or run a detection with violations.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            cards_html = ['<div class="tg-feed-container">']
            for a in alerts:
                ts_str = a.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts_str)
                    formatted_time = dt.strftime("%I:%M:%S %p").lstrip("0")
                except Exception:
                    formatted_time = ts_str

                is_sent = bool(a.get("telegram_sent"))
                status_class = "sent" if is_sent else "queued"
                status_label = "✅ Sent" if is_sent else "⚡ Dispatched"
                v_type = a.get("violation_type", "Violation")
                p_id = a.get("person_id", "1")
                if isinstance(p_id, bytes):
                    try:
                        p_id = int.from_bytes(p_id, "little")
                    except Exception:
                        p_id = 1

                cards_html.append(
                    f'<div class="tg-card">'
                    f'<div class="tg-card-main">'
                    f'<div class="tg-card-header">'
                    f'<span>🚨 VIOLATION REPORT #{a["id"]}</span>'
                    f'</div>'
                    f'<div class="tg-card-detail">'
                    f'👤 <b>Worker #{p_id}</b> &nbsp;•&nbsp; ⚠️ <span style="color:#f87171;font-weight:600;">{v_type}</span>'
                    f'</div>'
                    f'<div class="tg-card-meta">'
                    f'<span>🕐 {formatted_time}</span>'
                    f'<span>•</span>'
                    f'<span>📱 Telegram Channel</span>'
                    f'</div>'
                    f'</div>'
                    f'<div class="tg-status-pill {status_class}">{status_label}</div>'
                    f'</div>'
                )
            cards_html.append('</div>')
            st.markdown("".join(cards_html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# REAL-TIME DASHBOARD UPDATER
# ----------------------------------------------------------------------------
def update_dashboard_live(stat_cards_ph, violations_ph, ppe_rings_ph, recent_ph):
    """Dynamically refresh dashboard widgets during active video streaming without page reload."""
    latest_detections = logger.load_detections()
    stats = logger.compute_stats(latest_detections)
    violations = logger.compute_violations_breakdown(latest_detections)
    ppe_items = logger.compute_ppe_overview(latest_detections)
    recent = logger.compute_recent_events(n=6)

    if stat_cards_ph is not None:
        render_stat_cards(stats, placeholder=stat_cards_ph)
    if violations_ph is not None:
        render_violations_panel(violations, placeholder=violations_ph)
    if ppe_rings_ph is not None:
        render_ppe_overview(ppe_items, placeholder=ppe_rings_ph)
    if recent_ph is not None:
        render_recent_detections(recent, placeholder=recent_ph)


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        logo_b64 = _image_to_base64(LOGO_PATH)
        bottom_b64 = _image_to_base64(BOTTOM_PATH)

        logo_tag = f'<img src="{logo_b64}" class="sidebar-logo-img" alt="Logo" />' if logo_b64 else '<span style="font-size:3.5rem;margin-bottom:0.5rem;">🛡️</span>'

        st.markdown(
            f"""
            <div class="sidebar-brand">
                {logo_tag}
                <div class="sidebar-brand-title">RASED <span>AI</span></div>
                <div class="sidebar-brand-subtitle">Real-time AI Safety Equipment Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Navigation",
            options=["🏠 Dashboard", "📋 Detection History Log", "ℹ️ About"],
            index=0,
            label_visibility="collapsed",
            key="main_nav_radio",
        )

        if bottom_b64:
            st.markdown(
                f"""
                <div class="sidebar-bottom-wrap">
                    <img src="{bottom_b64}" class="sidebar-worker-img" alt="Safety Worker" />
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="sidebar-tip-card">
                <div class="sidebar-tip-header">Daily Safety Tip 💡</div>
                <div class="sidebar-tip-body">Always wear your PPE correctly and ensure it fits properly before entering active zones.</div>
                <div class="sidebar-tip-dots">
                    <span class="dot active"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return nav_choice


# ----------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard_page():
    render_header()

    detections_log = logger.load_detections()
    stats = logger.compute_stats(detections_log)
    violations = logger.compute_violations_breakdown(detections_log)
    ppe_items = logger.compute_ppe_overview(detections_log)
    recent = logger.compute_recent_events(n=6)

    stat_cards_ph = st.empty()
    render_stat_cards(stats, placeholder=stat_cards_ph)

    left_col, right_col = st.columns([2.1, 1], gap="medium")

    with right_col:
        violations_ph = st.empty()
        render_violations_panel(violations, placeholder=violations_ph)
        render_trend_panel(detections_log)

    bottom_col1, bottom_col2, bottom_col3 = st.columns([1, 1, 1], gap="medium")

    with bottom_col1:
        ppe_rings_ph = st.empty()
        render_ppe_overview(ppe_items, placeholder=ppe_rings_ph)

    with bottom_col2:
        render_telegram_feed()

    with bottom_col3:
        recent_ph = st.empty()
        render_recent_detections(recent, placeholder=recent_ph)

    def live_updater():
        update_dashboard_live(
            stat_cards_ph=stat_cards_ph,
            violations_ph=violations_ph,
            ppe_rings_ph=ppe_rings_ph,
            recent_ph=recent_ph,
        )

    with left_col:
        render_live_detection_panel(live_dashboard_updater=live_updater)


# ----------------------------------------------------------------------------
# PAGE 2: DETECTION HISTORY LOG
# ----------------------------------------------------------------------------
def render_history_log_page():
    now = datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%B %d, %Y")
    current_theme = st.session_state.get("app_theme", "dark")

    head_left, head_right = st.columns([2.6, 1.8], gap="small")
    with head_left:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <h1 style="font-size: 1.65rem; font-weight: 700; color: {'#0f172a' if current_theme == 'light' else '#f5f7fa'}; margin: 0 0 2px 0;">📋 Detection History Log</h1>
                <p style="color: {'#64748b' if current_theme == 'light' else '#8b93a7'}; margin: 0; font-size: 0.92rem;">Complete audit log of all recorded PPE violations and alert dispatches</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_right:
        h_col1, h_col2 = st.columns([1.2, 0.8], gap="small")
        with h_col1:
            st.markdown(
                f"""
                <div class="pill">
                    <div class="pill-time">{time_str}</div>
                    <div class="pill-date">{date_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h_col2:
            toggle_label = "🌙 Dark" if current_theme == "dark" else "☀️ Light"
            if st.button(toggle_label, key="hist_theme_toggle_btn", use_container_width=True):
                st.session_state["app_theme"] = "light" if current_theme == "dark" else "dark"
                st.rerun()

    if hasattr(database, "get_all_alerts"):
        alerts = database.get_all_alerts(limit=1000)
    else:
        alerts = database.get_recent_alerts(limit=1000)

    total_violations = len(alerts)
    helmet_v = sum(1 for a in alerts if "Helmet" in a.get("violation_type", ""))
    vest_v = sum(1 for a in alerts if "Vest" in a.get("violation_type", ""))
    glove_v = sum(1 for a in alerts if "Glove" in a.get("violation_type", ""))

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon red">🚨</div>
                <div>
                    <div class="stat-label">Total Logged Violations</div>
                    <div class="stat-value">{total_violations}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon purple">⛑️</div>
                <div>
                    <div class="stat-label">Helmet Violations</div>
                    <div class="stat-value">{helmet_v}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon blue">🦺</div>
                <div>
                    <div class="stat-label">Vest Violations</div>
                    <div class="stat-value">{vest_v}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon green">🧤</div>
                <div>
                    <div class="stat-label">Glove Violations</div>
                    <div class="stat-value">{glove_v}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.4, 1.3, 1.8, 1.2], gap="small")
        with f_col1:
            type_filter = st.selectbox("Violation Type", ["All Violations", "No-Helmet", "No-Vest", "No-Glove"], key="hist_type_filter")
        with f_col2:
            status_filter = st.selectbox("Dispatch Status", ["All Statuses", "✅ Telegram Sent", "⚡ Dispatched"], key="hist_status_filter")
        with f_col3:
            search_query = st.text_input("🔍 Search Worker / Date", placeholder="e.g. Worker 1 or 2026-08", key="hist_search_box")
        with f_col4:
            if alerts:
                df = pd.DataFrame(alerts)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export CSV",
                    data=csv_bytes,
                    file_name=f"rased_violations_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button("📥 Export CSV", disabled=True, use_container_width=True)

    filtered_alerts = []
    for a in alerts:
        v_type = a.get("violation_type", "")
        if type_filter != "All Violations" and type_filter not in v_type:
            continue
        is_sent = bool(a.get("telegram_sent"))
        if status_filter == "✅ Telegram Sent" and not is_sent:
            continue
        if status_filter == "⚡ Dispatched" and is_sent:
            continue
        if search_query.strip():
            sq = search_query.strip().lower()
            p_id_str = str(a.get("person_id", ""))
            ts_str = str(a.get("timestamp", ""))
            if sq not in p_id_str.lower() and sq not in ts_str.lower() and sq not in v_type.lower():
                continue
        filtered_alerts.append(a)

    if not filtered_alerts:
        st.info("No violation records match the selected filters.")
        return

    st.markdown(f"**Displaying {len(filtered_alerts)} violation record(s)**")

    grid_cols = st.columns(2, gap="medium")
    for idx, a in enumerate(filtered_alerts):
        col = grid_cols[idx % 2]
        with col:
            with st.container(border=True):
                c_img, c_info = st.columns([1, 2.2], gap="small")
                face_path = a.get("face_image_path")
                p_id = a.get("person_id", 1)
                if isinstance(p_id, bytes):
                    try:
                        p_id = int.from_bytes(p_id, "little")
                    except Exception:
                        p_id = 1
                v_type = a.get("violation_type", "Violation")
                ts_str = a.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts_str)
                    formatted_dt = dt.strftime("%B %d, %Y • %I:%M:%S %p")
                except Exception:
                    formatted_dt = ts_str

                is_sent = bool(a.get("telegram_sent"))
                status_badge = "✅ Sent to Telegram" if is_sent else "⚡ Dispatched"
                status_style = "background:#143525;color:#4ade80;border:1px solid #1e5236;" if is_sent else "background:#3b1219;color:#f87171;border:1px solid #5c1d2b;"

                with c_img:
                    if face_path and os.path.exists(face_path):
                        img_b64 = _image_to_base64(face_path)
                        st.markdown(
                            f"""
                            <div style="width:100%;height:115px;border-radius:12px;overflow:hidden;border:1px solid #ef4444;box-shadow:0 0 8px rgba(239,68,68,0.3);background:#0a0d16;display:flex;align-items:center;justify-content:center;">
                                <img src="{img_b64}" style="width:100%;height:100%;object-fit:cover;" alt="Violator Screenshot" />
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="width:100%;height:115px;border-radius:12px;background:#131826;border:1px solid #212739;display:flex;align-items:center;justify-content:center;font-size:2.4rem;">
                                ⚠️
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with c_info:
                    st.markdown(
                        f"""
                        <div style="display:flex;flex-direction:column;gap:4px;">
                            <div style="display:flex;align-items:center;justify-content:space-between;">
                                <span style="font-weight:700;font-size:1.05rem;color:{'#0f172a' if current_theme == 'light' else '#f5f7fa'};">👤 Worker #{p_id}</span>
                                <span style="font-size:0.75rem;padding:3px 8px;border-radius:6px;font-weight:600;{status_style}">{status_badge}</span>
                            </div>
                            <div style="font-size:0.92rem;font-weight:600;color:#ef4444;margin-top:2px;">
                                ⚠️ {v_type} Detected
                            </div>
                            <div style="font-size:0.78rem;color:{'#64748b' if current_theme == 'light' else '#8b93a7'};margin-top:2px;">
                                🕐 {formatted_dt}
                            </div>
                            <div style="font-size:0.75rem;color:{'#94a3b8' if current_theme == 'light' else '#64748b'};margin-top:4px;">
                                Incident Record: #{a.get('id', idx+1)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ----------------------------------------------------------------------------
# PAGE 3: ABOUT SECTION
# ----------------------------------------------------------------------------
def render_about_page():
    current_theme = st.session_state.get("app_theme", "dark")
    logo_b64 = _image_to_base64(LOGO_PATH)

    with st.container(border=True):
        b_col1, b_col2 = st.columns([1, 4], gap="medium")
        with b_col1:
            if logo_b64:
                st.markdown(f'<div style="text-align:center;padding-top:0.4rem;"><img src="{logo_b64}" style="width:115px;height:115px;object-fit:contain;" alt="RASED Logo" /></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:4rem;text-align:center;">🛡️</div>', unsafe_allow_html=True)
        with b_col2:
            st.markdown(
                f"""
                <div style="padding:0.2rem 0;">
                    <h1 style="font-size:1.9rem;font-weight:800;color:{'#0f172a' if current_theme == 'light' else '#f5f7fa'};margin:0 0 6px 0;">
                        RASED <span style="color:#f59e0b;font-size:1.3rem;font-weight:600;">(Real-time AI Safety Equipment Detection)</span>
                    </h1>
                    <p style="font-size:1rem;color:{'#334155' if current_theme == 'light' else '#c3c8d6'};line-height:1.5;margin:0 0 10px 0;">
                        An autonomous, edge-optimized Computer Vision safety management platform designed to monitor, track, and enforce Personal Protective Equipment (PPE) compliance in real-time across industrial workplaces, construction sites, and hazardous facilities.
                    </p>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;">
                        <span style="background:#1e1b4b;color:#a5b4fc;border:1px solid #3730a3;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;">🚀 YOLO11s Native</span>
                        <span style="background:#143525;color:#4ade80;border:1px solid #1e5236;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;">⚡ ByteTrack Kalman Tracker</span>
                        <span style="background:#0f2438;color:#229ED9;border:1px solid #1a3c5a;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;">✈️ Instant Telegram Dispatch</span>
                        <span style="background:#2b2512;color:#fbbf24;border:1px solid #4a3e1c;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;">🎯 Native Resolution</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown("### 🔍 Supported PPE Classes & Detection")
            st.markdown(
                """
                RASED continuously detects workers and associates protective gear with each unique tracked person:

                - ⛑️ **Hardhat / Helmet**: `Class 1 (Helmet)` vs `Class 3 (No-Helmet)`
                - 🦺 **High-Visibility Safety Vest**: `Class 6 (Vest)` vs `Class 4 (No-Vest)`
                - 🧤 **Protective Gloves**: `Class 0 (Gloves)` vs `Class 2 (No-Glove)`
                - 👤 **Worker Detection**: `Class 5 (Person)` with persistent ByteTrack tracking IDs

                > [!IMPORTANT]
                > **Adaptive Hand & Head Matching**: The system utilizes an expanded lateral margin and adaptive sensitivity floor for small objects like gloves and bare hands, ensuring reliable detection even during fast movement or extended reaching.
                """
            )

        with col_left:
            with st.container(border=True):
                st.markdown("### 🛠️ Technology Stack")
                st.markdown(
                    """
                    | Component | Technology / Library | Purpose |
                    | :--- | :--- | :--- |
                    | **Object Detection** | Ultralytics YOLO11s | High-speed multi-class inference |
                    | **Multi-Person Tracking** | ByteTrack (`bytetrack.yaml`) | Kalman Filter IoU tracking (30+ FPS CPU) |
                    | **Application Framework** | Streamlit | Responsive Web Dashboard & Controls |
                    | **Video Processing** | OpenCV (`cv2.CAP_DSHOW`) | Hardware-accelerated camera & video feed |
                    | **Database Storage** | SQLite (`ppe_guard.db`) | Persistent violation logs & worker states |
                    | **Data Visualization** | Plotly | Dynamic donut & compliance trend charts |
                    | **Alert Notifications** | Telegram Bot API | Instant photo violation dispatch |
                    """
                )

    with col_right:
        with st.container(border=True):
            st.markdown("### ⚡ End-to-End System Architecture")
            st.markdown(
                """
                1. **Input Stream Capture**:
                   DirectShow hardware camera capture or local MP4 video decoding.
                2. **Frame Preprocessing**:
                   Input normalized and resized directly to standard **640×640**.
                3. **YOLO Inference & ByteTrack Tracking**:
                   Detects workers and PPE items, maintaining static track IDs (`Worker #1`, `Worker #2`).
                4. **Spatial Overlap & Compliance Engine**:
                   Calculates bounding box intersections to determine Helmet, Vest, and Glove compliance per person.
                5. **Automated Incident Logging & Telegram Dispatch**:
                   Crops violator screenshot and dispatches real-time alerts to the Telegram safety channel.
                6. **Live Dashboard Streaming**:
                   Real-time dynamic placeholders update metrics, donut charts, and hourly trends live without page reloads.
                """
            )

        with col_right:
            with st.container(border=True):
                st.markdown("### 📋 Safety Compliance Standards & Rules")
                st.markdown(
                    """
                    - **Strict 3-Point Check**: A worker is marked **Compliant (✅)** only when wearing **all 3 items** (Helmet, Vest, Gloves).
                    - **Cooldown Alerts**: Duplicate alerts are rate-limited per worker (configurable cooldown) to prevent alert flooding.
                    - **Automatic Evidence Archiving**: Every violation captures an isolated face/upper body photo in `static/faces/` for audit compliance.
                    """
                )


# ----------------------------------------------------------------------------
# MAIN ROUTER
# ----------------------------------------------------------------------------
def main():
    inject_css()
    nav_choice = render_sidebar()

    if nav_choice == "🏠 Dashboard":
        render_dashboard_page()
    elif nav_choice == "📋 Detection History Log":
        render_history_log_page()
    elif nav_choice == "ℹ️ About":
        render_about_page()


if __name__ == "__main__":
    main()