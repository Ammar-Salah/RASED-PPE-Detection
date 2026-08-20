import os

# === Model Configuration ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# === Class Names (from YOLO11 training) ===
CLASS_NAMES = {
    0: 'Gloves',
    1: 'Helmet',
    2: 'No-Glove',
    3: 'No-Helmet',
    4: 'No-Vest',
    5: 'Person',
    6: 'Vest'
}

# PPE Classes (positive = wearing PPE)
PPE_POSITIVE = {0: 'Gloves', 1: 'Helmet', 6: 'Vest'}
# Violation Classes (negative = NOT wearing PPE)
PPE_NEGATIVE = {2: 'No-Glove', 3: 'No-Helmet', 4: 'No-Vest'}
# Map violation to its positive counterpart
VIOLATION_TO_PPE = {2: 'Gloves', 3: 'Helmet', 4: 'Vest'}
PERSON_CLASS_ID = 5

# === Colors for each class (BGR for OpenCV) ===
CLASS_COLORS = {
    0: (0, 255, 0),      # Gloves - Green
    1: (0, 255, 0),      # Helmet - Green
    2: (0, 0, 255),      # No-Glove - Red
    3: (0, 0, 255),      # No-Helmet - Red
    4: (0, 0, 255),      # No-Vest - Red
    5: (255, 165, 0),    # Person - Orange
    6: (0, 255, 0),      # Vest - Green
}

# === Telegram Bot Configuration ===
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'  # Get from @BotFather
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_HERE'      # Get from @userinfobot
TELEGRAM_ENABLED = False  # Set to True after configuring token and chat ID

# === Alert Configuration ===
ALERT_COOLDOWN = 60  # seconds between alerts for same person+violation
FACE_CROP_PADDING = 20  # pixels padding around face crop

# === Database ===
DB_PATH = os.path.join(os.path.dirname(__file__), 'ppe_guard.db')

# === Server ===
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False

# === Faces Storage ===
FACES_DIR = os.path.join(os.path.dirname(__file__), 'static', 'faces')
os.makedirs(FACES_DIR, exist_ok=True)
