import requests
import threading
import queue
import time
import os
import html
from datetime import datetime

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED
except ImportError:
    TELEGRAM_BOT_TOKEN = '8645512067:AAHwXrUH9z8HSJom8xjhrO3Rewfq5AiJE5A'
    TELEGRAM_CHAT_ID = '1331491729'
    TELEGRAM_ENABLED = False


class TelegramAlertBot:
    """Telegram alert system for PPE violations with asynchronous background queue processing."""

    def __init__(self, bot_token=None, chat_id=None, enabled=None):
        self.bot_token = bot_token if bot_token is not None else TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id if chat_id is not None else TELEGRAM_CHAT_ID
        self.enabled = enabled if enabled is not None else TELEGRAM_ENABLED
        self.api_url = f'https://api.telegram.org/bot{self.bot_token}'
        self.alert_queue = queue.Queue()
        self.running = False
        self._worker_thread = None

    def start(self):
        """Start the background alert sending thread."""
        if not self.enabled:
            print('[Telegram] Bot disabled. Set TELEGRAM_ENABLED=True in config.py')
            return
        if self.running:
            print('[Telegram] Alert bot is already running')
            return

        self.running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        print('[Telegram] Alert bot started')

    def stop(self):
        """Stop the background alert sending thread."""
        self.running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        print('[Telegram] Alert bot stopped')

    def send_alert(self, person_id, violation_type, face_image_path=None, timestamp=None):
        """Queue an alert to be sent."""
        if not self.enabled:
            return
        alert = {
            'person_id': person_id,
            'violation_type': violation_type,
            'face_image_path': face_image_path,
            'timestamp': timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.alert_queue.put(alert)
        print(f'[Telegram] Queued alert for Person #{person_id}: {violation_type}')

    def _process_queue(self):
        """Background worker to process alert queue."""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_telegram_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f'[Telegram] Error processing alert: {e}')

    def _send_telegram_alert(self, alert):
        """Send alert to Telegram with face photo or text message with retry logic."""
        person_id = alert.get('person_id', 'Unknown')
        violation_type = alert.get('violation_type', 'Unknown')
        timestamp = alert.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        face_image_path = alert.get('face_image_path')

        safe_person_id = html.escape(str(person_id))
        safe_violation_type = html.escape(str(violation_type))
        safe_timestamp = html.escape(str(timestamp))

        caption_text = (
            "🚨 <b>PPE VIOLATION ALERT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>Person ID:</b> #{safe_person_id}\n"
            f"⚠️ <b>Violation:</b> {safe_violation_type}\n"
            f"🕐 <b>Time:</b> {safe_timestamp}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📍 <i>Action Required!</i>"
        )

        has_photo = bool(face_image_path and os.path.isfile(face_image_path))
        max_retries = 3
        retry_delay = 2

        for attempt in range(1, max_retries + 1):
            try:
                if has_photo:
                    url = f"{self.api_url}/sendPhoto"
                    data = {
                        'chat_id': self.chat_id,
                        'caption': caption_text,
                        'parse_mode': 'HTML'
                    }
                    with open(face_image_path, 'rb') as photo_file:
                        files = {'photo': photo_file}
                        response = requests.post(url, data=data, files=files, timeout=15)
                else:
                    url = f"{self.api_url}/sendMessage"
                    payload = {
                        'chat_id': self.chat_id,
                        'text': caption_text,
                        'parse_mode': 'HTML'
                    }
                    response = requests.post(url, json=payload, timeout=15)

                if response.status_code == 200 and response.json().get('ok'):
                    print(f"[Telegram] Alert sent successfully for Person #{person_id} (Attempt {attempt})")
                    return True
                else:
                    print(f"[Telegram] Failed to send alert (Attempt {attempt}/{max_retries}): {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[Telegram] Exception sending alert (Attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                time.sleep(retry_delay)

        print(f"[Telegram] Could not send alert for Person #{person_id} after {max_retries} attempts")
        return False

    def test_connection(self):
        """Test if the bot token and chat ID are valid. Returns (success: bool, message: str)"""
        if not self.enabled:
            return False, 'Telegram bot is disabled in config'
        if not self.bot_token or self.bot_token == 'YOUR_BOT_TOKEN_HERE':
            return False, 'Telegram bot token is not configured'
        try:
            response = requests.get(f'{self.api_url}/getMe', timeout=10)
            if response.status_code == 200 and response.json().get('ok'):
                bot_info = response.json().get('result', {})
                return True, f"Connected as @{bot_info.get('username', 'unknown')}"
            return False, f'API error: {response.text}'
        except Exception as e:
            return False, f'Connection error: {e}'


if __name__ == '__main__':
    print("[Telegram] Testing TelegramAlertBot initialization...")
    bot = TelegramAlertBot()
    status, message = bot.test_connection()
    print(f"[Telegram] Connection test: {status} -> {message}")
