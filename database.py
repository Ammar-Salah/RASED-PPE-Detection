import sqlite3
import threading
from datetime import datetime
import os
from config import DB_PATH

# Thread lock for DB access
db_lock = threading.Lock()

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                person_id INTEGER,
                helmet_status TEXT,
                vest_status TEXT,
                gloves_status TEXT,
                is_compliant BOOLEAN
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                person_id INTEGER,
                violation_type TEXT,
                face_image_path TEXT,
                telegram_sent BOOLEAN DEFAULT 0,
                acknowledged BOOLEAN DEFAULT 0
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                source TEXT,
                total_persons INTEGER,
                total_violations INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

def log_detection(person_id, helmet, vest, gloves, is_compliant):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO detections (timestamp, person_id, helmet_status, vest_status, gloves_status, is_compliant)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, person_id, helmet, vest, gloves, is_compliant))
        conn.commit()
        conn.close()

def log_alert(person_id, violation_type, face_image_path):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO alerts (timestamp, person_id, violation_type, face_image_path)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, person_id, violation_type, face_image_path))
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alert_id

def mark_alert_sent(alert_id):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE alerts SET telegram_sent = 1 WHERE id = ?
        ''', (alert_id,))
        conn.commit()
        conn.close()

def get_recent_alerts(limit=50):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

def get_all_alerts(limit=1000):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

def get_recent_detections(limit=100):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

def get_statistics():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        # total detections
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        # total persons
        cursor.execute('SELECT COUNT(DISTINCT person_id) FROM detections')
        total_persons = cursor.fetchone()[0]
        
        # total violations (alerts count)
        cursor.execute('SELECT COUNT(*) FROM alerts')
        total_violations = cursor.fetchone()[0]
        
        # compliance rate based on detections
        cursor.execute('SELECT COUNT(*) FROM detections WHERE is_compliant = 1')
        compliant_detections = cursor.fetchone()[0]
        
        compliance_rate = (compliant_detections / total_detections * 100) if total_detections > 0 else 0
        
        # violations by type
        cursor.execute('SELECT violation_type, COUNT(*) as count FROM alerts GROUP BY violation_type')
        violations_by_type = {row['violation_type']: row['count'] for row in cursor.fetchall()}
        
        # class distribution could refer to detections by classes
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN helmet_status='yes' THEN 1 ELSE 0 END) as helmet_yes,
                SUM(CASE WHEN helmet_status='no' THEN 1 ELSE 0 END) as helmet_no,
                SUM(CASE WHEN vest_status='yes' THEN 1 ELSE 0 END) as vest_yes,
                SUM(CASE WHEN vest_status='no' THEN 1 ELSE 0 END) as vest_no,
                SUM(CASE WHEN gloves_status='yes' THEN 1 ELSE 0 END) as gloves_yes,
                SUM(CASE WHEN gloves_status='no' THEN 1 ELSE 0 END) as gloves_no
            FROM detections
        ''')
        dist_row = cursor.fetchone()
        class_distribution = dict(dist_row) if dist_row else {}
        
        # hourly stats for the last 24h
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, 
                   SUM(CASE WHEN is_compliant=1 THEN 1 ELSE 0 END) as compliant,
                   SUM(CASE WHEN is_compliant=0 THEN 1 ELSE 0 END) as violations
            FROM detections 
            GROUP BY hour
        ''')
        hourly_stats = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'total_persons': total_persons,
            'total_violations': total_violations,
            'compliance_rate': compliance_rate,
            'violations_by_type': violations_by_type,
            'hourly_stats': hourly_stats,
            'class_distribution': class_distribution
        }

def get_session_stats():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sessions ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

def start_session(source):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO sessions (start_time, source, total_persons, total_violations)
            VALUES (?, ?, 0, 0)
        ''', (timestamp, source))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id

def end_session(session_id, total_persons, total_violations):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, total_persons = ?, total_violations = ? 
            WHERE id = ?
        ''', (timestamp, total_persons, total_violations, session_id))
        conn.commit()
        conn.close()
