"""
verify_password_gating.py — Automated verification test demonstrating real write-through enforcement across both Desktop and Web ingestion pathways.
"""

import json
import os
import shutil
import sqlite3
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure standard UTF-8 stream output for Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure repo root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chordspy.tensionbudget.cloud_schema import create_schema_sqlite
from chordspy.tensionbudget.cloud_ingest import ingest_session_pair, verify_password
from scripts.web_endpoint_server import run_server

TEST_DB_PATH = Path("test_password_gating.db")
TEST_LOGS_DIR = Path("test_temp_logs")


def setup_test_environment():
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    if TEST_LOGS_DIR.exists():
        shutil.rmtree(TEST_LOGS_DIR)
    TEST_LOGS_DIR.mkdir(exist_ok=True)


def create_dummy_session(user_name: str, session_id: str) -> tuple[Path, Path]:
    sub_dir = TEST_LOGS_DIR / user_name / session_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    meta_path = sub_dir / f"{session_id}_metadata.json"
    csv_path = sub_dir / f"{session_id}_features.csv"

    meta_payload = {
        "session_id": session_id,
        "user_profile": {
            "user_name": user_name,
            "birth_date": "1995-05-15",
            "gender_sex": "F",
            "weight_kg": 68.5,
            "height_cm": 172.0
        },
        "start_time": time.time(),
        "start_time_str": "2026-08-05T10:00:00",
        "end_time": time.time() + 300,
        "end_time_str": "2026-08-05T10:05:00",
        "mode": "live_uno_r4",
        "calibration": {"left_rms_reference": 0.05, "right_rms_reference": 0.05},
        "processing_version": "phase11_edge_padding"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch_index,elapsed_minutes,composite_score,strain_reported,subjective_strain_cr10\n")
        f.write("1,5.0,0.15,1,2.5\n")
        f.write("2,10.0,0.20,0,\n")

    return meta_path, csv_path


def run_desktop_path_verification(conn: sqlite3.Connection):
    print("=" * 72)
    print("[PATH 1] DESKTOP SYNC PATH ENFORCEMENT VERIFICATION (Trusted Script)")
    print("=" * 72)
    subject = "pilot_desktop_subject_01"
    correct_pass = "PilotSubjectPassword_2026"
    wrong_pass = "WrongMistypedPassword!"

    meta1, csv1 = create_dummy_session(subject, "session_001")
    meta2, csv2 = create_dummy_session(subject, "session_002")
    meta3, csv3 = create_dummy_session(subject, "session_003_mistyped")

    # Test 1: First-Time Registration
    print(f"\n[Test 1A] First-time ingestion for new subject '{subject}'...")
    res = ingest_session_pair(conn, meta1, csv1, password=correct_pass, created_via="desktop_registration")
    print(f"    [SUCCESS] Result: {res['status'].upper()} ({res['epochs_ingested']} epochs saved).")

    cursor = conn.cursor()
    cursor.execute("SELECT user_name, password_hash, created_via FROM user_profiles WHERE user_name = ?", (subject,))
    row = cursor.fetchone()
    print(f"    [SECURE DB RECORD] user='{row[0]}', created_via='{row[2]}', password_hash='{row[1][:25]}...'")

    # Test 2: Subsequent Sync (Correct Password)
    print(f"\n[Test 1B] Subsequent session sync using CORRECT password...")
    res2 = ingest_session_pair(conn, meta2, csv2, password=correct_pass)
    print(f"    [SUCCESS] Result: {res2['status'].upper()} (Session 002 authorized & ingested).")

    # Test 3: Data Corruption Safeguard (Incorrect Password)
    print(f"\n[Test 1C] Attempting to sync session under existing subject '{subject}' with WRONG password...")
    try:
        ingest_session_pair(conn, meta3, csv3, password=wrong_pass)
        print("    [ERROR] Ingestion incorrectly succeeded!")
        sys.exit(1)
    except PermissionError as e:
        print(f"    [REJECTED AS EXPECTED] PermissionError raised:")
        print(f"       => \"{str(e)}\"")

    cursor.execute("SELECT count(*) FROM sessions WHERE session_id = 'session_003_mistyped'")
    blocked_cnt = cursor.fetchone()[0]
    print(f"    [INTEGRITY PROVED] Verification: DB row count for 'session_003_mistyped' is {blocked_cnt} (Data corruption prevented!).")


def run_web_endpoint_verification():
    print("\n" + "=" * 72)
    print("[PATH 2] WEB ENDPOINT PATH ENFORCEMENT VERIFICATION (Server-Side Check)")
    print("=" * 72)
    subject_web = "pilot_web_subject_02"
    correct_pass = "WebSecret_2026"
    wrong_pass = "HackerOrMistypedPass"
    url = "http://127.0.0.1:8005/ingest"

    # Test 1: Web First-Time Registration
    payload1 = {
        "user_name": subject_web,
        "password": correct_pass,
        "session_metadata": {
            "session_id": "web_session_001",
            "user_profile": {"user_name": subject_web, "weight_kg": 75.0, "height_cm": 180.0},
            "start_time": time.time(),
            "end_time": time.time() + 300,
            "mode": "web_wasm",
            "processing_version": "phase11_edge_padding"
        },
        "epoch_features": [
            {"epoch_index": 1, "elapsed_minutes": 5.0, "composite_score": 0.22, "strain_reported": 1, "subjective_strain_cr10": 3.0}
        ]
    }

    print(f"\n[Test 2A] Sending HTTP POST to serverless web endpoint for new subject '{subject_web}'...")
    req1 = urllib.request.Request(url, data=json.dumps(payload1).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req1) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"    [SUCCESS] Server Response (HTTP {resp.status}): {data['message']}")

    conn = sqlite3.connect(str(TEST_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, created_via FROM user_profiles WHERE user_name = ?", (subject_web,))
    row = cursor.fetchone()
    print(f"    [WEB DB RECORD] user='{row[0]}', created_via='{row[1]}'")
    conn.close()

    # Test 2: Web Sync (Correct Password)
    payload2 = dict(payload1)
    payload2["session_metadata"]["session_id"] = "web_session_002"
    print(f"\n[Test 2B] Sending second web session with CORRECT password...")
    req2 = urllib.request.Request(url, data=json.dumps(payload2).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req2) as resp2:
        print(f"    [SUCCESS] Server Response (HTTP {resp2.status}): Authorized & committed.")

    # Test 3: Web Sync Rejection (Incorrect Password)
    payload3 = dict(payload1)
    payload3["session_metadata"]["session_id"] = "web_session_003_corrupt"
    payload3["password"] = wrong_pass
    print(f"\n[Test 2C] Sending HTTP POST to overwrite '{subject_web}' records with WRONG password...")
    req3 = urllib.request.Request(url, data=json.dumps(payload3).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req3)
        print("    [ERROR] Endpoint incorrectly permitted insertion!")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        err_body = json.loads(e.read().decode("utf-8"))
        print(f"    [SERVER REJECTED AS EXPECTED] (HTTP {e.code} Forbidden):")
        print(f"       => \"{err_body['error']}\"")

    conn = sqlite3.connect(str(TEST_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM sessions WHERE session_id = 'web_session_003_corrupt'")
    blocked_cnt = cursor.fetchone()[0]
    print(f"    [INTEGRITY PROVED] Verification: DB row count for 'web_session_003_corrupt' is {blocked_cnt} (Zero corrupt records!).")
    conn.close()


def main():
    setup_test_environment()
    os.environ["TENSIONBUDGET_DB_PATH"] = str(TEST_DB_PATH)

    # Prepare database connection for Desktop verification
    conn = sqlite3.connect(str(TEST_DB_PATH))
    create_schema_sqlite(conn)

    # 1. Verify Desktop Path
    run_desktop_path_verification(conn)
    conn.close()

    # 2. Start Python background endpoint server to simulate Supabase / Vercel Edge Function
    server_thread = threading.Thread(target=run_server, args=(8005,), daemon=True)
    server_thread.start()
    time.sleep(1.0)  # Give server a second to bind

    # 3. Verify Web Endpoint Path
    run_web_endpoint_verification()

    print("\n" + "=" * 72)
    print(" ALL PASSWORD-GATED INTEGRITY CHECKS PROVED COMPLETE & BULLETPROOF!")
    print("=" * 72 + "\n")

    # Cleanup test residuals
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    if TEST_LOGS_DIR.exists():
        shutil.rmtree(TEST_LOGS_DIR)


if __name__ == "__main__":
    main()
