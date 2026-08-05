"""
web_endpoint_server.py — Python serverless endpoint handler and test server for web browser session ingestion.

Mirrors the Supabase Edge Function (supabase/functions/ingest-session/index.ts).
Acts as a trusted server-side enforcement checkpoint between web browser Wasm clients and the database.
Enforces PBKDF2 password verification before authorizing write operations.
"""

import json
import os
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Ensure standard UTF-8 stream output for Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure repository root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chordspy.tensionbudget.cloud_schema import create_schema_sqlite
from chordspy.tensionbudget.cloud_ingest import hash_password, verify_password, _parse_val

class WebIngestionHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if self.path != "/ingest":
            self.send_error(404, "Endpoint not found. Post to /ingest")
            return

        db_file = os.getenv("TENSIONBUDGET_DB_PATH", "tensionbudget_cloud.db")
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")

        try:
            payload = json.loads(post_data)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON payload format."})
            return

        user_name = payload.get("user_name")
        password = payload.get("password")
        session_meta = payload.get("session_metadata", {})
        epoch_features = payload.get("epoch_features", [])

        if not user_name or not password:
            self._send_json(400, {"error": "Missing required user_name and password attributes."})
            return

        conn = sqlite3.connect(db_file)
        create_schema_sqlite(conn)
        cursor = conn.cursor()

        try:
            # ── Server-Side Password & Data Integrity Check ────────────────────────
            cursor.execute("SELECT user_name, password_hash FROM user_profiles WHERE user_name = ?", (user_name,))
            existing = cursor.fetchone()

            if existing and existing[1]:
                # Existing subject — verify stored hash server-side using trusted database credentials
                stored_hash = existing[1]
                if not verify_password(password, stored_hash):
                    print(f"[REJECTED] [Web Endpoint Server] Blocked insertion for subject '{user_name}': Invalid password.")
                    conn.close()
                    self._send_json(403, {
                        "error": "Data Integrity Error: Incorrect password for existing subject account. Ingestion blocked server-side to prevent silent data corruption."
                    })
                    return
            else:
                # First-time user registration via web endpoint
                new_hash = hash_password(password)
                prof = session_meta.get("user_profile", {})
                birth_date = prof.get("birth_date", "1900-01-01")
                gender_sex = prof.get("gender_sex", "Unknown")
                weight_kg = _parse_val(prof.get("weight_kg"), "float")
                height_cm = _parse_val(prof.get("height_cm"), "float")

                if not existing:
                    cursor.execute(
                        """
                        INSERT INTO user_profiles (user_name, birth_date, gender_sex, weight_kg, height_cm, updated_at, password_hash, created_via)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (user_name, birth_date, gender_sex, weight_kg, height_cm, time.time(), new_hash, "web_endpoint")
                    )
                else:
                    cursor.execute("UPDATE user_profiles SET password_hash = ? WHERE user_name = ?", (new_hash, user_name))

            # ── Password authorized -> ingest session telemetry ─────────────────────
            if session_meta:
                sid = session_meta.get("session_id", f"web_{int(time.time())}")
                start_t = _parse_val(session_meta.get("start_time"), "float")
                end_t = _parse_val(session_meta.get("end_time"), "float")
                cal = session_meta.get("calibration", {})
                proc_ver = session_meta.get("processing_version", "phase11_edge_padding")

                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (sid,))
                if cursor.fetchone():
                    cursor.execute(
                        "UPDATE sessions SET user_name=?, start_time=?, end_time=?, mode=?, calibration_rms_left=?, calibration_rms_right=?, processing_version=? WHERE session_id=?",
                        (user_name, start_t, end_t, session_meta.get("mode", "web_wasm"), _parse_val(cal.get("left_rms_reference"), "float"), _parse_val(cal.get("right_rms_reference"), "float"), proc_ver, sid)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO sessions (session_id, user_name, start_time, end_time, mode, calibration_rms_left, calibration_rms_right, processing_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sid, user_name, start_t, end_t, session_meta.get("mode", "web_wasm"), _parse_val(cal.get("left_rms_reference"), "float"), _parse_val(cal.get("right_rms_reference"), "float"), proc_ver)
                    )

                for idx, row in enumerate(epoch_features):
                    ep_idx = _parse_val(row.get("epoch_index", idx), "int")
                    elapsed_min = _parse_val(row.get("elapsed_minutes", ep_idx * 5.0), "float")
                    comp = _parse_val(row.get("composite_score", 0.0), "float")
                    strain = _parse_val(row.get("strain_reported", 0), "int")
                    cr10 = _parse_val(row.get("subjective_strain_cr10"), "float")

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO epoch_features (session_id, epoch_index, elapsed_minutes, composite_score, strain_reported, subjective_strain_cr10)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (sid, ep_idx, elapsed_min, comp, strain, cr10)
                    )

            conn.commit()
            conn.close()
            print(f"[SUCCESS] [Web Endpoint Server] Authorized & saved session for subject '{user_name}'.")
            self._send_json(200, {"status": "success", "message": f"Session telemetry successfully committed for subject '{user_name}'."})

        except Exception as e:
            if conn:
                conn.close()
            print(f"[ERROR] [Web Endpoint Server] Error during database operation: {e}")
            self._send_json(500, {"error": str(e)})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine request polling prints
        return


def run_server(port=8001):
    db_file = os.getenv("TENSIONBUDGET_DB_PATH", "tensionbudget_cloud.db")
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, WebIngestionHandler)
    print(f"[WEB ENDPOINT] Listening on http://127.0.0.1:{port}/ingest (DB: {db_file})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    run_server(port)
