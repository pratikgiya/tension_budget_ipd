"""
cloud_ingest.py — Idempotent data ingestion engine for Phase 12 Cloud PostgreSQL & ML feature pipeline.

Scans local logging directories (output_logs/, output-data/) for paired JSON metadata manifests and feature CSV logs.
Normalizes explicit missingness cells (converting resting empty strings to SQL NULL while maintaining mdf_computed booleans),
safely segments legacy pre-Phase 11 sessions with the 'legacy_constant_padding' marker, and performs idempotent batch upserts.
"""

import csv
import hashlib
import hmac
import json
import sqlite3
import secrets
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .config import TBConfig

def hash_password(password: str) -> str:
    """Hashes a plaintext password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    hashed_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"pbkdf2:sha256:100000${salt}${hashed_bytes.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a plaintext password against a stored PBKDF2 hash."""
    if not stored_hash or not stored_hash.startswith("pbkdf2:sha256:"):
        return False
    parts = stored_hash.split("$")
    if len(parts) != 3:
        return False
    salt, expected_hex = parts[1], parts[2]
    computed_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return hmac.compare_digest(computed_bytes.hex(), expected_hex)


def _parse_val(val: Any, val_type: str = "float") -> Any:
    """Helper to convert string/raw CSV values to typed Python values for SQL insertion (None -> NULL)."""
    if val is None or val == "" or str(val).strip().lower() in ("nan", "none", "null"):
        return None
    try:
        if val_type == "int":
            return int(float(val))
        elif val_type == "bool":
            s = str(val).strip().lower()
            return 1 if s in ("1", "true", "t", "yes", "1.0") else 0
        elif val_type == "float":
            return float(val)
        elif val_type == "str":
            return str(val).strip()
    except (ValueError, TypeError):
        return None
    return val


class UniversalCursor:
    """Wraps a DB-API cursor to seamlessly translate SQLite placeholders (?) to PostgreSQL (%s) when connecting to Postgres."""
    def __init__(self, raw_cursor, is_postgres: bool):
        self._cursor = raw_cursor
        self._is_pg = is_postgres

    def execute(self, sql: str, params=None):
        if self._is_pg and "?" in sql:
            sql = sql.replace("?", "%s")
        if params is not None:
            return self._cursor.execute(sql, params)
        return self._cursor.execute(sql)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


def ingest_session_pair(
    conn: Any,
    json_path: Path,
    csv_path: Path,
    password: Optional[str] = None,
    interactive: bool = False,
    created_via: str = "desktop_registration"
) -> Dict[str, Any]:
    """
    Ingests a matched pair of session metadata JSON and feature CSV into the database connection.
    Enforces password gating BEFORE any write operation to prevent accidental subject data corruption.
    Returns summary statistics for the ingested session.
    """
    raw_cursor = conn.cursor()
    is_pg = "psycopg" in str(type(conn)).lower() or "postgres" in str(type(conn)).lower()
    cursor = UniversalCursor(raw_cursor, is_pg)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = str(meta.get("session_id", json_path.stem.replace("_metadata", "")))
    user_prof = meta.get("user_profile", {})
    user_name = str(user_prof.get("user_name", "Anonymous"))
    birth_date = user_prof.get("birth_date", "1900-01-01")
    gender_sex = user_prof.get("gender_sex", "Unknown")
    weight_kg = _parse_val(user_prof.get("weight_kg"), "float")
    height_cm = _parse_val(user_prof.get("height_cm"), "float")

    # ── Password Gating & Subject Data Integrity Check ────────────────────────
    try:
        cursor.execute("SELECT user_name, password_hash FROM user_profiles WHERE user_name = ?", (user_name,))
        existing = cursor.fetchone()
    except sqlite3.OperationalError:
        existing = None

    if existing and len(existing) > 1 and existing[1]:
        # Existing subject with stored password — verify BEFORE proceeding
        stored_hash = existing[1]
        if not password and interactive:
            import getpass
            password = getpass.getpass(f"[AUTH REQUIRED] Subject '{user_name}' is password protected. Enter password: ")
        if not password or not verify_password(password, stored_hash):
            raise PermissionError(f"Data Integrity Error: Invalid or missing password for subject '{user_name}'. Ingestion rejected to prevent silent data corruption.")
    else:
        # New user (or existing legacy profile without password) — set up password if provided/interactive
        if not password and interactive:
            import getpass
            password = getpass.getpass(f"[NEW SUBJECT] Subject '{user_name}' detected — set a password for data protection: ")
        new_hash = hash_password(password) if password else None
        
        if not existing:
            cursor.execute(
                """
                INSERT INTO user_profiles (user_name, birth_date, gender_sex, weight_kg, height_cm, updated_at, password_hash, created_via)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_name, birth_date, gender_sex, weight_kg, height_cm, time.time(), new_hash, created_via)
            )
        elif new_hash:
            cursor.execute("UPDATE user_profiles SET password_hash = ? WHERE user_name = ?", (new_hash, user_name))

    if existing:
        cursor.execute(
            """
            UPDATE user_profiles 
            SET birth_date = ?, gender_sex = ?, weight_kg = ?, height_cm = ?, updated_at = ?
            WHERE user_name = ?
            """,
            (birth_date, gender_sex, weight_kg, height_cm, time.time(), user_name)
        )

    # Tier 2: Upsert Session Metadata & Processing Version Segmentation
    start_time = _parse_val(meta.get("start_time"), "float")
    start_time_str = meta.get("start_time_str", "")
    end_time = _parse_val(meta.get("end_time"), "float")
    end_time_str = meta.get("end_time_str", "")
    mode = meta.get("mode", "unknown")
    calib = meta.get("calibration", {})
    calib_l = _parse_val(calib.get("left_rms_reference"), "float")
    calib_r = _parse_val(calib.get("right_rms_reference"), "float")

    # Check for processing_version (Phase 11+ marker vs older zero-padded sessions)
    proc_version = meta.get("processing_version", "legacy_constant_padding")

    cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if cursor.fetchone():
        cursor.execute(
            """
            UPDATE sessions
            SET user_name = ?, start_time = ?, start_time_str = ?, end_time = ?, end_time_str = ?,
                mode = ?, calibration_rms_left = ?, calibration_rms_right = ?, processing_version = ?
            WHERE session_id = ?
            """,
            (user_name, start_time, start_time_str, end_time, end_time_str, mode, calib_l, calib_r, proc_version, session_id)
        )
    else:
        cursor.execute(
            """
            INSERT INTO sessions (
                session_id, user_name, start_time, start_time_str, end_time, end_time_str, mode,
                calibration_rms_left, calibration_rms_right, processing_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_name, start_time, start_time_str, end_time, end_time_str, mode, calib_l, calib_r, proc_version)
        )

    # Tier 3: Ingest Epoch Feature Rows
    epochs_ingested = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_idx = _parse_val(row.get("epoch_index"), "int")
            if epoch_idx is None:
                continue

            # Extract features with safe missingness handling
            row_data = (
                _parse_val(row.get("elapsed_minutes"), "float"),
                _parse_val(row.get("composite_score"), "float"),
                _parse_val(row.get("eindex_live_left"), "float"),
                _parse_val(row.get("eindex_live_right"), "float"),
                _parse_val(row.get("eindex_cumulative_left"), "float"),
                _parse_val(row.get("eindex_cumulative_right"), "float"),
                _parse_val(row.get("short_suma_penalty_left"), "float"),
                _parse_val(row.get("short_suma_penalty_right"), "float"),
                _parse_val(row.get("total_suma_bursts_left"), "float"),
                _parse_val(row.get("total_suma_bursts_right"), "float"),
                _parse_val(row.get("gap_frequency_left"), "float"),
                _parse_val(row.get("gap_frequency_right"), "float"),
                _parse_val(row.get("apdf_10_left"), "float"),
                _parse_val(row.get("apdf_10_right"), "float"),
                _parse_val(row.get("apdf_50_left"), "float"),
                _parse_val(row.get("apdf_50_right"), "float"),
                _parse_val(row.get("apdf_90_left"), "float"),
                _parse_val(row.get("apdf_90_right"), "float"),
                _parse_val(row.get("asymmetry_index_ai"), "float"),
                _parse_val(row.get("asymmetry_penalty_applied"), "bool"),
                _parse_val(row.get("mdf_hz_left"), "float"),
                _parse_val(row.get("mdf_hz_right"), "float"),
                _parse_val(row.get("mnf_hz_left"), "float"),
                _parse_val(row.get("mnf_hz_right"), "float"),
                _parse_val(row.get("fatigue_slope_left"), "float"),
                _parse_val(row.get("fatigue_slope_right"), "float"),
                _parse_val(row.get("mdf_r_squared_left"), "float"),
                _parse_val(row.get("mdf_r_squared_right"), "float"),
                _parse_val(row.get("n_windows_left"), "float"),
                _parse_val(row.get("n_windows_right"), "float"),
                _parse_val(row.get("mdf_computed_left"), "bool"),
                _parse_val(row.get("is_fatiguing_left"), "bool"),
                _parse_val(row.get("mdf_computed_right"), "bool"),
                _parse_val(row.get("is_fatiguing_right"), "bool"),
                0 if _parse_val(row.get("strain_reported"), "int") is None else _parse_val(row.get("strain_reported"), "int"),
                _parse_val(row.get("subjective_strain_cr10"), "float"),
            )

            cursor.execute(
                "SELECT feature_id FROM epoch_features WHERE session_id = ? AND epoch_index = ?",
                (session_id, epoch_idx)
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE epoch_features
                    SET elapsed_minutes=?, composite_score=?, eindex_live_left=?, eindex_live_right=?,
                        eindex_cumulative_left=?, eindex_cumulative_right=?, short_suma_penalty_left=?, short_suma_penalty_right=?,
                        total_suma_bursts_left=?, total_suma_bursts_right=?, gap_frequency_left=?, gap_frequency_right=?,
                        apdf_10_left=?, apdf_10_right=?, apdf_50_left=?, apdf_50_right=?, apdf_90_left=?, apdf_90_right=?,
                        asymmetry_index_ai=?, asymmetry_penalty_applied=?, mdf_hz_left=?, mdf_hz_right=?,
                        mnf_hz_left=?, mnf_hz_right=?, fatigue_slope_left=?, fatigue_slope_right=?,
                        mdf_r_squared_left=?, mdf_r_squared_right=?, n_windows_left=?, n_windows_right=?,
                        mdf_computed_left=?, is_fatiguing_left=?, mdf_computed_right=?, is_fatiguing_right=?,
                        strain_reported=?, subjective_strain_cr10=?
                    WHERE session_id = ? AND epoch_index = ?
                    """,
                    (*row_data, session_id, epoch_idx)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO epoch_features (
                        session_id, epoch_index, elapsed_minutes, composite_score, eindex_live_left, eindex_live_right,
                        eindex_cumulative_left, eindex_cumulative_right, short_suma_penalty_left, short_suma_penalty_right,
                        total_suma_bursts_left, total_suma_bursts_right, gap_frequency_left, gap_frequency_right,
                        apdf_10_left, apdf_10_right, apdf_50_left, apdf_50_right, apdf_90_left, apdf_90_right,
                        asymmetry_index_ai, asymmetry_penalty_applied, mdf_hz_left, mdf_hz_right,
                        mnf_hz_left, mnf_hz_right, fatigue_slope_left, fatigue_slope_right,
                        mdf_r_squared_left, mdf_r_squared_right, n_windows_left, n_windows_right,
                        mdf_computed_left, is_fatiguing_left, mdf_computed_right, is_fatiguing_right,
                        strain_reported, subjective_strain_cr10
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, epoch_idx, *row_data)
                )
            epochs_ingested += 1

    conn.commit()
    return {
        "session_id": session_id,
        "user_name": user_name,
        "processing_version": proc_version,
        "epochs_ingested": epochs_ingested,
        "status": "success"
    }


def ingest_directory(
    conn: Any,
    directory_path: Path,
    password: Optional[str] = None,
    interactive: bool = False,
    created_via: str = "desktop_registration"
) -> Dict[str, Any]:
    """
    Scans a directory for all pairs of *_metadata.json and *_features.csv, ingesting each pair idempotently into the DB.
    Enforces password gating on subject accounts.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return {"error": f"Directory {directory_path} does not exist or is not a directory.", "total_sessions": 0}

    # Find all metadata JSONs
    json_files = list(dir_path.glob("**/*_metadata.json"))
    report = {
        "total_sessions": 0,
        "epochs_ingested": 0,
        "modern_sessions_count": 0,
        "legacy_sessions_count": 0,
        "sessions": [],
        "failures": []
    }

    for j_path in json_files:
        csv_path = Path(str(j_path).replace("_metadata.json", "_features.csv"))
        if not csv_path.exists():
            continue  # Skip un-paired incomplete logs

        try:
            res = ingest_session_pair(conn, j_path, csv_path, password=password, interactive=interactive, created_via=created_via)
            report["total_sessions"] += 1
            report["epochs_ingested"] += res["epochs_ingested"]
            report["sessions"].append(res)
            if res["processing_version"] == "legacy_constant_padding":
                report["legacy_sessions_count"] += 1
            else:
                report["modern_sessions_count"] += 1
        except PermissionError as e:
            print(f"\n[REJECTED] {j_path.parent.name}: {str(e)}")
    return report


def ingest_session_pair_via_http(
    json_path: Path,
    csv_path: Path,
    endpoint_url: str,
    anon_key: Optional[str] = None,
    password: Optional[str] = None,
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Transmits a session metadata & feature CSV pair over standard HTTPS (Port 443) to the deployed Edge Function.
    Bypasses corporate and campus database port blocking (ports 5432/6543) while enforcing server-side password gating.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = str(meta.get("session_id", json_path.stem.replace("_metadata", "")))
    proc_version = meta.get("processing_version", "legacy_constant_padding")

    # LocalSessionLogger writes user_name at root level and demographics under
    # "user_snapshot". Legacy ingests may use the old "user_profile" key.
    user_name = str(
        meta.get("user_name")
        or meta.get("user_profile", {}).get("user_name")
        or "Anonymous"
    )
    snapshot = meta.get("user_snapshot") or meta.get("user_profile") or {}
    # Build a normalised user_profile dict for the Edge Function payload
    user_prof = {
        "user_name": user_name,
        "birth_date": snapshot.get("birth_date", "1900-01-01"),
        "gender_sex": snapshot.get("gender_sex", "Unknown"),
        "weight_kg": snapshot.get("weight_kg"),
        "height_cm": snapshot.get("height_cm"),
    }

    if not password and interactive:
        import getpass
        password = getpass.getpass(f"[AUTH REQUIRED] Enter password for subject '{user_name}': ")

    int_cols = {"epoch_index", "asymmetry_penalty_applied", "mdf_computed_left", "is_fatiguing_left", "mdf_computed_right", "is_fatiguing_right", "strain_reported"}

    epoch_features = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_idx = _parse_val(row.get("epoch_index"), "int")
            if epoch_idx is None:
                continue
            item = {"session_id": session_id}
            for k, v in row.items():
                if k == "timestamp":
                    continue
                v_type = "int" if k in int_cols else "float"
                parsed_val = _parse_val(v, v_type)
                if parsed_val is not None:
                    item[k] = parsed_val
            epoch_features.append(item)

    # De-duplicate by epoch_index — keep the LAST entry for each index.
    # Duplicates arise when "Log Strain Marker" is pressed mid-epoch, writing a
    # second CSV row with the same epoch_index. PostgreSQL's ON CONFLICT DO UPDATE
    # cannot update the same row twice in one batch (raises HTTP 500), so we must
    # collapse duplicates before sending.
    dedup: dict = {}
    for item in epoch_features:
        dedup[item.get("epoch_index")] = item  # later row wins
    epoch_features = list(dedup.values())

    # ── Normalise timestamps ─────────────────────────────────────────────────
    # LocalSessionLogger stores start_time/end_time as ISO strings, not floats.
    # Convert to unix float for the DB column; keep the raw string as _str.
    def _iso_to_unix(s):
        if not s:
            return None
        try:
            return float(s)           # already a numeric unix ts
        except (ValueError, TypeError):
            pass
        try:
            from datetime import datetime as _dt
            return _dt.fromisoformat(str(s)).timestamp()
        except Exception:
            return None

    start_time_raw = meta.get("start_time", "")
    end_time_raw   = meta.get("end_time", "")

    # ── Normalise calibration ────────────────────────────────────────────────
    # LocalSessionLogger stores: {"left": <float>, "right": <float>}
    # Edge Function reads:       calibration.left_rms_reference / right_rms_reference
    calib_raw = meta.get("calibration_baselines_mv") or meta.get("calibration") or {}
    calib_normalised = {
        "left_rms_reference":  calib_raw.get("left")  or calib_raw.get("left_rms_reference"),
        "right_rms_reference": calib_raw.get("right") or calib_raw.get("right_rms_reference"),
    }

    payload = {
        "user_name": user_name,
        "password": password or "",
        "session_metadata": {
            "session_id": session_id,
            "start_time": _iso_to_unix(start_time_raw),
            "start_time_str": str(start_time_raw),
            "end_time": _iso_to_unix(end_time_raw),
            "end_time_str": str(end_time_raw),
            "mode": meta.get("mode", "unknown"),
            "calibration": calib_normalised,
            "processing_version": proc_version,
            "user_profile": user_prof,
            "self_reports": meta.get("self_reports", [])
        },
        "epoch_features": epoch_features
    }

    headers = {"Content-Type": "application/json"}
    if anon_key:
        headers["apikey"] = anon_key
        headers["Authorization"] = f"Bearer {anon_key}"

    req = urllib.request.Request(endpoint_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_msg = json.loads(err_body).get("error", str(e))
        except Exception:
            err_msg = err_body or str(e)
        raise PermissionError(f"Cloud Ingestion Rejected (HTTP {e.code}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Cloud Ingestion Failed: {str(e)}")

    return {
        "session_id": session_id,
        "user_name": user_name,
        "processing_version": proc_version,
        "epochs_ingested": len(epoch_features),
        "status": "success"
    }


def ingest_directory_via_http(
    directory_path: Path,
    endpoint_url: str,
    anon_key: Optional[str] = None,
    password: Optional[str] = None,
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Scans a local log directory and synchronizes all session pairs over HTTPS to the deployed Edge Function.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return {"error": f"Directory {directory_path} does not exist or is not a directory.", "total_sessions": 0}

    json_files = list(dir_path.glob("**/*_metadata.json"))
    report = {
        "total_sessions": 0,
        "epochs_ingested": 0,
        "modern_sessions_count": 0,
        "legacy_sessions_count": 0,
        "sessions": [],
        "failures": []
    }

    for j_path in json_files:
        csv_path = Path(str(j_path).replace("_metadata.json", "_features.csv"))
        if not csv_path.exists():
            continue

        try:
            res = ingest_session_pair_via_http(j_path, csv_path, endpoint_url=endpoint_url, anon_key=anon_key, password=password, interactive=interactive)
            report["total_sessions"] += 1
            report["epochs_ingested"] += res["epochs_ingested"]
            report["sessions"].append(res)
            if res["processing_version"] == "legacy_constant_padding":
                report["legacy_sessions_count"] += 1
            else:
                report["modern_sessions_count"] += 1
        except (PermissionError, RuntimeError) as e:
            print(f"\n[REJECTED] {j_path.parent.name}: {str(e)}")
            report["failures"].append({"path": str(j_path), "error": str(e)})

    return report


def sync_logs_to_postgres(
    log_dir: str = "output_logs",
    db_uri: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    anon_key: Optional[str] = None,
    password: Optional[str] = None,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Automatic cloud sync entry point called by LocalLogger upon session completion.
    Defaults to HTTPS Edge Function transport (Port 443) to reliably traverse campus/corporate firewalls.
    """
    import os
    endpoint = endpoint_url or os.getenv("SUPABASE_EDGE_URL") or "https://khsrpxzckidhmzdpihfu.supabase.co/functions/v1/ingest-session"
    key = anon_key or os.getenv("SUPABASE_ANON_KEY")

    # If targeting a plain SQLite file directly
    if db_uri and not db_uri.startswith("http") and not db_uri.startswith("postgres"):
        import sqlite3
        if not quiet:
            print(f"[CloudSync] Synchronizing directly via SQLite file connection: {db_uri}")
        conn = sqlite3.connect(db_uri)
        report = ingest_directory(conn, Path(log_dir), password=password)
        conn.close()
        return report

    if not quiet:
        print(f"[CloudSync] Synchronizing via HTTPS Edge Function transport: {endpoint}")
    return ingest_directory_via_http(Path(log_dir), endpoint_url=endpoint, anon_key=key, password=password)

