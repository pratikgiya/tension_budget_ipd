"""
cloud_ingest.py — Idempotent data ingestion engine for Phase 12 Cloud PostgreSQL & ML feature pipeline.

Scans local logging directories (output_logs/, output-data/) for paired JSON metadata manifests and feature CSV logs.
Normalizes explicit missingness cells (converting resting empty strings to SQL NULL while maintaining mdf_computed booleans),
safely segments legacy pre-Phase 11 sessions with the 'legacy_constant_padding' marker, and performs idempotent batch upserts.
"""

import csv
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def ingest_session_pair(conn: Any, json_path: Path, csv_path: Path) -> Dict[str, Any]:
    """
    Ingests a matched pair of session metadata JSON and feature CSV into the database connection.
    Returns summary statistics for the ingested session.
    """
    cursor = conn.cursor()

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = str(meta.get("session_id", json_path.stem.replace("_metadata", "")))
    user_prof = meta.get("user_profile", {})
    user_name = str(user_prof.get("user_name", "Anonymous"))
    birth_date = user_prof.get("birth_date", "1900-01-01")
    gender_sex = user_prof.get("gender_sex", "Unknown")
    weight_kg = _parse_val(user_prof.get("weight_kg"), "float")
    height_cm = _parse_val(user_prof.get("height_cm"), "float")

    # Tier 1: Upsert User Profile
    cursor.execute("SELECT user_name FROM user_profiles WHERE user_name = ?", (user_name,))
    if cursor.fetchone():
        cursor.execute(
            """
            UPDATE user_profiles 
            SET birth_date = ?, gender_sex = ?, weight_kg = ?, height_cm = ?, updated_at = ?
            WHERE user_name = ?
            """,
            (birth_date, gender_sex, weight_kg, height_cm, time.time(), user_name)
        )
    else:
        cursor.execute(
            """
            INSERT INTO user_profiles (user_name, birth_date, gender_sex, weight_kg, height_cm, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_name, birth_date, gender_sex, weight_kg, height_cm, time.time())
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
                        mdf_computed_left=?, is_fatiguing_left=?, mdf_computed_right=?, is_fatiguing_right=?
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
                        mdf_computed_left, is_fatiguing_left, mdf_computed_right, is_fatiguing_right
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def ingest_directory(conn: Any, directory_path: Path) -> Dict[str, Any]:
    """
    Scans a directory for all pairs of *_metadata.json and *_features.csv, ingesting each pair idempotently into the DB.
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
        "sessions": []
    }

    for j_path in json_files:
        csv_path = Path(str(j_path).replace("_metadata.json", "_features.csv"))
        if not csv_path.exists():
            continue  # Skip un-paired incomplete logs

        res = ingest_session_pair(conn, j_path, csv_path)
        report["total_sessions"] += 1
        report["epochs_ingested"] += res["epochs_ingested"]
        report["sessions"].append(res)
        if res["processing_version"] == "legacy_constant_padding":
            report["legacy_sessions_count"] += 1
        else:
            report["modern_sessions_count"] += 1

    return report
