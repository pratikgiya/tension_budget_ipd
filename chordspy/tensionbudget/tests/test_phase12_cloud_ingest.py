"""
test_phase12_cloud_ingest.py — Verification tests for Phase 12 Cloud PostgreSQL & ML Database Ingestion Pipeline.
"""

import csv
import json
import sqlite3
import pytest
from pathlib import Path

from chordspy.tensionbudget.cloud_schema import create_schema_sqlite
from chordspy.tensionbudget.cloud_ingest import ingest_directory, ingest_session_pair


class TestPhase12CloudIngestion:
    """Verifies relational DDL initialization, idempotent batch upserts, explicit missingness handling, and legacy processing version segmentation."""

    @pytest.fixture
    def mem_db(self):
        conn = sqlite3.connect(":memory:")
        create_schema_sqlite(conn)
        yield conn
        conn.close()

    def test_schema_creation_and_view(self, mem_db):
        """Verify that all 3 tables and the ML training pair SQL view are instantiated cleanly."""
        cursor = mem_db.cursor()
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view');")
        objects = {row[0]: row[1] for row in cursor.fetchall()}
        
        assert "user_profiles" in objects and objects["user_profiles"] == "table"
        assert "sessions" in objects and objects["sessions"] == "table"
        assert "epoch_features" in objects and objects["epoch_features"] == "table"
        assert "v_ml_training_pairs" in objects and objects["v_ml_training_pairs"] == "view"

    def test_idempotence_and_version_segmentation(self, mem_db, tmp_path):
        """
        Verify that repeated ingest_directory calls produce zero duplicate rows (idempotence) and that older
        sessions lacking processing_version are accurately flagged as 'legacy_constant_padding'.
        """
        # 1. Setup Session A (Modern Phase 11+ Session)
        meta_a = {
            "session_id": "session_modern_01",
            "start_time": 1700000000.0,
            "start_time_str": "2026-08-02 10:00:00",
            "mode": "live",
            "user_profile": {"user_name": "ModernUser", "birth_date": "1995-05-10", "gender_sex": "Female", "weight_kg": 62.0, "height_cm": 168.0},
            "calibration": {"left_rms_reference": 0.05, "right_rms_reference": 0.05},
            "processing_version": "phase11_edge_padding"
        }
        with open(tmp_path / "session_modern_01_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_a, f)

        rows_a = [
            {"epoch_index": "1", "elapsed_minutes": "5.0", "composite_score": "8.5", "mdf_computed_left": "1", "mdf_hz_left": "72.5", "mdf_r_squared_left": "0.91", "n_windows_left": "350"},
            {"epoch_index": "2", "elapsed_minutes": "10.0", "composite_score": "12.0", "mdf_computed_left": "1", "mdf_hz_left": "69.0", "mdf_r_squared_left": "0.88", "n_windows_left": "345"}
        ]
        with open(tmp_path / "session_modern_01_features.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
            writer.writeheader()
            writer.writerows(rows_a)

        # 2. Setup Session B (Legacy Pre-Phase 11 Session, NO processing_version flag, contains empty string resting MDF)
        meta_b = {
            "session_id": "session_legacy_02",
            "start_time": 1600000000.0,
            "mode": "offline",
            "user_profile": {"user_name": "LegacyUser", "birth_date": "1980-01-01", "gender_sex": "Male", "weight_kg": 80.0, "height_cm": 175.0},
            "calibration": {"left_rms_reference": 0.04, "right_rms_reference": 0.04}
        }
        with open(tmp_path / "session_legacy_02_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_b, f)

        rows_b = [
            {"epoch_index": "1", "elapsed_minutes": "5.0", "composite_score": "4.0", "mdf_computed_left": "0", "mdf_hz_left": "", "mdf_r_squared_left": "", "n_windows_left": "15"}
        ]
        with open(tmp_path / "session_legacy_02_features.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
            writer.writeheader()
            writer.writerows(rows_b)

        # First Ingestion Run
        report_1 = ingest_directory(mem_db, tmp_path)
        assert report_1["total_sessions"] == 2
        assert report_1["modern_sessions_count"] == 1
        assert report_1["legacy_sessions_count"] == 1
        assert report_1["epochs_ingested"] == 3

        # Verify SQL table state after First Run
        cursor = mem_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions;")
        assert cursor.fetchone()[0] == 2
        cursor.execute("SELECT COUNT(*) FROM epoch_features;")
        assert cursor.fetchone()[0] == 3

        # Check version segmentation flags in DB
        cursor.execute("SELECT session_id, processing_version FROM sessions WHERE session_id = 'session_legacy_02';")
        assert cursor.fetchone()[1] == "legacy_constant_padding", "Legacy session was not tagged with 'legacy_constant_padding'!"
        cursor.execute("SELECT session_id, processing_version FROM sessions WHERE session_id = 'session_modern_01';")
        assert cursor.fetchone()[1] == "phase11_edge_padding"

        # Check explicit missingness translation (empty string converted to NULL in DB, not zero)
        cursor.execute("SELECT mdf_computed_left, mdf_hz_left FROM epoch_features WHERE session_id = 'session_legacy_02' AND epoch_index = 1;")
        res = cursor.fetchone()
        assert res[0] == 0, "mdf_computed_left should be boolean 0 (false)"
        assert res[1] is None, f"mdf_hz_left should be SQL NULL (None in Python) during rest to avoid 0 Hz imputation, got {res[1]}"

        # Second Ingestion Run (Proving Idempotency and lack of primary key/unique collisions)
        report_2 = ingest_directory(mem_db, tmp_path)
        assert report_2["total_sessions"] == 2

        # Verify SQL row count remained unchanged (no duplicate rows)
        cursor.execute("SELECT COUNT(*) FROM sessions;")
        assert cursor.fetchone()[0] == 2
        cursor.execute("SELECT COUNT(*) FROM epoch_features;")
        assert cursor.fetchone()[0] == 3, "Duplicate epoch feature rows inserted on repeated sync!"

        # Verify ML View output
        cursor.execute("SELECT session_id, user_name, gender_sex, calibration_rms_left, processing_version, composite_score, mdf_r_squared_left, n_windows_left FROM v_ml_training_pairs WHERE session_id = 'session_modern_01' AND epoch_index = 1;")
        ml_row = cursor.fetchone()
        assert ml_row[0] == "session_modern_01"
        assert ml_row[1] == "ModernUser"
        assert ml_row[2] == "Female"
        assert ml_row[3] == 0.05
        assert ml_row[4] == "phase11_edge_padding"
        assert ml_row[5] == 8.5
        assert ml_row[6] == 0.91
        assert ml_row[7] == 350.0
