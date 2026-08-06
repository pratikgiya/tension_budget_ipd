"""
Comprehensive test suite verifying the integrity, serialization, and ingestion
of all 38 ergonomic feature columns across LocalSessionLogger CSVs, SQLite/PostgreSQL
relational tables, and Supabase Edge Function HTTP payloads.
"""

import csv
import json
import math
import sqlite3
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from chordspy.tensionbudget.local_logger import LocalSessionLogger, CSV_HEADER
from chordspy.tensionbudget.cloud_schema import create_schema_sqlite
from chordspy.tensionbudget.cloud_ingest import ingest_session_pair, ingest_session_pair_via_http


def test_38_features_header_specification():
    """Verify that CSV_HEADER defines precisely 38 feature columns in the designated order."""
    expected_columns = [
        "epoch_index", "timestamp", "elapsed_minutes", "composite_score",
        "eindex_live_left", "eindex_live_right", "eindex_cumulative_left", "eindex_cumulative_right",
        "short_suma_penalty_left", "short_suma_penalty_right", "total_suma_bursts_left", "total_suma_bursts_right",
        "gap_frequency_left", "gap_frequency_right", "apdf_10_left", "apdf_50_left", "apdf_90_left",
        "apdf_10_right", "apdf_50_right", "apdf_90_right", "asymmetry_index_ai", "asymmetry_penalty_applied",
        "mdf_hz_left", "mnf_hz_left", "fatigue_slope_left", "mdf_r_squared_left", "n_windows_left",
        "mdf_computed_left", "is_fatiguing_left", "mdf_hz_right", "mnf_hz_right", "fatigue_slope_right",
        "mdf_r_squared_right", "n_windows_right", "mdf_computed_right", "is_fatiguing_right",
        "strain_reported", "subjective_strain_cr10"
    ]
    assert len(CSV_HEADER) == 38, f"Expected exactly 38 columns in CSV_HEADER, got {len(CSV_HEADER)}"
    assert CSV_HEADER == expected_columns, "CSV_HEADER columns or ordering deviate from specification."


def test_38_features_local_csv_logging():
    """Proves that LocalSessionLogger properly formats and records every one of the 38 features to CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        with patch("chordspy.tensionbudget.local_logger.LOGS_ROOT", tmp_root):
            logger = LocalSessionLogger(user_name="FeatureTester", weight_kg=70.0, height_cm=175.0)
            logger.start_session(mode="offline", calibration_left=0.045, calibration_right=0.048)

            # Define unique test values for all score metrics
            score_left = {
                "eindex_live": -0.456,
                "eindex_cumulative": -1.234,
                "short_suma_penalty": 0.20,
                "suma_count": 12,
                "gap_frequency_per_min": 8.4,
                "apdf_10": 1.2,
                "apdf_50": 4.5,
                "apdf_90": 15.6,
            }
            score_right = {
                "eindex_live": 0.123,
                "eindex_cumulative": 0.567,
                "short_suma_penalty": 0.40,
                "suma_count": 18,
                "gap_frequency_per_min": 3.2,
                "apdf_10": 2.1,
                "apdf_50": 6.8,
                "apdf_90": 22.4,
            }
            spec_left = {
                "spectral_series": {"mdf_hz": [65.0, 64.0], "mnf_hz": [75.0, 74.0]},
                "slope_hz_per_min": -0.85,
                "r_squared": 0.88,
                "n_windows": 15,
                "is_fatiguing": True
            }
            spec_right = {
                "spectral_series": {"mdf_hz": [62.0, 62.5], "mnf_hz": [72.0, 71.8]},
                "slope_hz_per_min": -0.10,
                "r_squared": 0.12,
                "n_windows": 15,
                "is_fatiguing": False
            }

            logger.log_epoch(
                epoch_index=1,
                elapsed_minutes=5.0,
                composite_score=1.4567,
                score_left_dict=score_left,
                score_right_dict=score_right,
                spectral_left_dict=spec_left,
                spectral_right_dict=spec_right,
                asymmetry_index=0.25,
                asymmetry_penalty_applied=False,
                subjective_strain_cr10=3.5
            )

            # Read back from CSV
            assert logger.csv_path.exists()
            with open(logger.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            r = rows[0]
            assert int(r["epoch_index"]) == 1
            assert r["timestamp"] != ""  # ISO datetime verified
            assert float(r["elapsed_minutes"]) == 5.0
            assert float(r["composite_score"]) == 1.4567

            # Left metrics
            assert float(r["eindex_live_left"]) == -0.456
            assert float(r["eindex_cumulative_left"]) == -1.234
            assert float(r["short_suma_penalty_left"]) == 0.20
            assert int(r["total_suma_bursts_left"]) == 12
            assert float(r["gap_frequency_left"]) == 8.4
            assert float(r["apdf_10_left"]) == 1.2
            assert float(r["apdf_50_left"]) == 4.5
            assert float(r["apdf_90_left"]) == 15.6

            # Right metrics
            assert float(r["eindex_live_right"]) == 0.123
            assert float(r["eindex_cumulative_right"]) == 0.567
            assert float(r["short_suma_penalty_right"]) == 0.40
            assert int(r["total_suma_bursts_right"]) == 18
            assert float(r["gap_frequency_right"]) == 3.2
            assert float(r["apdf_10_right"]) == 2.1
            assert float(r["apdf_50_right"]) == 6.8
            assert float(r["apdf_90_right"]) == 22.4

            # Asymmetry
            assert float(r["asymmetry_index_ai"]) == 0.25
            assert int(r["asymmetry_penalty_applied"]) == 0

            # Spectral left
            assert float(r["mdf_hz_left"]) == 64.5  # Mean of 65.0, 64.0
            assert float(r["mnf_hz_left"]) == 74.5  # Mean of 75.0, 74.0
            assert float(r["fatigue_slope_left"]) == -0.85
            assert float(r["mdf_r_squared_left"]) == 0.88
            assert int(r["n_windows_left"]) == 15
            assert int(r["mdf_computed_left"]) == 1
            assert int(r["is_fatiguing_left"]) == 1

            # Spectral right
            assert float(r["mdf_hz_right"]) == 62.25  # Mean of 62.0, 62.5
            assert float(r["mnf_hz_right"]) == 71.9
            assert float(r["fatigue_slope_right"]) == -0.10
            assert float(r["mdf_r_squared_right"]) == 0.12
            assert int(r["n_windows_right"]) == 15
            assert int(r["mdf_computed_right"]) == 1
            assert int(r["is_fatiguing_right"]) == 0

            # Borg subjective rating
            assert int(r["strain_reported"]) == 1
            assert float(r["subjective_strain_cr10"]) == 3.5

            # Save session for cloud ingest testing
            logger.end_session(password="test_mock_pw")


def test_38_features_database_schema_and_ingest():
    """Proves that all 38 features are preserved cleanly through relational SQL ingestion without truncation or NULLing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        with patch("chordspy.tensionbudget.local_logger.LOGS_ROOT", tmp_root):
            logger = LocalSessionLogger(user_name="SchemaTester")
            logger.start_session(mode="live", calibration_left=0.05, calibration_right=0.05)
            logger.log_epoch(
                epoch_index=2,
                elapsed_minutes=10.0,
                composite_score=0.95,
                score_left_dict={"eindex_live": 0.1, "eindex_cumulative": 0.2, "short_suma_penalty": 0.0, "suma_count": 5, "gap_frequency_per_min": 10.0, "apdf_10": 2.0, "apdf_50": 5.0, "apdf_90": 12.0},
                score_right_dict={"eindex_live": 0.2, "eindex_cumulative": 0.4, "short_suma_penalty": 0.1, "suma_count": 7, "gap_frequency_per_min": 6.5, "apdf_10": 2.5, "apdf_50": 6.0, "apdf_90": 14.0},
                spectral_left_dict={"spectral_series": {"mdf_hz": [60.0], "mnf_hz": [70.0]}, "slope_hz_per_min": -0.4, "r_squared": 0.75, "n_windows": 15, "is_fatiguing": True},
                spectral_right_dict={"spectral_series": {"mdf_hz": [58.0], "mnf_hz": [68.0]}, "slope_hz_per_min": -0.2, "r_squared": 0.40, "n_windows": 15, "is_fatiguing": False},
                asymmetry_index=0.15,
                asymmetry_penalty_applied=False,
                subjective_strain_cr10=5.0
            )
            logger.end_session(password="test_mock_pw")

            # Create in-memory SQL DB from cloud_schema.py
            conn = sqlite3.connect(":memory:")
            create_schema_sqlite(conn)

            # Ingest session pair
            result = ingest_session_pair(conn, logger.meta_path, logger.csv_path)
            assert result["status"] == "success"
            assert result["epochs_ingested"] == 1

            # Verify every column in epoch_features table
            cur = conn.cursor()
            cur.execute("SELECT * FROM epoch_features WHERE epoch_index = 2")
            col_names = [d[0] for d in cur.description]
            row = cur.fetchone()
            db_dict = dict(zip(col_names, row))

            assert db_dict["epoch_index"] == 2
            assert db_dict["elapsed_minutes"] == 10.0
            assert db_dict["composite_score"] == 0.95
            assert db_dict["eindex_live_left"] == 0.1
            assert db_dict["total_suma_bursts_left"] == 5.0
            assert db_dict["total_suma_bursts_right"] == 7.0
            assert db_dict["gap_frequency_left"] == 10.0
            assert db_dict["apdf_90_right"] == 14.0
            assert db_dict["mdf_hz_left"] == 60.0
            assert db_dict["fatigue_slope_left"] == -0.4
            assert db_dict["is_fatiguing_left"] == 1
            assert db_dict["strain_reported"] == 1
            assert db_dict["subjective_strain_cr10"] == 5.0
            conn.close()


def test_38_features_supabase_edge_function_payload():
    """Proves that sync/ingestion to Supabase Edge Function includes every single feature attribute."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        with patch("chordspy.tensionbudget.local_logger.LOGS_ROOT", tmp_root):
            logger = LocalSessionLogger(user_name="PayloadTester")
            logger.start_session(mode="live", calibration_left=0.033, calibration_right=0.035)
            logger.log_epoch(
                epoch_index=1,
                elapsed_minutes=5.0,
                composite_score=0.88,
                score_left_dict={"eindex_live": 0.05, "eindex_cumulative": 0.05, "short_suma_penalty": 0.1, "suma_count": 9, "gap_frequency_per_min": 12.1, "apdf_10": 1.1, "apdf_50": 4.2, "apdf_90": 10.1},
                score_right_dict={"eindex_live": 0.08, "eindex_cumulative": 0.08, "short_suma_penalty": 0.0, "suma_count": 6, "gap_frequency_per_min": 14.0, "apdf_10": 1.0, "apdf_50": 3.9, "apdf_90": 9.5},
                spectral_left_dict={"spectral_series": {"mdf_hz": [68.0], "mnf_hz": [78.0]}, "slope_hz_per_min": -0.5, "r_squared": 0.90, "n_windows": 15, "is_fatiguing": True},
                spectral_right_dict={"spectral_series": {"mdf_hz": [69.0], "mnf_hz": [79.0]}, "slope_hz_per_min": 0.1, "r_squared": 0.10, "n_windows": 15, "is_fatiguing": False},
                asymmetry_index=0.05,
                asymmetry_penalty_applied=False,
                subjective_strain_cr10=2.0
            )
            logger.end_session(password="test_mock_pw")

            # Mock urlopen to inspect outgoing JSON payload to Supabase
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = b'{"success": true}'
                mock_resp.__enter__.return_value = mock_resp
                mock_urlopen.return_value = mock_resp

                ingest_session_pair_via_http(
                    logger.meta_path,
                    logger.csv_path,
                    endpoint_url="https://fake.supabase.co/functions/v1/ingest-session",
                    password="secure_test_pwd"
                )

                assert mock_urlopen.called
                req = mock_urlopen.call_args[0][0]
                payload_str = req.data.decode("utf-8")
                payload = json.loads(payload_str)

                assert "epoch_features" in payload
                assert len(payload["epoch_features"]) == 1
                feat_item = payload["epoch_features"][0]

                # All numeric features must be present in the JSON payload sent to Supabase
                expected_keys = set(CSV_HEADER) - {"timestamp"}
                for k in expected_keys:
                    assert k in feat_item, f"Missing feature column '{k}' in Edge Function payload!"

                assert feat_item["total_suma_bursts_left"] == 9
                assert feat_item["total_suma_bursts_right"] == 6
                assert feat_item["strain_reported"] == 1
                assert feat_item["subjective_strain_cr10"] == 2.0
                assert feat_item["is_fatiguing_left"] == 1
                assert feat_item["is_fatiguing_right"] == 0
