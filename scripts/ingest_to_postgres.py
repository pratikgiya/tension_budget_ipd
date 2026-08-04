"""
ingest_to_postgres.py — CLI tool for executing batch data synchronization and ingestion from local log directories into SQL databases.

Usage Example:
    python -m scripts.ingest_to_postgres --dir output-data/ --db tensionbudget_cloud.db
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chordspy.tensionbudget.cloud_schema import create_schema_sqlite
from chordspy.tensionbudget.cloud_ingest import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="TensionBudget Cloud & ML Database Ingestion Tool (Phase 12)")
    parser.add_argument("--dir", dest="directory", default="output-data", help="Path to local logs directory containing paired JSON/CSV files.")
    parser.add_argument("--db", dest="db_path", default="tensionbudget_cloud.db", help="Path to SQLite or database file for ingestion.")
    args = parser.parse_args()

    dir_path = Path(args.directory)
    db_path = Path(args.db_path)

    print("===============================================================")
    print(" TensionBudget Data Ingestion Engine - Phase 12 Synchronization")
    print("===============================================================")
    print(f"Source Directory : {dir_path.resolve()}")
    print(f"Target Database  : {db_path.resolve()}\n")

    if not dir_path.exists():
        print(f"Error: Source directory '{dir_path}' does not exist.")
        sys.exit(1)

    print("Connecting to database and verifying relational schema...")
    conn = sqlite3.connect(str(db_path))
    create_schema_sqlite(conn)
    print("Schema verified (Tables: user_profiles, sessions, epoch_features, self_reports, v_ml_training_pairs).\n")

    print(f"Scanning '{dir_path}' and performing idempotent batch ingestion...")
    report = ingest_directory(conn, dir_path)
    conn.close()

    print("\n--- Synchronization Summary Report ---")
    if "error" in report:
        print(f"Ingestion Failed: {report['error']}")
    else:
        print(f"Total Sessions Ingested : {report['total_sessions']}")
        print(f"  +-- Modern (Phase 11+) : {report['modern_sessions_count']}")
        print(f"  +-- Legacy Zero-Padded : {report['legacy_sessions_count']} (flagged as 'legacy_constant_padding')")
        print(f"Total Epoch Rows Saved  : {report['epochs_ingested']}")
        print("Status                  : SUCCESS - Ready for ML training and longitudinal analytics.")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
