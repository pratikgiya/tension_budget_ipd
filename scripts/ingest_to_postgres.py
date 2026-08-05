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
    parser.add_argument("--password", dest="password", default=None, help="Subject account password for non-interactive automated write authentication.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive terminal password prompts for protected user accounts.")
    parser.add_argument("--init-schema", dest="init_schema", action="store_true", help="Explicitly initialize schema tables. Do NOT use during routine syncs; schema creation is an intentional one-time action.")
    parser.add_argument("--direct", action="store_true", help="Use direct SQL database connection (ports 5432/6543) via psycopg2 instead of default HTTPS Edge Function transport (port 443).")
    parser.add_argument("--endpoint", default=os.getenv("SUPABASE_EDGE_URL", "https://khsrpxzckidhmzdpihfu.supabase.co/functions/v1/ingest-session"), help="HTTPS URL for deployed Edge Function endpoint.")
    parser.add_argument("--anon-key", default=os.getenv("SUPABASE_ANON_KEY"), help="Public anonymous API key for Supabase project.")
    args = parser.parse_args()

    dir_path = Path(args.directory)
    db_input = str(args.db_path)
    is_pg = db_input.startswith("postgres")
    use_http_transport = not args.direct and not args.init_schema and (is_pg or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_EDGE_URL") or args.endpoint)

    print("===============================================================")
    print(" TensionBudget Data Ingestion Engine - Password Gating & Sync  ")
    print("===============================================================")
    print(f"Source Directory : {dir_path.resolve()}")
    if use_http_transport:
        print(f"Transport Mode   : HTTPS Edge Function (Port 443 - Firewall Bypass Default)")
        print(f"Target Endpoint  : {args.endpoint}\n")
    else:
        print(f"Transport Mode   : Direct SQL Database Connection (Fallback / Local SQLite Mode)")
        print(f"Target Database  : {'[PostgreSQL Cloud Instance]' if is_pg else Path(db_input).resolve()}\n")

    if not dir_path.exists():
        print(f"Error: Source directory '{dir_path}' does not exist.")
        sys.exit(1)

    if use_http_transport:
        from chordspy.tensionbudget.cloud_ingest import ingest_directory_via_http
        print(f"Scanning '{dir_path}' and transmitting telemetry payloads via secure HTTPS POST...")
        report = ingest_directory_via_http(dir_path, endpoint_url=args.endpoint, anon_key=args.anon_key, password=args.password, interactive=args.interactive)
    else:
        print("Connecting to database...")
        if is_pg:
            if db_input.count('@') > 1:
                parts = db_input.rsplit('@', 1)
                db_input = parts[0].replace('@', '%40') + '@' + parts[1]
            try:
                import psycopg2 as pg_mod
            except ImportError:
                import psycopg as pg_mod
            conn = pg_mod.connect(db_input)
        else:
            conn = sqlite3.connect(db_input)

        if args.init_schema:
            if is_pg:
                print("[!] ERROR: `--init-schema` is disabled for cloud PostgreSQL instances.")
                print("    Please execute DDL manually in Supabase SQL Editor via `python -m scripts.export_postgres_ddl`.")
                conn.close()
                sys.exit(1)
            print("[CAUTION] Explicit `--init-schema` flag provided. Executing table creation DDL...")
            create_schema_sqlite(conn)
            print("Schema initialized (Tables: user_profiles, sessions, epoch_features, v_ml_training_pairs).\n")
        else:
            cursor = conn.cursor()
            if is_pg:
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_name='user_profiles';")
            else:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles';")
            if not cursor.fetchone():
                print("\n[!] ERROR: Target database lacks required tables ('user_profiles' missing).")
                print("    Per security safeguards, automatic schema creation is disabled during routine sync calls.")
                print("    Please execute SQL DDL once in your cloud query editor (via `python -m scripts.export_postgres_ddl`)")
                print("    or pass `--init-schema` explicitly when initializing local SQLite file databases.")
                conn.close()
                sys.exit(1)
            print("Required tables confirmed present in database (no DDL executed).\n")

        print(f"Scanning '{dir_path}' and performing password-gated batch ingestion...")
        report = ingest_directory(conn, dir_path, password=args.password, interactive=args.interactive)
        conn.close()

    print("\n--- Synchronization Summary Report ---")
    if "error" in report:
        print(f"Ingestion Failed: {report['error']}")
    else:
        print(f"Total Sessions Ingested : {report['total_sessions']}")
        print(f"  +-- Modern (Phase 11+) : {report['modern_sessions_count']}")
        print(f"  +-- Legacy Zero-Padded : {report['legacy_sessions_count']} (flagged as 'legacy_constant_padding')")
        print(f"Total Epoch Rows Saved  : {report['epochs_ingested']}")
        if report.get("failures"):
            print(f"\n[!] Security & Data Integrity Rejections ({len(report['failures'])} session(s) blocked):")
            for fail in report["failures"]:
                print(f"  [BLOCKED] {fail['error']}")
        else:
            print("Status                  : SUCCESS - Ready for ML training and longitudinal analytics.")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
