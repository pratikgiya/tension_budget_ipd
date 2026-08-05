#!/usr/bin/env python3
"""
Tension Budget - PostgreSQL DDL Exporter
Prints the complete, canonical relational schema DDL formatted for PostgreSQL / Supabase SQL Editor.
Run this script to inspect or copy the DDL before executing against a live cloud database.

Usage:
    python -m scripts.export_postgres_ddl
"""

import sys
from chordspy.tensionbudget.cloud_schema import get_postgres_ddl


def main():
    print("=" * 72)
    print("-- TENSION BUDGET CLOUD DATABASE (SUPABASE POSTGRESQL) SCHEMA DDL --")
    print("-- INSTRUCTIONS: Copy the SQL code below and execute once inside your --")
    print("--               Supabase project's SQL Query Editor.                  --")
    print("=" * 72 + "\n")

    ddl_statements = get_postgres_ddl()
    for stmt in ddl_statements:
        print(stmt.strip())
        print("\n")

    print("=" * 72)
    print("-- SCHEMA SUMMARY: 3 Tables (user_profiles, sessions, epoch_features)")
    print("--                 1 View (v_ml_training_pairs for ML model training)")
    print("=" * 72)


if __name__ == "__main__":
    main()
