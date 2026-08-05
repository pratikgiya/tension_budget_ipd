"""
cloud_schema.py — Phase 12 Relational Database Schema definition for TensionBudget multi-user cloud ingestion and ML pipelines.

Implements the verified 3-Tier Privacy & Longitudinal Telemetry Hierarchy:
  Tier 1: user_profiles (Demographics and physical customization vault)
  Tier 2: sessions (Working sitting, calibration baselines, and processing_version flags)
  Tier 3: epoch_features (Flattened, wide-format 5-minute longitudinal telemetry with embedded subjective strain/RPE ratings)

Designed for compatibility with both local SQLite archives and cloud PostgreSQL deployments.
"""

import sqlite3
from typing import Any, Dict, List, Optional


SQLITE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_name TEXT PRIMARY KEY,
        birth_date TEXT,
        gender_sex TEXT,
        weight_kg REAL,
        height_cm REAL,
        updated_at REAL,
        password_hash TEXT,
        created_via TEXT DEFAULT 'desktop_registration'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_name TEXT NOT NULL,
        start_time REAL,
        start_time_str TEXT,
        end_time REAL,
        end_time_str TEXT,
        mode TEXT,
        calibration_rms_left REAL,
        calibration_rms_right REAL,
        processing_version TEXT NOT NULL,
        FOREIGN KEY(user_name) REFERENCES user_profiles(user_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS epoch_features (
        feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        epoch_index INTEGER NOT NULL,
        elapsed_minutes REAL,
        composite_score REAL,
        eindex_live_left REAL,
        eindex_live_right REAL,
        eindex_cumulative_left REAL,
        eindex_cumulative_right REAL,
        short_suma_penalty_left REAL,
        short_suma_penalty_right REAL,
        total_suma_bursts_left REAL,
        total_suma_bursts_right REAL,
        gap_frequency_left REAL,
        gap_frequency_right REAL,
        apdf_10_left REAL,
        apdf_10_right REAL,
        apdf_50_left REAL,
        apdf_50_right REAL,
        apdf_90_left REAL,
        apdf_90_right REAL,
        asymmetry_index_ai REAL,
        asymmetry_penalty_applied INTEGER,
        mdf_hz_left REAL,
        mdf_hz_right REAL,
        mnf_hz_left REAL,
        mnf_hz_right REAL,
        fatigue_slope_left REAL,
        fatigue_slope_right REAL,
        mdf_r_squared_left REAL,
        mdf_r_squared_right REAL,
        n_windows_left REAL,
        n_windows_right REAL,
        mdf_computed_left INTEGER,
        is_fatiguing_left INTEGER,
        mdf_computed_right INTEGER,
        is_fatiguing_right INTEGER,
        strain_reported INTEGER NOT NULL DEFAULT 0,
        subjective_strain_cr10 REAL DEFAULT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id),
        UNIQUE(session_id, epoch_index)
    );
    """,
    """
    CREATE VIEW IF NOT EXISTS v_ml_training_pairs AS
    SELECT 
        s.user_name,
        u.birth_date,
        u.gender_sex,
        u.weight_kg,
        u.height_cm,
        s.calibration_rms_left,
        s.calibration_rms_right,
        s.processing_version,
        e.*
    FROM epoch_features e
    JOIN sessions s ON e.session_id = s.session_id
    LEFT JOIN user_profiles u ON s.user_name = u.user_name;
    """
]


def create_schema_sqlite(db_connection: sqlite3.Connection):
    """
    Initializes the TensionBudget relational database tables and views on a SQLite connection.
    Also executes non-breaking schema migrations for existing database files.
    """
    cursor = db_connection.cursor()
    for ddl_stmt in SQLITE_DDL:
        cursor.execute(ddl_stmt)
    # Ensure existing SQLite tables receive the password-gating integrity columns
    try:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN password_hash TEXT;")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN created_via TEXT DEFAULT 'desktop_registration';")
    except sqlite3.OperationalError:
        pass
    db_connection.commit()


def get_postgres_ddl() -> List[str]:
    """
    Returns equivalent PostgreSQL DDL statements for cloud database setup.
    """
    pg_ddl = []
    for stmt in SQLITE_DDL:
        s = stmt.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        s = s.replace("CREATE VIEW IF NOT EXISTS", "CREATE OR REPLACE VIEW")
        pg_ddl.append(s.strip() + "\n")
    return pg_ddl
