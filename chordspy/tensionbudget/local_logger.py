"""
Local Session Logger & User Profile Manager
===========================================
Manages per-user local folders under `output_logs/<user_name>/`, saving profile
metadata (`user_profile.json`), dynamic Age and BMI computations, and logging
5-minute periodic ergonomic feature rows into wide-format CSVs with automatic
cloud PostgreSQL / Supabase ingestion upon session completion.
"""

import csv
import json
import math
import os
import re
from datetime import datetime, date
from pathlib import Path

LOGS_ROOT = Path("output_logs")
PROCESSING_VERSION = "phase11_edge_padding"

CSV_HEADER = [
    "epoch_index",
    "timestamp",
    "elapsed_minutes",
    "composite_score",
    "eindex_live_left",
    "eindex_live_right",
    "eindex_cumulative_left",
    "eindex_cumulative_right",
    "short_suma_penalty_left",
    "short_suma_penalty_right",
    "total_suma_bursts_left",
    "total_suma_bursts_right",
    "gap_frequency_left",
    "gap_frequency_right",
    "apdf_10_left",
    "apdf_50_left",
    "apdf_90_left",
    "apdf_10_right",
    "apdf_50_right",
    "apdf_90_right",
    "asymmetry_index_ai",
    "asymmetry_penalty_applied",
    "mdf_hz_left",
    "mnf_hz_left",
    "fatigue_slope_left",
    "mdf_r_squared_left",
    "n_windows_left",
    "mdf_computed_left",
    "is_fatiguing_left",
    "mdf_hz_right",
    "mnf_hz_right",
    "fatigue_slope_right",
    "mdf_r_squared_right",
    "n_windows_right",
    "mdf_computed_right",
    "is_fatiguing_right",
    "strain_reported",
    "subjective_strain_cr10",
]

RAW_CSV_HEADER = [
    "sample_index",
    "timestamp_s",
    "epoch_index",
    "raw_adc_left",
    "raw_adc_right",
]


class LocalSessionLogger:
    def __init__(
        self,
        user_name: str,
        birth_date_str: str = "",
        gender_sex: str = "Unspecified",
        weight_kg: float = None,
        height_cm: float = None,
        db_uri: str = None,
    ):
        # Sanitize folder name (replace spaces and special chars with underscores)
        raw_name = user_name.strip() if user_name.strip() else "Anonymous_User"
        self.user_folder_name = re.sub(r"[^\w\-]", "_", raw_name)
        self.user_dir = LOGS_ROOT / self.user_folder_name
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self.user_name = raw_name
        self.birth_date_str = birth_date_str.strip()
        self.gender_sex = gender_sex
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.db_uri = db_uri or os.getenv("SUPABASE_DB_URI") or os.getenv("TENSION_BUDGET_DB_URI")

        self.session_timestamp_str = None
        self.session_dir = None
        self.csv_path = None
        self.meta_path = None
        self.raw_path = None
        self.raw_buffer = []
        self.is_active = False

        # Compute dynamic metrics and save/update profile
        self.age_years, self.bmi_value = self._update_user_profile()

    def _calculate_age(self, reference_date=None) -> float:
        if not self.birth_date_str:
            return None
        try:
            if reference_date is None:
                reference_date = datetime.now().date()
            b_date = datetime.strptime(self.birth_date_str, "%Y-%m-%d").date()
            days = (reference_date - b_date).days
            if days >= 0:
                return round(days / 365.25, 1)
        except Exception:
            pass
        return None

    def _calculate_bmi(self) -> float:
        try:
            if self.weight_kg is not None and self.height_cm is not None and self.height_cm > 0 and self.weight_kg > 0:
                height_m = self.height_cm / 100.0
                return round(self.weight_kg / (height_m * height_m), 2)
        except Exception:
            pass
        return None

    def _update_user_profile(self):
        age = self._calculate_age()
        bmi = self._calculate_bmi()
        profile_path = self.user_dir / "user_profile.json"
        
        # Load existing if available to preserve created_at or fill defaults
        existing = {}
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        created_at = existing.get("created_at", datetime.now().isoformat())
        
        profile_data = {
            "user_name": self.user_name,
            "birth_date": self.birth_date_str,
            "gender_sex": self.gender_sex,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "current_computed_age": age,
            "current_computed_bmi": bmi,
            "created_at": created_at,
            "last_updated_at": datetime.now().isoformat(),
        }

        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save user_profile.json: {e}")

        return age, bmi

    def start_session(self, mode: str, calibration_left: float, calibration_right: float, source_info: str = ""):
        """Initializes a new recording session within the user's folder."""
        now = datetime.now()
        self.session_timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        
        base_filename = f"session_{self.session_timestamp_str}"
        self.session_dir = self.user_dir / base_filename
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.session_dir / f"{base_filename}_features.csv"
        self.meta_path = self.session_dir / f"{base_filename}_metadata.json"
        self.raw_path = self.session_dir / f"{base_filename}_raw.csv"
        self.raw_buffer.clear()

        meta_data = {
            "session_id": self.session_timestamp_str,
            "user_name": self.user_name,
            "mode": mode,
            "source_info": source_info,
            "start_time": now.isoformat(),
            "end_time": None,
            "status": "active",
            "processing_version": PROCESSING_VERSION,
            "user_snapshot": {
                "birth_date": self.birth_date_str,
                "gender_sex": self.gender_sex,
                "weight_kg": self.weight_kg,
                "height_cm": self.height_cm,
                "age_years_at_session": self.age_years,
                "bmi_at_session": self.bmi_value,
            },
            "calibration_baselines_mv": {
                "left": calibration_left,
                "right": calibration_right,
            },
        }

        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save session metadata: {e}")

        # Write CSV Headers
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            with open(self.raw_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(RAW_CSV_HEADER)
        except Exception as e:
            print(f"Warning: Could not initialize session CSVs: {e}")

        self.is_active = True
        print(f"\n[LocalLogger] Session folder initialized for user '{self.user_name}' -> {self.session_dir}")

    def log_epoch(
        self,
        epoch_index: int,
        elapsed_minutes: float,
        composite_score: float,
        score_left_dict: dict,
        score_right_dict: dict,
        spectral_left_dict: dict,
        spectral_right_dict: dict,
        asymmetry_index: float,
        asymmetry_penalty_applied: bool,
        subjective_strain_cr10: float = None,
    ):
        """Appends one 5-minute epoch feature row to the session's CSV file, using explicit boolean flags for missing Borg CR-10 data."""
        if not self.is_active or not self.csv_path:
            return

        now_iso = datetime.now().isoformat()

        def _val(d, key, default=""):
            v = d.get(key, default)
            return "" if v is None or (isinstance(v, float) and math.isnan(v)) else v

        def _get_spec_mean(d, key):
            series = d.get("spectral_series", {}).get(key, [])
            valid = [x for x in series if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if valid and len(valid) > 0:
                return round(float(sum(valid) / len(valid)), 4)
            return ""

        mdf_l = _get_spec_mean(spectral_left_dict, "mdf_hz")
        mnf_l = _get_spec_mean(spectral_left_dict, "mnf_hz")
        mdf_r = _get_spec_mean(spectral_right_dict, "mdf_hz")
        mnf_r = _get_spec_mean(spectral_right_dict, "mnf_hz")

        row = [
            epoch_index,
            now_iso,
            round(elapsed_minutes, 2),
            round(composite_score, 4) if composite_score is not None and not math.isnan(composite_score) else "",
            _val(score_left_dict, "eindex_live"),
            _val(score_right_dict, "eindex_live"),
            _val(score_left_dict, "eindex_cumulative"),
            _val(score_right_dict, "eindex_cumulative"),
            _val(score_left_dict, "short_suma_penalty"),
            _val(score_right_dict, "short_suma_penalty"),
            _val(score_left_dict, "suma_count"),
            _val(score_right_dict, "suma_count"),
            _val(score_left_dict, "gap_frequency_per_min"),
            _val(score_right_dict, "gap_frequency_per_min"),
            _val(score_left_dict, "apdf_10"),
            _val(score_left_dict, "apdf_50"),
            _val(score_left_dict, "apdf_90"),
            _val(score_right_dict, "apdf_10"),
            _val(score_right_dict, "apdf_50"),
            _val(score_right_dict, "apdf_90"),
            round(asymmetry_index, 4) if asymmetry_index is not None and not math.isnan(asymmetry_index) else "",
            "1" if asymmetry_penalty_applied else "0",
            mdf_l,
            mnf_l,
            _val(spectral_left_dict, "slope_hz_per_min"),
            _val(spectral_left_dict, "r_squared"),
            _val(spectral_left_dict, "n_windows"),
            "1" if mdf_l != "" else "0",  # mdf_computed_left explicit boolean
            "1" if spectral_left_dict.get("is_fatiguing", False) else "0",
            mdf_r,
            mnf_r,
            _val(spectral_right_dict, "slope_hz_per_min"),
            _val(spectral_right_dict, "r_squared"),
            _val(spectral_right_dict, "n_windows"),
            "1" if mdf_r != "" else "0",  # mdf_computed_right explicit boolean
            "1" if spectral_right_dict.get("is_fatiguing", False) else "0",
            "1" if (subjective_strain_cr10 is not None and not (isinstance(subjective_strain_cr10, float) and (math.isnan(subjective_strain_cr10) or subjective_strain_cr10 < 0))) else "0",  # strain_reported boolean flag
            round(float(subjective_strain_cr10), 1) if (subjective_strain_cr10 is not None and not (isinstance(subjective_strain_cr10, float) and (math.isnan(subjective_strain_cr10) or subjective_strain_cr10 < 0))) else "",  # NULL when unreported
        ]

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[LocalLogger] Logged epoch #{epoch_index} ({elapsed_minutes:.1f} min) to {self.csv_path.name}")
        except Exception as e:
            print(f"Warning: Could not append row to features CSV: {e}")

    def log_raw_chunk(self, chunk_left: list, chunk_right: list, base_sample_idx: int, fs: float, epoch_idx: int):
        """Buffers raw high-frequency telemetry waveforms and flushes once every second."""
        if not self.is_active or not self.raw_path:
            return
        dt = 1.0 / max(1, fs)
        rows = []
        for idx, (l_val, r_val) in enumerate(zip(chunk_left, chunk_right)):
            s_idx = base_sample_idx + idx
            t_s = round(s_idx * dt, 4)
            rows.append([s_idx, t_s, epoch_idx, round(float(l_val), 2), round(float(r_val), 2)])
        self.raw_buffer.extend(rows)
        if len(self.raw_buffer) >= 500:
            self.flush_raw_buffer()

    def flush_raw_buffer(self):
        """Writes buffered raw rows to disk cleanly without IO bottlenecking."""
        if not self.raw_buffer or not self.raw_path:
            return
        try:
            with open(self.raw_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(self.raw_buffer)
            self.raw_buffer.clear()
        except Exception as e:
            print(f"Warning: Could not write raw telemetry buffer: {e}")

    def update_calibration(self, calibration_left: float, calibration_right: float):
        """Updates the session metadata file when a live shrug calibration completes."""
        if not self.is_active or not self.meta_path or not self.meta_path.exists():
            return
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["calibration_baselines_mv"] = {"left": calibration_left, "right": calibration_right}
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not update calibration in metadata JSON: {e}")

    def end_session(self, password: str = None):
        """Marks the session as completed, compresses raw telemetry to Parquet, and updates metadata JSON."""
        self.flush_raw_buffer()
        # Convert staging raw CSV into ultra-compressed Parquet archive
        if self.raw_path and self.raw_path.exists():
            try:
                import pandas as pd
                parquet_path = self.raw_path.with_suffix(".parquet")
                df_raw = pd.read_csv(self.raw_path)
                df_raw.to_parquet(parquet_path, index=False)
                self.raw_path.unlink(missing_ok=True)
                print(f"[LocalLogger] Compressed raw high-frequency telemetry to Parquet: {parquet_path.name}")
                self.raw_path = parquet_path
            except Exception as e:
                print(f"Warning: Could not compress raw telemetry to Parquet (retaining plain CSV): {e}")
        if not self.is_active or not self.meta_path or not self.meta_path.exists():
            return
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["end_time"] = datetime.now().isoformat()
            data["status"] = "completed"
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"[LocalLogger] Session completed -> saved to {self.session_dir}")
        except Exception as e:
            print(f"Warning: Could not finalize metadata JSON: {e}")
        finally:
            self.is_active = False
            if os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_EDGE_URL") or (hasattr(self, "db_uri") and self.db_uri):
                try:
                    from .cloud_ingest import sync_logs_to_postgres
                    print(f"[LocalLogger] Automatically synchronizing completed session via HTTPS Edge Function transport...")
                    # Sync ONLY this session's folder — not the whole output_logs tree.
                    # This prevents re-attempting every historical session on every stop.
                    sync_logs_to_postgres(log_dir=str(self.session_dir), db_uri=getattr(self, "db_uri", None), quiet=False, password=password)
                    print(f"[LocalLogger] Automatic cloud sync complete.")
                except Exception as sync_err:
                    print(f"[LocalLogger] Notice: Automatic cloud database sync skipped or failed: {sync_err}")

