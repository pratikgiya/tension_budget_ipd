"""
test_phase8_app.py — Verification tests for Phase 8 Real-Time & Offline Bilateral Monitoring App.
"""

from pathlib import Path
import yaml
import pytest
import numpy as np
from chordspy.tensionbudget.config import TBConfig


class TestPhase8AppConfiguration:
    """Verifies app configuration, YAML registration, and hardware channel mappings."""

    def test_app_registered_in_yaml(self):
        """Verify tensionbudget_app is registered in chordspy/config/apps.yaml."""
        apps_yaml_path = Path(__file__).resolve().parent.parent.parent / "config" / "apps.yaml"
        assert apps_yaml_path.exists(), f"Expected apps.yaml at {apps_yaml_path}"
        
        with open(apps_yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            
        apps_list = config_data.get("apps", [])
        tb_app = next((a for a in apps_list if a.get("script") == "tensionbudget_app"), None)
        
        assert tb_app is not None, "tensionbudget_app not found in apps.yaml script list!"
        assert tb_app["title"] == "TensionBudget Monitor"
        assert tb_app["category"] == "EMG"
        assert "Bilateral" in tb_app["description"] or "sEMG" in tb_app["description"]

    def test_default_channel_map(self):
        """Verify TBConfig.CHANNEL_MAP is configured for dual-Arduino LSL channels (0=Left, 1=Right)."""
        cfg = TBConfig()
        assert cfg.CHANNEL_MAP == {"left": 0, "right": 1}, (
            f"Expected default CHANNEL_MAP to be {{'left': 0, 'right': 1}}, got {cfg.CHANNEL_MAP}"
        )

    def test_import_and_math_logic(self):
        """Verify that chordspy.tensionbudget_app can be imported and moving RMS logic is sound."""
        try:
            from chordspy import tensionbudget_app
        except ImportError as e:
            pytest.skip(f"Skipping import due to missing system dependency: {e}")
            
        assert hasattr(tensionbudget_app, "TensionBudgetApp")
        assert hasattr(tensionbudget_app, "main")
        
        # Test calculate_moving_rms directly without spawning GUI window
        app_cls = tensionbudget_app.TensionBudgetApp
        
        # Synthetic constant signal: envelope of constant 5.0 should equal 5.0
        const_sig = np.ones(100) * 5.0
        rms_res = app_cls.calculate_moving_rms(None, const_sig, window_size=10)
        
        assert len(rms_res) == 100
        assert np.allclose(rms_res, 5.0, atol=1e-5)

        # Synthetic short signal fallback
        short_sig = np.array([2.0, -2.0, 2.0])
        rms_short = app_cls.calculate_moving_rms(None, short_sig, window_size=10)
        assert len(rms_short) == 3
        assert np.allclose(rms_short, [2.0, 2.0, 2.0], atol=1e-5)


class TestAppLoggerIntegration:
    """Regression test for multi-window epoch logging and cumulative EIndex tracking."""

    def test_cumulative_eindex_steps_upward_and_logs_cleanly(self, tmp_path):
        """
        Regression test for BUG 1 (eindex_cumulative_left/right always logs as 0.0).
        Verifies end-to-end through the live app update loop and logger path that:
          1) submit_completed_window() is called at each epoch boundary.
          2) eindex_cumulative_left and eindex_cumulative_right step upward across windows.
          3) Values are serialized into the CSV log columns and are never stuck at 0.0 past window 1.
          4) R² columns (mdf_r_squared_left/right) exist in the logged output schema.
        """
        try:
            from PyQt5.QtWidgets import QApplication
            from chordspy.tensionbudget_app import TensionBudgetApp
            from chordspy.tensionbudget import local_logger
        except ImportError as e:
            pytest.skip(f"Skipping GUI test due to missing dependency: {e}")

        app = QApplication.instance() or QApplication([])

        # Redirect logging directory to temporary test path
        orig_logs_root = local_logger.LOGS_ROOT
        local_logger.LOGS_ROOT = tmp_path

        try:
            win = TensionBudgetApp()
            win.mode = "offline"
            win.ref_rms_left = 0.001   # Low baseline calibration to ensure high muscle activation (%RVE > 100%)
            win.ref_rms_right = 0.001

            # Provide synthetic active EMG signal (enough samples for multiple chunk ticks)
            t = np.linspace(0, 2.0, 1000)
            sig = 0.05 * np.sin(2 * np.pi * 30 * t) + 0.02 * np.random.randn(len(t))
            win.offline_data = {"left": sig, "right": sig}
            win.offline_idx = 0
            win.stream_active = True

            # Pre-fill rolling buffers with active EMG signal to prevent default zeros (P_Rest) from dominating
            win.raw_left = np.random.uniform(0.02, 0.05, win.buffer_size)
            win.raw_right = np.random.uniform(0.02, 0.05, win.buffer_size)

            # Initialize local logger session
            win.local_logger = local_logger.LocalSessionLogger(user_name="TestSubject", birth_date_str="2000-01-01")
            win.local_logger.start_session(mode="offline", calibration_left=0.001, calibration_right=0.001)
            csv_path = win.local_logger.csv_path

            # Warm up buffers with active signal (10 ticks = 200 ms, triggers real-time analytics)
            for _ in range(10):
                win.update_loop()
            assert hasattr(win, "last_score_res"), "App failed to generate score results during loop."

            # Fast-forward sample count to just before Epoch #1 boundary (300 seconds @ 500 Hz = 150,000 samples)
            win.session_sample_count = int(500 * 300) - 5
            win.update_loop()  # Crosses epoch boundary -> logs Epoch 1 & submits completed window #1

            cum_l_1 = win.accumulator_l.eindex_session_cumulative
            cum_r_1 = win.accumulator_r.eindex_session_cumulative
            assert cum_l_1 > 0.0, f"Cumulative EIndex Left remained 0.0 after first completed window! Got {cum_l_1}"
            assert win.accumulator_l.completed_windows == 1

            # Fast-forward sample count to just before Epoch #2 boundary (600 seconds @ 500 Hz = 300,000 samples)
            win.session_sample_count = int(500 * 300 * 2) - 5
            win.update_loop()  # Crosses epoch boundary -> logs Epoch 2 & submits completed window #2

            cum_l_2 = win.accumulator_l.eindex_session_cumulative
            cum_r_2 = win.accumulator_r.eindex_session_cumulative
            assert cum_l_2 > cum_l_1, f"Cumulative EIndex Left did not step upward! (win 1: {cum_l_1}, win 2: {cum_l_2})"
            assert win.accumulator_l.completed_windows == 2

            # Close logger to flush CSV
            win.local_logger.end_session()
            assert csv_path.exists(), f"Expected CSV log file at {csv_path}"

            # Inspect CSV rows to confirm schema and non-zero cumulative eindex logging
            with open(csv_path, "r", encoding="utf-8") as f:
                import csv
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) >= 2, f"Expected at least 2 epoch rows logged, found {len(rows)}"
            
            # Verify Epoch 1 log
            assert float(rows[0]["eindex_cumulative_left"]) == pytest.approx(cum_l_1, rel=1e-3)
            assert float(rows[0]["eindex_cumulative_left"]) > 0.0
            assert "mdf_r_squared_left" in rows[0], "Missing R² quality metric in CSV columns!"
            assert "n_windows_left" in rows[0], "Missing n_windows_left metric in CSV columns!"

            # Verify Epoch 2 log steps upward
            assert float(rows[1]["eindex_cumulative_left"]) == pytest.approx(cum_l_2, rel=1e-3)
            assert float(rows[1]["eindex_cumulative_left"]) > float(rows[0]["eindex_cumulative_left"])

            # Verify metadata JSON includes processing_version marker
            with open(win.local_logger.meta_path, "r", encoding="utf-8") as f:
                import json
                meta_data = json.load(f)
            assert meta_data.get("processing_version") == "phase11_edge_padding", (
                f"Expected processing_version 'phase11_edge_padding', got {meta_data.get('processing_version')}"
            )

        finally:
            local_logger.LOGS_ROOT = orig_logs_root

