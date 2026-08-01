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
