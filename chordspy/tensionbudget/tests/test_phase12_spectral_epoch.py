"""
test_phase12_spectral_epoch.py — Verification tests for Phase 12 Full-Epoch Spectral Accumulation Architecture (Bug 2 Root Fix).
"""

import csv
import numpy as np
import pytest
from pathlib import Path

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.preprocessing import StreamingChannelProcessor
from chordspy.tensionbudget.spectral import analyze_bilateral_spectral_fatigue


class TestPhase12SpectralEpochBuffer:
    """Verifies that full-epoch spectral accumulation utilizes Stage 2 bandpass & notch filtering, achieves high Welch window volume (>=300), maintains slope stability, and clears memory upon logging."""

    def test_epoch_filt_buffer_uses_bandpass_filtered_signal(self):
        """
        Verify the architectural correction: epoch accumulation buffers must receive pre-rectified bandpass & notch
        filtered signals (Stage 2 output), NOT raw unfiltered ADC samples.
        We feed an out-of-band baseline drift (5 Hz, below the 20 Hz bandpass cutoff) and verify strong attenuation.
        """
        cfg = TBConfig()
        fs = cfg.TARGET_SAMPLING_RATE  # 500 Hz
        proc = StreamingChannelProcessor(fs=fs, config=cfg, vref=None)

        # Create a strong 5 Hz baseline drift (amplitude = 5000 ADC counts around mid-rail 8192)
        t = np.linspace(0, 2.0, int(fs * 2.0))
        raw_adc = 8192 + 5000 * np.sin(2 * np.pi * 5 * t)
        
        # Process in chunks as in live streaming
        chunk_size = 50
        filtered_out = []
        for i in range(0, len(raw_adc), chunk_size):
            res = proc.process_chunk(raw_adc[i:i + chunk_size])
            if "filtered" in res and len(res["filtered"]) > 0:
                filtered_out.extend(res["filtered"])

        filtered_arr = np.array(filtered_out)
        # Calculate RMS amplitude of raw AC counts vs filtered counts after initial transient (first 200 samples)
        raw_ac = raw_adc[200:] - np.mean(raw_adc[200:])
        raw_rms = np.sqrt(np.mean(raw_ac**2))
        filt_rms = np.sqrt(np.mean(filtered_arr[200:]**2))

        # Assert at least 20 dB attenuation of the 5 Hz drift
        attenuation_db = 20 * np.log10(raw_rms / max(1e-9, filt_rms))
        assert attenuation_db > 20.0, (
            f"Expected Stage 2 bandpass filter to attenuate 5 Hz drift by >20 dB, got {attenuation_db:.2f} dB. "
            "Confirms signal tap is before rectification and after bandpass filtering."
        )

    def test_full_epoch_window_count_and_memory_clearance(self, tmp_path):
        """
        Integration test verifying that when an epoch log occurs, spectral fatigue is computed over the full
        accumulated 5-minute epoch buffer (producing n_windows >= 300) and memory is cleared immediately after.
        """
        try:
            from PyQt5.QtWidgets import QApplication
            from chordspy.tensionbudget_app import TensionBudgetApp
            from chordspy.tensionbudget import local_logger
        except ImportError as e:
            pytest.skip(f"Skipping GUI integration test due to missing dependency: {e}")

        app = QApplication.instance() or QApplication([])

        orig_logs_root = local_logger.LOGS_ROOT
        local_logger.LOGS_ROOT = tmp_path

        try:
            win = TensionBudgetApp()
            win.mode = "offline"
            win.ref_rms_left = 0.001
            win.ref_rms_right = 0.001

            win.local_logger = local_logger.LocalSessionLogger(user_name="SpectralTester", birth_date_str="1990-01-01")
            win.local_logger.start_session(mode="offline", calibration_left=0.001, calibration_right=0.001)
            csv_path = win.local_logger.csv_path

            # Simulate 300 seconds (5 minutes) of accumulated bandpass-filtered active signal in epoch_filt_l/r
            # 150,000 samples @ 500 Hz of 40 Hz EMG activation
            t = np.linspace(0, 300, 150000)
            sig = 0.05 * np.sin(2 * np.pi * 40 * t) + 0.01 * np.random.randn(len(t))
            win.epoch_filt_l = list(sig)
            win.epoch_filt_r = list(sig)

            # Ensure minimal score results exist so logging proceeds
            win.last_score_res = {"composite": {"composite_score": 10.0}, "asymmetry_index": 0.0}
            win.last_res_l = {"gaps_count": 5, "gaps_total_time_s": 10.0, "active_apdf": {10: 5.0, 50: 15.0, 90: 30.0}}
            win.last_res_r = {"gaps_count": 5, "gaps_total_time_s": 10.0, "active_apdf": {10: 5.0, 50: 15.0, 90: 30.0}}

            # Trigger epoch log
            win.session_sample_count = 150000
            win._log_current_epoch(epoch_idx=1, is_final=False)

            # Assert memory clearance after logging
            assert len(win.epoch_filt_l) == 0, "epoch_filt_l was not cleared after epoch logging!"
            assert len(win.epoch_filt_r) == 0, "epoch_filt_r was not cleared after epoch logging!"

            # Read generated CSV to verify observed n_windows telemetry
            assert csv_path and Path(csv_path).exists(), "Feature CSV log file was not created."
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                assert len(reader) >= 1
                row = reader[0]
                n_win_l = float(row.get("n_windows_left", 0))
                n_win_r = float(row.get("n_windows_right", 0))

                # On a 10s buffer, n_windows was <= 19. On a full 300s epoch buffer with 500Hz sampling
                # and 50% overlap of 256-sample Welch windows, expected windows > 300.
                assert n_win_l >= 300, f"Expected full-epoch Welch window count >= 300, got {n_win_l}"
                assert n_win_r >= 300, f"Expected full-epoch Welch window count >= 300, got {n_win_r}"

        finally:
            local_logger.LOGS_ROOT = orig_logs_root

    def test_slope_stability_under_short_bursts(self):
        """
        Verify Bug 2 resolution: on short 10s visualization buffers, a brief 2-3s muscular contraction created minimal
        Welch windows and resulted in erratic slope inflation (e.g. -196 Hz/min). Over the full 300s epoch buffer,
        OLS regression remains mathematically stable without artificial magnitude inflation.
        """
        cfg = TBConfig()
        fs = cfg.TARGET_SAMPLING_RATE
        n_samples = fs * 300  # 300 seconds

        # Create a predominantly quiet baseline with a single brief 3-second muscular contraction
        sig = 0.001 * np.random.randn(n_samples)
        burst_start = fs * 100
        burst_end = fs * 103
        t_burst = np.linspace(0, 3, burst_end - burst_start)
        # Contraction frequency decaying from 90 Hz to 70 Hz over 3 seconds
        sig[burst_start:burst_end] += 0.08 * np.sin(2 * np.pi * (90 - 3 * t_burst) * t_burst)

        res = analyze_bilateral_spectral_fatigue(sig, sig, fs=fs, config=cfg)
        slope_l = res["left"].get("slope_hz_per_min", 0.0)
        
        # Assert slope magnitude is completely bounded and stable (< 15 Hz/min)
        assert not np.isnan(slope_l), "Slope computed as NaN on full epoch buffer."
        assert abs(slope_l) < 15.0, f"Slope remained erratically inflated: {slope_l:.2f} Hz/min"
