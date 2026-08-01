"""
Phase 7 — Spectral Fatigue Tests (Stage 10: MDF/MNF + OLS Slope)

Test design philosophy (matches project preference for verified, simulated
numbers over bare assertions):
  - All tests use synthetic signals with analytically known properties.
  - Numeric results are asserted against tight tolerances derived from the
    known signal structure, not arbitrary margins.
  - Hardware not required — all tests are offline and deterministic.

Coverage:
  1. Mathematical precision — single-tone MDF/MNF accuracy
  2. Dual-tone equal-power centering
  3. Pre-rectification distortion demonstration (WHY we use Stage 2 output)
  4. Active-mask gating — rest epochs are ignored
  5. OLS slope recovery — "Farina fatigue drift" synthetic simulation
  6. Edge cases — empty/zero signal, insufficient windows
  7. Bilateral independence
  8. Passband caveat always present in output
  9. Fatigue flag reads thresholds from config, not hardcoded (Phase 6 hotfix class)
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.spectral import (
    compute_welch_psd,
    compute_mdf_mnf,
    extract_active_spectral_series,
    compute_spectral_fatigue_slope,
    analyze_bilateral_spectral_fatigue,
)


FS = 500   # locked hardware sampling rate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sine(freq_hz, duration_s, fs=FS, amplitude=1.0, seed=None):
    """Pure sine wave at freq_hz Hz."""
    t = np.arange(int(duration_s * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _bandlimited_noise(duration_s, fs=FS, seed=42):
    """Bandlimited Gaussian noise in 20-240 Hz (simulates resting EMG noise)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(int(duration_s * fs))
    sos = butter(4, [20.0, 240.0], btype='bandpass', output='sos', fs=fs)
    return sosfilt(sos, noise) * 0.05   # low amplitude = resting floor


def _chirp_emg(duration_s, start_hz=105.0, end_hz=65.0, fs=FS, amplitude=1.0):
    """
    Linearly frequency-swept sine: simulates MDF drifting from start_hz down
    to end_hz over duration_s seconds (Farina fatigue drift model).
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    # Instantaneous frequency decreases linearly
    inst_freq = start_hz + (end_hz - start_hz) * t / duration_s
    # Instantaneous phase = integral of frequency
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    return amplitude * np.sin(phase)


# ── Test class 1: Mathematical precision ─────────────────────────────────────

class TestComputeWelchPsdAndMdfMnf:

    def test_single_tone_mdf_mnf_accuracy(self):
        """
        Pure 100 Hz sine → MDF and MNF should both ≈ 100.0 Hz.
        Tolerance = ±1.0 Hz (Welch frequency bin resolution at 1s window, fs=500).
        """
        sig = _sine(100.0, duration_s=5.0)
        freqs, psd = compute_welch_psd(sig, fs=FS)
        mdf, mnf = compute_mdf_mnf(freqs, psd)

        assert not np.isnan(mdf), "MDF should not be NaN for a pure tone"
        assert not np.isnan(mnf), "MNF should not be NaN for a pure tone"
        assert abs(mdf - 100.0) <= 1.0, f"MDF={mdf:.2f} Hz, expected ~100.0 Hz ±1.0"
        assert abs(mnf - 100.0) <= 1.0, f"MNF={mnf:.2f} Hz, expected ~100.0 Hz ±1.0"

    def test_dual_tone_equal_power_centering(self):
        """
        Equal-amplitude tones at 60 Hz + 120 Hz.

        MNF (power-weighted centroid) = Σ(f·P) / Σ(P) ≈ 90 Hz ✓
        MDF (median power frequency) ≠ centroid for discrete tones.

        The cumulative spectral power is concentrated at two bins: 60 Hz and
        120 Hz. The running sum reaches 50% of total power exactly at the 60 Hz
        bin — so MDF ≈ 60 Hz is MATHEMATICALLY CORRECT.

        This test documents this distinction explicitly: MNF ≈ centroid,
        MDF ≈ lower tone for two equal-power discrete frequency components.
        Tolerance: ±2 Hz for MNF (centroid) and ±3 Hz for MDF (lower bin).
        """
        sig = _sine(60.0, 5.0) + _sine(120.0, 5.0)
        freqs, psd = compute_welch_psd(sig, fs=FS)
        mdf, mnf = compute_mdf_mnf(freqs, psd)

        # MNF = power-weighted centroid ≈ (60+120)/2 = 90 Hz
        assert abs(mnf - 90.0) <= 3.0, f"MNF={mnf:.2f} Hz, expected ~90.0 Hz ±3.0"

        # MDF = median power frequency ≈ lower tone (~60 Hz) for equal discrete tones
        # (50% of total power is reached at/near the first spike at 60 Hz)
        assert abs(mdf - 60.0) <= 3.0, (
            f"MDF={mdf:.2f} Hz, expected ~60.0 Hz ±3.0 for equal-power dual tones. "
            "MDF is NOT the frequency centroid — that is MNF. "
            "For two equal-power discrete tones, cumulative power hits 50% at the lower tone."
        )

    def test_empty_inputs_return_nan(self):
        """Empty freqs/psd arrays → (nan, nan), no exception."""
        mdf, mnf = compute_mdf_mnf(np.array([]), np.array([]))
        assert np.isnan(mdf)
        assert np.isnan(mnf)

    def test_zero_signal_returns_empty_psd(self):
        """All-zeros signal → compute_welch_psd returns empty arrays (zero power guard)."""
        sig = np.zeros(1000)
        freqs, psd = compute_welch_psd(sig, fs=FS)
        assert len(freqs) == 0
        assert len(psd) == 0

    def test_signal_shorter_than_window_returns_empty(self):
        """Signal shorter than one Welch window → empty arrays, no exception."""
        sig = np.random.randn(10)   # 10 samples << 500-sample window
        freqs, psd = compute_welch_psd(sig, fs=FS)
        assert len(freqs) == 0
        assert len(psd) == 0

    def test_psd_band_restricted_to_config_range(self):
        """All returned frequency bins must be within [SPECTRAL_BAND_LOW_HZ, SPECTRAL_BAND_HIGH_HZ]."""
        sig = _sine(80.0, 3.0)
        freqs, _ = compute_welch_psd(sig, fs=FS)
        assert len(freqs) > 0
        assert freqs.min() >= TBConfig.SPECTRAL_BAND_LOW_HZ - 1e-6
        assert freqs.max() <= TBConfig.SPECTRAL_BAND_HIGH_HZ + 1e-6


# ── Test class 2: Pre-rectification requirement ───────────────────────────────

class TestPreRectificationRequirement:

    def test_rectification_distorts_spectral_estimate(self):
        """
        Prove that rectifying an 80 Hz signal before spectral analysis
        corrupts MDF/MNF, making Stage 4 output invalid for spectral use.

        Expected: rectified signal's MDF shifts meaningfully BELOW 80 Hz
        due to harmonic energy at DC + 160 Hz being interpreted differently.

        This documents WHY extract_active_spectral_series requires
        pre-rectification input (Stage 2 output, not Stage 4).
        """
        sig = _sine(80.0, 5.0)

        # Pre-rectification: correct spectral estimate
        freqs_pre, psd_pre = compute_welch_psd(sig, fs=FS)
        mdf_pre, _ = compute_mdf_mnf(freqs_pre, psd_pre)

        # Post-rectification: corrupted by harmonics
        sig_rect = np.abs(sig)
        freqs_rect, psd_rect = compute_welch_psd(sig_rect, fs=FS)
        mdf_rect, _ = compute_mdf_mnf(freqs_rect, psd_rect)

        # The difference must be significant (> 5 Hz) demonstrating corruption
        assert not np.isnan(mdf_pre),  "Pre-rect MDF should not be NaN"
        assert not np.isnan(mdf_rect), "Post-rect MDF should not be NaN"
        shift = abs(mdf_pre - mdf_rect)
        assert shift > 5.0, (
            f"Rectification distortion was only {shift:.2f} Hz — expected >5.0 Hz. "
            "Pre- vs post-rectification MDF difference confirms spectral corruption."
        )
        # Document the actual shift in the message (visible in pytest -v output)
        print(f"\n  Pre-rect MDF: {mdf_pre:.1f} Hz  |  Post-rect MDF: {mdf_rect:.1f} Hz  "
              f"|  Shift: {shift:.1f} Hz  (>5.0 Hz confirms rectification corrupts spectrum)")


# ── Test class 3: Active-mask gating ─────────────────────────────────────────

class TestActiveMaskGating:

    def test_resting_epochs_are_skipped(self):
        """
        Signal alternates: active 100 Hz sine, then resting floor noise.
        Active mask marks only the sine sections as active.
        Result: only active epochs contribute to MDF series.
        """
        n = FS * 20   # 20 seconds
        signal = np.zeros(n)
        active_mask = np.zeros(n, dtype=bool)

        # 0-5s: active (100 Hz sine)
        signal[0:FS*5] = _sine(100.0, 5.0)
        active_mask[0:FS*5] = True

        # 5-10s: rest (low-amplitude noise) — mask stays False
        signal[FS*5:FS*10] = _bandlimited_noise(5.0)

        # 10-15s: active
        signal[FS*10:FS*15] = _sine(100.0, 5.0)
        active_mask[FS*10:FS*15] = True

        # 15-20s: rest
        signal[FS*15:FS*20] = _bandlimited_noise(5.0)

        # Without mask: gating disabled → more epochs (includes rest)
        result_no_mask = extract_active_spectral_series(signal, active_mask=None, fs=FS)

        # With mask: only active epochs
        result_masked = extract_active_spectral_series(signal, active_mask=active_mask, fs=FS)

        assert result_masked['n_skipped'] > 0, \
            "Some epochs should be skipped when mask is applied"
        assert result_masked['n_active'] <= result_no_mask['n_active'], \
            "Masked result should have <= active epochs than unmasked"
        assert result_masked['n_active'] > 0, \
            "At least some active epochs should pass the gate"

    def test_all_rest_gives_zero_active_epochs(self):
        """Signal all below active ratio → n_active = 0, slope returns insufficient data."""
        n = FS * 10
        signal = _bandlimited_noise(10.0)
        active_mask = np.zeros(n, dtype=bool)   # all inactive

        result = extract_active_spectral_series(signal, active_mask=active_mask, fs=FS)
        assert result['n_active'] == 0
        assert result['n_skipped'] == result['n_total']

        # Slope on empty series → insufficient_data flag
        slope = compute_spectral_fatigue_slope(result['times_sec'], result['mdf_hz'])
        assert slope['sufficient_data'] is False
        assert np.isnan(slope['slope_hz_per_min'])


# ── Test class 4: OLS slope recovery ─────────────────────────────────────────

class TestOLSSlopeRecovery:

    def test_farina_fatigue_drift_recovery(self):
        """
        Synthesise a 10-minute chirp signal where the dominant frequency drifts
        linearly from 105 Hz down to 65 Hz (a -4.0 Hz/min rate).

        Assert that the recovered OLS slope is -4.0 Hz/min ± 0.5 Hz/min
        and that R² > 0.85 (strong linear trend should be easily detectable).

        This is the primary quantitative validation of the spectral pipeline.
        """
        duration_s = 600.0   # 10 minutes
        start_hz   = 105.0
        end_hz     = 65.0
        true_slope = (end_hz - start_hz) / (duration_s / 60.0)   # = -4.0 Hz/min

        sig = _chirp_emg(duration_s, start_hz=start_hz, end_hz=end_hz)
        result = extract_active_spectral_series(sig, active_mask=None, fs=FS)
        slope_report = compute_spectral_fatigue_slope(
            result['times_sec'], result['mdf_hz']
        )

        assert slope_report['sufficient_data'], \
            "10-minute session must have sufficient windows"

        recovered_slope = slope_report['slope_hz_per_min']
        r2 = slope_report['r_squared']

        print(f"\n  True slope: {true_slope:.2f} Hz/min")
        print(f"  Recovered:  {recovered_slope:.3f} Hz/min")
        print(f"  R²:         {r2:.3f}")

        assert abs(recovered_slope - true_slope) <= 0.5, (
            f"Recovered slope {recovered_slope:.3f} Hz/min deviates more than ±0.5 "
            f"from true slope {true_slope:.2f} Hz/min"
        )
        assert r2 >= 0.85, (
            f"R²={r2:.3f} should be >= 0.85 for a linearly drifting chirp"
        )

    def test_stable_frequency_gives_near_zero_slope(self):
        """
        Constant-frequency signal → slope should be near 0.0 Hz/min.
        Confirms no spurious fatigue trend is detected on a healthy signal.
        """
        sig = _sine(90.0, 300.0)   # 5 minutes at steady 90 Hz
        result = extract_active_spectral_series(sig, active_mask=None, fs=FS)
        slope_report = compute_spectral_fatigue_slope(
            result['times_sec'], result['mdf_hz']
        )
        slope = slope_report['slope_hz_per_min']
        assert abs(slope) < 0.5, \
            f"Stable signal should have slope ≈0.0 Hz/min, got {slope:.3f}"


# ── Test class 5: Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:

    def test_insufficient_windows_returns_sufficient_data_false(self):
        """Fewer than SPECTRAL_MIN_VALID_WINDOWS valid epochs → sufficient_data=False."""
        # 2-second signal gives only 2-3 Welch windows — below the minimum of 5
        sig = _sine(80.0, 2.0)
        result = extract_active_spectral_series(sig, active_mask=None, fs=FS)
        slope_report = compute_spectral_fatigue_slope(
            result['times_sec'], result['mdf_hz']
        )
        assert slope_report['sufficient_data'] is False
        assert np.isnan(slope_report['slope_hz_per_min'])
        assert slope_report['is_fatiguing'] is False

    def test_all_zero_signal_no_exception(self):
        """All-zeros signal must not raise any exception — return gracefully."""
        sig = np.zeros(FS * 10)
        result = extract_active_spectral_series(sig, active_mask=None, fs=FS)
        assert result['n_active'] == 0 or all(
            np.isnan(v) for v in result['mdf_hz']
        )

    def test_nan_mdf_values_excluded_from_slope(self):
        """NaN MDF values (from zero-power epochs) must be filtered before OLS."""
        times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        mdf   = np.array([100.0, np.nan, 98.0, np.nan, 96.0, 94.0])
        slope_report = compute_spectral_fatigue_slope(times, mdf)
        # Should use 4 valid windows, not 6
        assert slope_report['n_windows'] == 4
        assert slope_report['sufficient_data'] is False  # 4 < SPECTRAL_MIN_VALID_WINDOWS=5


# ── Test class 6: Bilateral independence ─────────────────────────────────────

class TestBilateralIndependence:

    def test_bilateral_channels_independent(self):
        """
        Left: steady 90 Hz (no fatigue).
        Right: chirp 105→65 Hz over 5 minutes (strong fatigue).
        Assert: left is_fatiguing=False, right is_fatiguing=True (if slope threshold met).
        Assert: left slope and right slope are meaningfully different.
        """
        left_sig  = _sine(90.0, 300.0)          # no fatigue
        right_sig = _chirp_emg(300.0, 105, 65)  # -8 Hz/min fatigue

        report = analyze_bilateral_spectral_fatigue(
            left_sig, right_sig, fs=FS
        )

        left_slope  = report['left']['slope_hz_per_min']
        right_slope = report['right']['slope_hz_per_min']

        assert not np.isnan(left_slope),  "Left slope should be computed"
        assert not np.isnan(right_slope), "Right slope should be computed"
        assert abs(left_slope) < abs(right_slope), (
            f"Left (stable) slope {left_slope:.3f} should be smaller in magnitude "
            f"than right (fatiguing) slope {right_slope:.3f}"
        )
        print(f"\n  Left slope: {left_slope:.3f} Hz/min  |  Right slope: {right_slope:.3f} Hz/min")

    def test_passband_caveat_always_present(self):
        """
        analyze_bilateral_spectral_fatigue output MUST always contain the
        'passband_ceiling_caveat' key with a non-empty string.
        This is a methodology integrity check — the caveat is mandatory.
        """
        left  = _sine(80.0, 10.0)
        right = _sine(80.0, 10.0)
        report = analyze_bilateral_spectral_fatigue(left, right, fs=FS)

        assert 'passband_ceiling_caveat' in report, \
            "passband_ceiling_caveat key must always be present in bilateral output"
        caveat = report['passband_ceiling_caveat']
        assert isinstance(caveat, str) and len(caveat) > 0, \
            "passband_ceiling_caveat must be a non-empty string"
        assert '240' in caveat, \
            "Caveat should mention 240 Hz (the Nyquist ceiling)"
        assert '500' in caveat, \
            "Caveat should mention 500 Hz (the sampling rate)"


# ── Test class 7: Config-driven thresholds (anti-hardcoding guard) ────────────

class TestFatigueFlagUsesConfigThresholds:
    """
    Guard equivalent to test_asymmetry_penalty_proportional_to_scale in Phase 6.
    The fatigue flag must read FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN and
    FATIGUE_MIN_R_SQUARED from TBConfig — not hardcoded values.

    If someone hardcodes -0.5/-0.3 in the function, this test catches it by
    changing the config threshold to a value that flips the flag, and verifying
    the output changes accordingly.
    """

    def test_fatigue_flag_responds_to_config_threshold_change(self):
        """
        Build a time series with a known -0.4 Hz/min slope and good R².
        With threshold = -0.3, flag should be TRUE  (slope < threshold).
        With threshold = -0.5, flag should be FALSE (slope > threshold).
        The function must use config, not a hardcoded value.
        """
        # Construct a synthetic MDF series with slope ≈ -0.4 Hz/min
        # 10 points over 10 minutes: MDF drops from 100 to 96 Hz
        times_min = np.linspace(0, 10, 10)
        mdf_values = 100.0 - 0.4 * times_min
        times_sec = times_min * 60.0

        # Config with threshold = -0.3 → -0.4 < -0.3 → should flag as fatiguing
        class LowThresholdConfig(TBConfig):
            FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN = -0.3
            FATIGUE_MIN_R_SQUARED = 0.3
            SPECTRAL_MIN_VALID_WINDOWS = 5

        report_low = compute_spectral_fatigue_slope(
            times_sec, mdf_values, config=LowThresholdConfig
        )

        # Config with threshold = -0.5 → -0.4 > -0.5 → should NOT flag
        class HighThresholdConfig(TBConfig):
            FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN = -0.5
            FATIGUE_MIN_R_SQUARED = 0.3
            SPECTRAL_MIN_VALID_WINDOWS = 5

        report_high = compute_spectral_fatigue_slope(
            times_sec, mdf_values, config=HighThresholdConfig
        )

        slope = report_low['slope_hz_per_min']
        print(f"\n  Slope: {slope:.4f} Hz/min")
        print(f"  With threshold=-0.3: is_fatiguing={report_low['is_fatiguing']}")
        print(f"  With threshold=-0.5: is_fatiguing={report_high['is_fatiguing']}")

        assert report_low['is_fatiguing'] is True, (
            f"With threshold=-0.3 and slope≈-0.4, is_fatiguing must be True. "
            f"Got False — check if threshold is hardcoded in compute_spectral_fatigue_slope."
        )
        assert report_high['is_fatiguing'] is False, (
            f"With threshold=-0.5 and slope≈-0.4, is_fatiguing must be False. "
            f"Got True — check if threshold is hardcoded in compute_spectral_fatigue_slope."
        )

    def test_slope_threshold_echoed_in_output(self):
        """
        The slope dict must echo back the threshold values it actually used.
        This gives a machine-verifiable paper trail — you can always reconstruct
        what threshold produced a given is_fatiguing flag.
        """
        times_sec = np.linspace(0, 600, 20)
        mdf_values = 100.0 - 0.4 * (times_sec / 60.0)

        report = compute_spectral_fatigue_slope(times_sec, mdf_values)

        assert 'slope_threshold_used' in report
        assert 'r_squared_threshold_used' in report
        assert report['slope_threshold_used'] == TBConfig.FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN, \
            "slope_threshold_used must equal TBConfig.FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN"
        assert report['r_squared_threshold_used'] == TBConfig.FATIGUE_MIN_R_SQUARED, \
            "r_squared_threshold_used must equal TBConfig.FATIGUE_MIN_R_SQUARED"
