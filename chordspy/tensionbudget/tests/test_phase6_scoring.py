"""
TensionBudget — Phase 6 Tests: Composite Scoring Engine.

Tests all Phase 6 scoring components:
  - EIndex formula correctness and bounds
  - Short-SUMA frequency penalty (count-based, cap, per-window only)
  - Per-side score composition
  - Asymmetry index (signed, locked formula)
  - Composite score (fusion order, asymmetry penalty, floor clamp)
  - EIndexAccumulator (live vs session_cumulative separation)

Regression tests:
  - 8-hour session at constant moderate activity:
      * eindex_live stays in [-2, +2] throughout
      * eindex_session_cumulative only steps at discrete 10-min boundaries
        (flat between boundaries)

Design rule: tests use local config subclasses to avoid mutating the
global TBConfig (test-isolation bug documented in project context).
"""

import numpy as np
import pytest

from chordspy.tensionbudget.scoring import (
    compute_eindex_for_window,
    compute_short_suma_freq_penalty,
    compute_per_side_score,
    compute_asymmetry_index,
    compute_composite_score,
    EIndexAccumulator,
    map_eindex_to_display_scale,
)
from chordspy.tensionbudget.config import TBConfig


# ── Local config for isolated tests ──────────────────────────────────

class TestConfig(TBConfig):
    """Local subclass — overrides only what the test needs.
    Never mutates TBConfig class-level attributes directly."""
    EINDEX_REST_THRESHOLD_PCT = 0.5
    EINDEX_HIGH_THRESHOLD_PCT = 7.0
    SHORT_SUMA_PENALTY_PER_EVENT = 0.1
    ASYMMETRY_PENALTY_THRESHOLD = 0.5
    # ASYMMETRY_PENALTY_POINTS intentionally NOT overridden here —
    # inherits from TBConfig so the proportionality test catches
    # any future scale mismatch in the parent config.
    GAP_REST_THRESHOLD_PCT = 3.0


# ── 1. EIndex formula ─────────────────────────────────────────────────

class TestComputeEindex:

    def test_all_rest_gives_minus_two(self):
        """100% rest → EIndex = -2.0 (minimum possible value)."""
        arr = np.zeros(1000)  # all 0.0, below EINDEX_REST_THRESHOLD_PCT (0.5%)
        result = compute_eindex_for_window(arr, config=TestConfig)
        assert result == pytest.approx(-2.0)

    def test_all_high_gives_plus_two(self):
        """100% high load → EIndex = +2.0 (maximum possible value)."""
        arr = np.full(1000, 10.0)  # all 10%, above EINDEX_HIGH_THRESHOLD_PCT (7%)
        result = compute_eindex_for_window(arr, config=TestConfig)
        assert result == pytest.approx(2.0)

    def test_all_low_gives_plus_one(self):
        """100% low load → EIndex = +1.0."""
        # Low = [0.5%, 7%) — use 3.5% (mid-range)
        arr = np.full(1000, 3.5)
        result = compute_eindex_for_window(arr, config=TestConfig)
        assert result == pytest.approx(1.0)

    def test_equal_thirds_gives_one_third(self):
        """Equal split across rest/low/high → (-2+1+2)/3 = 1/3."""
        n = 300
        arr = np.concatenate([
            np.zeros(100),          # rest (< 0.5%)
            np.full(100, 3.5),      # low ([0.5%, 7%))
            np.full(100, 10.0),     # high (>= 7%)
        ])
        result = compute_eindex_for_window(arr, config=TestConfig)
        expected = (-2 * (100/300)) + (1 * (100/300)) + (2 * (100/300))
        assert result == pytest.approx(expected, abs=1e-9)

    def test_empty_window_returns_zero(self):
        """Empty window → 0.0 (neutral)."""
        result = compute_eindex_for_window(np.array([]), config=TestConfig)
        assert result == 0.0

    def test_bounded_below_minus_two(self):
        """Result can never fall below -2.0."""
        arr = np.zeros(500)
        result = compute_eindex_for_window(arr, config=TestConfig)
        assert result >= -2.0

    def test_bounded_above_plus_two(self):
        """Result can never exceed +2.0."""
        arr = np.full(500, 100.0)
        result = compute_eindex_for_window(arr, config=TestConfig)
        assert result <= 2.0


# ── 2. Short-SUMA frequency penalty ──────────────────────────────────

class TestShortSumaFreqPenalty:

    def _make_bins(self, n_short_first=0, n_short_second=0):
        """Helper: build suma_bins dict with given short-bin counts."""
        return {
            "1.5-5s": n_short_first,
            "5-10s": n_short_second,
            "10-20s": 0,
            "20-60s": 0,
            "1-2min": 0,
            "2-4min": 0,
            "4-8min": 0,
            "8-10min": 0,
            "10-20min": 0,
            ">20min": 0,
        }

    def test_zero_events_gives_zero_penalty(self):
        bins = self._make_bins(0, 0)
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == 0.0

    def test_one_event_gives_point_one(self):
        bins = self._make_bins(1, 0)
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == pytest.approx(0.1)

    def test_ten_events_gives_exactly_one(self):
        """10 events → exactly 1.0 (the cap)."""
        bins = self._make_bins(5, 5)  # 5+5 = 10
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == pytest.approx(1.0)

    def test_more_than_ten_events_still_capped_at_one(self):
        """20 events → still 1.0 — cap is hard."""
        bins = self._make_bins(10, 10)
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == pytest.approx(1.0)

    def test_nine_events_not_capped(self):
        """9 events → 0.9, not yet at cap."""
        bins = self._make_bins(9, 0)
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == pytest.approx(0.9)

    def test_only_counts_short_bins(self):
        """Long-bin events (>10s) contribute zero to the frequency penalty."""
        bins = {
            "1.5-5s": 0,
            "5-10s": 0,
            "10-20s": 50,   # many long events — should not count
            "20-60s": 50,
            "1-2min": 50,
            "2-4min": 50,
            "4-8min": 50,
            "8-10min": 0,
            "10-20min": 0,
            ">20min": 0,
        }
        result = compute_short_suma_freq_penalty(bins, config=TestConfig)
        assert result == 0.0

    def test_empty_bins_dict_gives_zero(self):
        """Missing keys default to 0 gracefully."""
        result = compute_short_suma_freq_penalty({}, config=TestConfig)
        assert result == 0.0

    def test_result_always_in_zero_one_range(self):
        """Exhaustive: penalty is always in [0.0, 1.0]."""
        for n in range(0, 101):
            bins = {"1.5-5s": n, "5-10s": 0}
            result = compute_short_suma_freq_penalty(bins, config=TestConfig)
            assert 0.0 <= result <= 1.0


# ── 3. Asymmetry index ────────────────────────────────────────────────

class TestAsymmetryIndex:

    def test_symmetric_gives_zero(self):
        ai = compute_asymmetry_index(10.0, 10.0)
        assert ai == pytest.approx(0.0)

    def test_all_right_gives_plus_one(self):
        ai = compute_asymmetry_index(0.0, 10.0)
        # denom = 10, (10-0)/10 = +1.0
        assert ai == pytest.approx(1.0)

    def test_all_left_gives_minus_one(self):
        ai = compute_asymmetry_index(10.0, 0.0)
        # (0-10)/10 = -1.0
        assert ai == pytest.approx(-1.0)

    def test_both_zero_gives_zero(self):
        """Both sides at zero → symmetric by convention."""
        ai = compute_asymmetry_index(0.0, 0.0)
        assert ai == 0.0

    def test_nan_left_gives_nan(self):
        ai = compute_asymmetry_index(np.nan, 10.0)
        assert np.isnan(ai)

    def test_nan_right_gives_nan(self):
        ai = compute_asymmetry_index(10.0, np.nan)
        assert np.isnan(ai)

    def test_bounded_minus_one_to_plus_one(self):
        """Exhaustive: AI is always in [-1, +1] for any non-negative inputs."""
        vals = np.linspace(0.0, 50.0, 51)
        for l in vals:
            for r in vals:
                ai = compute_asymmetry_index(l, r)
                if not np.isnan(ai):
                    assert -1.0 <= ai <= 1.0

    def test_signed_formula_asymmetry(self):
        """Right > Left → positive AI. Left > Right → negative AI."""
        ai_r = compute_asymmetry_index(5.0, 15.0)  # right dominant
        ai_l = compute_asymmetry_index(15.0, 5.0)  # left dominant
        assert ai_r > 0
        assert ai_l < 0
        # Must be equal magnitude (symmetric formula)
        assert abs(ai_r) == pytest.approx(abs(ai_l))


# ── 4. Composite score (fusion + penalty + clamp) ─────────────────────

class TestCompositeScore:

    def test_worst_side_drives(self):
        """min(left, right) is used as the base — the higher-loaded side wins."""
        result = compute_composite_score(2.0, 0.5, 0.0, config=TestConfig)
        assert result["composite_score"] == pytest.approx(0.5)  # right was worse
        assert result["worst_side"] == "right"

    def test_worst_side_drives_left(self):
        result = compute_composite_score(0.3, 2.0, 0.0, config=TestConfig)
        assert result["composite_score"] == pytest.approx(0.3)
        assert result["worst_side"] == "left"

    def test_tied_sides(self):
        result = compute_composite_score(1.0, 1.0, 0.0, config=TestConfig)
        assert result["worst_side"] == "tied"
        assert result["composite_score"] == pytest.approx(1.0)

    def test_asymmetry_penalty_applied_when_above_threshold(self):
        """AI = 0.8 > 0.5 threshold → penalty subtracted; result still positive.

        Uses max valid per-side score (+3.0 = EIndex max +2 + freq max +1).
        The penalty must be small enough that max_score - penalty > 0 —
        if it's not, that's the kill-switch bug (see config.py comment).
        """
        max_score = 3.0  # EIndex +2.0 + freq penalty +1.0
        ai_above_threshold = 0.8
        result = compute_composite_score(max_score, max_score, ai_above_threshold, config=TestConfig)
        assert result["asymmetry_penalty_applied"] is True
        assert result["asymmetry_penalty_amount"] == pytest.approx(TestConfig.ASYMMETRY_PENALTY_POINTS)
        # Must be positive — kill-switch check
        assert result["composite_score"] > 0.0
        assert result["composite_score"] == pytest.approx(max_score - TestConfig.ASYMMETRY_PENALTY_POINTS)

    def test_asymmetry_penalty_not_applied_when_below_threshold(self):
        """AI = 0.3 < 0.5 threshold → no penalty."""
        result = compute_composite_score(2.0, 2.0, 0.3, config=TestConfig)
        assert result["asymmetry_penalty_applied"] is False
        assert result["composite_score"] == pytest.approx(2.0)

    def test_floor_clamp_prevents_negative(self):
        """If penalty drives score negative, floor clamp brings it to 0.0.

        Uses a score just below the penalty threshold so the subtraction
        produces a negative pre-clamp value.
        """
        # score = 0.1, penalty = 0.25 → pre-clamp = -0.15 → clamped to 0.0
        tiny_score = 0.1
        result = compute_composite_score(tiny_score, tiny_score, 0.9, config=TestConfig)
        assert result["composite_score"] == 0.0  # clamped
        assert result["pre_clamp_score"] < 0.0   # was negative before clamp

    def test_floor_clamp_is_last(self):
        """Verify: penalty is applied first, THEN clamp (order must be preserved).

        score = 0.1, penalty = 0.25 → pre-clamp = -0.15 → clamped = 0.0
        """
        score = 0.1
        result = compute_composite_score(score, score, 0.9, config=TestConfig)
        assert result["pre_clamp_score"] == pytest.approx(score - TestConfig.ASYMMETRY_PENALTY_POINTS)
        assert result["composite_score"] == 0.0

    def test_nan_ai_skips_penalty(self):
        """NaN asymmetry index → no penalty applied."""
        result = compute_composite_score(1.5, 1.5, np.nan, config=TestConfig)
        assert result["asymmetry_penalty_applied"] is False
        assert result["composite_score"] == pytest.approx(1.5)

    def test_asymmetry_penalty_proportional_to_scale(self):
        """Proportionality guard: the penalty must not be a kill switch.

        At MAXIMUM valid per-side score (+3.0 = EIndex +2 + freq penalty +1),
        triggering the asymmetry penalty must still leave a positive composite.
        If this fails, the penalty has been rescaled for a different output range
        (e.g., a 0-100 STAMI-mapped scale) but not updated for the current raw
        [-2, +3] units — the exact bug this test was written to catch.

        Also asserts the penalty is less than 50% of the full output range,
        so the penalty is always proportionate rather than dominating.
        """
        max_per_side_score = 2.0 + 1.0   # EIndex max + freq penalty max
        min_per_side_score = -2.0         # EIndex min + 0 freq penalty
        output_range = max_per_side_score - min_per_side_score  # 5.0 units

        ai_triggered = 0.8  # |AI| > ASYMMETRY_PENALTY_THRESHOLD (0.5)

        result = compute_composite_score(
            max_per_side_score, max_per_side_score, ai_triggered, config=TestConfig
        )

        # Core kill-switch check: at max score, penalty must NOT zero the composite
        assert result["composite_score"] > 0.0, (
            f"Asymmetry penalty ({TestConfig.ASYMMETRY_PENALTY_POINTS:.3f}) is a kill switch: "
            f"at max per-side score ({max_per_side_score}), composite was zero-clamped "
            f"(pre-clamp={result['pre_clamp_score']:.3f}). "
            f"Penalty must be < max_per_side_score. "
            f"If the score scale changed (e.g., STAMI 0-100 mapping applied), "
            f"update ASYMMETRY_PENALTY_POINTS in config.py to match."
        )

        # Proportionality check: penalty < 50% of output range
        assert TestConfig.ASYMMETRY_PENALTY_POINTS < output_range * 0.5, (
            f"Penalty ({TestConfig.ASYMMETRY_PENALTY_POINTS}) exceeds 50% of the "
            f"{output_range}-unit output range — disproportionate to the scale."
        )

        # Exact value check: penalty = 5% of output range (design intent)
        expected_penalty = output_range * 0.05
        assert TestConfig.ASYMMETRY_PENALTY_POINTS == pytest.approx(expected_penalty, abs=1e-9), (
            f"Penalty ({TestConfig.ASYMMETRY_PENALTY_POINTS}) is no longer 5% of the "
            f"output range ({expected_penalty:.3f}). Either the range or the penalty "
            f"changed without the other being updated."
        )

    def test_output_schema_complete(self):
        """All expected output keys are present."""
        result = compute_composite_score(1.0, 2.0, 0.2, config=TestConfig)
        required_keys = {
            "composite_score", "score_left", "score_right",
            "worst_side", "asymmetry_index", "asymmetry_penalty_applied",
            "asymmetry_penalty_amount", "pre_clamp_score", "components",
        }
        assert required_keys.issubset(result.keys())


# ── 5. EIndexAccumulator ──────────────────────────────────────────────

class TestEIndexAccumulator:

    def test_initial_state(self):
        """Both outputs start at 0.0 before any data."""
        acc = EIndexAccumulator(config=TestConfig)
        assert acc.eindex_live == 0.0
        assert acc.eindex_session_cumulative == 0.0
        assert acc.completed_windows == 0

    def test_update_live_replaces_not_accumulates(self):
        """update_live() REPLACES eindex_live — never adds to it."""
        acc = EIndexAccumulator(config=TestConfig)

        # First update: moderate activity (all low-load)
        w1 = np.full(3000, 3.5)  # 100% low → EIndex = +1.0
        acc.update_live(w1)
        assert acc.eindex_live == pytest.approx(1.0)

        # Second update: all rest
        w2 = np.zeros(3000)  # 100% rest → EIndex = -2.0
        acc.update_live(w2)
        # Must be -2.0, not -2.0 + 1.0 = -1.0
        assert acc.eindex_live == pytest.approx(-2.0)

    def test_update_live_does_not_change_cumulative(self):
        """update_live() never affects eindex_session_cumulative."""
        acc = EIndexAccumulator(config=TestConfig)
        for _ in range(100):
            acc.update_live(np.full(1000, 10.0))
        assert acc.eindex_session_cumulative == 0.0

    def test_submit_window_increments_cumulative(self):
        """submit_completed_window() increments cumulative by one window's value."""
        acc = EIndexAccumulator(config=TestConfig)
        w = np.full(3000, 10.0)  # all high → EIndex = +2.0 per window
        acc.submit_completed_window(w)
        assert acc.eindex_session_cumulative == pytest.approx(2.0)
        assert acc.completed_windows == 1

    def test_submit_two_windows_adds_correctly(self):
        """Two discrete windows sum correctly."""
        acc = EIndexAccumulator(config=TestConfig)
        w_high = np.full(3000, 10.0)  # +2.0
        w_rest = np.zeros(3000)        # -2.0
        acc.submit_completed_window(w_high)
        acc.submit_completed_window(w_rest)
        # cumulative = +2.0 + (-2.0) = 0.0
        assert acc.eindex_session_cumulative == pytest.approx(0.0)
        assert acc.completed_windows == 2

    def test_live_and_cumulative_are_independent(self):
        """Mixing live updates and window submissions: the two never interfere."""
        acc = EIndexAccumulator(config=TestConfig)
        high = np.full(3000, 10.0)
        rest = np.zeros(3000)

        # Live: update many times
        for _ in range(50):
            acc.update_live(rest)  # sets live to -2.0 each time

        # Cumulative: submit one high window
        acc.submit_completed_window(high)

        assert acc.eindex_live == pytest.approx(-2.0)
        assert acc.eindex_session_cumulative == pytest.approx(2.0)


# ── 6. REGRESSION: 8-hour session simulation ─────────────────────────

class TestEightHourRegression:
    """
    Regression test: simulate a full 8-hour session at constant moderate
    activity and verify:
        1. eindex_live stays within [-2, +2] at EVERY update (never diverges)
        2. eindex_session_cumulative ONLY changes at discrete 10-min
           boundaries (flat between boundaries)

    This test was specified to catch the verified unbounded-divergence bug:
    under constant moderate activity, the old implementation reached ~5760
    by session end. The fix: eindex_live is a direct current-window value,
    never summed across updates.

    Session parameters:
        Duration: 8 hours = 480 minutes
        Activity: constant moderate load (~40% in low bucket, 60% in high)
        Live update interval: every 5 seconds
        10-min window: 600 seconds of RMS samples
        RMS step: 20ms → 50 samples/second of RMS output
    """

    # Simulation constants — do not change without updating comments above
    SESSION_DURATION_S = 8 * 3600       # 8 hours in seconds
    LIVE_UPDATE_INTERVAL_S = 5          # Seconds between live updates
    WINDOW_DURATION_S = 10 * 60         # 10 minutes in seconds
    RMS_SAMPLES_PER_SECOND = 50         # 1000ms / 20ms step

    def _make_moderate_window(self, duration_s):
        """Generate moderate-activity %RVE samples: 40% low, 60% high."""
        n_samples = int(duration_s * self.RMS_SAMPLES_PER_SECOND)
        n_low  = int(n_samples * 0.4)
        n_high = n_samples - n_low
        arr = np.concatenate([
            np.full(n_low,  3.5),   # low bucket [0.5%, 7%)
            np.full(n_high, 10.0),  # high bucket >= 7%
        ])
        # Expected EIndex = -2*0 + 1*0.4 + 2*0.6 = 0.4 + 1.2 = 1.6
        return arr

    def test_eindex_live_bounded_throughout(self):
        """
        eindex_live must stay in [-2, +2] at every single live update
        over an 8-hour session.
        """
        acc = EIndexAccumulator(config=TestConfig)
        window_10min = self._make_moderate_window(self.WINDOW_DURATION_S)

        n_updates = self.SESSION_DURATION_S // self.LIVE_UPDATE_INTERVAL_S
        violated = False

        for _ in range(n_updates):
            acc.update_live(window_10min)
            if not (-2.0 <= acc.eindex_live <= 2.0):
                violated = True
                break

        assert not violated, (
            f"eindex_live exceeded [-2, +2]: last value = {acc.eindex_live:.4f}. "
            "The live EIndex must REPLACE its value on each update, not accumulate."
        )

    def test_eindex_live_correct_value(self):
        """
        At constant 40%/60% low/high, eindex_live should be ~+1.6.
        (Not the unbounded diverging number the old implementation produced.)
        """
        acc = EIndexAccumulator(config=TestConfig)
        window = self._make_moderate_window(self.WINDOW_DURATION_S)
        acc.update_live(window)
        assert acc.eindex_live == pytest.approx(1.6, abs=0.01)

    def test_cumulative_only_steps_at_boundaries(self):
        """
        eindex_session_cumulative must be FLAT between 10-min boundaries
        and only increment when submit_completed_window() is called.

        Simulates 480 live updates (at 5s intervals = 40 min total)
        with 4 discrete window submissions at 10-min marks.
        Checks cumulative value is flat between submissions and
        steps exactly at submission points.
        """
        acc = EIndexAccumulator(config=TestConfig)
        window_10min = self._make_moderate_window(self.WINDOW_DURATION_S)

        # Simulate 40 minutes: live updates every 5s, window at 10/20/30/40 min
        updates_per_window = self.WINDOW_DURATION_S // self.LIVE_UPDATE_INTERVAL_S  # 120

        cumulative_values = []
        submission_indices = []  # which update number triggered each step

        for window_num in range(4):  # 4 discrete 10-min windows
            for i in range(updates_per_window):
                acc.update_live(window_10min)
                cumulative_values.append(acc.eindex_session_cumulative)

            # Submit the completed window at the 10-min boundary
            before = acc.eindex_session_cumulative
            acc.submit_completed_window(window_10min)
            after = acc.eindex_session_cumulative
            submission_indices.append(len(cumulative_values) - 1)
            cumulative_values.append(acc.eindex_session_cumulative)

            # Must have stepped up at this boundary
            assert after > before, (
                f"Window {window_num+1}: cumulative did not increase at boundary "
                f"(before={before:.4f}, after={after:.4f})"
            )

        # Verify the cumulative was flat WITHIN each window period
        # i.e., all values within a window block are equal (no drift)
        start_idx = 0
        for window_num in range(4):
            block = cumulative_values[start_idx:start_idx + updates_per_window]
            assert all(v == block[0] for v in block), (
                f"Window {window_num+1}: cumulative was NOT flat between boundaries. "
                f"Values changed during {updates_per_window} live updates."
            )
            start_idx += updates_per_window + 1  # +1 for the submission append

    def test_cumulative_does_not_diverge(self):
        """
        After 48 discrete 10-min windows (8 hours), cumulative should be
        exactly 48 × per-window-EIndex — a finite, predictable number.
        NOT the ~5760 the old implementation would have produced.
        """
        acc = EIndexAccumulator(config=TestConfig)
        window = self._make_moderate_window(self.WINDOW_DURATION_S)

        n_windows = self.SESSION_DURATION_S // self.WINDOW_DURATION_S  # 48

        for _ in range(n_windows):
            acc.submit_completed_window(window)

        # Expected: 48 × 1.6 = 76.8
        expected = 48 * 1.6
        assert acc.eindex_session_cumulative == pytest.approx(expected, abs=0.1)
        assert acc.completed_windows == n_windows

        # Must NOT be anywhere near 5760 (the old buggy value)
        assert acc.eindex_session_cumulative < 200, (
            f"Cumulative diverged to {acc.eindex_session_cumulative:.1f} — "
            "this looks like the old unbounded-summation bug."
        )


# ── 7. STAMI display-scale mapping ──────────────────────────────────────

class TestStamiMapping:

    def test_stami_mapping_endpoints(self):
        """
        map_eindex_to_display_scale maps eindex_session_cumulative to [0, 100]
        anchored to STAMI S1_Dataset.sav 5th (-3.854) and 95th (49.635) percentiles.
        """
        assert map_eindex_to_display_scale(-3.854) == pytest.approx(0.0, abs=1e-3)
        assert map_eindex_to_display_scale(49.635) == pytest.approx(100.0, abs=1e-3)
        assert map_eindex_to_display_scale((49.635 + (-3.854)) / 2.0) == pytest.approx(50.0, abs=1e-3)

    def test_stami_mapping_clipping_and_edge_cases(self):
        """
        Values outside [EI_FLOOR, EI_CEIL] must be cleanly clipped to 0.0 and 100.0.
        Equal floor and ceil must return 0.0 without zero division.
        """
        assert map_eindex_to_display_scale(-50.0) == 0.0
        assert map_eindex_to_display_scale(150.0) == 100.0
        assert map_eindex_to_display_scale(25.0, floor=10.0, ceil=10.0) == 0.0

    def test_structural_guard_no_live_or_composite_mapping(self):
        """
        Structural guard against scale mismatch bugs: map_eindex_to_display_scale()
        is strictly scoped to eindex_session_cumulative (full-workday cumulative value).
        It must NEVER be called on eindex_live, per-side scores, or composite_score
        anywhere in the codebase.
        """
        import os

        chordspy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        forbidden_keywords = ['live', 'composite', 'per_side', 'score_l', 'score_r']

        for root, dirs, files in os.walk(chordspy_root):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for idx, line in enumerate(lines, 1):
                        if 'map_eindex_to_display_scale(' in line and 'def map_eindex_to_display_scale' not in line:
                            line_lower = line.lower()
                            for bad_kw in forbidden_keywords:
                                assert bad_kw not in line_lower, (
                                    f"Scale violation in {file}:{idx}: map_eindex_to_display_scale() "
                                    f"was called with forbidden argument/keyword '{bad_kw}'. "
                                    f"STAMI display mapping applies exclusively to eindex_session_cumulative."
                                )
