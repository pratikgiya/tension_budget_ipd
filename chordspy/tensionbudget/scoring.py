"""
TensionBudget — Phase 6: Composite Tension Budget Scoring Engine.

Implements the per-side exposure index, short-SUMA frequency penalty,
bilateral fusion, and asymmetry penalty to produce a composite score.

Math sources (all cited, no guesses):
    EIndex base formula:  Koch et al. (2024) — Exposure Index
    Short-SUMA frequency: Koch et al. (2025) — short-event frequency
                          predicts pain; this per-window count-based
                          formula is the project's novel real-time
                          extension (design choice, not paper-supplied).
    Asymmetry index:      Project design choice (signed laterality index),
                          locked to the tested version — do not switch
                          to the unsigned variant.
    Worst-side fusion:    Project design choice, documented rationale below.
    Asymmetry penalty:    Project design choice, threshold from TBConfig.

All tuneable constants are in TBConfig — no magic numbers here.

Pipeline order (MUST be preserved):
    1. Per-side: compute EIndex (base + frequency term)
    2. Per-side: independent score
    3. Fusion:   Composite = max(Score_L, Score_R)  [worst-side-drives, higher=worse]
    4. Penalty:  add ASYMMETRY_PENALTY_POINTS if |AI| > ASYMMETRY_PENALTY_THRESHOLD
    5. Return unclamped value as the analytical / ML-training composite_score.
       (range approx [-2.0, +3.25] after fixes)
    6. Display only: map_composite_to_display_scale() clamps to [0.0, 3.25] — UI only,
       NEVER written back to epoch_features or used as training target.

EIndex split (fixes unbounded-growth bug, verified via simulation):
    eindex_live:
        - Computed on a SLIDING 10-min window, refreshed every few seconds.
        - Uses the current window value DIRECTLY — never summed across
          recomputes. Bounded to [-2, +2] by construction at all times.
        - Drives the real-time composite score.

    eindex_session_cumulative:
        - Computed on DISCRETE, NON-OVERLAPPING 10-min windows (matches
          Koch's exact published method).
        - Summed only once per completed 10-min boundary.
        - Intentionally flat between boundaries — this is a session
          report-card statistic, not a live signal.
        - Used for end-of-session reporting and literature comparison only.

STAMI mapping (eindex_session_cumulative only):
    The EIndex→0-100 display scale is anchored against the STAMI S1_Dataset.sav
    (SPSS format, Section 5.2 of the project's dataset review). This mapping applies
    EXCLUSIVELY to eindex_session_cumulative (full-workday exposure accumulation).
    Do NOT apply to eindex_live, per-side scores, or composite_score.
"""

import numpy as np

from chordspy.tensionbudget.config import TBConfig


# ── 1. Per-side EIndex computation ────────────────────────────────────

def compute_eindex_for_window(normalized_rms_window, config=None):
    """
    Compute the Koch (2024) Exposure Index for a single time window.

    Formula:
        EIndex = -2*P_Rest + P_Low + 2*P_High

    Where proportions are fractions of all RMS samples in the window:
        P_Rest: proportion of samples < EINDEX_REST_THRESHOLD_PCT  (< 0.5% MVE)
        P_Low:  proportion of samples in [EINDEX_REST_THRESHOLD_PCT, EINDEX_HIGH_THRESHOLD_PCT)
        P_High: proportion of samples >= EINDEX_HIGH_THRESHOLD_PCT  (>= 7% MVE)

    Result is bounded to [-2, +2] by construction because:
        P_Rest + P_Low + P_High = 1.0  (exhaustive partition)
        min: all rest  → -2*1 + 0 + 0 = -2
        max: all high  →  0 + 0 + 2*1 = +2

    This is the Koch (2024) exact formula. The two distinct rest
    thresholds in this project (3.0% for gap detection, 0.5% for EIndex)
    must NOT be merged — they come from different papers and serve
    different purposes.

    Args:
        normalized_rms_window: 1-D numpy array of %RVE values for the
            window. Should be the RMS envelope, not raw signal.
        config: TBConfig class.

    Returns:
        float: EIndex in [-2, +2].
    """
    cfg = config or TBConfig
    arr = np.asarray(normalized_rms_window, dtype=float)

    if arr.size == 0:
        return 0.0

    n = float(arr.size)
    p_rest = float(np.sum(arr < cfg.EINDEX_REST_THRESHOLD_PCT)) / n
    p_high = float(np.sum(arr >= cfg.EINDEX_HIGH_THRESHOLD_PCT)) / n
    p_low = 1.0 - p_rest - p_high

    return -2.0 * p_rest + p_low + 2.0 * p_high


def compute_short_suma_freq_penalty(suma_bins, config=None):
    """
    Compute the short-SUMA frequency penalty term.

    Novel extension of the Koch (2025) finding that short-event
    frequency (not total duration) predicts neck/shoulder pain, especially
    in women. This is a project design choice for real-time monitoring —
    not a formula supplied by Koch 2025 directly.

    Formula:
        short_event_count = count of events in (1.5–5s bin) + (5–10s bin)
        short_suma_freq_penalty = min(short_event_count * 0.1, 1.0)

    Interpretation:
        - Each short event contributes 0.1 penalty points.
        - 10 or more short events in the scoring window → full penalty of 1.0.
        - Cap is per-window — NEVER accumulated across windows.
        - Bounded [0.0, 1.0] by construction.

    Args:
        suma_bins: Dict mapping bin label to event count, as returned by
            analytics.detect_sustained_activity().
            Expected keys: "1.5-5s", "5-10s", ...
        config: TBConfig class (unused, reserved for future cap override).

    Returns:
        float: Short-SUMA frequency penalty in [0.0, 1.0].
    """
    cfg = config or TBConfig
    short_count = suma_bins.get("1.5-5s", 0) + suma_bins.get("5-10s", 0)
    return min(short_count * cfg.SHORT_SUMA_PENALTY_PER_EVENT, 1.0)


def compute_per_side_score(eindex_live, short_suma_penalty, config=None):
    """
    Combine EIndex and short-SUMA frequency penalty into a per-side score.

    The two terms compound intentionally (documented design decision):
        - EIndex measures the PROPORTION of time under load — amplitude domain.
        - Short-SUMA penalty measures the FRAGMENTATION PATTERN — event-frequency
          domain. A muscle that alternates rapid short bursts produces the same
          EIndex as one sustaining the same load continuously, but different risk.
        - Koch (2025) validates these as orthogonal risk factors, justifying
          additive compounding rather than averaging.

    Formula:
        per_side_score = eindex_live + short_suma_freq_penalty

    Range: [-2.0, +3.0] (EIndex in [-2,+2], penalty in [0,+1]).
    This is an intermediate score — the bilateral fusion and floor clamp
    are applied AFTER this step in compute_composite_score().

    Args:
        eindex_live: float in [-2, +2] from compute_eindex_for_window().
        short_suma_penalty: float in [0, 1] from compute_short_suma_freq_penalty().
        config: TBConfig class (reserved for future weighting).

    Returns:
        float: Per-side score in [-2.0, +3.0].
    """
    return eindex_live + short_suma_penalty


# ── 2. Asymmetry index ────────────────────────────────────────────────

def compute_asymmetry_index(apdf50_left, apdf50_right):
    """
    Compute the signed Laterality Index from Active APDF-50 values.

    Formula (locked — do not switch to the unsigned variant):
        AI = (APDF50_R - APDF50_L) / (APDF50_R + APDF50_L)

    Range: [-1, +1]
        -1.0 = fully left dominant
         0.0 = perfectly symmetric
        +1.0 = fully right dominant

    This is a project design choice — no direct literature citation.
    Uses Active APDF-50 (samples >= 3% RVE), not Session APDF-50,
    because active periods are what drive musculoskeletal load.

    The asymmetry penalty threshold in TBConfig.ASYMMETRY_PENALTY_THRESHOLD
    was calibrated against this specific signed formula. Switching to the
    unsigned variant would silently invalidate that threshold.

    Args:
        apdf50_left: Median %RVE of left trapezius (active periods only).
        apdf50_right: Median %RVE of right trapezius (active periods only).

    Returns:
        float: Laterality Index in [-1.0, +1.0], or np.nan if either
            input is NaN (e.g. one side was fully resting).
    """
    if np.isnan(apdf50_left) or np.isnan(apdf50_right):
        return np.nan

    denom = apdf50_left + apdf50_right
    if denom == 0.0:
        return 0.0  # Both sides at zero — symmetric by convention

    return float((apdf50_right - apdf50_left) / denom)


# ── 3. Bilateral fusion ───────────────────────────────────────────────

def compute_composite_score(
    score_left,
    score_right,
    asymmetry_index,
    config=None
):
    """
    Fuse per-side scores into the final composite Tension Budget score.

    Score convention: HIGHER = WORSE (more exposure/risk).
        Per-side scores from compute_per_side_score() range [-2.0, +3.0]
        where -2.0 = fully resting, +3.0 = maximum load + maximum SUMA penalty.

    Fusion rule: WORST-SIDE-DRIVES (project design choice).
        Rationale: The trapezius works as a bilateral muscle group under
        shared neural control. In asymmetric desk work (mousing posture,
        monitor offset, etc.), one side typically accumulates load faster.
        Using the MAXIMUM score (= the side with higher exposure/risk)
        ensures the composite reflects the worst-off side, not an
        averaged-away signal. A user overloading their right side while
        the left rests will see a composite driven by the right.

    Pipeline order (MUST be followed exactly):
        1. base = max(score_left, score_right)    [worst-side-drives, higher=worse]
        2. If |AI| > ASYMMETRY_PENALTY_THRESHOLD:
               base += ASYMMETRY_PENALTY_POINTS   [spinal-torque risk added on top]
        3. Return base UNCLAMPED as composite_score (analytical / ML-training value).
           Range: approx [-2.0, +3.25] after both fixes.

    Floor clamp — analytical vs display split:
        composite_score is returned unclamped so the full [-2.0, +3.25] range
        is available for ML training. Clamping the negative (resting) range to
        0.0 would flatten "deeply resting" and "borderline active" into one value,
        destroying variance the regression model needs.

        For GUI display use map_composite_to_display_scale() which clamps to
        [0.0, 3.25]. That function MUST NOT be used to produce values written
        back into epoch_features, local_logger, or cloud ingestion paths.

    Compounding rationale (documented intentional design):
        - Worst-side-drives captures amplitude-domain load asymmetry.
        - Asymmetry penalty captures the spinal torque / uneven mechanical
          load risk that EXISTS IN ADDITION to per-side load — a user with
          symmetric high load has a different risk profile than one with
          identical worst-side load but severe lateral imbalance.

    Args:
        score_left: Per-side score for left trapezius (from compute_per_side_score).
        score_right: Per-side score for right trapezius.
        asymmetry_index: Signed AI from compute_asymmetry_index(), or np.nan.
        config: TBConfig class.

    Returns:
        dict with keys:
            'composite_score': float, UNCLAMPED, range approx [-2.0, +3.25].
                This is the analytical / ML-training value. Do NOT clamp before
                writing to epoch_features. For display, use
                map_composite_to_display_scale() separately.
            'score_left': float (input, unchanged)
            'score_right': float (input, unchanged)
            'worst_side': 'left' | 'right' | 'tied'
            'asymmetry_index': float or None
            'asymmetry_penalty_applied': bool
            'asymmetry_penalty_amount': float
            'components': dict (breakdown for explainability)
    """
    cfg = config or TBConfig

    # Step 1: worst-side fusion — higher score = worse, so use max()
    if score_left >= score_right:
        base = score_left
        worst_side = "left"
    else:
        base = score_right
        worst_side = "right"

    if score_left == score_right:
        worst_side = "tied"

    # Step 2: asymmetry penalty — ADD on top (higher = worse direction)
    penalty_applied = False
    penalty_amount = 0.0

    if not np.isnan(asymmetry_index):
        if abs(asymmetry_index) > cfg.ASYMMETRY_PENALTY_THRESHOLD:
            penalty_amount = cfg.ASYMMETRY_PENALTY_POINTS
            base += penalty_amount
            penalty_applied = True

    # Step 3: return UNCLAMPED (analytical / ML-training value)
    # Use map_composite_to_display_scale() for GUI display.
    composite = base

    return {
        "composite_score": float(composite),  # UNCLAMPED — do not clamp before logging
        "score_left": float(score_left),
        "score_right": float(score_right),
        "worst_side": worst_side,
        "asymmetry_index": float(asymmetry_index) if not np.isnan(asymmetry_index) else None,
        "asymmetry_penalty_applied": penalty_applied,
        "asymmetry_penalty_amount": float(penalty_amount),
        "components": {
            "base_worst_side_score": float(score_left if worst_side in ("left", "tied") else score_right),
            "asymmetry_penalty": float(penalty_amount),
        },
    }


# ── 4. EIndex session accumulator ────────────────────────────────────

class EIndexAccumulator:
    """
    Tracks the two EIndex outputs correctly across a session.

    eindex_live:
        Sliding 10-min window, value used DIRECTLY each time it is computed.
        NEVER summed across recomputes. Bounded [-2, +2] by construction.
        Refreshed on every call to update_live().

    eindex_session_cumulative:
        Discrete, non-overlapping 10-min windows (Koch 2024 exact method).
        Incremented ONCE per completed 10-min boundary — never between.
        Grows monotonically over the session but deterministically so
        (N completed windows × per-window EIndex, not a re-summing error).

    The split fixes the verified unbounded-growth bug: if EIndex were
    literally summed every few seconds on a sliding window, the same
    activity would be double/triple/hundred-counted across overlapping
    windows. Simulation confirms this diverges to ~5760 by the end of an
    8-hour session of constant moderate activity. The split prevents this.

    Usage:
        acc = EIndexAccumulator(config=TBConfig)
        # Called periodically (e.g. every 2 seconds):
        acc.update_live(normalized_rms_last_10min)
        live_val = acc.eindex_live  # Use current value directly

        # Called automatically when a full 10-min window of raw RMS is passed:
        acc.submit_completed_window(normalized_rms_10min_window)
        cumulative_val = acc.eindex_session_cumulative  # Steps once here
    """

    def __init__(self, config=None):
        self._cfg = config or TBConfig
        self.eindex_live = 0.0
        self.eindex_session_cumulative = 0.0
        self._completed_window_count = 0

    def update_live(self, normalized_rms_sliding_window):
        """
        Refresh eindex_live from the current sliding 10-min window.

        This REPLACES the stored value — does NOT add to it.
        Safe to call every few seconds.

        Args:
            normalized_rms_sliding_window: 1-D numpy array of %RVE values
                from the last LIVE_BUFFER_MINUTES (default 10 min) of
                RMS envelope samples.
        """
        self.eindex_live = compute_eindex_for_window(
            normalized_rms_sliding_window, config=self._cfg
        )

    def submit_completed_window(self, normalized_rms_10min_window):
        """
        Record one completed discrete 10-min window into the cumulative total.

        Call this ONCE when a full, non-overlapping 10-min block finishes.
        Must NOT be called more frequently — doing so would re-introduce
        the double-counting bug this split was designed to prevent.

        The cumulative value increments deterministically:
            eindex_session_cumulative = sum of per-window EIndex values
            for N completed windows, one increment per boundary.

        Args:
            normalized_rms_10min_window: 1-D numpy array of %RVE values
                for the just-completed, non-overlapping 10-min window.
        """
        window_eindex = compute_eindex_for_window(
            normalized_rms_10min_window, config=self._cfg
        )
        self.eindex_session_cumulative += window_eindex
        self._completed_window_count += 1

    @property
    def completed_windows(self):
        """Number of discrete 10-min windows submitted so far."""
        return self._completed_window_count


# ── 5. Display-scale mappings ─────────────────────────────────────────

def map_eindex_to_display_scale(cumulative_eindex, floor=None, ceil=None, config=None):
    """
    Maps eindex_session_cumulative onto a [0,100] display scale, anchored
    to the 5th/95th percentile of Koch et al. 2024's 731-subject STAMI
    pooled dataset (S1_Dataset.sav).

    IMPORTANT — labeling requirement: this is a population REFERENCE for
    display calibration only. It is NOT a validation of TensionBudget's
    own scoring methodology (EIndex + short-SUMA frequency term + fusion +
    asymmetry penalty), which has no equivalent in the STAMI dataset.
    Actual methodology validation comes from this project's own pilot
    study (self-report correlation), not from this mapping.

    Only valid for eindex_session_cumulative. Do not apply to eindex_live,
    per-side scores, or composite_score — those are point-in-time
    snapshots, not comparable to STAMI's full-workday cumulative values.
    """
    cfg = config or TBConfig
    if floor is None:
        floor = cfg.EI_FLOOR
    if ceil is None:
        ceil = cfg.EI_CEIL

    if ceil == floor:
        return 0.0
    normalized = ((cumulative_eindex - floor) / (ceil - floor)) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))


def map_composite_to_display_scale(composite_score):
    """
    Clamp the analytical composite_score to a GUI-safe display range.

    DISPLAY-ONLY. This function exists solely to drive gauge widgets and
    coaching banners in the GUI. It MUST NOT be used to produce values
    written back into epoch_features, local_logger, cloud_ingest, or any
    path that stores data or trains the ML model.

    The analytical composite_score returned by compute_composite_score() is
    intentionally unclamped (range approx [-2.0, +3.25]) to preserve full
    variance for the Bayesian/ordinal regression model. Clamping the
    negative (resting) range to 0.0 in storage would flatten "deeply
    resting" and "borderline active" into one value, destroying exactly the
    resolution the ML target needs.

    Display mapping:
        composite_score < 0.0  → display 0.0  (negative scores are valid
                                                analytically but visually
                                                displayed as "no detected load")
        composite_score in [0.0, 3.25] → unchanged
        composite_score > 3.25 → display 3.25 (theoretical maximum;
                                                 EIndex max +2.0 + freq max
                                                 +1.0 + asymmetry +0.25)

    Structural guard: the test_structural_guard_no_composite_display_leak()
    test in test_phase6_scoring.py asserts that this function is never
    called from local_logger.py, cloud_ingest.py, or any path that assigns
    its output to composite_score before storage.

    Args:
        composite_score: float — raw unclamped analytical output from
            compute_composite_score()['composite_score'].

    Returns:
        float in [0.0, 3.25] — safe for GUI gauge/banner display only.
    """
    DISPLAY_FLOOR = 0.0
    DISPLAY_CEIL = 3.25  # EIndex max(2.0) + freq max(1.0) + asymmetry penalty(0.25)
    return float(np.clip(composite_score, DISPLAY_FLOOR, DISPLAY_CEIL))


# ── 6. Full scoring pass (convenience wrapper) ────────────────────────

def score_bilateral_window(
    rms_left,
    rms_right,
    suma_bins_left,
    suma_bins_right,
    active_apdf50_left,
    active_apdf50_right,
    eindex_accumulator_left,
    eindex_accumulator_right,
    config=None
):
    """
    Run a complete scoring pass for one bilateral window.

    This is the convenience entry point that wires together all Phase 6
    sub-functions in the correct order. For live use, call this every
    LIVE_UPDATE_INTERVAL_S seconds with the last LIVE_BUFFER_MINUTES of
    RMS data.

    Args:
        rms_left: 1-D numpy array of %RVE values for left trapezius
            (last 10-min sliding window).
        rms_right: 1-D numpy array of %RVE values for right trapezius.
        suma_bins_left: Dict from analytics.detect_sustained_activity()
            for left channel. Used for short-SUMA frequency penalty.
        suma_bins_right: Same for right channel.
        active_apdf50_left: Median Active APDF for left channel (float).
            Pass np.nan if not yet computed (e.g. fully resting buffer).
        active_apdf50_right: Same for right channel.
        eindex_accumulator_left: EIndexAccumulator instance for left side.
            Its eindex_live is updated in-place by this call.
        eindex_accumulator_right: EIndexAccumulator instance for right side.
        config: TBConfig class.

    Returns:
        dict with full scoring output:
            'eindex_live_left': float in [-2, +2]
            'eindex_live_right': float in [-2, +2]
            'eindex_session_cumulative_left': float (sum of completed windows)
            'eindex_session_cumulative_right': float
            'short_suma_penalty_left': float in [0, 1]
            'short_suma_penalty_right': float in [0, 1]
            'score_left': float (per-side combined)
            'score_right': float (per-side combined)
            'asymmetry_index': float or None
            'composite': dict (from compute_composite_score)
    """
    cfg = config or TBConfig

    # Update live EIndex for each side (replaces, never accumulates)
    eindex_accumulator_left.update_live(rms_left)
    eindex_accumulator_right.update_live(rms_right)

    ei_live_l = eindex_accumulator_left.eindex_live
    ei_live_r = eindex_accumulator_right.eindex_live

    # Short-SUMA frequency penalty (per-window, never cumulative)
    penalty_l = compute_short_suma_freq_penalty(suma_bins_left, config=cfg)
    penalty_r = compute_short_suma_freq_penalty(suma_bins_right, config=cfg)

    # Per-side scores
    score_l = compute_per_side_score(ei_live_l, penalty_l, config=cfg)
    score_r = compute_per_side_score(ei_live_r, penalty_r, config=cfg)

    # Asymmetry index
    ai = compute_asymmetry_index(active_apdf50_left, active_apdf50_right)

    # Bilateral fusion + asymmetry penalty + floor clamp
    composite = compute_composite_score(score_l, score_r, ai, config=cfg)

    return {
        "eindex_live_left": ei_live_l,
        "eindex_live_right": ei_live_r,
        "eindex_session_cumulative_left": eindex_accumulator_left.eindex_session_cumulative,
        "eindex_session_cumulative_right": eindex_accumulator_right.eindex_session_cumulative,
        "short_suma_penalty_left": penalty_l,
        "short_suma_penalty_right": penalty_r,
        "score_left": score_l,
        "score_right": score_r,
        "asymmetry_index": composite["asymmetry_index"],
        "composite": composite,
    }
