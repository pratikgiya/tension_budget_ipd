"""
TensionBudget — Central configuration.

All thresholds, window sizes, weights, and tuneable parameters live here.
No magic numbers in any other module.

Design constraints (from prompt):
    - 500 Hz sampling, 14-bit ADC, UNO-R4 via USB serial at 230400 baud
    - 6 raw channels transmitted per packet; only 2 carry real EMG (left/right)
    - Bandpass 20–240 Hz (De Luca 1997 / Farina 2002, truncated to Nyquist)
    - RMS window 100 ms / 50 samples (Marker & Maluf 2016)
    - All parameters configurable, never hardcoded inline
"""


class TBConfig:
    """
    Central configuration for the TensionBudget pipeline.

    Every tuneable parameter across all three stages lives here.
    Helper classmethods convert time-domain values to sample counts.
    """

    # ── Stage 1: Acquisition ───────────────────────────────────────────

    # Target hardware defaults (auto-detected, but these are the expected values)
    TARGET_BOARD = "UNO-R4"
    TARGET_BAUD_RATE = 230400
    TARGET_SAMPLING_RATE = 500          # Hz
    TARGET_RESOLUTION_BITS = 14         # ADC resolution for UNO-R4
    TARGET_NUM_RAW_CHANNELS = 6         # Firmware always sends 6 channel slots

    # Channel mapping — which LSL stream indices carry real EMG signal.
    #
    # Hardware-confirmed dual-Arduino setup:
    #   Left Arduino (COM6) pin A2  → LSL channel 0 ("Channel1" in CSV)
    #   Right Arduino (COM5) pin A2 → LSL channel 1 ("Channel2" in CSV)
    CHANNEL_MAP = {"left": 0, "right": 1}  # type: dict[str, int] | None

    # Channels not in CHANNEL_MAP are flagged as unused/floating in the manifest.
    # If CHANNEL_MAP is None, all channels are recorded and none are flagged.

    # CSV recording
    CSV_FILENAME_PREFIX = "TB"
    CSV_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    CSV_METADATA_COMMENT_CHAR = "#"     # Lines starting with this are metadata

    # ── Stage 2: Preprocessing ─────────────────────────────────────────

    # Bandpass filter (De Luca 1997 / Farina et al. 2002)
    # Upper cutoff truncated to 240 Hz to stay safely below 250 Hz Nyquist
    # at 500 Hz sampling. Do NOT set above ~240 Hz at this sample rate.
    BANDPASS_LOW_HZ = 20.0
    BANDPASS_HIGH_HZ = 240.0
    BANDPASS_ORDER = 4                  # 4th-order Butterworth, applied via filtfilt

    # Notch filter (mains interference)
    NOTCH_FREQ_HZ = 50.0               # 50 Hz (India/EU) or 60 Hz (US)
    NOTCH_QUALITY_FACTOR = 30.0
    NOTCH_ENABLED = "auto"              # True, False, or "auto" (detects spike)

    # RMS envelope (Marker & Maluf 2016)
    # 100 ms window = 50 samples at 500 Hz
    # 20 ms step = 10 samples at 500 Hz
    RMS_WINDOW_MS = 100
    RMS_STEP_MS = 20

    # ── Stage 3: Analytics (Phase 5) ──────────────────────────────────

    # Gap/rest detection (Marker & Maluf 2016)
    # IMPORTANT: This 3.0% threshold is DISTINCT from EINDEX_REST_THRESHOLD_PCT
    # below — two different papers, two different purposes. Do not merge.
    GAP_REST_THRESHOLD_PCT = 3.0        # Below 3.0% RVE = muscular rest (gap)
    MIN_GAP_DURATION_S = 0.125          # Minimum relaxation gap to count (s)

    # Sustained activity (SUMA-style, Koch 2024 Cinderella-fiber threshold)
    SUSTAINED_ACTIVITY_THRESHOLD_PCT = 0.5  # Above 0.5% RVE = sustained tension
    MIN_SUSTAINED_DURATION_S = 1.5      # Minimum sustained-activity run to flag (s)

    # Live buffer parameters
    LIVE_BUFFER_MINUTES = 10.0          # Sliding window size for live EIndex
    LIVE_UPDATE_INTERVAL_S = 2.0        # How often to refresh live EIndex (s)

    # APDF percentiles
    APDF_PERCENTILES = [10, 50, 90]

    # ── Stage 3: Phase 6 Scoring ───────────────────────────────────────

    # EIndex bucket thresholds (Koch 2024)
    # DISTINCT from GAP_REST_THRESHOLD_PCT (3.0%) — different paper, different purpose.
    EINDEX_REST_THRESHOLD_PCT = 0.5     # < 0.5% MVE = rest bucket (Koch 2024)
    EINDEX_HIGH_THRESHOLD_PCT = 7.0     # >= 7.0% MVE = high-load bucket (Koch 2024)

    # Short-SUMA frequency penalty (project novelty, extending Koch 2025)
    # Formula: min(short_event_count * SHORT_SUMA_PENALTY_PER_EVENT, 1.0)
    # 10 events = full penalty of 1.0; per-window only, NEVER cumulative.
    SHORT_SUMA_PENALTY_PER_EVENT = 0.1  # 0.1 per event in (1.5-5s) or (5-10s) bins

    # Asymmetry penalty (project design choice — signed AI formula locked)
    # Applied if |AI| > threshold, subtracted BEFORE floor clamp.
    ASYMMETRY_PENALTY_THRESHOLD = 0.5   # |AI| > 0.5 triggers penalty

    # ASYMMETRY_PENALTY_POINTS — scale-matched to the per-side score range [-2, +3].
    #
    # The original design intent was a ~5% reduction penalty (from an old 0-100
    # budget example: 5.0 pts on a 0-100 scale). The per-side score range is
    # 5 units wide (from -2 to +3), so the proportionate equivalent is:
    #     5% × 5 units = 0.25
    #
    # Using the original 5.0 here is a kill switch: max per-side score is +3.0,
    # so 3.0 - 5.0 = -2.0 → always clamped to 0. Every trigger would zero the
    # composite regardless of actual load — a verified bug.
    #
    # Note: ASYMMETRY_PENALTY_POINTS remains fixed at 0.25 for the live/composite
    # path ([-2, +3] scale). STAMI mapping applies ONLY to eindex_session_cumulative,
    # and does NOT touch per-side scores or the composite score at all.
    ASYMMETRY_PENALTY_POINTS = 0.25     # 5% of 5-unit [-2,+3] output range

    # Fusion rule: "worst_side" = min(Score_L, Score_R)
    # This is locked — do not change without updating scoring.py logic.
    FUSION_RULE = "worst_side"

    # ── STAMI S1_Dataset.sav Cumulative Display-Scale Mapping ──────────
    # Anchored to the 5th/95th percentile of Koch et al. 2024's 731-subject STAMI
    # pooled dataset (S1_Dataset.sav, SPSS format).
    # NOTE ON /100 PERCENTAGE-SCALE CORRECTION:
    # STAMI's stored proportions are on a 0-100 percentage scale, not 0-1 fractions.
    # Applying our EI formula (-2*P_Rest + P_Low + 2*P_High) to raw STAMI data
    # produces values 100x too large. Do not re-derive from raw file values without
    # applying this /100 division (verified via top-5 subject per-window averages and
    # full population RRT distributions).
    EI_FLOOR: float = -3.854            # STAMI S1_Dataset.sav, 5th pctile, corrected /100 (was -385.3945 raw)
    EI_CEIL: float = 49.635             # STAMI S1_Dataset.sav, 95th pctile, corrected /100 (was 4963.4760 raw)

    # ── Stage 10: Spectral Fatigue (MDF/MNF) ──────────────────────────

    # Welch PSD estimation window and overlap.
    # Window must be 0.5–2.0s per the locked spec; 1.0s @ 500 Hz = 500 samples.
    SPECTRAL_WINDOW_S   = 1.0           # Welch epoch length (s)
    SPECTRAL_OVERLAP_S  = 0.5           # 50% overlap between consecutive epochs

    # Valid frequency band for PSD evaluation.
    # Matches Stage 2 bandpass exactly — anything outside this band is noise.
    # NOTE: 240 Hz ceiling is due to Nyquist (fs=500Hz). De Luca/Farina use 450 Hz
    # (requires ≥1000 Hz sampling). Our MDF/MNF are internal trend indicators only —
    # not directly comparable to published absolute baselines.
    SPECTRAL_BAND_LOW_HZ  = 20.0       # Hz — matches BANDPASS_LOW_HZ
    SPECTRAL_BAND_HIGH_HZ = 240.0      # Hz — matches BANDPASS_HIGH_HZ

    # Minimum valid windows before computing OLS slope.
    # Fewer than this gives a meaningless 2-point regression.
    SPECTRAL_MIN_VALID_WINDOWS = 5

    # Active-epoch gating threshold.
    # An epoch is skipped for spectral estimation if fewer than this fraction
    # of its samples are above GAP_REST_THRESHOLD_PCT (3.0% RVE).
    # Rationale: PSD on electrical resting-floor noise is meaningless.
    # ⚠️ PROJECT HEURISTIC — not literature-sourced. Same category as
    # ASYMMETRY_PENALTY_THRESHOLD (0.5). Retune if pilot data shows gating
    # is too aggressive or too lenient.
    SPECTRAL_MIN_ACTIVE_RATIO = 0.5

    # Fatigue detection thresholds for the OLS slope flag (is_fatiguing).
    #
    # FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN:
    #   ⚠️ NOT LITERATURE-DERIVED. No published MDF slope threshold exists for
    #   low-level sustained occupational EMG (<5% MVC desk-work loads).
    #   The only comparable data is from high-force fatigue studies (e.g. biceps
    #   curls at 60-80% MVC), which show ~-1.2 Hz/min MDF slopes — NOT applicable
    #   at our load range (far lower intensity → far smaller slopes expected).
    #   This -0.3 Hz/min is a conservative downward extrapolation. PLACEHOLDER v1.
    #   ➜ Retune using pilot session slope distributions (v2, post-data-collection).
    #
    # FATIGUE_MIN_R_SQUARED:
    #   Minimum linear fit quality to declare a trend. R²=0.3 means 30% of MDF
    #   variance explained by time — low threshold that accepts noisy real data.
    #   ⚠️ PROJECT HEURISTIC — same status as above. Retune in v2.
    FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN = -0.3   # Hz/min — see note above
    FATIGUE_MIN_R_SQUARED               = 0.3    # minimum R² to flag as fatiguing

    # ── Calibration (placeholder) ─────────────────────────────────────

    CALIBRATION_DURATION_S = 5.0        # Duration of reference contraction capture

    CALIBRATION_TYPE = "RVE"            # "RVE" (Reference Voluntary Exertion) or "MVC"

    # ── Subjective Self-Reporting (Borg CR-10 Scale for Bayesian Hierarchical ML) ──
    # Full continuous-feeling 0-10 resolution for ordinal regression and Bayesian priors.
    # Missing data is represented by None (NULL in SQL / empty string in CSV) paired with
    # an explicit boolean flag `strain_reported`, NEVER a sentinel number like -1.
    STRAIN_SCALE_MIN = 0.0
    STRAIN_SCALE_MAX = 10.0
    STRAIN_TARGET_LABELS = {
        0: "0 — Nothing at all (Complete Rest)",
        1: "1 — Very weak (Just noticeable effort)",
        2: "2 — Weak (Light effort)",
        3: "3 — Moderate (Comfortable working level)",
        4: "4 — Somewhat strong",
        5: "5 — Strong (Heavy working fatigue)",
        6: "6 — Very noticeable fatigue",
        7: "7 — Very strong (Severe strain)",
        8: "8 — Extremely strong (Near failure)",
        9: "9 — Approaching maximum tolerance",
        10: "10 — Absolute maximum (Intolerable pain/fatigue)"
    }

    # ── Derived helpers ───────────────────────────────────────────────

    @classmethod
    def rms_window_samples(cls):
        """Compute RMS window size in samples from ms and sampling rate."""
        return int(cls.RMS_WINDOW_MS * cls.TARGET_SAMPLING_RATE / 1000)

    @classmethod
    def rms_step_samples(cls):
        """Compute RMS step size in samples from ms and sampling rate."""
        return int(cls.RMS_STEP_MS * cls.TARGET_SAMPLING_RATE / 1000)

    @classmethod
    def sustained_min_samples(cls):
        """Compute sustained-activity minimum duration in RMS samples."""
        return int(cls.MIN_SUSTAINED_DURATION_S * (1000 / cls.RMS_STEP_MS))

    @classmethod
    def min_gap_samples(cls):
        """Compute minimum gap duration in RMS samples."""
        return int(cls.MIN_GAP_DURATION_S * (1000 / cls.RMS_STEP_MS))

    @classmethod
    def active_channel_indices(cls):
        """
        Return the list of raw channel indices that carry real EMG data.
        If CHANNEL_MAP is None, returns all channel indices.
        """
        if cls.CHANNEL_MAP is None:
            return list(range(cls.TARGET_NUM_RAW_CHANNELS))
        return list(cls.CHANNEL_MAP.values())

    @classmethod
    def unused_channel_indices(cls):
        """
        Return raw channel indices that are floating/unused.
        Empty list if CHANNEL_MAP is None (all channels treated as active).
        """
        if cls.CHANNEL_MAP is None:
            return []
        active = set(cls.CHANNEL_MAP.values())
        return [i for i in range(cls.TARGET_NUM_RAW_CHANNELS) if i not in active]
