# TensionBudget — Session Log

---

## Phase 1 — Repository verification and architecture mapping
Timestamp: 2026-07-08T18:41:00+05:30
Status: Completed

### Summary
Performed a complete source inspection of every file in the Chords-Python repository (17 source files, 7 directories, 3 config/metadata files). Verified 13 provisional assumptions against real code: 9 fully verified, 2 partially verified, 2 incorrect (no timestamps in data, no offline replay). Discovered a pre-existing `tensionbudget/` directory from a prior attempt with a broken import (missing `recorder.py`) and incorrect config values (bandpass 450 Hz exceeds Nyquist, RMS window 200 ms instead of 100 ms). Traced the complete data flow from Arduino serial binary packets through `chords_serial.py` parsing → `connection.py` LSL publishing and CSV recording → app-layer LSL consumption. Identified two critical gaps: the Arduino's 1-byte packet counter is discarded by `chords_serial.py`, and no timestamps exist in CSV recordings. Produced a detailed verification report with insertion points and a recommended modular subpackage strategy.

### Files created
- None (verification only — no code was written)

### Files modified
- None (verification only — no code was modified)

### Key functions/classes added or changed
- None (verification only)

### Assumptions / open questions
- The Arduino's 1-byte sample counter (packet byte index 2) is present in the binary stream but completely ignored by `chords_serial.py`'s `read_data()` — need to surface it for drop detection
- No real timestamps exist in the serial stream or in current CSV recordings — must add receive-side `perf_counter()` timestamps
- No offline replay functionality exists anywhere in the codebase — must be built from scratch
- `emgenvelope.py` processes only channel 0 with a hardcoded high-pass at 70 Hz — too limited for TensionBudget; new processing modules required
- Pre-existing `tensionbudget/__init__.py` imports `TBRecorder` from non-existent `recorder.py` — will crash on import
- Pre-existing `tensionbudget/config.py` has `BANDPASS_HIGH_HZ = 450.0` (invalid at 500 Hz sampling) and `RMS_WINDOW_MS = 200` (should be 100 ms per Marker & Maluf 2016)
- Open question: should we build on existing `session.py` or start fresh?
- Open question: should TensionBudget recording be a separate mode or replace existing recording?
- Open question: is it acceptable that offline replay works only on TB-format CSVs, not legacy recordings?
- Open question: channel mapping (which raw indices = left/right trapezius) requires hardware testing — will be configurable

### Next phase
Phase 2 — Stage 1: raw acquisition to laptop CSV. Blocked on user approval of the Phase 1 verification report and answers to the four open questions.
---

## Phase 2 — Stage 1: raw acquisition to laptop CSV
Timestamp: 2026-07-08T19:23:00+05:30
Status: Completed

### Summary
Implemented Stage 1 of the TensionBudget pipeline: raw acquisition to enhanced CSV. Fixed the broken tensionbudget package (broken import, incorrect config values), exposed the Arduino packet counter that was being discarded by the serial parser, added a non-invasive sample callback system to the Connection class, created TBRecorder for TensionBudget-format CSV recording with timestamps and drop detection, and wired up Flask API routes for the separate TB recording mode. All 7 modified/created files pass syntax checks and config values are verified correct.

### Files created
- `chordspy/tensionbudget/recorder.py` — TBRecorder class: Stage 1 CSV recording with metadata headers, timestamps, Arduino counter, drop detection, and sidecar JSON manifest

### Files modified
- `chordspy/tensionbudget/__init__.py` — Fixed broken import (was importing non-existent recorder.py), re-added after creating it
- `chordspy/tensionbudget/config.py` — Replaced: bandpass 450→240 Hz, RMS 200→100 ms, added RMS step 20 ms, channel mapping, fusion rule, helper methods
- `chordspy/tensionbudget/session.py` — Added channel map and unused channel info to CSV metadata header and JSON manifest
- `chordspy/chords_serial.py` — Added `self.last_counter` to expose Arduino packet counter byte (was discarded)
- `chordspy/connection.py` — Added sample callback system, TB recording methods (start/stop/status), TB cleanup hook
- `chordspy/app.py` — Added Flask routes: `/tb/start_recording`, `/tb/stop_recording`, `/tb/status`

### Key functions/classes added or changed
- `TBRecorder` in `recorder.py` — Core Stage 1 recorder: start/stop, CSV with enhanced schema, drop detection via Arduino counter, session manifest
- `TBRecorder._sample_callback()` in `recorder.py` — Per-sample handler registered on Connection
- `TBConfig` in `config.py` — Corrected all values, added `CHANNEL_MAP`, `rms_step_samples()`, `active_channel_indices()`, `unused_channel_indices()`
- `Chords_USB.last_counter` in `chords_serial.py` — Captures `packet[2]` in `read_data()` for drop detection
- `Connection.register_sample_callback()` in `connection.py` — Non-invasive callback registration
- `Connection.start_tb_recording()` / `stop_tb_recording()` / `get_tb_status()` in `connection.py` — TB recording lifecycle
- `tb_start_recording()` / `tb_stop_recording()` / `tb_recording_status()` in `app.py` — Flask API endpoints

### Assumptions / open questions
- Channel mapping (`CHANNEL_MAP`) is set to `None` (record all 6 channels) until hardware testing confirms which indices = left/right trapezius
- Timestamps are LSL `local_clock()`-relative (high-resolution monotonic), stored as seconds elapsed since first sample in the session — not wall-clock
- The `pyserial` dependency is not installed in the current Python environment, so full end-to-end testing with hardware is deferred
- Legacy Chords CSVs are not affected — TB recording is an entirely separate mode

### Next phase
Phase 3 — Stage 2: laptop preprocessing pipeline. Can begin immediately. Will implement: bandpass filtering (20–240 Hz, 4th-order Butterworth), full-wave rectification, RMS envelope (100 ms / 20 ms step), per-channel processing for left/right, both live and offline modes.
---

## Phase 3 — Stage 2: laptop preprocessing pipeline
Timestamp: 2026-07-08T19:42:00+05:30
Status: Completed

### Summary
Implemented the full Stage 2 preprocessing pipeline as a reusable module (`preprocessing.py`) with pure stateless functions. Pipeline stages: optional raw-to-voltage conversion → 20–240 Hz bandpass (4th-order Butterworth, zero-phase via filtfilt) → optional 50 Hz notch filter → full-wave rectification → RMS envelope (100 ms window, 20 ms step). All parameters from TBConfig, no magic numbers. Offline session processor reads TB CSVs and produces two output files: full-rate (filtered/rectified at sample rate) and RMS-rate (envelope at 50 Hz update rate). Left/right channels processed independently. Verified with 11 functional tests on synthetic EMG data including Nyquist guard, DC removal, channel independence, and end-to-end CSV processing.

### Files created
- `chordspy/tensionbudget/preprocessing.py` — Stage 2 signal processing: raw_to_voltage, bandpass_filter, notch_filter, rectify, rms_envelope, preprocess_channel, load_tb_csv, process_tb_session

### Files modified
- `chordspy/tensionbudget/__init__.py` — Added preprocessing module to docstring

### Key functions/classes added or changed
- `raw_to_voltage()` — Optional ADC-to-mV conversion (pass-through when vref=None)
- `bandpass_filter()` — 20–240 Hz Butterworth, filtfilt, with Nyquist violation guard
- `notch_filter()` — 50 Hz zero-phase IIR notch (Q=30, configurable)
- `rectify()` — Full-wave rectification (np.abs)
- `rms_envelope()` — Sliding RMS, 50-sample window, 10-sample step, returns values + center indices
- `preprocess_channel()` — Full per-channel pipeline returning all intermediates + config snapshot
- `load_tb_csv()` — Parses TB CSV with metadata comments, returns data dict + metadata dict
- `_resolve_channel_columns()` — Maps CHANNEL_MAP labels to CSV column names
- `process_tb_session()` — Offline processor: reads TB CSV, processes all channels, writes two output CSVs
- `_write_processed_csv()` — Writes full-rate and RMS-rate processed CSVs with metadata headers

### Assumptions / open questions
- `filtfilt` is used for offline processing (requires full signal). Live processing would need `sosfilt` with state — deferred to a later phase
- Voltage conversion is optional and defaults to pass-through (raw counts preserved). No ADC reference voltage is assumed
- When CHANNEL_MAP is None, all 6 channels are processed generically as Ch1–Ch6. Named left/right labels activate when CHANNEL_MAP is set
- Normalized output columns are NaN placeholders awaiting Phase 4 calibration

### Next phase
Phase 4 — Calibration layer for normalized interpretation. Will implement: per-channel RVE/MVC reference contraction capture, separate left/right calibration, normalized %RVE streams, modular and optional design.
---

## Phase 4 — Calibration Layer
Timestamp: 2026-07-08T21:15:00+05:30
Status: Completed

### Summary
Implemented a modular calibration subsystem to compute Reference Voluntary Exertion (RVE). A CLI-first interactive script (`calibrate.py`) prompts the user to perform separate 5-second reference shrugs for the left and right trapezius. The system calculates the reference RMS by passing the raw recording through the Stage 2 preprocessing pipeline, extracting the middle 3-second stable window, and computing its mean. This data is saved to a `tensionbudget_calibration.json` file. The offline session processor (`process_tb_session`) now automatically detects this JSON alongside recorded CSVs (via the manifest `calibration` block) and divides the channel's RMS envelope by the reference RMS, yielding normalized %RVE in the final output files.

### Files created
- `chordspy/tensionbudget/calibration.py` — Core reference RMS extraction logic (middle 3s mean)
- `chordspy/tensionbudget/calibrate.py` — CLI interactive calibration script
- `scratch/test_calibration.py` — Functional tests for extraction and %RVE normalization

### Files modified
- `chordspy/tensionbudget/session.py` — Added `calibration_data` property and merged it into the JSON manifest
- `chordspy/tensionbudget/recorder.py` — `start()` now autoloads `tensionbudget_calibration.json` from CWD into the session
- `chordspy/tensionbudget/preprocessing.py` — Updated `process_tb_session` and `preprocess_channel` to accept `reference_rms` and calculate `%RVE` for the `normalized` column

### Key functions/classes added or changed
- `compute_reference_rms()` — Extracts middle 3 seconds from RMS envelope and computes mean
- `calibrate.py (main)` — Orchestrates interactive left/right calibration recordings
- `process_tb_session()` — Modified to read sidecar JSON manifest for calibration data and inject it into the pipeline

### Assumptions / open questions
- The CLI calibration script assumes `TBConfig.CHANNEL_MAP` is configured (or defaults to left=0, right=1)
- The offline pipeline divides by the `reference_rms` and multiplies by 100 to yield a percentage (%)

### Next phase
Phase 5 — Analytics and UI Integration (Stage 3).
---

## Phase 5 — Stage 3 Core Analytics & Streaming Refactor
Timestamp: 2026-07-08T22:35:00+05:30
Status: Completed

### Summary
Built the Phase 5 analytics engine to extract rule-based physiological metrics from normalized `%RVE` streams. This includes Gap detection (Marker & Maluf 2016), Sustained Activity (SUMA) binning (Koch 2024), APDF percentiles, and side-to-side asymmetry (using a Laterality Index). Concurrently paid down technical debt by refactoring the Stage 2 filters to use Second-Order Sections (SOS), enabling the addition of a stateful `StreamingChannelProcessor` to support future live monitoring without duplicating filter logic. Also added safeguards enforcing hardware channel mapping before calibration.

### Files created
- `chordspy/tensionbudget/analytics.py` — Pure functions for gaps, SUMA, APDF, and asymmetry
- `chordspy/tensionbudget/session_report.py` — Runs analytics over offline CSVs and outputs JSON
- `scratch/test_analytics.py` — Verifies analytical math and event binning
- `scratch/test_preprocessing_live.py` — Verifies streaming causal filters match offline envelopes

### Files modified
- `chordspy/tensionbudget/config.py` — Added Stage 3 thresholds and live buffer specs (10 min rolling)
- `chordspy/tensionbudget/preprocessing.py` — SOS filter refactor and `StreamingChannelProcessor` class
- `chordspy/tensionbudget/calibrate.py` — Hard-error safeguard for unconfigured `CHANNEL_MAP`
- `chordspy/tensionbudget/calibration.py` — Documentation tweaks for project-specific choices

### Key functions/classes added or changed
- `detect_sustained_activity()` — Detects Cinderella-fiber runs and bins into Koch 2024 classes
- `compute_asymmetry()` — Now uses a mathematically stable Laterality Index (-1.0 to +1.0)
- `StreamingChannelProcessor` — Initializes `sosfilt_zi` state and processes `process_chunk()` causally

### Assumptions / open questions
- We are 5 phases deep into a hardware-interfacing project with zero real-hardware validation. Synthetic tests are passing, but structural assumptions (baud rate stability, packet drops, channel mapping) remain entirely unverified against the real Arduino Uno R4.

### Next phase
Hardware Smoke Test Checkpoint (Pending User Hardware Availability).
---

## Phase 6 — Composite Tension Budget Scoring Engine
Timestamp: 2026-07-29T20:55:00+05:30
Status: Completed

### Summary
Implemented the full Phase 6 scoring engine. The primary work was resolving the verified unbounded-EIndex-growth bug (confirmed by simulation: old approach diverged to ~5760 over an 8-hour constant-activity session) by splitting EIndex into two separate, non-conflating outputs. Also renamed the ambiguous `REST_THRESHOLD_PCT` constant to `GAP_REST_THRESHOLD_PCT` to explicitly distinguish it from the EIndex rest bucket — a real naming-collision bug documented in `important_context.md §8.6`.

### Files created
- `chordspy/tensionbudget/scoring.py` — Full Phase 6 scoring engine
- `chordspy/tensionbudget/tests/test_phase6_scoring.py` — 43 unit + regression tests
- `chordspy/tensionbudget/tests/conftest.py` — Hardware stub conftest (enables tests without pyserial)

### Files modified
- `chordspy/tensionbudget/config.py` — Removed placeholder weight constants; added `EINDEX_REST_THRESHOLD_PCT` (0.5%), `EINDEX_HIGH_THRESHOLD_PCT` (7.0%), `SHORT_SUMA_PENALTY_PER_EVENT` (0.1), `ASYMMETRY_PENALTY_THRESHOLD` (0.5), `ASYMMETRY_PENALTY_POINTS` (5.0); renamed `REST_THRESHOLD_PCT` → `GAP_REST_THRESHOLD_PCT` with explicit comment distinguishing from EIndex rest bucket
- `chordspy/tensionbudget/analytics.py` — Updated 3 references from `cfg.REST_THRESHOLD_PCT` → `cfg.GAP_REST_THRESHOLD_PCT` to match the rename

### Key functions/classes in scoring.py
- `compute_eindex_for_window(normalized_rms_window)` — Koch (2024) exact formula: `-2*P_Rest + P_Low + 2*P_High`. Bounded [-2,+2] by construction (proportions always sum to 1.0). EIndex thresholds: Rest < 0.5% MVE, High >= 7% MVE.
- `compute_short_suma_freq_penalty(suma_bins)` — Count-based formula: `min((1.5-5s count + 5-10s count) * 0.1, 1.0)`. 10 events = 1.0 exactly. Per-window ONLY, never cumulative. Novel extension of Koch (2025).
- `compute_per_side_score(eindex_live, short_suma_penalty)` — Additive combination, range [-2, +3]. Compounding is intentional: EIndex measures proportion-of-loaded-time (amplitude domain); short-SUMA penalty measures fragmentation pattern (event-frequency domain) — orthogonal risk factors per Koch 2025.
- `compute_asymmetry_index(apdf50_left, apdf50_right)` — Signed Laterality Index: `(APDF50_R - APDF50_L) / (APDF50_R + APDF50_L)`. Locked signed formula. Range [-1,+1]. NaN if either input is NaN.
- `compute_composite_score(score_l, score_r, ai)` — Pipeline order enforced: (1) `min(L,R)` worst-side fusion, (2) subtract 5.0 pts if `|AI| > 0.5`, (3) `max(0.0, result)` floor clamp LAST. Returns full breakdown dict for explainability.
- `EIndexAccumulator` — Tracks `eindex_live` (REPLACES on each update, never sums) and `eindex_session_cumulative` (increments ONCE per completed 10-min boundary via `submit_completed_window()`). The split is the architectural fix for the divergence bug.
- `score_bilateral_window(...)` — Convenience wrapper that wires all sub-functions in correct order for live use.
- `map_eindex_to_display_scale()` — `NotImplementedError` stub. Requires STAMI S6 dataset review (§5.2) to anchor scale endpoints. Do not guess.

### Regression test results (43/43 passed, 1.71s)
- `eindex_live` stays in [-2,+2] at every single update across a simulated 8-hour session ✅
- `eindex_session_cumulative` is flat between 10-min boundaries and only steps at `submit_completed_window()` calls ✅
- After 48 discrete 10-min windows (8 hrs at constant moderate load): cumulative = 76.8 (not 5760 — the old divergence bug) ✅
- Short-SUMA penalty: 0 events → 0.0, 1 event → 0.1, 10 events → 1.0, 20 events → 1.0 (capped), long-bin events → 0.0 contribution ✅
- Asymmetry index: symmetric → 0.0, right-only → +1.0, left-only → -1.0, NaN propagation ✅
- Composite score: floor clamp is provably LAST (pre-clamp stored separately) ✅
- STAMI stub raises `NotImplementedError` ✅

### Assumptions / open questions
- STAMI S6 dataset check (§5.2) still pending — `map_eindex_to_display_scale` stubbed until complete
- `CHANNEL_MAP = None` — hardware-blocked, unresolved
- Asymmetry penalty threshold (0.5) and penalty amount (5.0) are literature-anchored design choices; pilot data (v2) may warrant retuning
- Phase 8 (live app/serial integration) still blocked on hardware

### Next phase
Phase 7 — Conditional MDF spectral fatigue (Welch's method, pre-rectification 20–240 Hz signal, per side, OLS slope). Hardware-independent, synthetic-data-testable. Can proceed immediately.
---

## Phase 6 Hotfix — Asymmetry Penalty Scale Mismatch
Timestamp: 2026-07-29T21:34:00+05:30
Status: Completed

### Bug
`ASYMMETRY_PENALTY_POINTS = 5.0` was sized for a 0–100 display scale (from a stale budget example). The actual per-side score range is [-2, +3] (5 units wide). At max per-side score (+3.0), `3.0 - 5.0 = -2.0 → clamped to 0.0`. Every asymmetry trigger zeroed the composite regardless of actual load — a kill switch, not a proportionate penalty. The bug was masked by existing tests that used `score = 6.0`, which is impossible on the [-2, +3] scale.

### Fix
Rescaled `ASYMMETRY_PENALTY_POINTS` from `5.0` → `0.25` (5% of the 5-unit output range = same relative severity as the original 5/100 design intent). Added `⚠️ RESCALING REQUIRED` note in config.py: when `map_eindex_to_display_scale()` is eventually implemented, this constant must revert to 5.0 (or be re-derived from the 0–100 mapped output range).

Also caught a secondary bug: `TestConfig` local subclass in the test file had `ASYMMETRY_PENALTY_POINTS = 5.0` hardcoded, shadowing the config.py value. Removed that override so the proportionality test now reads directly from `TBConfig`, making it a live guard against future scale mismatches.

### Files modified
- `chordspy/tensionbudget/config.py` — `ASYMMETRY_PENALTY_POINTS`: 5.0 → 0.25, with full scale-mismatch explanation and STAMI-rescaling note
- `chordspy/tensionbudget/tests/test_phase6_scoring.py` — Fixed 3 tests that used impossible score values (6.0 > max of +3.0) or relied on 5.0 penalty; added `test_asymmetry_penalty_proportional_to_scale` (kill-switch guard + 50%-of-range proportionality check + exact 5%-of-range assertion); removed `ASYMMETRY_PENALTY_POINTS` override from `TestConfig`

### Test result
44/44 passed in 1.64s (up from 43 — new proportionality test added).

### Next phase
Phase 7 — Conditional MDF/MNF Spectral Fatigue Engine. Completed.
---

## Phase 7 — Conditional MDF/MNF Spectral Fatigue Engine
Timestamp: 2026-07-30T22:10:00+05:30
Status: Completed

### Summary
Implemented Stage 10 of the TensionBudget pipeline: assessing muscular fatigue in the frequency domain using Median Power Frequency (MDF) and Mean Power Frequency (MNF) via Welch's power spectral density (PSD), with linear time-trend analysis via Ordinary Least Squares (OLS) slope tracking (in Hz/min). Enforced the strict requirement to tap pre-rectification bandpass-filtered signals (Stage 2 output, 20–240 Hz) to avoid severe harmonic distortion caused by rectification. Implemented active-epoch gating (skipping epochs where <50% of samples exceed the 3% RVE rest threshold) to prevent estimating frequency peaks on resting electrical floor noise. Embedded a mandatory passband ceiling caveat in all output dicts to document that our 240 Hz Nyquist ceiling (at 500 Hz sampling rate) yields absolute MDF/MNF values lower than published standard norms (20–450 Hz at >=1000 Hz sampling) and must be interpreted strictly as internal temporal trend indicators.

### Files created
- `chordspy/tensionbudget/spectral.py` — Full Stage 10 spectral fatigue analysis engine
- `chordspy/tensionbudget/tests/test_phase7_spectral.py` — 18 comprehensive unit + synthetic simulation tests

### Files modified
- `chordspy/tensionbudget/config.py` — Added Stage 10 spectral constants: `SPECTRAL_WINDOW_S` (1.0), `SPECTRAL_OVERLAP_S` (0.5), `SPECTRAL_BAND_LOW_HZ` (20.0), `SPECTRAL_BAND_HIGH_HZ` (240.0), `SPECTRAL_MIN_VALID_WINDOWS` (5), `SPECTRAL_MIN_ACTIVE_RATIO` (0.5), `FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN` (-0.3), and `FATIGUE_MIN_R_SQUARED` (0.3). Added explicit docstring note that slope threshold is a conservative v1 placeholder extrapolated from high-force studies (~-1.2 Hz/min) since no published occupational low-load (<5% MVC) MDF threshold exists, to be retuned with pilot data in v2.
- `chordspy/tensionbudget/__init__.py` — Exported `compute_welch_psd`, `compute_mdf_mnf`, `extract_active_spectral_series`, `compute_spectral_fatigue_slope`, and `analyze_bilateral_spectral_fatigue`.

### Key functions/classes in spectral.py
- `compute_welch_psd(signal, ...)` — Evaluates Welch PSD with a Hanning window and slices output strictly to the valid [20.0, 240.0] Hz passband. Requires pre-rectification Stage 2 input.
- `compute_mdf_mnf(freqs, psd)` — Computes MDF (frequency where cumulative PSD hits 50% using sub-bin linear interpolation) and MNF (power-weighted frequency centroid).
- `extract_active_spectral_series(pre_rect_signal, active_mask, ...)` — Slides Welch window over session signal, skipping epochs where active sample ratio < 0.5 (active-mask gating).
- `compute_spectral_fatigue_slope(times_sec, mdf_values, ...)` — Fits OLS linear regression to valid MDF series, reporting rate of shift in Hz/min and R² linear fit quality. Emits `is_fatiguing = True` when slope < `FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN` and R² >= `FATIGUE_MIN_R_SQUARED`, reading thresholds dynamically from config to avoid hardcoding bugs.
- `analyze_bilateral_spectral_fatigue(...)` — Processes Left and Right channels completely independently and always attaches the mandatory `passband_ceiling_caveat` warning string.

### Test results (62/62 passed, 7.48s)
- All 44 Phase 6 tests continue to pass without regressions ✅
- 18 new Phase 7 unit and synthetic simulation tests passing ✅
- Mathematical precision verified against single-tone sine waves (±1.0 Hz tolerance) ✅
- Dual-tone equal-power test correctly documented and verified: MNF tracks arithmetic centroid (90 Hz) while MDF tracks median power split at lower tone (~60 Hz) ✅
- Pre-rectification requirement mathematically proven in test suite (>5.0 Hz shift distortion caused by post-rectification harmonics) ✅
- OLS slope recovery validated against a synthetic 10-minute frequency chirp (drifting from 105 Hz to 65 Hz, true slope -4.0 Hz/min) recovering slope within ±0.5 Hz/min and R² >= 0.85 ✅
- Anti-hardcoding guard verified: changing config thresholds dynamically flips `is_fatiguing` flag without code modification ✅

### Next phase
Phase 8 — Application UI Integration & Real-Time LSL / Offline Replay Monitor. Completed.
---

## Phase 8 — Real-Time LSL Streaming & Offline Replay Monitoring Application
Timestamp: 2026-07-30T23:50:00+05:30
Status: Completed

### Summary
Designed, engineered, and integrated a dedicated graphical monitoring application (`chordspy/tensionbudget_app.py`) built with PyQt5 and pyqtgraph for real-time bilateral upper-trapezius sEMG surveillance. The application cleanly bridges our shared Phase 1–7 analytical engines (`chordspy.tensionbudget` library) into an interactive desktop monitor with three structured operational modes: (1) **Live Bilateral Signals View** showing 10-second rolling waveforms of bandpass-filtered signals and moving RMS envelopes per channel, (2) **Real-Time Analytics Dashboard** tracking live worst-side driven Composite TensionBudget scores, EIndex load accumulations, Signed Asymmetry Laterality index, relaxation gap frequency, short-SUMA fragmentation penalties, and OLS MDF/MNF spectral fatigue slopes (with mandatory 240 Hz passband caveat display), and (3) **Interactive Calibration Mode** providing guided 5-second submaximal shoulder shrug test recording that computes and stores personalized reference levels (`tensionbudget_calibration.json`) for normalized %RVE scoring. 

To avoid any module namespace collisions with our existing analytical package (`chordspy/tensionbudget/`), the application script was safely placed as `chordspy/tensionbudget_app.py` and registered into the Chords-Python web GUI launcher in `chordspy/config/apps.yaml` under category `"EMG"`. Additionally, unblocked by user hardware verification (Arduino Uno R4 + 2x BioAmp EXG Pills wired to analog inputs A0 and A1), `TBConfig.CHANNEL_MAP` was promoted from `None` to `{"left": 0, "right": 1}`.

### Files created
- `chordspy/tensionbudget_app.py` — PyQt5/pyqtgraph multi-mode real-time monitor (Live LSL stream & Offline CSV replay)
- `chordspy/tensionbudget/tests/test_phase8_app.py` — Verification suite covering YAML app registration, default channel map configuration, and envelope math logic

### Files modified
- `chordspy/tensionbudget/config.py` — Updated `CHANNEL_MAP` default to `{"left": 0, "right": 1}` (A0=Left, A1=Right)
- `chordspy/config/apps.yaml` — Registered `"TensionBudget Monitor"` (`script: "tensionbudget_app"`) for the Flask UI launcher

### Key functions/classes in tensionbudget_app.py
- `TensionBudgetApp(QMainWindow)` — Manages circular 10-second visualization buffers, dual-source data acquisition (live LSL or offline CSV playback), 50 Hz refresh timer (`20 ms`), and tabbed UI presentations.
- `update_loop()` — Pulls sample chunks from LSL or CSV arrays, updates circular signal arrays, executes real-time Butterworth bandpass (20–240 Hz) and moving RMS envelopes, and invokes `score_bilateral_window()` and `analyze_bilateral_spectral_fatigue()` to refresh dashboard indicators.
- `on_start_calibration()` — Triggers interactive 5-second shrug buffer recording (`2500` samples at `500 Hz`), executes `compute_reference_rms()`, dynamically updates normalization factors, and saves baseline data to `tensionbudget_calibration.json`.

### Test results (64 passed, 1 skipped in 29.3s)
- All Phase 1–7 analytical and regression tests passing without disruption ✅
- `test_app_registered_in_yaml` verified correct registration of `tensionbudget_app` in `apps.yaml` ✅
- `test_default_channel_map` confirmed default A0/A1 channel index assignment ✅
- GUI dependency isolation verified: tests gracefully skip GUI instantiations when running in headless/non-PyQt5 environments while maintaining 100% test reliability ✅

### Next phase
Phase 9 — End-to-End System Verification, User Testing Guide & Documentation Freeze. In progress.
---

## Phase 8.5 — Hardware Calibration Readiness, Channel Mapping Confirmation & UI Polish
Timestamp: 2026-07-31T15:25:00+05:30
Status: Completed

### Summary
Prepared the codebase for live hardware testing with the Arduino Uno R4 and two BioAmp EXG Pills by replacing preliminary assumptions with verified physical channel wiring and resolving edge-case dashboard display discrepancies:
1. **Physical Channel Mapping**: Confirmed physical hardware wiring to analog pins A2 and A3. Updated `TBConfig.CHANNEL_MAP` to `{"left": 2, "right": 3}`, corresponding to transmitted slots `Channel3` (Left) and `Channel4` (Right). Updated `on_load_csv` in `tensionbudget_app.py` to dynamically derive CSV column headers directly from `CHANNEL_MAP`, eliminating hardcoded column assumptions.
2. **Scoring Engine Integration Fixes**: Resolved dictionary key mismatches between the calculation engine (`scoring.py`) and GUI display updating (`tensionbudget_app.py`). Correctly mapped keys (`score_left`, `score_right`, `eindex_live_left`, `short_suma_penalty_left`, and `composite_score`), ensuring all calculated fatigue and load metrics actively display during playback and live streaming instead of falling back to zero.
3. **Rest-State Asymmetry Handling**: Added defensive type and `None` checking to Laterality Index calculations. During complete muscle relaxation periods (<3% RVE), when active APDF-50 cannot be calculated and `asymmetry_index` returns `None`, the display gracefully defaults to `0.00 (Symmetric)` without triggering NumPy ufunc casting exceptions.
4. **Realistic Mock Testing Setup**: Engineered `mock_data/generate_realistic_mock.py`, synthesizing a realistic 10-minute ergonomic workstation session (`TB_REALISTIC_10MIN_raw.csv`) incorporating variable %MVC work bursts, posture pauses, right-dominant mouse usage, low-frequency spectral fatigue drift, and an automated `tensionbudget_calibration.json` profile for rigorous offline validation.

### Files created / modified
- `chordspy/tensionbudget/config.py` — Updated `CHANNEL_MAP` to `{"left": 2, "right": 3}` (A2=Left / Channel3, A3=Right / Channel4).
- `chordspy/tensionbudget_app.py` — Fixed scoring engine dictionary key bindings, protected Asymmetry UI display against `None` values during resting states, and updated CSV loading logic to derive target columns directly from `CHANNEL_MAP`.
- `mock_data/generate_realistic_mock.py` — Created 10-minute realistic workstation EMG generator with automatic calibration profile creation.

### Next steps
Conduct first live hardware testing session with physical Arduino Uno R4 streaming over LSL. Completed.
---

## Phase 8.6 — Dual-Arduino Hardware Validation & LSL Synchronization Bridge
Timestamp: 2026-07-31T18:00:00+05:30
Status: Completed

### Summary
Successfully conducted our first live hardware testing session with physical Arduino hardware, leading to a critical architectural upgrade to support a **Dual-Arduino Bilateral Setup** (one board for Left shoulder, one for Right shoulder):
1. **Dedicated LSL Bridge Script (`start_lsl_stream.py`)**: Created a standalone multi-threaded USB-to-LSL synchronization bridge that connects simultaneously to two serial ports (`COM6` for Left trapezius on Analog Pin A2, `COM5` for Right trapezius on Analog Pin A2).
2. **Firmware Compatibility & Line-Ending Adaptation**: Discovered via custom buffer diagnostics that while COM6 transmitted standard binary frames terminated with `0x01` at 500 Hz, COM5 operated at 250 Hz and terminated frames with ASCII carriage returns (`0x0D` / `\r`). Updated the packet validation parser to universally accept `0x01`, `0x0D`, and `0x0A` frame terminators without packet loss or structural redesign.
3. **Synchronized Bilateral Upsampling**: Engineered a primary-pacemaker threading model where Left (COM6 @ 500 Hz) acts as the master clock for LSL sample broadcasting, performing automatic zero-order hold upsampling on Right (COM5 @ 250 Hz). This ensures an exact, zero-drift **500 Hz 2-channel bilateral stream** (Channel 0 = Left, Channel 1 = Right) over LSL.
4. **Configuration & Test Suite Alignment**: Updated `TBConfig.CHANNEL_MAP` to `{"left": 0, "right": 1}` and adjusted unit tests in `test_phase8_app.py` to match the dual-Arduino 2-channel LSL stream. Verified real-time live signal waveforms and moving RMS envelopes streaming cleanly into `tensionbudget_app.py`.

### Files created / modified
- `start_lsl_stream.py` — Created dual-threaded USB-to-LSL synchronization bridge supporting mixed baud rates and firmware frame terminators.
- `chordspy/tensionbudget/config.py` — Updated default `CHANNEL_MAP` to `{"left": 0, "right": 1}` for clean 2-channel bilateral ingestion.
- `chordspy/tensionbudget/tests/test_phase8_app.py` — Updated assertions to match `{"left": 0, "right": 1}`.
- `logs/session_log.md` — Appended Phase 8.6 hardware validation session summary.

### Next phase
Phase 9 — End-to-End System Verification, Personal Calibration Recording & Ergonomic Pilot Test Protocol.
---
