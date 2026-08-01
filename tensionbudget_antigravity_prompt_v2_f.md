# TensionBudget — Chords-Python Implementation Prompt (for Antigravity)

---

## Prompt (copy everything below into Antigravity)

I want you to modify **Chords-Python** into a project-specific implementation for my **TensionBudget** system. This is not a generic EMG visualizer anymore — it must become a structured acquisition + preprocessing + analysis pipeline for low-cost upper-trapezius sEMG monitoring in desk-work conditions.

### Context

**Hardware**
- Arduino Uno R4 + 2x BioAmp EXG Pill sensors, one per side — left upper trapezius and right upper trapezius. 5 electrodes total: 2 left (bipolar pair), 2 right (bipolar pair), 1 shared reference at the neck (confirmed as the vendor's own documented multi-module wiring method).
- Confirmed acquisition rate: 500 Hz (verified from firmware source, `#define SAMP_RATE 500.0` in `UNO-R4.ino`).
- Serial streaming at 230400 baud (verified: `#define BAUD_RATE 230400`).
- **Firmware always transmits 6 channel slots per packet** (`#define NUM_CHANNELS 6` in `UNO-R4.ino`), regardless of how many electrodes are physically wired. Only 2 of those 6 raw channels will carry real EMG signal (left, right); the rest are floating/unused and must not be treated as data. Phase 1 must confirm which raw channel indices the left and right electrodes actually land on once wired — do not assume channel 1 = left, channel 2 = right without checking.
- There is no 3rd physical EMG channel. A "3rd channel" referenced anywhere in prior notes is a **derived left/right asymmetry metric**, computed downstream from the two real channels — not a third sensor, not present in the raw ADC stream.
- Wireless (BLE/WiFi) is **out of scope for this implementation.** Arduino Uno R4 Minima has no radio hardware; Uno R4 WiFi's onboard ESP32-S3 is not used by the current `UNO-R4.ino` firmware (Serial-only). Chords-Python's `chords_ble.py`/`chords_wifi.py` are built around a different board (NPG-Lite, 3-channel, 12-bit ADC) and don't apply here. Stay USB serial only unless explicitly told otherwise in a later prompt.
- Data is recorded on a laptop through the Chords-Python acquisition flow.

**About the architecture notes below**
Everything in the next section (`connection.py`, `chords_serial.py`, `emgenvelope.py`, `config/apps.yaml`, `csvplotter.py`, and how they interact) is drawn from my own written summaries of the codebase, not from you having read the actual source yet. Treat it as a **provisional map only** — a hypothesis to verify, not a spec to build from. Your first job in this project is to open the real repository and confirm, correct, or replace this map before anything else happens.

**Provisional architecture (unverified)**
- `connection.py` — believed to be the unified broker for acquisition, LSL publishing, and CSV recording.
- `chords_serial.py`, `chords_wifi.py`, `chords_ble.py` — believed to be transport handlers.
- `app.py` + `config/apps.yaml` — believed to control app launching from the web UI.
- `emgenvelope.py` — believed to be the closest existing EMG-specific processing app.
- `csvplotter.py` and related utilities — believed to handle offline inspection.

**Do not rebuild from scratch.** Reuse Chords-Python's acquisition, recording, app-launching, and stream-subscription architecture wherever possible — once you've confirmed what that architecture actually is.

**Purpose of TensionBudget**
This system estimates cumulative trapezius load, detects sustained low-level activity and reduced relaxation, and computes a personalized **Tension Budget** score — it is built for continuous low-level monitoring in desk workers/students, not high-amplitude sports EMG.

**Three-stage pipeline**
1. **Stage 1** — raw acquisition to laptop CSV
2. **Stage 2** — laptop-side preprocessing
3. **Stage 3** — analysis / algorithms / model-ready outputs for tension, fatigue, and sustained-load interpretation

---

## Non-negotiable design constraints

1. **Verify before you build.** Never assume a file, function, class, or CSV schema exists because a summary said so — confirm it by reading the actual source first. If the real repo differs from the provisional map above, follow the real repo and tell me what changed.
2. **No monolith files.** Keep acquisition, preprocessing, analytics, and scoring in separate modules.
3. **Raw data is sacred.** Raw acquisition must remain recoverable and stored separately from any processed/derived data.
4. **Support both live and offline workflows.** The pipeline must process newly acquired data *and* replay old CSVs.
5. **Design for sustained low-level trapezius activity**, not generic gym/sports EMG.
6. **No magic numbers.** All thresholds, windows, and weights go in a central config, never hardcoded inline.
7. **Model-ready by design.** Even with no ML model yet, Stage 3 outputs must be structured for future model consumption.
8. **Report after every phase — no exceptions.** Do not silently chain phases together, and do not begin implementation (Phase 2 onward) until the Phase 1 verification report is reviewed and approved.
9. **Design for two real EMG channels (left, right), independently — not one generic channel.** Left and right upper trapezius do not share a calibration baseline, filter state, or APDF/SUMA computation. Every per-channel stage (filtering, RMS, calibration, APDF, gap detection, SUMA) must run on left and right separately. The Tension Budget score in Phase 6 must define an explicit left/right fusion rule (e.g. worst-side-drives-alert, average, or asymmetry-as-its-own-term) — this is a real design decision to make and justify, not something that falls out of the math by default.

---

## Reporting format

At the end of **every phase**, report back using exactly this structure:

1. Phase completed
2. Files modified
3. Files created
4. Key code changes (functions/classes added or modified)
5. Current data flow
6. What I can now run
7. What outputs I now get
8. Open questions / assumptions
9. Recommendation for next phase

Phase 1 has its own, more detailed reporting format (see below) since no implementation happens in it. Do not proceed to the next phase without the relevant report and my go-ahead.

---

## Phase 1 — Repository verification and architecture mapping

This phase is **verification only**. Do not write or modify any implementation code in this phase.

### What you must do

**1. Walk the real repository tree**
- Enumerate the directory structure.
- Identify the true package root.
- Locate all files relevant to acquisition, CSV logging, EMG processing, app launching, config, and replay tools.

**2. Open and inspect the real source files**
At minimum, inspect: `app.py`, `connection.py`, `chords_serial.py`, `emgenvelope.py`, `csvplotter.py`, `config/apps.yaml`, and any helper modules they depend on.

If the actual structure differs from the provisional map above, **follow the real structure** and flag the difference explicitly — do not silently reconcile it.

**3. Verify key assumptions, from code, not inference**
- Where serial data enters the system.
- How baud rate is configured.
- How samples are parsed.
- Whether timestamps or sample counters exist.
- Where CSV logging actually happens.
- Whether apps consume LSL or some other stream abstraction.
- How `emgenvelope.py` currently filters, rectifies, smooths, and plots EMG.
- How new apps are registered in the UI.
- Whether offline replay already exists and how it works.

**4. Search for hidden coupling**
Check whether EMG logic is spread outside `emgenvelope.py` — shared helper files, utility modules, frontend launch logic, tests, notebooks, parser prototypes.

**5. Check real constraints before designing**
- Current dependencies and install requirements.
- Whether scipy/numpy/pandas are already in use.
- Whether an existing buffering model is in place.
- Whether file paths / app launching assume a particular structure.
- Whether a CSV schema is already fixed anywhere else in the codebase.

### Exploration priority order

1. Repository root, `README`, `requirements.txt` / `pyproject.toml`
2. `app.py`
3. `connection.py`
4. Transport files such as `chords_serial.py`
5. `emgenvelope.py`
6. `config/apps.yaml`
7. `csvplotter.py`
8. Any helper/shared modules referenced by those files
9. Tests and notebooks — only if they influence the implementation

### Behavioral rules for this phase

- Do not assume a file exists just because a summary said so; verify it in the repo first.
- Do not assume a function name, class name, or CSV schema without reading the source.
- Do not decide whether to modify `emgenvelope.py` or create `tensionbudget.py` until after inspection.
- Do not start Stage 1 implementation (Phase 2) until the verification report below is complete.
- If files are missing, renamed, or structured differently than assumed, adapt to the actual repository and explain the difference.
- If the repo has multiple plausible entry points, compare them and justify which one should be used.

### Decision gate

Stop after the verification report. Do not proceed to Phase 2 until you have verified: the real file structure, the real acquisition/data path, the real EMG processing path, and the safest insertion points for Stage 1, Stage 2, and Stage 3.

### Required report for Phase 1

1. **Verification phase completed**
2. **Repository files inspected**
3. **Verified architecture map** — actual relevant files/folders, one-line purpose for each
4. **Assumption check table** — for each provisional assumption above (e.g. "`connection.py` owns CSV logging," "`chords_serial.py` owns serial parsing," "`emgenvelope.py` is the correct EMG entry point," "`apps.yaml` controls app registration," "apps subscribe through LSL"), mark: Verified / Partially verified / Incorrect / Not yet found
5. **Actual data flow** — traced from hardware input → serial parser → connection manager → stream/logging layer → app consumer → EMG app/output
6. **Best insertion points for TensionBudget** — for Stage 1, Stage 2, and Stage 3 specifically
7. **Recommended implementation strategy** — one of: extend existing `emgenvelope.py`; create a new dedicated `tensionbudget.py`; create reusable shared modules plus a thin app wrapper; refactor acquisition/processing boundaries first. Must be justified by verified code structure, not guesswork.
8. **Risks / unknowns** — where implementing TensionBudget may break existing behavior (app launching, CSV compatibility, LSL consumers, real-time performance, replay tooling)
9. **What should be implemented next**

Only after I approve this report should you begin Phase 2.

---

## Phase 2 — Stage 1: raw acquisition to laptop CSV

Implement Stage 1 as the stable raw-data layer, using the actual files and structure confirmed in Phase 1 (not the provisional names from the context section).

**Requirements**
- Standardize serial acquisition for Arduino Uno R4 at 230400 baud.
- Standardize 500 Hz sample handling, including how timestamps/counters are reconstructed and recorded.
- Preserve raw channel values exactly as received — no preprocessing at this stage.
- Structured, analysis-friendly CSV output. The firmware sends 6 raw channel slots per packet regardless of wiring, so the schema must reflect all 6, e.g.:
  `sample_index, timestamp, ch1_raw, ch2_raw, ch3_raw, ch4_raw, ch5_raw, ch6_raw`
- As part of this phase, confirm and record which raw channel indices correspond to the left and right trapezius electrodes once wired (do not assume it's channels 1 and 2), and note the remaining channels as unused/floating in the session manifest rather than silently dropping them from the schema.

**Design goals:** deterministic logging, easy offline replay, live/offline compatibility, clean separation of raw vs. derived data.

**If needed**, capture metadata (sampling rate, baud rate, board type, channel count, session ID, recording start time) via file header comments, sidecar JSON, or a session manifest.

**Report:**
- actual raw CSV schema
- whether timestamps are acquisition, receive, or reconstructed timestamps
- whether packet drops / irregular intervals can be detected
- exactly what a user gets at the end of Stage 1

---

## Phase 3 — Stage 2: laptop preprocessing pipeline

Implement preprocessing as a reproducible pipeline that works both live and offline (on previously recorded CSVs).

**Required outputs, from raw CSV/stream:**
1. raw-to-voltage conversion (if appropriate)
2. filtered EMG signal
3. rectified EMG signal
4. RMS envelope
5. normalized-output placeholder for later %MVC / %RVE calibration

**2A. Raw interpretation** — determine whether volt/mV conversion belongs in preprocessing or stays optional. If ADC reference/scaling isn't guaranteed from incoming data, make the conversion layer configurable, not assumed.

**2B. Filtering** — support bandpass filtering for EMG and optional notch filtering for mains interference, with parameters in a central config. Default starting band: **20–240 Hz, 4th-order Butterworth**, applied via `filtfilt` for zero-phase — this is the literature-correct band (De Luca 1997 / Farina et al. 2002) truncated to fit under the 250 Hz Nyquist limit at 500 Hz sampling, with margin for numerical stability. Do not use the vendor demo filter (74.5–149.5 Hz from `EMGFilter.ino`) — that's a narrow visualization-only band, not appropriate for fatigue/spectral work. Keep it config-driven, not hardcoded, so it can still be tuned after real signal validation — but 20–240 Hz is the correct default, not an arbitrary placeholder. Apply independently to left and right channels.

**2C. Rectification** — explicit full-wave rectification as its own step, per channel.

**2D. RMS envelope** — rolling RMS with configurable window length and update step, per channel. Default: 100 ms window (50 samples at 500 Hz), updated every 10 samples (20 ms) — this is Marker & Maluf's (2016) exact specification and also matches Chords-Python's own existing `emgenvelope.py` default (`int(0.1 * sampling_rate)`), so it's consistent with both the literature and the codebase you're extending.

**2E. Processed output** — usable live (plotting), saved to processed CSV, and directly consumable by Stage 3. Use explicit left/right naming once Phase 1 confirms the real raw channel mapping, e.g.:
`sample_index, timestamp, ch_L_raw, ch_L_filtered, ch_L_rectified, ch_L_rms, ch_R_raw, ch_R_filtered, ch_R_rectified, ch_R_rms`

**Architecture preference:** reusable modules, e.g. `signal_processing/emg_pipeline.py`, `signal_processing/filters.py`, `signal_processing/features.py` — adjust this layout if Phase 1's findings suggest a cleaner fit within the real repo's existing structure. Do not bury logic inside a UI script.

**Report:**
- exact preprocessing stages implemented
- all configurable parameters and defaults
- raw vs. processed data structures
- whether the pipeline works live, offline, or both
- exactly what the user gets at the end of Stage 2

---

## Phase 4 — Calibration layer for normalized interpretation

Add a calibration subsystem so Stage 3 doesn't operate only on raw RMS values.

**Goal:** support a session-start reference-contraction workflow to compute a normalization denominator — Reference Voluntary Exertion (preferred), with MVC-style reference configurable later.

**Requirements**
- calibration mode/session step
- capture **two separate** short reference recordings per session — one calibration shrug for the left side, one for the right. Left and right trapezius do not share a baseline; a single shared reference will misrepresent whichever side wasn't calibrated against.
- compute reference RMS independently for each channel
- store calibration results in a structured way, keyed by channel/side (not a single scalar for the session)
- expose normalized streams such as `%MVC_L`/`%RVE_L` and `%MVC_R`/`%RVE_R`

**Important:** keep this modular and optional — I may test Stage 3 on unnormalized RMS before enforcing calibration every session.

**Report:**
- how calibration is triggered
- what file/state stores the calibration result
- what normalized outputs are produced
- exactly what the user gets at the end of this phase

---

## Phase 5 — Stage 3 core analytics: rule-based physiological metrics

Build the first analysis layer as rule-based features and state logic — not ML-first. Every metric below (5A–5D) runs independently on left and right channels — do not merge or average left/right before computing these; keep them as separate per-side outputs until the fusion step in Phase 6.

**5A. Relaxation / active mask** — detect active vs. resting muscle state; produce a boolean/categorical active-rest mask over time, per channel.

**5B. Gap detection** — detect relaxation gaps / micro-rests; log start, end, duration, and frequency, per channel.

**5C. APDF metrics** — amplitude distribution metrics over normalized activity: APDF-10, APDF-50, APDF-90, per channel. Make the analysis window configurable (session-wide, rolling, or fixed-interval).

**5D. SUMA-style sustained-activity detection** — thresholded runs over the normalized signal, with event extraction and duration logging, per channel. Don't reduce this to a single binary alarm — I want event-level outputs with duration bins so short vs. long events can be weighted differently later.

**5E. Session summaries** — total active time, total gap time, gap count, APDF values, sustained-activity event table, per-channel summaries, plus a derived left/right asymmetry summary (e.g. difference in RMS, APDF, or active-time between sides over the session) computed from the two channels above — this asymmetry output is the "3rd channel" referenced elsewhere in project notes; it does not need its own acquisition path, only a computation step here.

**Architecture requirement:** analytics exportable as Python objects/dicts, CSV/JSON summaries, and optionally UI-readable status blocks.

**Report:**
- what metrics are now computed
- exact formulas/logic used
- what files receive the outputs
- exactly what the user gets at the end of Stage 3 core analytics

---

## Phase 6 — Tension Budget scoring engine

Build a first-pass composite score combining:
- muscle activation exposure
- sustained-activity burden
- relaxation scarcity
- optional future fatigue-marker hooks

**Requirements**
- score must be explainable, not black-box
- weights/thresholds centralized in config
- output includes both a scalar score and the components/reasons behind it
- **explicit left/right fusion rule required.** All Phase 5 inputs exist as separate left/right values — pick and implement one fusion approach (e.g. `f(max(left, right))` for worst-side-drives-alert, `f(mean(left, right))` for average load, or folding the Phase 5E asymmetry summary in as its own weighted term), document why it was chosen, and keep it swappable via config rather than hardwired into the scoring function.

**Desired outputs:** current budget remaining, risk level, contributing factors, event log, human-readable explanation payload.

**Important:** this is a first implementation — design for revision, don't hardwire assumptions that block later tuning.

**Report:**
- exact scoring formulation implemented
- configurable parameters
- outputs exposed to UI/logs/files
- exactly what the user gets at the end of this phase

---

## Phase 7 — Optional spectral fatigue hooks

Add a modular placeholder or optional implementation for spectral fatigue metrics (Median/Mean Frequency trends), using raw or minimally processed signal segments rather than the RMS envelope — specifically the pre-rectification, bandpass-filtered (20–240 Hz) signal, computed per channel using Welch's method in short windows.

**Requirements:** keep optional, triggerable conditionally, and never let this complicate the core acquisition/preprocessing path. Because the 20–240 Hz band is truncated relative to the full De Luca/Farina 20–450 Hz convention (a consequence of the fixed 500 Hz sample rate and its 250 Hz Nyquist limit — do not attempt a 450 Hz upper cutoff at this sample rate, it's mathematically invalid), treat MDF/MNF output as an internal trend/slope indicator per channel (is it dropping over the session?), not as directly comparable to published absolute MDF baselines. State this limitation in code comments/docs rather than letting it surface later as an unflagged discrepancy.

**Report:**
- whether implemented fully or stubbed cleanly
- where it plugs into the pipeline
- what future work remains

---

## Phase 8 — UI / app integration into Chords-Python

Integrate into the existing app-launching structure.

**Requirements**
- register the new TensionBudget app in UI/config if appropriate
- expose meaningful run modes: raw acquisition, preprocessing view, analytics view, calibration mode
- don't break existing apps

**Preferred result:** a dedicated `tensionbudget.py` app using shared processing modules, runnable in live-stream or offline-replay mode.

**Report:**
- UI/config files changed
- how the new app is launched
- what modes the user can now run

---

## Phase 9 — Validation, test harness, and documentation

**Requirements**
- offline replay on sample CSV files
- sanity checks for sample-rate consistency
- basic detection of missing data / malformed rows
- unit-testable processing functions where practical
- concise developer documentation for the new pipeline

**Report:**
- test strategy
- what validations were added
- how to run the pipeline end-to-end
- known limitations

---

## Expected outputs at a system level

**Stage 1:** raw CSV/log stream from hardware to laptop, stable channelized recording, inspectable acquisition metadata.

**Stage 2:** processed EMG signal stream — filtered / rectified / RMS values, optional normalized values after calibration.

**Stage 3:** APDF metrics, relaxation-gap metrics, sustained-activity metrics, Tension Budget score, exported summaries suitable for future model development.

---

## Final preference

- Keep Chords-Python as the acquisition and app framework base — once its real structure is confirmed, not assumed.
- Add a new dedicated TensionBudget pipeline on top of it.
- Reuse existing EMG-related code where useful, but don't force-fit everything into `emgenvelope.py` if the verified code suggests a cleaner modular path.

**Start with Phase 1 only. Stop after the verification report — do not proceed to Phase 2 without my go-ahead.**


# TensionBudget — Session Logging Add-on

Paste this **in addition to** the master implementation prompt.

---

### Session logging instruction

In addition to reporting back to me in chat after each phase, maintain a single persistent session log file on disk that accumulates entries across the entire project. Do not create a new log per phase — append to the same file every time.

**Log file**
- Create `logs/session_log.md` at the project root the first time you log (create the `logs/` folder if it doesn't exist).
- Never overwrite or truncate this file. Every phase adds a new entry underneath the previous ones.

**When to log**
Write a new entry to this file immediately after completing each phase — right alongside your chat report to me. Logging is not optional and is not a substitute for the chat report; both happen, every phase, no exceptions.

**What each entry must contain**

Use this exact structure per entry:

```
## Phase <number> — <phase name>
Timestamp: <ISO 8601 timestamp>
Status: <Completed | Partially completed | Blocked>

### Summary
<2-4 sentence plain-language summary of what was actually done>

### Files created
- `<path>` — <one-line purpose>

### Files modified
- `<path>` — <one-line description of the change>

### Key functions/classes added or changed
- `<name>` in `<file>` — <what it does>

### Assumptions / open questions
- <anything unresolved or assumed>

### Next phase
<what should happen next, and any blockers>
---
```

If a section has nothing to report (e.g. no files created that phase), write "None" rather than omitting the heading — I want a consistent, scannable log across all phases, not a variable-shaped one.

**Why this matters**
This log is my audit trail for the whole implementation. I need to be able to open one file at any point and reconstruct exactly what changed, when, and why, without re-reading the whole chat history or diffing the repo myself.

**Confirmation**
Show me the log file path and the first entry once Phase 1 is logged, so I can confirm the format before you continue.