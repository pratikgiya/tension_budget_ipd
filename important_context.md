# TensionBudget — Full Project Context (paste this into a new chat to resume)

Real-time trapezius muscle fatigue detection/monitoring wearable. B.Tech-level research project, intent to eventually publish. Uses Arduino Uno R4 + BioAmp EXG Pill sensors, built on a modified fork of Chords-Python (upsidedownlabs). Implementation is being done via an agentic coding tool called Antigravity; Claude's role has been reviewing/correcting Antigravity's plans and session logs, not writing code directly — instructions to Antigravity are drafted by Claude, pasted in by the user (Mohit).

---

## 1. Hardware — locked decisions

- **Board**: Arduino Uno R4 **Minima** (not WiFi variant — Minima has zero radio hardware; even the R4 WiFi's onboard ESP32-S3 is not used by the current firmware, which is USB-serial only).
- **Sample rate**: 500 Hz. Verified from actual firmware source (`#define SAMP_RATE 500.0` in `UNO-R4.ino`). Considered switching to 1000 Hz (would enable true 20–450 Hz De Luca/Farina-standard bandpass, verified technically feasible — one-line firmware change + one-line change to `chords_serial.py`'s board table, serial baud has headroom). **Decided against it** — real-time detection goal doesn't need it; core scoring (APDF/SUMA/EIndex) is amplitude-domain and rate-independent; MDF is used as trend-only, not absolute-value comparison. Cons of switching also documented (storage doubles, breaks comparability with already-collected 500Hz pilot data and with the NeckCheck benchmark dataset).
- **Baud rate**: 230400 (verified: `#define BAUD_RATE 230400`).
- **Connection**: USB serial only. **Wireless (BLE/WiFi) confirmed out of scope** — Minima has no radio; `chords_ble.py`/`chords_wifi.py` in Chords-Python are built for a different board entirely (NPG-Lite, UDL's dedicated wireless product, 3-channel, 12-bit ADC, not compatible with this setup).
- **ADC**: 14-bit (`analogReadResolution(14)`).
- **Firmware channel behavior**: `NUM_CHANNELS = 6` is hardcoded in `UNO-R4.ino` and always transmits 6 channel slots per packet, **regardless of how many electrodes are physically wired**. Never assume raw channel count = electrode count.
- **Electrode plan**: Current single-device setup (3 electrodes: one right, one left, one ref at neck) is actually a **cross-body differential channel** (right minus left), not two independent channels — flagged as not comparable to literature single-muscle thresholds. Interim fix recommended: wire both current electrodes onto one side only (clean single-muscle signal) until upgrading.
  - **Planned upgrade**: 2× BioAmp EXG Pill boards, 5 electrodes total (2 left bipolar, 2 right bipolar, 1 shared reference at neck) → 2 true simultaneous channels. Shared single REF across two Pill boards confirmed as the vendor's own documented method.
  - There is **no 3rd physical channel**. Any "3rd channel" mentioned in early project docs is a **derived** left/right asymmetry metric computed downstream, not a raw ADC channel.
- **Hardware amplification confirmed real**: BioAmp EXG Pill does genuine analog amplification (instrumentation amp, TL074-based) before the ADC, gain fixed via resistor R6, exact multiplier not publicly published. This doesn't require any math changes — gain cancels out in the %RVE normalization ratio (same gain in numerator and denominator).
- **Vendor's separate demo filter** (`BioAmp-EXG-Pill/software/EMGFilter/EMGFilter.ino`, band 74.5–149.5 Hz) is a **different repository**, never used in this pipeline — don't confuse with the actual bandpass filter (see below).

## 2. Software stack

Modifying/extending **Chords-Python** (github.com/upsidedownlabs/Chords-Python), not building from scratch. Verified directly from pip-installed source (`chordspy` v0.2.0):
- `chordspy/emgenvelope.py` (original, pre-modification) used a single high-pass filter at 70 Hz, 4th-order Butterworth, via `scipy.filtfilt` (already zero-phase). RMS window was already `int(0.1 * sampling_rate)` = 100 ms default — a lucky coincidence that already matched the Marker & Maluf spec.
- `chordspy/chords_serial.py` board table: `"UNO-R4": {"sampling_rate": 500, "Num_channels": 6, "resolution": 14}`.
- `Chords-Arduino-Firmware/UNO-R4/UNO-R4.ino` (separate firmware repo): confirmed `SAMP_RATE=500.0`, `BAUD_RATE=230400`, `NUM_CHANNELS=6`, `HEADER_LEN=3`, `PACKET_LEN=16 bytes`, uses `FspTimer` (real hardware timer on the Renesas RA4M1 chip). No onboard filtering — raw passthrough only; all filtering happens laptop-side in Python.

## 3. Pipeline math — locked and cited, per stage

All thresholds below are locked and tested unless marked otherwise. L/R = computed independently per side unless noted.

- **Stage 0 (Acquisition)**: 500 Hz, 14-bit, 6 raw channel columns always present (firmware behavior above); only 2 real once upgraded, rest unused. Actual raw channel **index** mapping (which raw channel = left vs right) still requires physical hardware confirmation — never assumed.
- **Stage 0.5 (Board config)**: bridge solder pads below "bandpass" label on both EXG Pill boards for EMG/ECG mode — vendor recommended, doesn't lock a specific digital filter band.
- **Stage 1 (Voltage conversion)**: `V(t) = (raw(t)/16383) × Vref`, per side.
- **Stage 2 (Bandpass filter)**: **20–240 Hz**, NOT 20–450 Hz (De Luca/Farina's literal spec is invalid at 500 Hz sampling — Nyquist = 250 Hz; 450 Hz needs ≥1000 Hz sampling). 4th-order Butterworth, **SOS** (cascaded biquads, avoids float coefficient drift at high order), applied independently L/R.
  - Offline: `sosfiltfilt` (zero-phase, non-causal, needs full signal).
  - Live: `sosfilt` + persistent `zi` state across chunks (causal, real phase delay — the only version usable for streaming). Both implemented (`StreamingChannelProcessor` class), sharing the same downstream analytics functions.
- **Stage 3 (Notch filter)**: conditional, NOT default. Inspect spectrum first; only apply 50/100/150 Hz if a real mains peak is confirmed. Check L/R independently. Verified working via standalone test (`test_notch.py`): 43.13 dB attenuation at 50 Hz, 0.00 dB effect at 100 Hz (muscle proxy) — independently proven, not just inferred from a live-vs-offline comparison (which would've been circular, since both paths share config).
- **Stage 4 (Rectification)**: full-wave, `|x_filtered(t)|`, per side.
- **Stage 5 (RMS envelope)**: window = 50 samples (100 ms @ 500 Hz), step every 10 samples (20 ms, 80% overlap) — Marker & Maluf (2016) exact spec. Live implementation is an O(1) rolling sum. Per side independent.
- **Stage 6 (Calibration + normalization)**: 5 s reference hold per side, keep **middle 3 s only** (trims ramp-up/relaxation transients) → `EMG_RVE = RMS(middle 3s)`. `%RVE(t) = EMG_RMS(t)/EMG_RVE × 100`. **Left and right do NOT share a calibration reference** — separate shrug required per side. RVE preferred over MVC (Mathiassen 1995 — avoids repeated max-effort in occupational subjects). `calibrate.py` hard-errors if `CHANNEL_MAP` isn't set (fixed from an earlier version that silently defaulted to left=0/right=1 — a real bug).
- **Stage 7 (Gap/rest detection)**: `%RVE(t) < 3.0%` for `≥0.125s` = gap (Marker & Maluf 2016; corrected from an earlier wrong draft value of 5%/0.25s). Gap freq = count/min. Muscular rest % = sum(gap durations)/total time × 100. **Different threshold from Stage 9/11's rest bucket (0.5% MVE, Koch)** — kept as explicitly separate, distinctly named config values: `GAP_REST_THRESHOLD_PCT` (3.0%) vs `EINDEX_REST_THRESHOLD_PCT` (0.5%). This naming collision was a real bug caught and fixed.
- **Stage 8 (Active + Session APDF)**: strip <3% RVE samples first, then percentile (10th/50th/90th) = Active APDF (drives live scoring). Session APDF = same calc without stripping (retrospective/literature comparison only). Both computed separately L/R → 4 percentile sets total. Jonsson (1982) safety bands: Static(P10) <2–5% MVC, Median(P50) <10–14% MVC, Peak(P90) <50–70% MVC (continuous hourly work). On-device fallback (P² algorithm, Jain & Chlamtac 1985) flagged as contingency only, not currently needed (analytics run laptop-side).
  - Bug found & fixed: `compute_apdf(active_only=True)` could crash on an empty array (a fully-resting live buffer — a normal, expected scenario, e.g. right after calibration or during a break) — fixed with an explicit NaN-return guard, verified via dedicated test.
- **Stage 9 (SUMA state machine)**: continuous activity `>0.5% MVE` lasting `>1.5s`. Bins (Koch 2024 exact): 1.5–5s, 5–10s, 10–20s, 20–60s, 1–2min, 2–4min, 4–8min, 8–10min, 10–20min, >20min. Per side → 2 histograms. Boundary tests (exact 5.0s/60.0s/1200.0s edges) verified correct, consistent `[lower, upper)` convention. **Novelty**: weight by SHORT-bin FREQUENCY, not just total duration (Koch 2025 finding — short-event frequency predicts pain better than duration, especially in women). This is the project's stated core novelty and repeatedly got dropped during Phase 6 iterations (see section 5).
- **Stage 10 (Conditional MDF)**: Welch's method, **pre-rectification**, 20–240 Hz filtered signal, 0.5–2s windows, per side. Equal-energy split for MDF; MNF as standard companion metric. Slope via OLS over session time. Healthy baseline ~80–120 Hz → ~60–70 Hz = fatigue signature (Farina 2002). **Explicit caveat required in methodology**: band truncated to 240 Hz vs. the De Luca/Farina 450 Hz convention → report MDF as an internal trend/slope indicator only, not equivalent to published absolute baselines.
- **Stage 11 (Scoring)**: see Section 5 below — in progress, not finalized.
- **Stage 12 (Asymmetry/Laterality Index)**: not literature-cited, project's own design choice. **Locked to the already-implemented, tested version**: `AI = (APDF50_R − APDF50_L) / (APDF50_R + APDF50_L)` — signed, bounded [-1,+1], based on Active APDF-50. A different unsigned version (`|Score_L−Score_R| / (0.5×(Score_L+Score_R)) × 100`) appeared in a later audit doc but was explicitly rejected in favor of keeping the tested version, since switching would silently invalidate the already-calibrated asymmetry-penalty threshold.
- **Stage 13 (L/R fusion rule)**: "**Worst-Side-Drives**": `Composite_Score = min(Score_Left, Score_Right)`. Asymmetry penalty: subtract 5.0 points if `|AI| > 0.5`. Floor clamp: `max(0.0, Composite_Score)` applied **last**, after the asymmetry subtraction. Compounding decisions **explicitly documented as intentional**: (a) EIndex/exposure term + short-event-frequency term compound (orthogonal risk factors per Koch 2025 — one measures proportion of loaded time, the other measures fracturing pattern); (b) Worst-Side-Drives + asymmetry penalty compound (worst side drives the base, asymmetry adds spinal-torque risk on top).

## 4. Session log — actual implementation status (Antigravity)

- **Phase 1** (repo verification): Complete. Found and will fix a pre-existing broken `tensionbudget/` dir from a prior attempt (broken import, bandpass 450Hz>Nyquist, RMS window 200ms) in Phase 2.
- **Phase 2** (Stage 1, raw acquisition→CSV): Complete. `recorder.py`, fixed config, exposed Arduino packet counter (was discarded before), sample-callback system, Flask routes. `CHANNEL_MAP` deliberately left `None` pending hardware test.
- **Phase 3** (Stage 2, preprocessing): Complete. `preprocessing.py` — bandpass, rectify, RMS. 11 tests passed on synthetic data.
- **Phase 4** (Calibration): Complete. `calibration.py` + `calibrate.py` CLI, separate L/R, middle-3s extraction, auto-loads calibration JSON into session processing.
- **Phase 5** (Stage 3 analytics + live preprocessing): Complete, with a full hardening round. `analytics.py` (gaps, SUMA bins, APDF split, Laterality Index) + `preprocessing.py` refactored to SOS filtering with a `StreamingChannelProcessor` for live mode (shares the same analytics functions as offline — not duplicated logic). Bugs found and fixed: empty-array APDF crash, test-isolation global-config-mutation bug, missing SUMA boundary tests, weak live-vs-offline check upgraded to Pearson correlation (0.9278 — judged acceptable given the system's own timescales, not millisecond-critical), missing standalone notch-filter test (added, verified 43.13dB attenuation).
- **Phase 6** (Composite Tension Budget Score): **In progress, not complete.** See Section 5 for full history and current blockers.
- **Zero real-hardware validation** across everything above — `pyserial` not installed in dev environment, `CHANNEL_MAP` genuinely unresolved (can only be resolved by physical testing). Hardware smoke test still deferred (user doesn't currently have hardware in hand). Decision made: keep building Phase 6/7 (hardware-independent, synthetic-data-testable) but do **not** start Phase 8 (live app/actual serial integration — requires real hardware by definition) until hardware arrives.

## 5. Phase 6 — full history, why it matters, current state

Went through many iterations, each fixing a real bug in the previous one:

1. **v1**: absolute point-deduction table per SUMA bin, event-triggered (lump sum at event completion). Rejected — step-function/alert-fatigue UX problem, plus a verified math bug: cumulative drain saturates the 100-point budget by ~6.35 minutes, making 3 of the 10 bins (8–10min, 10–20min, >20min) permanently unreachable.
2. **v2**: continuous accrual (dt-based rate) instead of lump sum — fixed the event-trigger problem, but kept a discrete per-bin rate table, so the same saturation issue persisted via a different mechanism; gap restoration (+2.0pts/sec flat) also massively outweighed early-bin drain, causing near-binary score behavior.
3. Explored budget-*proportional* exponential decay (`dB/dt = -k(t)×B`) — this was Claude's proposal, but the user clarified this was a misread: they wanted drain-rate proportional to the **severity of the detected fatigue signal itself**, not proportional to remaining budget.
4. Corrected to a continuous power-law severity function (`rate(t_active) = c×(t/τ)^p`) — smooth, no bin-boundary cliffs, verified via simulation to avoid premature saturation.
5. **Ideation session** (user-requested "let's ideate") explored real alternative model families: directly running Koch's EIndex as-is; the **Three-Compartment Controller (3CC/3CCr)** model (Xia & Frey-Law 2008 — verified real, published, purpose-built muscle fatigue/recovery compartmental model with Active/Fatigued/Resting states and F/R rate constants — flagged as the strongest available option at the time); Banister's sports-science fitness-fatigue impulse-response model (cross-domain import); a discrete hysteresis state-machine (Fresh/Elevated/Fatigued/Critical) as an easier-to-validate alternative to a continuous score.
6. **Critical check-back** (user asked "what did our original math actually suggest us to do?"): going back to source docs revealed that **every single idea explored in steps 1–5 had drifted from the original spec**, which from the very first literature pass said: use Koch's EIndex (-2/+1/+2 weights) *directly* as the base skeleton, summed cumulatively, **plus a short-event-frequency term** (the actual stated novelty, citing Koch 2025) — this frequency term had been **missing from every iteration**, despite being flagged as the core novelty contribution since the earliest literature-extraction pass in this project.
7. **Reset**: base term = Koch's EIndex computed directly (Rest<0.5%/Low 0.5–7%/High>7% buckets, `-2×P_Rest+P_Low+2×P_High`); frequency term = separate additive component counting short SUMA events (1.5–5s, 5–10s bins), kept distinct in output, not folded into the base term. Continuous-accrual and drain/restore-consistency learnings from the detour were kept (still valid regardless of which formula is used); the discarded curve ideas (power-law, 3CC) were preserved as a documented "Alternatives Considered" methods-section note, not deleted.
8. **Bug found in the reset plan**: "rolling window" was ambiguous — if EIndex is computed on a sliding window but *literally summed* every few seconds, this causes unbounded divergence (same activity double/triple/hundred-counted across heavily overlapping windows). **Verified via simulation**: under constant moderate activity alone, literal summing hits 100 by ~8.3 minutes and reaches **5760** by the end of an 8-hour session. This exact bug was independently confirmed by a separate audit document the user later provided (good convergent validation — treated as fully confirmed, not just suspected).
9. **Current fix (locked, sent to Antigravity)**: split into two separate outputs —
   - `eindex_live`: sliding 10-min window, recomputed every few seconds, uses the **current** value directly each time, **never summed across recomputes**. Drives the real-time score. Bounded to [-2,+2] by construction at all times.
   - `eindex_session_cumulative`: **true discrete, non-overlapping** 10-min windows (matches Koch's actual published method exactly), summed only across these. Updates once per real 10-min boundary (intentionally flat between boundaries — it's an end-of-session report-card statistic, not fed into the live score).
   - Regression test specified: simulate a full 8hr session at constant moderate activity, assert `eindex_live` stays within [-2,+2] throughout and `eindex_session_cumulative` only changes in discrete 10-min steps.
   - Note: an earlier version of the message to Antigravity cited weak "evidence" for this bug (a single JSON example value of 3.4 being "outside per-window range" — not actually valid proof, since a cumulative field is *supposed* to exceed per-window bounds once summed). Corrected to cite the full divergence trajectory (12→24→48→100→360→720→5760 over 8hrs, no plateau) instead.

### Still open / blocking, unresolved as of the latest turn
- **#1 open blocker**: the short-SUMA-frequency term has a stated contradiction — spec text says "capped at +1.0 max (10 events × 0.1 each)" but the plan's own illustrative JSON example shows `"short_suma_freq_penalty": 5.0`, 5× over its own stated ceiling. Never resolved whether this is meant to be a live per-window value (bounded 0–1.0) or session-cumulative (no real ceiling). Needs explicit resolution before implementation proceeds.
- **STAMI S6_Dataset.csv check** (Section 5.2 of the project's own dataset review) — still pending, needed to anchor the EIndex→0-100 display scale mapping instead of guessing it.
- **Do not use**: a "Leaky-Integrator Budget" formula (`Bn = Bn-1 - α·Drain + β·Recovery`) appeared in one audit document's "flagged, not verified" section. Traced through the whole conversation — does not match anything actually agreed here (closest relative is Claude's own earlier proportional-decay idea, which the user explicitly overruled). Treat as unverified/likely-misremembered until its actual source is found.
- **Where fusion/asymmetry sit in the pipeline** (last question asked, not yet answered before this context dump was requested): both L and R run through the *entire* pipeline (Stages 1–10, plus each side's own EIndex/frequency scoring) completely independently, in parallel, with zero mixing. The fusion rule (`min(L,R)`) and the asymmetry penalty are the **very last step** — they only run after each side already has its own complete, independent score. Asymmetry itself is a comparison *between* two already-fully-computed values (APDF-50 for L and R), not a channel of its own and not computed earlier in the pipeline.

## 6. Pilot study plan (human data collection)

- **12–15 participants** (upgraded from an earlier "5–10" plan), varied conditions/duration, tracking **age and gender** as factors — directly motivated by Koch et al. 2025's own finding of sex differences in EMG gap patterns (already cited in the project's literature review, ref 26).
- Protocol: extended sessions of simulated/replicated office work; participant pool partly drawn from an actual office-desk-work age demographic.
- **Periodic self-report** (1–10 tension/discomfort rating) at fixed intervals (10–15 min) — recommended by the LLM Council (Section 7) and formally adopted. **Still needs to be added to the actual written pilot protocol** — flagged multiple times, not yet done.
- **Ethics/consent**: flagged as needing verification with the institution before starting, since this is now real human-subjects data collection intended for eventual publication. **Not yet confirmed.**
- **Purpose**: (1) preliminary ground-truth signal to correlate the composite score trajectory against self-reported perceived exertion — upgrades the publication claim from "we guessed weights" to "literature-anchored weights with preliminary validation"; (2) legitimate light calibration of the small number of free parameters in the scoring model (not a full model-fit — n=12–15 can constrain ~2–4 parameters, not a full multi-factor interaction model).
- **Two-stage plan**: v1 = literature-anchored placeholder constants, shipped now, used *during* the pilot study; v2 = refit the small parameter set *after* the study using real self-report + score-trajectory correlation data.
- **Statistical honesty required**: n=12–15 has no power for full age×gender×duration interaction effects — treat as exploratory only. Protocol needs to be fixed and written down before starting (task type, session duration, break schedule) for reproducibility.

## 7. LLM Council session (ran once)

Convened on the question "how should we decide Phase 6's scoring coefficients" (trial-and-error vs RL vs self-updating vs something else). 5 simulated personas + peer review + chairman synthesis (no live sub-agent tool available, so run as a single-pass simulation). **Verdict**: reject RL/self-updating/blind trial-and-error at this data scale (no ground-truth signal to learn against). Recommended: freeze constants as v1, explicitly labeled as "expert-elicitation heuristic, not data-fitted" (legitimate precedented methodology, same category as NIOSH's lifting equation or RULA/REBA); validate the function's *structure* (monotonic, bounded, ordinally consistent with cited literature), not its exact numeric precision. Recommended adding cheap self-report capture to pilot sessions — this recommendation directly led to the pilot study plan being upgraded (Section 6). This council verdict was later partially superseded once the 12–15-person + self-report pilot plan was finalized — the two-stage v1/v2 plan in Section 6 is the updated guidance. Output files: `council-report-20260709.html` + `council-transcript-20260709.md`.

## 8. Key corrections / "don't repeat these mistakes" list

1. Vendor's 74.5–149.5Hz `EMGFilter.ino` demo filter is a different repo, never used here.
2. 20–450Hz bandpass is invalid at 500Hz sampling (Nyquist=250Hz) — use 20–240Hz.
3. Firmware always emits 6 channels regardless of electrode count — never assume raw channel count = electrode count.
4. Never assume channel index 1=left/2=right without physical hardware confirmation.
5. Left and right trapezius do not share a calibration baseline.
6. Two different "rest" thresholds exist (Marker&Maluf 3.0% for gaps, Koch 0.5% for EIndex buckets) — must stay distinctly named in config.
7. Current single-device 3-electrode wiring gives a cross-body differential signal, not two independent channels.
8. `compute_apdf(active_only=True)` needs explicit empty-array/NaN handling — fully-resting live buffers are normal, not an edge case.
9. Never mutate shared config classes globally in tests — use local subclass/override.
10. Event-triggered (lump-sum) scoring recreates the alert-fatigue/step-function problem — must be continuous accrual.
11. Multiple Phase 6 iterations had cumulative-sum/saturation bugs — verify via simulation, don't assume a formula is bounded just because it looks reasonable.
12. The short-SUMA-frequency term (the actual project novelty) was missing from every Phase 6 design until caught by checking back against original source docs — check back against `correct_math_flow.md`/`paper_math_tension_budget.md`/`initial_memory.md` at the *start* of each new design round, not just when explicitly asked.
13. Frequency-term cap-vs-cumulative contradiction — still unresolved, top open item.
14. Wireless is not currently feasible with the current hardware+firmware+software combo without real hardware/firmware changes.
15. 1000Hz sampling is technically possible but was decided against.
16. Keep the tested signed asymmetry formula; don't switch to the unsigned variant that appeared in a later audit doc.
17. The "leaky-integrator" formula in one audit doc is unverified — don't use without finding its actual source.

## 9. Files known to exist in this project

- `paper_math_tension_budget.md` — literature math extraction
- `Untitled-document-1.docx` — full literature review (sections 2.1–2.4, gaps table, dataset section, refs 1–26)
- `correct_math_flow.md` — corrected end-to-end pipeline spec
- `paper_review.pdf` — project file, not deeply explored in this thread
- `TensionBudget_Pipeline_Corrected_Final_v2.md` — Claude-corrected pipeline doc (channel/calibration fixes)
- `tensionbudget_antigravity_prompt_v2.md` — Claude-corrected Antigravity prompt
- `implementation_plan.md` — Antigravity's own plan doc
- Multiple Antigravity session-log documents (Phase 1–5 completion reports)
- `council-report-20260709.html` + `council-transcript-20260709.md` — LLM Council outputs
- User's own "final reconciled math, bilateral" audit document — mostly confirmed the built pipeline, flagged the asymmetry-formula choice and the EIndex divergence root cause (both resolved above), and the unverified leaky-integrator formula (not resolved, don't use).

## 10. How this project has been worked (for whoever picks this up)

- `/caveman` tag used inconsistently by the user for compressed responses — appears to be per-message, not a persistent mode; respond normally unless the tag is present on that specific message.
- `/llm-council` used once, for high-stakes methodology decisions (5-persona simulation + peer review + chairman synthesis, output as HTML + markdown transcript).
- Strong preference for **verified, simulated numbers over assertions** — responds well to actual Python simulations proving claims (e.g. the 6.35min saturation calc, the 5760 divergence calc, the notch dB verification) rather than assertions alone.
- Standard practice now: when there's drift or disagreement about the math, **go back to the original source docs** as the arbiter of what's correct, rather than continuing to layer fixes on top of fixes.
- Claude's role throughout has been reviewing and correcting Antigravity's plans/session logs, drafting the exact instruction text the user pastes into Antigravity — not writing implementation code directly in this thread.