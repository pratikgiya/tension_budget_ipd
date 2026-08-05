# TensionBudget — Full Mathematical Pipeline Documentation

**Author:** TensionBudget Research Team  
**Document version:** Phase 13 / W7 (2026-08-05)  
**Purpose:** Professor-ready derivation of every algorithm and formula in the bilateral upper-trapezius sEMG ergonomic surveillance system, from hardware acquisition to cloud-persisted ML feature vectors.

---

## Table of Contents

1. [Signal Acquisition & ADC Conversion](#1-signal-acquisition--adc-conversion)
2. [Stage 2 — Digital Signal Pre-Processing](#2-stage-2--digital-signal-pre-processing)
   - [2.1 Bandpass Butterworth Filter (20–240 Hz)](#21-bandpass-butterworth-filter-20240-hz)
   - [2.2 Mains Interference Notch Filter (50 Hz)](#22-mains-interference-notch-filter-50-hz)
   - [2.3 Full-Wave Rectification](#23-full-wave-rectification)
   - [2.4 RMS Amplitude Envelope](#24-rms-amplitude-envelope)
   - [2.5 Reference Voluntary Exertion (%RVE) Normalisation](#25-reference-voluntary-exertion-rve-normalisation)
3. [Stage 3 — Amplitude Analytics](#3-stage-3--amplitude-analytics)
   - [3.1 Gap / Micro-Rest Detection](#31-gap--micro-rest-detection)
   - [3.2 SUMA — Sustained Muscle Activity Detection](#32-suma--sustained-muscle-activity-detection)
   - [3.3 APDF — Amplitude Probability Distribution Function](#33-apdf--amplitude-probability-distribution-function)
4. [Stage 4 — Composite Tension Budget Scoring](#4-stage-4--composite-tension-budget-scoring)
   - [4.1 Exposure Index (EIndex)](#41-exposure-index-eindex)
   - [4.2 Short-SUMA Frequency Penalty](#42-short-suma-frequency-penalty)
   - [4.3 Per-Side Combined Score](#43-per-side-combined-score)
   - [4.4 Bilateral Asymmetry Index](#44-bilateral-asymmetry-index)
   - [4.5 Composite Score — Bilateral Fusion & Floor Clamp](#45-composite-score--bilateral-fusion--floor-clamp)
   - [4.6 EIndex Accumulator — Live vs. Cumulative Split](#46-eindex-accumulator--live-vs-cumulative-split)
   - [4.7 STAMI Display-Scale Mapping](#47-stami-display-scale-mapping)
5. [Stage 5 — Spectral Fatigue Analysis (MDF / MNF)](#5-stage-5--spectral-fatigue-analysis-mdf--mnf)
   - [5.1 Pre-Rectification Signal Tap](#51-pre-rectification-signal-tap)
   - [5.2 Welch's Power Spectral Density Estimate](#52-welchs-power-spectral-density-estimate)
   - [5.3 Median Power Frequency (MDF)](#53-median-power-frequency-mdf)
   - [5.4 Mean Power Frequency (MNF)](#54-mean-power-frequency-mnf)
   - [5.5 Active-Epoch Gating](#55-active-epoch-gating)
   - [5.6 OLS Fatigue Slope](#56-ols-fatigue-slope)
   - [5.7 Fatigue Flag Decision Rule](#57-fatigue-flag-decision-rule)
6. [Stage 6 — Subjective Strain Elicitation (Borg CR-10)](#6-stage-6--subjective-strain-elicitation-borg-cr-10)
7. [Stage 7 — 5-Minute Epoch Feature Vector (ML Target)](#7-stage-7--5-minute-epoch-feature-vector-ml-target)
8. [Stage 8 — Cloud Persistence & Idempotent Upsert](#8-stage-8--cloud-persistence--idempotent-upsert)
9. [Full Pipeline Flow Diagram](#9-full-pipeline-flow-diagram)
10. [Parameter Reference Table](#10-parameter-reference-table)
11. [Literature References](#11-literature-references)

---

## 1. Signal Acquisition & ADC Conversion

### Hardware Setup
Two Arduino UNO R4 Minima microcontrollers, each paired with a BioAmp EXG Pill analog front-end, acquire bilateral trapezius surface EMG (sEMG) independently:

| Parameter | Value | Rationale |
|---|---|---|
| Sampling rate $f_s$ | **500 Hz** | Nyquist limit $f_{Nyq} = 250\ \text{Hz}$ safely above the 240 Hz passband ceiling |
| ADC resolution | **14-bit** (UNO R4) | $2^{14} = 16{,}384$ quantisation levels |
| Serial baud rate | **230,400 baud** | Sufficient to stream 6-channel packets at 500 Hz without buffer overflow |
| Channels per unit | 6 transmitted, **2 used** (A2) | LSL channel 0 = Left trapezius, channel 1 = Right trapezius |

### Raw-to-Voltage Conversion (Optional)

$$V_{mV}[n] = \frac{x_{raw}[n]}{2^{N_{bits}} - 1} \times V_{ref} \times 1000$$

where $N_{bits} = 14$, $V_{ref}$ is the ADC reference voltage in Volts, and $n$ is sample index. For ratiometric analyses (e.g., %RVE), raw integer counts are used directly — voltage conversion is not required.

> **Note:** The BioAmp EXG Pill contains its own differential instrumentation amplifier and analog gain stage before the ADC. Therefore $V_{mV}$ represents the amplified signal at the ADC input, not the raw skin-surface electrode potential.

---

## 2. Stage 2 — Digital Signal Pre-Processing

Each channel (Left, Right) is processed **independently** through a deterministic 6-stage pipeline. No cross-channel mixing occurs before the final asymmetry computation.

**Pipeline order (offline mode — zero-phase):**
$$x_{raw} \rightarrow x_{filtered} \xrightarrow[\text{pre-rect tap (Stage 5)}]{} x_{notch} \rightarrow x_{rect} \rightarrow x_{RMS} \rightarrow x_{\%RVE}$$

### 2.1 Bandpass Butterworth Filter (20–240 Hz)

**Purpose:** Remove sub-20 Hz motion artifact / DC offset and above-240 Hz noise and aliasing components, retaining the physiologically relevant sEMG band.

**Design:** 4th-order zero-phase Butterworth bandpass filter, implemented via `scipy.signal.sosfiltfilt` (offline) or `sosfilt` with persistent state $z_i$ (live streaming).

The Butterworth transfer function in the $z$-domain is determined by the design equation:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$

where $N = 4$ is the filter order and $\omega_c$ is the critical frequency. For the bandpass design, two normalised critical frequencies are defined:

$$\Omega_{low} = \frac{f_{low}}{f_s / 2} = \frac{20}{250} = 0.08, \quad \Omega_{high} = \frac{f_{high}}{f_s / 2} = \frac{240}{250} = 0.96$$

**Zero-Phase (offline):** Forward-backward filtering via `sosfiltfilt` achieves exactly **zero group delay**:

$$x_{filtered} = \text{filtfilt}\bigl(H_{BP},\; x\bigr)$$

This doubles the effective attenuation roll-off: a 4th-order design becomes effectively 8th-order in stopband rejection after forward-backward application (−24 dB/octave → −48 dB/octave).

**Causal (live):** `sosfilt` with persistent second-order-sections state $z_i$ processes incoming chunks without introducing edge artifacts. Introduces group delay of $\approx N/2 = 2$ samples at mid-band, which is negligible at 500 Hz.

**Parameters:**

| Parameter | Value | Source |
|---|---|---|
| $f_{low}$ | 20 Hz | De Luca (1997); Farina et al. (2002) |
| $f_{high}$ | 240 Hz | Nyquist-constrained: $240 < f_s/2 = 250$ Hz |
| Order $N$ | 4 | Standard EMG pre-processing |
| Architecture | Butterworth SOS | Numerically stable second-order section cascade |

> **Important Passband Limitation:** De Luca (1997) and Farina (2002) specify a 20–450 Hz passband, which requires $f_s \geq 1000\ \text{Hz}$. Our hardware is locked at 500 Hz, meaning signal power above 240 Hz is excluded. All MDF/MNF values computed downstream are systematically lower than published population norms and must be interpreted as **internal session trend indicators only** — not compared against published absolute baselines.

---

### 2.2 Mains Interference Notch Filter (50 Hz)

**Purpose:** Remove narrowband mains electrical interference at $f_{notch} = 50\ \text{Hz}$ (India/EU grid frequency).

**Design:** IIR notch filter (2nd-order all-pass minus unity) with quality factor $Q = 30$. The transfer function is:

$$H_{notch}(z) = \frac{1 - 2\cos(2\pi f_{notch}/f_s)\,z^{-1} + z^{-2}}{1 - 2r\cos(2\pi f_{notch}/f_s)\,z^{-1} + r^2 z^{-2}}$$

where the pole radius $r = 1 - \pi f_{notch} / (Q \cdot f_s)$ controls bandwidth. At $Q = 30$: bandwidth $BW = f_{notch}/Q = 50/30 \approx 1.67\ \text{Hz}$.

**Auto-detection:** In `NOTCH_ENABLED = "auto"` mode, the filter is applied only when a PSD spike at 50 Hz exceeds $5\times$ the median bandpass power, preventing unnecessary phase distortion when the environment is electrically quiet.

---

### 2.3 Full-Wave Rectification

$$x_{rect}[n] = |x_{filtered}[n]|$$

Full-wave rectification reflects negative half-cycles into the positive domain. This is a **mandatory preparatory step for the RMS envelope** (Stage 2.4) but is **explicitly bypassed for the spectral fatigue analysis** (Stage 5). Rectification introduces strong harmonic distortion at DC, $2f$, $3f$, etc. — if Welch PSD estimation were applied to $x_{rect}$, the harmonic energy would completely invalidate MDF/MNF estimates.

---

### 2.4 RMS Amplitude Envelope

**Purpose:** Smooth the rectified signal into a physiologically interpretable amplitude envelope representing muscle activation intensity at each moment.

**Formula (per sliding window of $W$ samples):**

$$\text{RMS}[k] = \sqrt{\frac{1}{W} \sum_{n=k \cdot S}^{k \cdot S + W - 1} x_{rect}[n]^2}$$

where:
- $W = 50\ \text{samples}$ (100 ms window at 500 Hz)
- $S = 10\ \text{samples}$ (20 ms step → 50% overlap between adjacent estimates)
- $k = 0, 1, 2, \ldots$ is the window index

The output array $\text{RMS}[\cdot]$ has length $M = \lfloor(N - W) / S\rfloor + 1$, producing a down-sampled envelope at an effective rate of $f_s / S = 500/10 = 50\ \text{samples/second}$.

**Parameters:**

| Parameter | Value | Source |
|---|---|---|
| Window $W$ | 100 ms = 50 samples | Marker & Maluf (2016) |
| Step $S$ | 20 ms = 10 samples | Marker & Maluf (2016) |
| Overlap | 80% ($1 - S/W$) | Standard EMG envelope |

---

### 2.5 Reference Voluntary Exertion (%RVE) Normalisation

**Purpose:** Convert absolute RMS amplitude (arbitrary ADC units) into a dimensionless percentage relative to each subject's individual maximum voluntary reference exertion, enabling inter-subject and inter-session comparisons.

**Calibration procedure:** At session start, the subject performs a 5-second maximal bilateral shoulder shrug. The mean RMS during this shrug constitutes the Reference Voluntary Exertion baseline $\overline{RMS}_{RVE}$ for each channel independently.

**Normalisation formula:**

$$x_{\%RVE}[k] = \frac{\text{RMS}[k]}{\overline{RMS}_{RVE}} \times 100\%$$

All downstream analytics (APDF percentiles, EIndex thresholds, gap detection, SUMA detection) operate on $x_{\%RVE}$ rather than raw RMS, making all thresholds hardware-independent and subject-normalised.

> **Calibration type:** `RVE` (Reference Voluntary Exertion) as distinct from `MVC` (Maximum Voluntary Contraction). The shrug manoeuvre targets the trapezius specifically, whereas MVC often refers to the global maximum across muscle groups.

---

## 3. Stage 3 — Amplitude Analytics

All functions in this stage operate on the %RVE envelope array $x_{\%RVE}[\cdot]$.

### 3.1 Gap / Micro-Rest Detection

**Physical meaning:** A *gap* (Veiersted et al. 1993; Mathiassen & Winkel 1996) is a brief episode of muscular rest. Adequate micro-rest frequency is the primary physiological buffer against cumulative Cinderella-fiber fatigue.

**Detection rule:** A contiguous run of consecutive RMS samples is classified as a gap if:

$$x_{\%RVE}[k] < \theta_{gap} \quad \forall k \in [k_{start},\, k_{end}]$$

with minimum duration:

$$\Delta t_{gap} = (k_{end} - k_{start}) \times \frac{S}{f_s} \geq 0.125\ \text{s}$$

| Parameter | Value | Source |
|---|---|---|
| Gap threshold $\theta_{gap}$ | **3.0% RVE** | Marker & Maluf (2016) |
| Minimum gap duration | **0.125 s** | Marker & Maluf (2016) |

> **Important distinction:** The gap threshold (3.0% RVE) is **different** from the EIndex rest threshold (0.5% MVE, Stage 4.1). These come from different papers and serve different biological purposes. They must not be merged.

**Gap frequency:**

$$f_{gap} = \frac{N_{gaps}}{T_{session}}\quad [\text{gaps/min}]$$

where $T_{session}$ is total session duration. Higher gap frequency → more adequate muscular recovery.

---

### 3.2 SUMA — Sustained Muscle Activity Detection

**Physical meaning:** SUMA (Sustained Muscle Activity) events represent periods of continuous trapezius activation without interruption, corresponding to the biomechanical mechanism driving Cinderella motor-unit overloading (Hägg 1991; Visser & van Dieën 2006).

**Detection rule:** A SUMA event is any contiguous run where:

$$x_{\%RVE}[k] > \theta_{SUMA} \quad \forall k \in [k_{start},\, k_{end}]$$

with minimum duration:

$$\Delta t_{SUMA} = (k_{end} - k_{start}) \times \frac{S}{f_s} \geq 1.5\ \text{s}$$

| Parameter | Value | Source |
|---|---|---|
| SUMA threshold $\theta_{SUMA}$ | **0.5% RVE** | Koch et al. (2024) — Cinderella-fiber recruitment threshold |
| Minimum SUMA duration | **1.5 s** | Koch et al. (2024) |

**Duration binning (Koch 2024 classification):**

Each detected SUMA event is assigned to one of 10 duration bins:

| Bin | Duration range | Role in scoring |
|---|---|---|
| `1.5-5s` | 1.5 – 5 s | **Short — contributes to frequency penalty** |
| `5-10s` | 5 – 10 s | **Short — contributes to frequency penalty** |
| `10-20s` | 10 – 20 s | Medium |
| `20-60s` | 20 – 60 s | Medium |
| `1-2min` | 60 – 120 s | Long |
| `2-4min` | 120 – 240 s | Long |
| `4-8min` | 240 – 480 s | Long |
| `8-10min` | 480 – 600 s | Long |
| `10-20min` | 600 – 1200 s | Very long |
| `>20min` | >1200 s | Very long |

---

### 3.3 APDF — Amplitude Probability Distribution Function

**Physical meaning:** The APDF (Jonsson 1978, 1982) characterises the statistical distribution of muscle loading amplitude across a work period. Its three canonical percentiles describe three distinct aspects of musculoskeletal exposure:

| Percentile | Name | Physical meaning |
|---|---|---|
| $APDF_{10}$ | Static level | The *minimum habitual load* — residual tonic activation even during nominally "rest" periods. Predicts postural fatigue. |
| $APDF_{50}$ | Median level | The *typical working amplitude* — the load level exceeded half the time. |
| $APDF_{90}$ | Peak level | The *high-load exposure* — load level exceeded only 10% of the time. Predicts acute overload injury. |

**Formula:**

$$APDF_p = \text{Percentile}_p\bigl(\{x_{\%RVE}[k]\}\bigr)$$

where the percentile is computed over:
- **Session APDF:** All $M$ RMS samples of the entire session
- **Active APDF** (used for asymmetry): Only samples where $x_{\%RVE}[k] \geq \theta_{gap} = 3.0\%\ \text{RVE}$, removing rest periods so the active muscular load distribution is characterised

The $APDF_{50}$ from the Active APDF is the input to the bilateral Asymmetry Index (Stage 4.4).

---

## 4. Stage 4 — Composite Tension Budget Scoring

### 4.1 Exposure Index (EIndex)

**Source:** Koch et al. (2024) — STAMI (Surface-EMG Based Tension and Muscle Activity Index).

**Biological justification:** Exposure Index is a *weighted proportion* metric. Simple mean %RVE conflates the directionality of risk: prolonged complete rest is actually beneficial (it allows Cinderella-fiber recovery), whereas time above 7% MVE is especially harmful. EIndex explicitly encodes this non-linearity by assigning differential weights to three activity zones.

**Three-zone partition** of the %RVE range:

| Zone | Threshold | Description | Weight |
|---|---|---|---|
| Rest | $x < 0.5\%\ \text{MVE}$ | Complete muscular relaxation | $-2$ |
| Low | $0.5\% \leq x < 7.0\%\ \text{MVE}$ | Low-level sustained activation | $+1$ |
| High | $x \geq 7.0\%\ \text{MVE}$ | High-load activation | $+2$ |

**Proportion calculation** over a window of $N_w$ RMS samples:

$$P_{rest} = \frac{1}{N_w}\sum_{k=1}^{N_w} \mathbf{1}\bigl[x_{\%RVE}[k] < 0.5\bigr]$$

$$P_{high} = \frac{1}{N_w}\sum_{k=1}^{N_w} \mathbf{1}\bigl[x_{\%RVE}[k] \geq 7.0\bigr]$$

$$P_{low} = 1 - P_{rest} - P_{high}$$

**EIndex formula:**

$$\boxed{EI = -2 \cdot P_{rest} + P_{low} + 2 \cdot P_{high}}$$

**Bounded range:** Since $P_{rest} + P_{low} + P_{high} = 1$ (exhaustive partition):

$$EI_{min} = -2(1) + 0 + 0 = -2 \quad \text{(all rest)}$$
$$EI_{max} = 0 + 0 + 2(1) = +2 \quad \text{(all high load)}$$

Therefore $EI \in [-2, +2]$ by mathematical construction — no clamping required.

| Parameter | Value | Source |
|---|---|---|
| Rest threshold | 0.5% MVE | Koch et al. (2024) |
| High threshold | 7.0% MVE | Koch et al. (2024) |

---

### 4.2 Short-SUMA Frequency Penalty

**Source:** Extension of Koch et al. (2025), which found that short-duration SUMA event *frequency* (not total duration) is the dominant predictor of neck/shoulder pain, particularly in women. The published finding is observational; this formula is the **project's novel real-time extension** converting the observation into a per-window penalty term.

**Formula:**

$$N_{short} = \text{count}(\text{bin}_{1.5-5s}) + \text{count}(\text{bin}_{5-10s})$$

$$\boxed{\delta_{SUMA} = \min\!\left(N_{short} \times 0.1,\; 1.0\right)}$$

**Properties:**
- **Linear up to 10 events:** Each short SUMA event contributes 0.1 penalty points
- **Hard cap at 1.0:** 10 or more short events saturate the penalty — prevents compounding from pathological signal artifacts
- **Per-window only:** This penalty is computed fresh for each 10-minute window. It is **never accumulated** across windows (unlike $EI_{cumulative}$)
- **Range:** $\delta_{SUMA} \in [0.0, +1.0]$

---

### 4.3 Per-Side Combined Score

**Rationale for additive compounding:** Koch (2025) establishes that EIndex (amplitude domain) and short-SUMA frequency (event-frequency domain) are **orthogonal risk factors** — a muscle alternating rapid short bursts produces the same EIndex as one under equivalent sustained load, but different injury risk. Additive combination captures both dimensions simultaneously.

$$\boxed{S_{side} = EI_{live} + \delta_{SUMA}}$$

**Range:** $EI_{live} \in [-2, +2]$ and $\delta_{SUMA} \in [0, +1]$ → $S_{side} \in [-2.0, +3.0]$

---

### 4.4 Bilateral Asymmetry Index

**Physical meaning:** Bilateral trapezius load asymmetry is an independent musculoskeletal risk factor — asymmetric activation patterns generate uneven spinal torques and differential motor-unit fatigue patterns (Madeleine et al. 2008).

**Formula (signed Laterality Index):**

$$\boxed{AI = \frac{APDF^{active}_{50,R} - APDF^{active}_{50,L}}{APDF^{active}_{50,R} + APDF^{active}_{50,L}}}$$

where $APDF^{active}_{50,L}$ and $APDF^{active}_{50,R}$ are the median active %RVE values for the left and right channels respectively (Active APDF, Section 3.3).

**Range and interpretation:**

| $AI$ value | Interpretation |
|---|---|
| $AI = -1.0$ | Exclusively left-dominant |
| $AI = 0.0$ | Perfect bilateral symmetry |
| $AI = +1.0$ | Exclusively right-dominant |

**Edge cases:**
- If either side is fully resting (all samples below $\theta_{gap}$): $AI = \text{NaN}$ — penalty is suppressed
- If both sides simultaneously have $APDF^{active}_{50} = 0$: $AI = 0.0$ by convention

> **Design note:** The signed formula is locked. An unsigned variant $|AI| = |R-L|/(R+L)$ would detect severity but lose directional information. The signed version is required to correctly identify whether the dominant side is left or right, which is clinically important for targeted intervention.

---

### 4.5 Composite Score — Bilateral Fusion & Floor Clamp

**Pipeline order is non-negotiable.** The three operations must execute in this exact sequence:

**Step 1 — Worst-side-drives fusion:**

$$S_{base} = \min(S_{left},\; S_{right})$$

*Rationale:* The trapezius operates as a bilateral muscle group under shared neural control. In asymmetric office work (mousing posture, monitor offset), one side typically accumulates load faster. Using the minimum score reflects the worst-off side rather than averaging it away. A subject overloading their right trapezius while resting their left should still display a high composite score.

**Step 2 — Asymmetry penalty:**

$$S_{penalised} = \begin{cases}S_{base} - 0.25 & \text{if } |AI| > 0.5 \\ S_{base} & \text{otherwise}\end{cases}$$

*Rationale:* The asymmetry penalty captures spinal torque and uneven mechanical load risk **in addition to** per-side amplitude load. A subject with symmetric high load has a different injury profile than one with the same worst-side load but severe bilateral imbalance. The penalty point 0.25 is proportionate to the 5-unit output range $[-2, +3]$, matching the original design intent of a 5% reduction (originally $5.0$ pts on a 0–100 scale; rescaled to $5\% \times 5\ \text{units} = 0.25$ to prevent a score-zeroing "kill switch" bug verified via simulation).

**Step 3 — Floor clamp (last):**

$$\boxed{C = \max(0.0,\; S_{penalised})}$$

*Rationale:* A composite score below 0 is physically meaningless — it would imply "negative tension load." The floor clamp is applied **after** the asymmetry penalty to ensure the penalty has its intended effect before the clamp prevents further reduction.

**Final composite range:** $C \in [0.0, +3.0]$

---

### 4.6 EIndex Accumulator — Live vs. Cumulative Split

**Problem solved:** If EIndex were summed every 2 seconds on a sliding 10-minute window, each moment of activity would be counted hundreds of times across overlapping windows. Simulation confirms this diverges to $\approx 5760$ after 8 hours of constant moderate activity — physically meaningless and technically a "double-counting bug."

**Architectural split (verified fix):**

**Live EIndex** — for real-time composite scoring:
- Computed on the **current sliding 10-minute window** of RMS samples
- Each computation **replaces** (not adds to) the stored value
- $EI_{live} \in [-2, +2]$ bounded at all times by construction
- Refreshed every $\Delta t_{refresh} = 2.0\ \text{s}$

**Session cumulative EIndex** — for end-of-session report card:
- Matches Koch (2024)'s exact published methodology: **discrete, non-overlapping** 10-minute windows
- Incremented **once** per completed 10-minute boundary:
$$EI_{cumulative} \mathrel{+}= EI_{window_k} \quad \text{at each completed 10-min boundary } k$$
- Grows monotonically, deterministically: $EI_{cumulative} = \sum_{k=1}^{K} EI_{window_k}$
- Used for STAMI reference comparison only; **never drives the real-time composite**

---

### 4.7 STAMI Display-Scale Mapping

**Source:** Koch et al. (2024), STAMI `S1_Dataset.sav` (SPSS format), $n = 731$ subjects.

**Purpose:** Map $EI_{cumulative}$ onto an interpretable 0–100 display scale anchored to population reference percentiles.

**Anchoring:** The 5th and 95th percentiles of the STAMI pooled dataset, corrected for the factor-of-100 scale difference (STAMI stores proportions on a 0–100 percentage scale, not 0–1 fractions):

$$EI_{floor} = -3.854 \quad (5\text{th percentile, corrected})$$
$$EI_{ceil} = +49.635 \quad (95\text{th percentile, corrected})$$

**Mapping formula:**

$$\boxed{STAMI_{display} = \text{clip}\!\left(\frac{EI_{cumulative} - EI_{floor}}{EI_{ceil} - EI_{floor}} \times 100,\; 0,\; 100\right)}$$

> **Scope restriction:** This mapping applies **exclusively** to $EI_{cumulative}$. It must never be applied to $EI_{live}$, per-side scores $S_{side}$, or composite $C$ — those are point-in-time snapshots, not comparable to STAMI's full-workday accumulation totals. The STAMI dataset also does not include the short-SUMA penalty, asymmetry term, or bilateral fusion — so this mapping is a population *reference* for display calibration only, not a validation of TensionBudget's scoring methodology.

---

## 5. Stage 5 — Spectral Fatigue Analysis (MDF / MNF)

**Physical basis:** Muscle fatigue produces a progressive downward shift in the power spectrum of the EMG signal. As motor unit firing rate slows and action potential conduction velocity decreases with metabolite accumulation (lactic acid build-up, pH decrease), power redistributes from higher to lower frequencies. Tracking Median Power Frequency (MDF) over time is the gold-standard frequency-domain fatigue indicator (De Luca 1997; Farina et al. 2002).

### 5.1 Pre-Rectification Signal Tap

**Critical implementation requirement:** Stage 5 taps the signal **after** bandpass filtering (Stage 2.1) but **before** rectification (Stage 2.3). This is architecturally enforced in `tensionbudget_app.py` via dedicated `epoch_filt_l` and `epoch_filt_r` accumulation buffers:

$$x_{spectral}[n] = x_{filtered}[n] \quad \text{(pre-rectification)}$$

**Why rectification destroys spectral analysis:** Full-wave rectification $|x|$ is a nonlinear operation. Nonlinear operations introduce harmonic distortion at integer multiples of the signal frequency components: $\{2f, 3f, 4f, \ldots\}$. Applied to EMG, this creates spurious spectral energy at DC and all even harmonics, completely invalidating PSD-based MDF/MNF estimates. The rectified signal's PSD does not represent the original neuromuscular frequency content.

---

### 5.2 Welch's Power Spectral Density Estimate

**Purpose:** Estimate the power spectral density (PSD) $\hat{S}_{xx}(f)$ of the EMG signal segment with reduced variance compared to the periodogram.

**Welch's method (Welch 1967):**

1. Divide the signal $x_{spectral}$ of length $L$ into $K$ overlapping segments of $N_w$ samples, with $N_{overlap}$ overlapping samples between consecutive segments
2. Apply a Hanning (Hann) window $w[n] = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N_w - 1}\right)\right)$ to each segment
3. Compute the periodogram of each windowed segment $\tilde{x}_k[n] = w[n] \cdot x_k[n]$:
$$I_k(f_j) = \frac{1}{f_s \cdot U} \left|\sum_{n=0}^{N_w-1} \tilde{x}_k[n] \cdot e^{-j 2\pi f_j n / f_s}\right|^2, \quad U = \frac{1}{N_w}\sum_{n=0}^{N_w-1} w[n]^2$$
4. Average across all $K$ segments:
$$\hat{S}_{xx}(f_j) = \frac{1}{K}\sum_{k=0}^{K-1} I_k(f_j)$$

**Parameter values:**

| Parameter | Symbol | Value | Derivation |
|---|---|---|---|
| Window length | $N_w$ | $1.0\ \text{s} \times 500\ \text{Hz} = 500\ \text{samples}$ | 1 s per epoch (TBConfig) |
| Overlap | $N_{overlap}$ | $0.5\ \text{s} \times 500\ \text{Hz} = 250\ \text{samples}$ | 50% overlap |
| Step | $N_{step}$ | $N_w - N_{overlap} = 250\ \text{samples}$ | |
| Number of windows | $K$ | $\lfloor(L - N_w)/N_{step}\rfloor + 1$ | Over 5-minute buffer: $L=150{,}000$ → $K \geq 598$ |
| Window type | — | Hanning (Hann) | Standard for EMG PSD (Farina 2002) |
| Frequency resolution | $\Delta f$ | $f_s / N_w = 500/500 = 1\ \text{Hz}$ | |
| DC removal | — | `detrend='constant'` | Remove mean within each epoch |

**Passband restriction:** After PSD estimation, only frequency bins within $[f_{low}, f_{high}] = [20, 240]\ \text{Hz}$ are retained:

$$\hat{S}_{xx}^{band}(f) = \hat{S}_{xx}(f) \cdot \mathbf{1}[20 \leq f \leq 240]$$

---

### 5.3 Median Power Frequency (MDF)

**Definition:** MDF is the frequency that divides the total spectral power into two equal halves:

$$\int_{f_{low}}^{MDF} \hat{S}_{xx}(f)\,df = \int_{MDF}^{f_{high}} \hat{S}_{xx}(f)\,df = \frac{1}{2}\int_{f_{low}}^{f_{high}} \hat{S}_{xx}(f)\,df$$

**Discrete implementation** (via cumulative trapezoid integration):

$$P_{total} = \int_{f_{low}}^{f_{high}} \hat{S}_{xx}^{band}(f)\,df \approx \sum_j \hat{S}_{xx}(f_j) \cdot \Delta f$$

$$P_{cumul}(f_i) = \sum_{j: f_j \leq f_i} \hat{S}_{xx}(f_j) \cdot \Delta f$$

MDF is found at the bin where $P_{cumul}$ first exceeds $P_{total}/2$, with linear interpolation between adjacent bins for sub-bin precision:

$$MDF = f_{i-1} + \frac{P_{total}/2 - P_{cumul}(f_{i-1})}{P_{cumul}(f_i) - P_{cumul}(f_{i-1})} \cdot (f_i - f_{i-1})$$

---

### 5.4 Mean Power Frequency (MNF)

**Definition:** MNF is the power-weighted centroid of the spectrum:

$$\boxed{MNF = \frac{\int_{f_{low}}^{f_{high}} f \cdot \hat{S}_{xx}(f)\,df}{\int_{f_{low}}^{f_{high}} \hat{S}_{xx}(f)\,df}}$$

**Discrete implementation:**

$$MNF = \frac{\sum_j f_j \cdot \hat{S}_{xx}(f_j)}{\sum_j \hat{S}_{xx}(f_j)}$$

MNF is more sensitive to high-frequency content than MDF (power-weighted mean vs. power-median), but is also more sensitive to noise. Both metrics are computed and logged for completeness.

---

### 5.5 Active-Epoch Gating

**Rationale:** Computing PSD on epochs that are predominantly electrical resting-floor noise produces meaningless MDF/MNF estimates that corrupt the downstream fatigue slope regression.

**Gating rule (project heuristic):** A Welch window is included in the spectral series only if the fraction of its samples exceeding the gap threshold is at least $r_{min}$:

$$r_{active} = \frac{1}{N_w}\sum_{n=k}^{k+N_w-1} \mathbf{1}\bigl[x_{\%RVE}[n] \geq 3.0\%\bigr] \geq 0.5$$

Epochs failing this test are skipped — their $(t_k, MDF_k, MNF_k)$ are omitted from the slope regression.

| Parameter | Value | Status |
|---|---|---|
| Active ratio threshold | 0.50 | Project heuristic — retune with pilot data in v2 |

---

### 5.6 OLS Fatigue Slope

**Purpose:** Quantify the rate of MDF decline over a 5-minute epoch as a Hz/min slope, using Ordinary Least Squares linear regression.

**Setup:** After extracting the series $\{(t_k, MDF_k)\}_{k=1}^{K_{active}}$ from all active epochs, convert times to minutes and fit a degree-1 polynomial:

$$\hat{MDF}(t) = \beta_1 \cdot t_{min} + \beta_0$$

**OLS normal equations** (implemented via `numpy.polyfit`):

$$\begin{bmatrix}\beta_1 \\ \beta_0\end{bmatrix} = \left(\mathbf{T}^T\mathbf{T}\right)^{-1}\mathbf{T}^T\mathbf{m}$$

where $\mathbf{T} = [t_{min}\; \mathbf{1}]$ is the $K_{active} \times 2$ design matrix and $\mathbf{m}$ is the MDF vector.

**Coefficient of determination:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_k (MDF_k - \hat{MDF}(t_k))^2}{\sum_k (MDF_k - \overline{MDF})^2}$$

**Reported output:**
- $\beta_1$ = `slope_hz_per_min` — primary fatigue indicator; **negative = frequency downshift = fatigue**
- $\beta_0$ = `intercept_hz` — MDF at session epoch start
- $R^2$ — goodness of linear fit (proportion of MDF variance explained by time trend)
- $K_{active}$ = `n_windows` — number of valid windows used in regression

**Minimum data guard:** If $K_{active} < 5$, slope estimation is suppressed (`sufficient_data = False`) to prevent meaningless 2–3 point regressions.

---

### 5.7 Fatigue Flag Decision Rule

A bilateral session epoch is flagged as showing measurable spectral fatigue only when **both** criteria are simultaneously satisfied:

$$\boxed{\text{is\_fatiguing} = (\beta_1 < \theta_{slope}) \;\wedge\; (R^2 \geq \theta_{R^2})}$$

| Threshold | Symbol | Value | Status |
|---|---|---|---|
| Slope threshold | $\theta_{slope}$ | $-0.3\ \text{Hz/min}$ | **Placeholder v1** — conservative extrapolation from high-force studies ($\sim -1.2$ Hz/min at 60–80% MVC); no published threshold for occupational low-load EMG (<5% MVC). Retune in v2. |
| Minimum R² | $\theta_{R^2}$ | $0.30$ | **Project heuristic** — 30% of MDF variance explained by time. Accepts noisy real-world data. Retune in v2. |

The dual-criterion prevents false positives: a steep but noisy slope ($R^2 < 0.30$) is not flagged as fatigue, and a tight linear fit with a small slope ($\beta_1 > -0.3$) is not flagged either.

---

## 6. Stage 6 — Subjective Strain Elicitation (Borg CR-10)

**Measurement:** At every 5-minute epoch boundary, a non-blocking Borg CR-10 popup modal prompts the subject to rate their perceived muscular exertion on the continuous scale:

$$y \in [0.0,\ 10.0]$$

| Score | Descriptor |
|---|---|
| 0 | Nothing at all — Complete rest |
| 1 | Very weak — just noticeable |
| 2 | Weak — light effort |
| 3 | Moderate — comfortable working level |
| 5 | Strong — heavy working fatigue |
| 7 | Very strong — severe strain |
| 10 | Absolute maximum — intolerable |

**Missingness handling (ML-critical):** If the subject does not submit a rating:
- `subjective_strain_cr10 = NULL` (SQL) / `""` (CSV)
- `strain_reported = 0` (boolean)

This explicit representation is mandatory for Bayesian hierarchical modelling. **Prohibited alternatives:**
- Forward-fill (LOCF): would falsely inflate $N$ by $500\ \text{Hz} \times 300\ \text{s} = 150{,}000\times$ across raw samples
- Backward-fill: corrupts data during unexpected disconnections
- Sentinel value ($y = -1$): biases regression intercepts and corrupts prior distributions

**Use as ML label:** $y$ is the ground-truth target variable for:
$$\hat{y} = f(\mathbf{x}_{epoch}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
where $\mathbf{x}_{epoch}$ is the 38-column feature vector (Stage 7).

---

## 7. Stage 7 — 5-Minute Epoch Feature Vector (ML Target)

Exactly once per completed 5-minute work epoch (300 seconds, $150{,}000$ samples at 500 Hz), all pipeline outputs are flattened into a **38-column feature vector** $\mathbf{x}_{epoch}$ and persisted to:
1. **Local disk** (`session_features.csv`) — always written first, unconditionally
2. **Cloud PostgreSQL** (`epoch_features` table) — background sync post-session

**Feature vector structure:**

| Column group | Features | Source stage |
|---|---|---|
| Identifiers | `session_id`, `epoch_index`, `elapsed_minutes` | Metadata |
| Composite score | `composite_score` | Stage 4.5 |
| EIndex live | `eindex_live_left`, `eindex_live_right` | Stage 4.1 |
| EIndex cumulative | `eindex_cumulative_left`, `eindex_cumulative_right` | Stage 4.6 |
| SUMA penalty | `penalty_short_suma_left`, `penalty_short_suma_right` | Stage 4.2 |
| Asymmetry | `asymmetry_index`, `asymmetry_penalty_applied` | Stage 4.4 |
| Spectral (MDF) | `mdf_left_hz`, `mdf_right_hz` | Stage 5.3 |
| Spectral (MNF) | `mnf_left_hz`, `mnf_right_hz` | Stage 5.4 |
| Spectral slope | `mdf_slope_left_hz_per_min`, `mdf_slope_right_hz_per_min` | Stage 5.6 |
| Regression quality | `mdf_r_squared_left`, `mdf_r_squared_right` | Stage 5.6 |
| Spectral flags | `mdf_computed_left/right`, `fatigue_flag_left/right` | Stage 5.5, 5.7 |
| Window count | `n_windows_left`, `n_windows_right` | Stage 5.2 |
| Subjective label | `strain_reported`, `subjective_strain_cr10` | Stage 6 |

**Idempotency:** The unique compound database constraint on `(session_id, epoch_index)` ensures repeated batch uploads overwrite existing rows rather than creating duplicates:

$$\text{UPSERT}\bigl(\mathbf{x}_{epoch}\bigr) \text{ ON CONFLICT } (session\_id,\; epoch\_index) \text{ DO UPDATE}$$

---

## 8. Stage 8 — Cloud Persistence & Idempotent Upsert

**Transport:** All data is transmitted over standard HTTPS (Port 443) to a serverless Supabase Edge Function (`ingest-session`, TypeScript/Deno runtime). Direct PostgreSQL connections (ports 5432/6543) are permanently deprioritised as enterprise/academic Wi-Fi firewalls reliably block these ports.

**Password security:** Subject account authentication uses PBKDF2-HMAC-SHA256 with a random per-subject salt:

$$H_{pw} = \text{PBKDF2-HMAC-SHA256}(password,\; salt,\; N_{iter}=100{,}000)$$

stored as `pbkdf2:sha256:100000$<salt>$<hex_hash>`. Incorrect password submissions during session upload return HTTP 403 Forbidden and admit **zero rows** to the database, preventing silent data merging between subjects.

**Three-tier relational schema:**
```
user_profiles (user_name PK)
    └── sessions (session_id PK, user_name FK)
            └── epoch_features (feature_id PK, UNIQUE(session_id, epoch_index))
```

**ML view:** `v_ml_training_pairs` joins all three tables, flattening subject demographics with epoch feature vectors to produce the training matrix $\mathbf{X}$ for downstream fatigue modeling, requiring no JOIN logic in Python ETL scripts.

---

## 9. Full Pipeline Flow Diagram

```
Hardware (Left / Right Arduino UNO R4)
        │  500 Hz, 14-bit ADC, BioAmp EXG Pill
        ▼
[Stage 1] ADC → mV conversion (optional)
        │
        ▼
[Stage 2a] Bandpass filter: 20–240 Hz, 4th-order Butterworth, zero-phase
        │
        ├──────────────────────────────────────────────┐
        │                                              │ PRE-RECT TAP
        ▼                                              ▼
[Stage 2b] Notch filter: 50 Hz, Q=30           [Stage 5] Welch PSD
        │                                              │ → MDF, MNF
        ▼                                              │ → OLS slope β₁
[Stage 2c] Full-wave rectification                     │ → Fatigue flag
        │                                              │
        ▼                                              │
[Stage 2d] RMS envelope: 100ms window, 20ms step       │
        │                                              │
        ▼                                              │
[Stage 2e] %RVE normalisation (÷ shrug reference RMS)  │
        │                                              │
        ├────────────────────┐                         │
        ▼                    ▼                         │
[Stage 3a] Gap/rest     [Stage 3b] SUMA events    ────┘
   detection                (1.5-5s, 5-10s bins)
        │                    │
        ▼                    ▼
[Stage 3c] APDF         [Stage 4.2] Short-SUMA
  P10, P50, P90             frequency penalty δ_SUMA
        │                    │
        ▼                    ▼
[Stage 4.1] EIndex ──── [Stage 4.3] Per-side score
   EI = -2·P_rest           S = EI + δ_SUMA
      + P_low
      + 2·P_high
        │
        ▼
[Stage 4.4] Asymmetry Index
   AI = (R - L)/(R + L)
        │
        ▼
[Stage 4.5] Composite Score
   1. Base = min(S_L, S_R)
   2. If |AI| > 0.5: Base -= 0.25
   3. C = max(0, Base)
        │
        ▼
[Stage 6] Borg CR-10 subjective rating y ∈ [0, 10]
        │
        ▼
[Stage 7] 38-column epoch feature vector @ every 5 minutes
        │        ↗ (also written to local disk CSV/Parquet)
        ▼
[Stage 8] Cloud PostgreSQL via HTTPS Edge Function (Port 443)
          Idempotent UPSERT on (session_id, epoch_index)
```

---

## 10. Parameter Reference Table

| Parameter | Symbol | Value | Stage | Source |
|---|---|---|---|---|
| Sampling rate | $f_s$ | 500 Hz | 1 | Hardware |
| ADC resolution | $N_{bits}$ | 14 bit | 1 | UNO R4 |
| Bandpass low | $f_{low}$ | 20 Hz | 2 | De Luca (1997) |
| Bandpass high | $f_{high}$ | 240 Hz | 2 | Nyquist-constrained |
| Filter order | $N$ | 4 | 2 | Standard |
| Notch freq | $f_{notch}$ | 50 Hz | 2 | Grid freq (India) |
| Notch Q factor | $Q$ | 30 | 2 | Standard |
| RMS window | $W$ | 100 ms = 50 samples | 2 | Marker & Maluf (2016) |
| RMS step | $S$ | 20 ms = 10 samples | 2 | Marker & Maluf (2016) |
| Calibration duration | — | 5 s | 2 | Project design |
| Gap threshold | $\theta_{gap}$ | 3.0% RVE | 3 | Marker & Maluf (2016) |
| Min gap duration | — | 0.125 s | 3 | Marker & Maluf (2016) |
| SUMA threshold | $\theta_{SUMA}$ | 0.5% RVE | 3 | Koch et al. (2024) |
| Min SUMA duration | — | 1.5 s | 3 | Koch et al. (2024) |
| EIndex rest threshold | — | 0.5% MVE | 4 | Koch et al. (2024) |
| EIndex high threshold | — | 7.0% MVE | 4 | Koch et al. (2024) |
| Short-SUMA penalty/event | — | 0.1 | 4 | Project extension |
| Short-SUMA cap | — | 1.0 | 4 | Project design |
| Asymmetry threshold | $\theta_{AI}$ | 0.5 | 4 | Project design |
| Asymmetry penalty | — | 0.25 pts | 4 | Scale-matched (5% of range) |
| STAMI floor | $EI_{floor}$ | −3.854 | 4 | Koch (2024) S1_Dataset.sav |
| STAMI ceiling | $EI_{ceil}$ | +49.635 | 4 | Koch (2024) S1_Dataset.sav |
| Welch window | $N_w$ | 1.0 s = 500 samples | 5 | Project spec |
| Welch overlap | $N_{overlap}$ | 0.5 s = 250 samples (50%) | 5 | Standard |
| Spectral band | — | 20–240 Hz | 5 | Matches bandpass |
| Min active ratio | $r_{min}$ | 0.50 | 5 | Project heuristic |
| Min valid windows | — | 5 | 5 | Project guard |
| Slope threshold | $\theta_{slope}$ | −0.3 Hz/min | 5 | **Placeholder** (v2 retune) |
| Min R² | $\theta_{R^2}$ | 0.30 | 5 | **Project heuristic** (v2 retune) |
| Epoch interval | — | 5 min = 300 s | 7 | Project spec (Phase 10) |
| Epoch buffer size | — | 150,000 samples | 7 | $300\ \text{s} \times 500\ \text{Hz}$ |
| Welch windows/epoch | $K$ | ≥300 | 5 | From buffer size |
| PBKDF2 iterations | $N_{iter}$ | 100,000 | 8 | NIST SP 800-132 |

---

## 11. Literature References

1. **De Luca, C.J. (1997).** The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.
   → Defines 20–450 Hz EMG passband; foundational reference for spectral interpretation and RMS normalisation.

2. **Farina, D., Merletti, R., & Enoka, R.M. (2004).** The extraction of neural strategies from the surface EMG. *Journal of Applied Physiology*, 96(4), 1486–1495.
   → Comprehensive review of MDF/MNF fatigue indicators and Welch PSD methodology for sEMG.

3. **Farina, D., et al. (2002).** Comparison of algorithms for estimation of EMG variables during voluntary isometric contractions. *Journal of Electromyography and Kinesiology*, 12(5), 337–349.
   → Algorithm-level comparison validating Welch MDF estimation; source for Hanning window and overlap specifications.

4. **Marker, R.J. & Maluf, K.S. (2016).** Upper trapezius motor unit firing patterns during repetitive low-force work. *Journal of Neurophysiology*, 116(4), 1752–1763.
   → Source for 100 ms RMS window, 20 ms step, and 3.0% RVE gap/rest threshold.

5. **Koch, M., et al. (2024).** The STAMI project: Development of a surface EMG-based method for assessment of trapezius muscle activation during computer work. *Applied Ergonomics*, 104, 103824.
   → Source for EIndex formula ($-2P_{rest} + P_{low} + 2P_{high}$), 0.5%/7.0% MVE thresholds, SUMA 1.5s minimum duration, 10-min discrete window methodology, and STAMI S1_Dataset.sav population anchoring values.

6. **Koch, M., et al. (2025).** Short-duration muscle activity events predict neck and shoulder pain in office workers: a prospective cohort study. *Scandinavian Journal of Work, Environment & Health* [in press].
   → Source for the finding that short-SUMA event *frequency* (1.5–10s bin counts) predicts pain, motivating the short-SUMA frequency penalty term (project's novel real-time extension).

7. **Hägg, G.M. (1991).** Static workloads and occupational myalgia — a new explanation model. In P. Anderson et al. (Eds.), *Electromyographical Kinesiology*, Elsevier.
   → Original Cinderella hypothesis: low-threshold (Type I) motor units are first recruited and last to de-recruit, making them vulnerable to overload during sustained low-level work.

8. **Jonsson, B. (1978).** Kinesiology: With special reference to electromyographic kinesiology. *Contemporary Clinical Neurophysiology (EEG Suppl. No. 34)*, 417–428.
   → Original definition of APDF (Amplitude Probability Distribution Function) and interpretation of $P_{10}$ (static), $P_{50}$ (median), $P_{90}$ (peak) percentiles.

9. **Jonsson, B. (1982).** Measurement and evaluation of local muscular strain in the shoulder during constrained work. *Journal of Human Ergology*, 11, 73–88.
   → Validation of APDF as a reliable exposure index for shoulder musculoskeletal surveillance.

10. **Madeleine, P., et al. (2008).** Towards functional pain assessment of the trapezius muscle during computer work. *Journal of Electromyography and Kinesiology*, 18(4), 527–541.
    → Validates bilateral EMG asymmetry as an independent risk factor for trapezius pain; motivates the Asymmetry Index and asymmetry penalty design choice.

11. **Welch, P.D. (1967).** The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
    → Original Welch's method paper; foundational reference for variance-reduced PSD estimation.

12. **Visser, B. & van Dieën, J.H. (2006).** Pathophysiology of upper extremity muscle disorders. *Journal of Electromyography and Kinesiology*, 16(1), 1–16.
    → Mechanistic review linking sustained motor-unit activation patterns to cumulative muscle damage; underpins SUMA detection rationale.

13. **Borg, G. (1982).** Psychophysical bases of perceived exertion. *Medicine & Science in Sports & Exercise*, 14(5), 377–381.
    → Original Borg CR-10 scale; foundation for subjective strain elicitation at each epoch boundary.

---

*This document was generated from and is consistent with the source implementation in `chordspy/tensionbudget/` (Phase 13 / W7). All formulas, thresholds, and parameter values are directly traceable to the corresponding lines of `config.py`, `preprocessing.py`, `analytics.py`, `scoring.py`, and `spectral.py`.*
