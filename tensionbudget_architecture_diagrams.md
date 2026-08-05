# TensionBudget — System Architecture Diagrams

**Version:** Phase 13 / W7 · 2026-08-05  
**Audience:** Professor review, academic presentation, research documentation

> This document contains five complementary architecture diagrams that together describe the full TensionBudget bilateral trapezius ergonomic surveillance system — from physical hardware through digital signal processing, live scoring, storage, cloud sync, and ML feature extraction.

---

## Diagram 1 — Full System Overview

*Top-level view of all layers: Hardware → Preprocessing → Analytics → Scoring → Storage → Cloud → ML*

```mermaid
flowchart TD
    subgraph HW["🔌 LAYER 1 — Hardware Acquisition"]
        direction LR
        L_HW["Left Arduino UNO R4\nBioAmp EXG Pill\nLeft Trapezius\nCOM6 · Pin A2"]
        R_HW["Right Arduino UNO R4\nBioAmp EXG Pill\nRight Trapezius\nCOM5 · Pin A2"]
    end

    subgraph LSL["📡 LSL Transport Layer"]
        direction LR
        LSL_STREAM["Lab Streaming Layer\nstart_lsl_stream.py\n500 Hz · 14-bit ADC\n230,400 baud"]
    end

    subgraph DSP["⚙️ LAYER 2 — Signal Pre-Processing  (preprocessing.py)"]
        direction TB
        BP["Bandpass Filter\n20–240 Hz · 4th-order\nButterworth · zero-phase"]
        NOTCH["Notch Filter\n50 Hz · Q=30\nauto-detect mains noise"]
        RECT["Full-Wave Rectification\n|x[n]|"]
        RMS_ENV["RMS Envelope\n100ms window · 20ms step\n→ 50 samples/sec"]
        RVE["% RVE Normalisation\nRMS ÷ shrug_calibration_RMS × 100\n→ hardware-independent %"]
        BP --> NOTCH --> RECT --> RMS_ENV --> RVE
    end

    subgraph ANA["📊 LAYER 3 — Amplitude Analytics  (analytics.py)"]
        direction LR
        GAPS["Gap Detection\n< 3.0% RVE\n≥ 0.125s duration\n→ gap_frequency"]
        SUMA["SUMA Detection\n> 0.5% RVE\n≥ 1.5s duration\n10 duration bins"]
        APDF["APDF Percentiles\nP10 · P50 · P90\nSession & Active variants"]
    end

    subgraph SPEC["🔬 LAYER 5 — Spectral Fatigue  (spectral.py)"]
        PRETAP["Pre-Rectification\nSignal Tap ← before |x|"]
        WELCH["Welch PSD\n1.0s window · 50% overlap\nHanning taper · 1Hz resolution\n20–240 Hz band"]
        MDF_MNF["MDF · MNF\nMedian & Mean\nPower Frequency"]
        OLS["OLS Fatigue Slope\nβ₁ Hz/min · R²\nis_fatiguing flag"]
        PRETAP --> WELCH --> MDF_MNF --> OLS
    end

    subgraph SCORE["🎯 LAYER 4 — Composite Scoring Engine  (scoring.py)"]
        direction TB
        EI["EIndex\n−2·P_rest + P_low + 2·P_high\n∈ [−2, +2]"]
        SFPEN["Short-SUMA Penalty\nmin(N_short × 0.1, 1.0)\n∈ [0, +1]"]
        PERSIDE["Per-Side Score\nEI + δ_SUMA\n∈ [−2, +3]"]
        AI["Asymmetry Index\n(R − L)/(R + L)\n∈ [−1, +1]"]
        FUSE["Worst-Side Fusion\nbase = min(Score_L, Score_R)"]
        APENALTY["Asymmetry Penalty\n|AI| > 0.5 → base −= 0.25"]
        CLAMP["Floor Clamp\nC = max(0.0, base)\n∈ [0.0, 3.0]"]
        EI --> PERSIDE
        SFPEN --> PERSIDE
        PERSIDE --> FUSE
        AI --> APENALTY
        FUSE --> APENALTY --> CLAMP
    end

    subgraph BORG["📝 LAYER 6 — Subjective Label"]
        CR10["Borg CR-10 Modal\nEvery 5 minutes\ny ∈ [0.0, 10.0] or NULL\nstrain_reported: 0 or 1"]
    end

    subgraph EPOCH["📦 LAYER 7 — 5-Minute Epoch Feature Vector"]
        VEC["38-column feature vector\nAssembled every 300 seconds\n150,000 raw samples → 1 row\nUNIQUE(session_id, epoch_index)"]
    end

    subgraph STORE["💾 LAYER 8A — Local Disk Storage  (local_logger.py)"]
        direction LR
        CSV_F["session_features.csv\n38 cols · 1 row/epoch"]
        CSV_R["session_raw.csv → .parquet\n5 cols · 500 rows/sec\n80–90% compression"]
        META["session_metadata.json\ncalibration · timestamps\nuser snapshot"]
    end

    subgraph CLOUD["☁️ LAYER 8B — Cloud Persistence  (cloud_ingest.py)"]
        direction TB
        EDGE["Supabase Edge Function\ningest-session (TypeScript)\nHTTPS Port 443\nfirewall-immune"]
        PG["PostgreSQL\nuser_profiles\nsessions\nepoch_features"]
        ML_VIEW["v_ml_training_pairs\n3-table JOIN view\nflat ML matrix"]
        EDGE --> PG --> ML_VIEW
    end

    HW --> LSL_STREAM
    LSL_STREAM --> DSP
    DSP --> ANA
    DSP -.->|"pre-rect tap"| SPEC
    ANA --> SCORE
    SPEC --> EPOCH
    SCORE --> EPOCH
    CR10 --> EPOCH
    EPOCH -->|"disk-first\nalways written"| STORE
    EPOCH -->|"background async\nHTTPS · non-blocking"| CLOUD
```

---

## Diagram 2 — Real-Time Signal Processing Pipeline (Per Channel)

*Detailed transformation chain applied to each of the two independent EMG channels. Left and Right are never mixed before the asymmetry step.*

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        RAW["x_raw[n]\nRaw ADC counts\n14-bit · 0–16383\n500 Hz"]
    end

    subgraph CONV["① ADC → mV\n(optional)"]
        VOLT["V_mV = (x / 16383)\n× V_ref × 1000\n\nSkipped for %RVE\nanalysis (ratios)"]
    end

    subgraph BPASS["② Bandpass Filter\nDe Luca 1997 / Farina 2002"]
        BPF["4th-order Butterworth\n20 Hz — 240 Hz\nZero-phase: sosfiltfilt\n(offline)\nCausal: sosfilt + z_i\n(live streaming)\n\nΩ_low = 20/250 = 0.08\nΩ_high = 240/250 = 0.96"]
    end

    subgraph SPECTAP["③ Pre-Rect\nSignal Tap"]
        TAP["x_filtered → Stage 5\nfor Welch PSD\n\nMust happen BEFORE\nrectification — |x|\ncreates 2f,3f harmonics\nthat invalidate MDF/MNF"]
    end

    subgraph NOTCH_F["④ Notch Filter\n(auto or manual)"]
        NF["IIR Notch · 50 Hz · Q=30\nBandwidth = f₀/Q = 1.67 Hz\nApplied only if PSD spike\nat 50 Hz > 5× median power"]
    end

    subgraph RECTF["⑤ Full-Wave Rectification"]
        RECT2["x_rect[n] = |x_filtered[n]|\n\nAll values ≥ 0\nPreserves instantaneous\namplitude envelope"]
    end

    subgraph RMSF["⑥ RMS Envelope\nMarker & Maluf 2016"]
        RMS2["RMS[k] = √( (1/W) Σ x²[n] )\n\nWindow W = 100ms = 50 samples\nStep S = 20ms = 10 samples\nOverlap = 80%\nOutput rate = 50 frames/sec\nN_frames per epoch ≈ 15,000"]
    end

    subgraph CALIB["⑦ % RVE Normalisation"]
        NORM["x_%RVE[k] = RMS[k] / RMS_ref × 100\n\nRMS_ref = mean RMS during\n5-second maximal shrug\n(stored in sessions table)\n\nResult: hardware-independent\ndimensionless % — enables\ncross-subject comparisons"]
    end

    subgraph OUT["Output to Analytics"]
        OUT1["x_%RVE[k] → analytics.py\n  • Gap detection (3.0% threshold)\n  • SUMA detection (0.5% threshold)\n  • APDF percentiles\n  • EIndex proportion counting"]
        OUT2["x_filtered[n] → spectral.py\n  • Welch PSD\n  • MDF / MNF\n  • OLS slope"]
    end

    INPUT --> CONV --> BPASS --> SPECTAP --> NOTCH_F --> RECTF --> RMSF --> CALIB --> OUT1
    SPECTAP -.->|pre-rect| OUT2
```

---

## Diagram 3 — Composite Scoring Engine

*All 5 steps of the scoring pipeline in order. Step order is architecturally non-negotiable.*

```mermaid
flowchart TD
    subgraph INPUTS["Inputs from Analytics (per side, per 10-min window)"]
        direction LR
        RMS_IN["x_%RVE[k]\nRMS envelope samples\n(15,000 frames / 5-min epoch)"]
        SUMA_IN["SUMA bin counts\n{1.5-5s: N₁, 5-10s: N₂, ...}\nfrom analytics.detect_sustained_activity()"]
        APDF_IN["Active APDF-50\nP50 of x_%RVE during\nactive periods only\n(samples ≥ 3.0% RVE)"]
    end

    subgraph EI_STEP["① Exposure Index (EIndex)\nKoch et al. 2024"]
        ZONES["Three-zone partition:\n\nP_rest = fraction of samples < 0.5% MVE\nP_high = fraction of samples ≥ 7.0% MVE\nP_low  = 1 − P_rest − P_high\n\n(P_rest + P_low + P_high = 1 exactly)"]
        EI_FORMULA["EI = −2·P_rest + P_low + 2·P_high\n\nRange: [−2, +2] by construction\n  −2 = fully resting all window\n  −1 = all time in low activation\n  0  = mixed rest and low\n  +1 = half high / half low\n  +2 = all time in high activation"]
        ZONES --> EI_FORMULA
    end

    subgraph SUMA_STEP["② Short-SUMA Frequency Penalty\nExtending Koch et al. 2025"]
        SHORT_COUNT["N_short = bin[1.5-5s] + bin[5-10s]\n\nEach short event = 0.1 penalty pts\n10+ events = full penalty"]
        PENALTY_FORMULA["δ_SUMA = min(N_short × 0.1, 1.0)\n\nRange: [0.0, +1.0]\nPer-window only — NEVER accumulated\nacross epochs"]
        SHORT_COUNT --> PENALTY_FORMULA
    end

    subgraph PERSIDE_STEP["③ Per-Side Score"]
        PS_FORMULA["S_side = EI_live + δ_SUMA\n\nRange: [−2.0, +3.0]\n(EI: −2..+2, penalty: 0..+1)\n\nComputed independently for\nLeft and Right trapezius"]
    end

    subgraph AI_STEP["④ Bilateral Asymmetry Index"]
        AI_FORMULA["AI = (APDF50_R − APDF50_L)\n     ─────────────────────\n     (APDF50_R + APDF50_L)\n\nRange: [−1.0, +1.0]\n  −1 = fully left-dominant\n   0 = perfect symmetry\n  +1 = fully right-dominant\n\nNaN if either side fully resting"]
    end

    subgraph FUSE_STEP["⑤ Bilateral Fusion + Penalty + Clamp\n(ORDER IS NON-NEGOTIABLE)"]
        STEP1["Step 1 — Worst-Side-Drives\nbase = min(Score_Left, Score_Right)\n\nRationale: One overloaded side\ncannot be hidden by a resting\ncontralateral side"]
        STEP2["Step 2 — Asymmetry Penalty\nIF |AI| > 0.5:\n  base −= 0.25\n\n0.25 = 5% of 5-unit range [−2,+3]\n(preserves original 5% intent\nwithout zeroing the score)"]
        STEP3["Step 3 — Floor Clamp  ← LAST\nC = max(0.0, base)\n\nFinal range: [0.0, +3.0]\nNegative = physically meaningless\nFloor applied AFTER penalty\nso penalty has full effect"]
        STEP1 --> STEP2 --> STEP3
    end

    subgraph OUTPUT_SCORE["Output: composite_score ∈ [0.0, 3.0]"]
        GUIDE["0.0–0.5  Minimal load — adequate recovery\n0.5–1.0  Low risk\n1.0–1.8  Moderate — prompt micro-break\n1.8–2.5  High load — alert user\n2.5–3.0  Severe — urgent intervention"]
    end

    subgraph STAMI["⑥ STAMI Display-Scale (cumulative only)"]
        STAMI_FORMULA["STAMI = clip( (EI_cumulative − (−3.854))\n               ──────────────────────────  × 100,  0, 100 )\n               (49.635 − (−3.854))\n\nAnchored to Koch 2024 STAMI\nS1_Dataset.sav · n=731 subjects\n5th/95th population percentiles\n\nApplied to EI_cumulative ONLY —\nnot to EI_live or composite_score"]
    end

    RMS_IN --> EI_STEP
    SUMA_IN --> SUMA_STEP
    APDF_IN --> AI_STEP
    EI_STEP --> PERSIDE_STEP
    SUMA_STEP --> PERSIDE_STEP
    PERSIDE_STEP --> FUSE_STEP
    AI_STEP --> FUSE_STEP
    FUSE_STEP --> OUTPUT_SCORE
    EI_STEP -.->|eindex_cumulative| STAMI
```

---

## Diagram 4 — Dual Storage Architecture & Cloud Sync

*How data flows from live capture to permanent local storage and then to cloud synchronisation. The architecture guarantees disk-first unconditional persistence.*

```mermaid
flowchart TD
    subgraph LIVE["🔴 LIVE SESSION  (zero network calls during recording)"]
        direction LR
        SENSORS["Hardware Sensors\n500 Hz · bilateral"]
        BUFFER["In-Memory Buffers\n• epoch_filt_left / right\n• rms_buffer_left / right\n• raw_sample_buffer\nNo disk I/O during processing"]
        SENSORS --> BUFFER
    end

    subgraph EPOCH_TRIGGER["⏱️ Every 5 Minutes (epoch boundary)"]
        FEATURE_ROW["Compute 38-column\nfeature vector\nfrom in-memory buffers"]
        BORG_POPUP["Borg CR-10 popup\n(non-blocking modal)\n→ strain_reported\n→ subjective_strain_cr10"]
    end

    subgraph LOCAL_WRITE["💾 LOCAL DISK  (always written, unconditionally)"]
        direction TB
        L_DIR["output_logs/\n  └── Subject_Name/\n        └── session_YYYYMMDD_HHMMSS/"]
        CSV_FEAT["session_features.csv\n38 columns · 1 row/epoch\nPlain text · human-readable\nPrimary research record"]
        CSV_RAW["session_raw.csv\n5 cols · 500 rows/sec\nStreamed per-second to disk\nNever lost even on crash"]
        META_JSON["session_metadata.json\n• session_id\n• user profile snapshot\n• calibration baselines\n• processing_version flag\n• hardware config"]
        PROFILE["user_profile.json\n• demographics\n• BMI at session time\n• dynamic age"]
        L_DIR --> CSV_FEAT & CSV_RAW & META_JSON & PROFILE
        PARQUET["On end_session():\nCSV → .parquet\n80–90% size reduction\n• 500 Hz: 100MB → 8MB\n• Retains full fidelity"]
        CSV_RAW --> PARQUET
    end

    subgraph CLOUD_SYNC["☁️ CLOUD SYNC  (background · async · non-blocking)"]
        direction TB
        TRIGGER["Trigger: end_session()\nAfter disk write completes\nRuns in background thread"]
        HTTPS["HTTPS POST · Port 443\nBypasses corporate firewall\n(ports 5432/6543 blocked\n on campus WiFi)"]
        EDGE_FN["Supabase Edge Function\ningest-session (Deno/TypeScript)\nPassword verification\n→ HTTP 403 on mismatch\n(zero unauthorised rows admitted)"]
        UPSERT["PostgreSQL UPSERT\nON CONFLICT (session_id, epoch_index)\nDO UPDATE\n→ Idempotent: re-upload\nis safe, no duplicates"]
        TRIGGER --> HTTPS --> EDGE_FN --> UPSERT
    end

    subgraph FALLBACK["🔄 Fallback & Retry"]
        BATCH["python -m scripts.ingest_to_postgres\n--log-dir output_logs\n\nBatch uploader: scans all\nlocal session folders and\nsyncs any missing cloud rows\nSafe to run repeatedly"]
    end

    subgraph ENV_CRED["🔑 Credentials (.env)"]
        ENV_FILE[".env file (git-ignored)\nSUPABASE_URL=...\nSUPABASE_ANON_KEY=...\n\nTeam shares keys out-of-band\n.env.example template committed\nAuto-loaded by config.py\nNo OS env vars needed"]
    end

    LIVE --> EPOCH_TRIGGER
    EPOCH_TRIGGER --> FEATURE_ROW
    BORG_POPUP --> FEATURE_ROW
    FEATURE_ROW -->|"① FIRST — always"| LOCAL_WRITE
    LOCAL_WRITE -->|"② AFTER disk write"| CLOUD_SYNC
    CLOUD_SYNC -->|"if network fails"| FALLBACK
    FALLBACK -->|"reads local files"| LOCAL_WRITE
    ENV_CRED --> HTTPS
```

---

## Diagram 5 — Relational Database Schema (Entity-Relationship)

*The complete cloud database structure with all columns, types, constraints, and relationships. This is the schema as implemented in `cloud_schema.py`.*

```mermaid
erDiagram
    user_profiles {
        TEXT user_name PK "Anonymised bridge ID — only this crosses to research tables"
        TEXT birth_date "YYYY-MM-DD — age computed dynamically at query time"
        TEXT gender_sex "M / F / Other / Unspecified — sex moderates SUMA effects (Koch 2025)"
        REAL weight_kg "Used with height_cm to compute BMI = weight/(height/100)²"
        REAL height_cm "Standing height in cm"
        REAL updated_at "Unix timestamp of last profile update"
        TEXT password_hash "PBKDF2-SHA256 · 100k iterations — prevents unauthorised row injection"
        TEXT created_via "desktop_registration | web_registration"
    }

    sessions {
        TEXT session_id PK "user_name_YYYYMMDD_HHMMSS — unique per recording"
        TEXT user_name FK "FK → user_profiles — links session to subject"
        REAL start_time "Unix timestamp (float) — timezone-independent"
        TEXT start_time_str "YYYY-MM-DD HH:MM:SS — human-readable convenience"
        REAL end_time "NULL if session terminated abnormally (crash/power loss)"
        TEXT end_time_str "Human-readable end time"
        TEXT mode "live_lsl | replay | web — acquisition path stratifier"
        REAL calibration_rms_left "RMS during 5s shrug — denominator for left %RVE normalisation"
        REAL calibration_rms_right "RMS during 5s shrug — denominator for right %RVE normalisation"
        TEXT processing_version "phase11_edge_padding | legacy_constant_padding — ML covariate"
    }

    epoch_features {
        INTEGER feature_id PK "Auto-increment — no scientific meaning"
        TEXT session_id FK "FK → sessions"
        INTEGER epoch_index "0-based epoch counter · join key to _raw.csv"
        REAL elapsed_minutes "epoch_index × 5.0 — time axis for trend plots"
        REAL composite_score "Final ergonomic score ∈ [0.0, 3.0]"
        REAL eindex_live_left "EIndex sliding window · left ∈ [−2, +2]"
        REAL eindex_live_right "EIndex sliding window · right ∈ [−2, +2]"
        REAL eindex_cumulative_left "Session-total EIndex · left (Koch 2024 exact method)"
        REAL eindex_cumulative_right "Session-total EIndex · right"
        REAL short_suma_penalty_left "Short-SUMA freq penalty · left ∈ [0, 1]"
        REAL short_suma_penalty_right "Short-SUMA freq penalty · right ∈ [0, 1]"
        REAL total_suma_bursts_left "Total SUMA event count · left (all duration bins)"
        REAL total_suma_bursts_right "Total SUMA event count · right"
        REAL gap_frequency_left "Micro-rests per minute · left (≥0 · higher = better recovery)"
        REAL gap_frequency_right "Micro-rests per minute · right"
        REAL apdf_10_left "10th percentile %RVE · left (static load level)"
        REAL apdf_10_right "10th percentile %RVE · right"
        REAL apdf_50_left "50th percentile %RVE · left (median working load)"
        REAL apdf_50_right "50th percentile %RVE · right"
        REAL apdf_90_left "90th percentile %RVE · left (peak load level)"
        REAL apdf_90_right "90th percentile %RVE · right"
        REAL asymmetry_index_ai "AI=(R−L)/(R+L) ∈ [−1,+1] · NULL if either side resting"
        INTEGER asymmetry_penalty_applied "1 if |AI|>0.5 and 0.25pts subtracted · explainability flag"
        REAL mdf_hz_left "Median Power Frequency · left · Hz (20–240 band)"
        REAL mdf_hz_right "Median Power Frequency · right · Hz"
        REAL mnf_hz_left "Mean Power Frequency · left · Hz"
        REAL mnf_hz_right "Mean Power Frequency · right · Hz"
        REAL fatigue_slope_left "OLS dMDF/dt · left · Hz/min (negative = fatigue)"
        REAL fatigue_slope_right "OLS dMDF/dt · right · Hz/min"
        REAL mdf_r_squared_left "OLS R² · left (≥0.3 required to flag fatigue)"
        REAL mdf_r_squared_right "OLS R² · right"
        REAL n_windows_left "Active Welch windows used · left (quality metric)"
        REAL n_windows_right "Active Welch windows used · right"
        INTEGER mdf_computed_left "1=MDF available · 0=epoch too resting to compute"
        INTEGER mdf_computed_right "1=MDF available · 0=epoch too resting"
        INTEGER is_fatiguing_left "1 if slope<−0.3Hz/min AND R²≥0.30 · left"
        INTEGER is_fatiguing_right "1 if slope<−0.3Hz/min AND R²≥0.30 · right"
        INTEGER strain_reported "1=subject submitted Borg rating · 0=skipped (NEVER NULL)"
        REAL subjective_strain_cr10 "Borg CR-10 target label y ∈ [0.0–10.0] · NULL if not reported"
    }

    raw_telemetry_csv {
        INTEGER sample_index "Global 0-based sample counter from session start"
        REAL timestamp_s "Seconds since session start (float)"
        INTEGER epoch_index "JOIN KEY → epoch_features.epoch_index"
        INTEGER raw_adc_left "14-bit ADC value · left channel · 0–16383"
        INTEGER raw_adc_right "14-bit ADC value · right channel · 0–16383"
    }

    user_profiles ||--o{ sessions : "one subject → many sessions"
    sessions ||--o{ epoch_features : "one session → many 5-min epochs"
    epoch_features ||--o{ raw_telemetry_csv : "one epoch → 150,000 raw samples"
```

---

## Diagram 6 — Two-Clock Principle & EIndex Split

*Why EIndex is split into a live sliding window and a discrete cumulative counter — and how the 5-minute feature epoch fits into the broader time architecture.*

```mermaid
timeline
    title Session Time Architecture (example 30-min session)

    section Hardware Clock (500 Hz · continuous)
        0s–300s    : Clock 1 ↺ ADC samples streaming at 500 Hz
                   : Raw buffer flushed to CSV once per second
                   : 150000 samples accumulated for Epoch 0
        300s–600s  : Epoch 1 accumulation
        600s–900s  : Epoch 2 accumulation
        900s–1800s : Epochs 3–5 accumulation

    section Feature Clock (5-min · discrete)
        t=5min  : Epoch 0 complete → compute 38-col feature vector → CSV row written → Borg CR-10 popup
        t=10min : Epoch 1 complete → feature vector → Borg popup
        t=15min : Epoch 2 complete → feature vector → Borg popup
        t=20min : Epoch 3 complete → feature vector
        t=25min : Epoch 4 complete → feature vector
        t=30min : Epoch 5 complete → end_session() called

    section EIndex Live (sliding 10-min window · refreshed every 2s)
        0–2s    : EI_live = EIndex(last 10 min of RMS)  REPLACES stored value
        2–4s    : EI_live refreshed again  (never summed — sliding window)
        ...     : Continues every 2s throughout session

    section EIndex Cumulative (discrete 10-min boundaries · Koch 2024)
        t=10min : EI_cumulative += EIndex(window_0)  incremented ONCE
        t=20min : EI_cumulative += EIndex(window_1)  incremented ONCE
        t=30min : EI_cumulative += EIndex(window_2)  incremented ONCE

    section Cloud Sync (HTTPS · async · post-session)
        t=30min : end_session() → disk write FIRST → background sync to Supabase
```

---

## Quick Reference — Key Parameters

| Layer | Parameter | Value | Source |
|---|---|---|---|
| Hardware | Sampling rate $f_s$ | **500 Hz** | Hardware |
| Hardware | ADC resolution | **14-bit** (UNO R4) | Hardware |
| Bandpass | Passband | **20 – 240 Hz** | De Luca 1997 |
| Bandpass | Filter order | **4th-order Butterworth** | Standard |
| Notch | Centre frequency | **50 Hz, Q=30** | India grid |
| RMS | Window | **100 ms = 50 samples** | Marker & Maluf 2016 |
| RMS | Step | **20 ms = 10 samples** | Marker & Maluf 2016 |
| Calibration | Duration | **5 seconds** (shrug) | Project design |
| Gap detection | Threshold | **3.0% RVE, ≥ 0.125 s** | Marker & Maluf 2016 |
| SUMA detection | Threshold | **0.5% RVE, ≥ 1.5 s** | Koch 2024 |
| EIndex | Rest zone | **< 0.5% MVE, weight −2** | Koch 2024 |
| EIndex | High zone | **≥ 7.0% MVE, weight +2** | Koch 2024 |
| Short-SUMA penalty | Per event | **0.1, cap 1.0** | Project extension |
| Asymmetry penalty | Threshold / points | **|AI| > 0.5 → −0.25 pts** | Project design |
| Composite score | Range | **[0.0, +3.0]** | Mathematical |
| STAMI anchors | 5th / 95th pctile | **−3.854 / +49.635** | Koch 2024, n=731 |
| Welch PSD | Window / overlap | **1.0 s / 50%** | Farina 2002 |
| Fatigue slope | Threshold | **−0.3 Hz/min (v1 placeholder)** | Extrapolated |
| Fatigue flag | Min R² | **0.30** | Project heuristic |
| Epoch interval | Feature emission | **5 minutes = 300 s** | Phase 10 lock |
| Epoch buffer | Raw samples | **150,000 per epoch** | $300 \times 500$ |
| Cloud transport | Protocol / port | **HTTPS / Port 443** | Firewall constraint |
| Auth | Hash algorithm | **PBKDF2-SHA256, 100k iter** | NIST SP 800-132 |

---

*All diagrams are derived directly from the implemented source code in `chordspy/tensionbudget/`. Parameter values are locked in `config.py` and propagate automatically to all modules — no magic numbers exist outside the config class.*
