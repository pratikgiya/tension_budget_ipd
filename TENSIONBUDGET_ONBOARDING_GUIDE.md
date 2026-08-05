# TensionBudget — Collaborator Setup & Onboarding Guide

Welcome to the **TensionBudget Bilateral sEMG Ergonomic Surveillance & Machine Learning System**. This guide provides exact, step-by-step instructions for team members and collaborators to set up their local Python environment, run simulated experiments without physical hardware, bridge real dual-Arduino hardware streams, capture initial shoulder shrug calibrations, and log longitudinal machine-learning training feature rows.

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Environment Setup & Installation](#2-environment-setup--installation)
3. [Running the Standalone Monitor App](#3-running-the-standalone-monitor-app)
   * [Subject Profile & Metadata Entry](#step-1-enter-subject-profile-metadata)
   * [Mode A: Offline CSV Replay & Simulation (No Hardware Needed)](#mode-a-offline-csv-replay--simulation-no-hardware-needed)
   * [Mode B: Live Hardware Streaming (Dual Arduino UNO R4s)](#mode-b-live-hardware-streaming-dual-arduino-uno-r4s)
   * [Performing the 5-Second Shrug Calibration](#step-3-perform-the-5-second-shoulder-shrug-calibration-live-mode)
4. [The Automated Local Logging Vault (`output_logs/`)](#4-the-automated-local-logging-vault-output_logs)
5. [Generating Custom Mock Datasets](#5-generating-custom-mock-datasets)
6. [Troubleshooting & Common FAQs](#6-troubleshooting--common-faqs)

---

## 1. System Architecture Overview

TensionBudget is built around two foundational physiological guidelines:
* **Zero DSP Rewrite & Verified Engineering**: Digital signal processing (20–240 Hz bandpass filtering, moving root-mean-square extraction, Welch's median power frequency estimation) runs through verified NumPy and SciPy modules in `chordspy.tensionbudget`.
* **The Two-Clock Principle & Hybrid Telemetry**: 
  * **Clock 1 (High-Frequency Live Stream & Raw Vault)**: Processes 500 Hz sEMG streams in RAM to power real-time graphical displays at 50 Hz refresh rates while simultaneously buffering and streaming raw millivolt telemetry directly to disk (`_raw.csv`) with relational `epoch_index` binding for deep waveform modeling and Bayesian priors.
  * **Clock 2 (5-Minute Discrete Summary & Target Logger)**: Exactly once every 5 minutes of elapsed working duration (300 seconds), a 38-column feature vector is computed and deposited into a dedicated session subdirectory under `output_logs/`, accompanied by an interactive **Borg CR-10 Popup Modal** to capture precise perceived muscular strain ($y$) without thread blocking or stale defaults.

---

## 2. Environment Setup & Installation

### Step 1: Clone or Pull the Repository
Open a terminal (PowerShell on Windows, Terminal on macOS/Linux) and navigate to your workspace:
```powershell
git clone https://github.com/<your-repo-link>/Chords-Python-main.git
cd Chords-Python-main
```

### Step 2: Create & Activate a Python Virtual Environment
Requires Python 3.9 or higher. Create an isolated virtual environment (`.venv`):
```powershell
python -m venv .venv
```

**Activate the virtual environment:**
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\activate
  ```
  *(Note: If you encounter an "Execution Policy Restricted" error in PowerShell, temporarily run: `Set-ExecutionPolicy Unrestricted -Scope Process` and retry activation).*
* **macOS / Linux (Bash):**
  ```bash
  source .venv/bin/activate
  ```
*(When active, your terminal prompt will be prefixed with `(.venv)`).*

### Step 3: Install Required Dependencies
Install the required packages (`pylsl`, `PyQt5`, `pyqtgraph`, `scipy`, `pandas`, `pyserial`, etc.):
```powershell
pip install -r requirements.txt
```

### Step 4: Database Credential Setup & Cloud Sync (Collaborator Onboarding)
To enable automatic, firewall-immune synchronization to our shared cloud research database without exposing private credentials in git commits:
1. Duplicate the `.env.example` template file in the project root and save it as `.env`.
2. Request the public project **Supabase Anon Key** from the research lead and paste it into `SUPABASE_ANON_KEY=...` inside your local `.env` file. (Note: The `.env` file is permanently ignored by git in `.gitignore` and will never leak online).
3. The application features an **automatic zero-dependency loader**: whenever you launch the GUI or execute CLI ingestion scripts, Python reads `.env` directly from disk into memory. You do not need to manually set operating system environment variables!

---

## 3. Running the Standalone Monitor App

To launch the real-time application GUI from your terminal, execute:
```powershell
python chordspy/tensionbudget_app.py
```

### Step 1: Enter Subject Profile Metadata
At the top of the GUI window, locate the **Subject Metadata & Personalization Vault** control bar:
* **Subject ID / Name**: Enter an identifier (e.g., `Mohit_K`, `Subject_001`). This name determines the storage folder inside `output_logs/<subject_name>/`.
* **DOB (YYYY-MM-DD)**: Used by the storage engine to compute real-time biological age at the precise hour of each session without requiring static manual updates.
* **Sex**: Select `Unspecified`, `M`, `F`, or `Other`. (Muscle mass and motor-unit firing dynamics differ across biological sexes; preserving this improves machine learning predictive accuracy).
* **Weight (kg) & Height (cm)**: Used by the logger to dynamically calculate Body Mass Index ($BMI = \frac{\text{weight in kg}}{(\text{height in meters})^2}$).

---

### Mode A: Offline CSV Replay & Simulation (No Hardware Needed)
This mode allows researchers to analyze recorded trials, test scoring dynamics, and verify logging behavior using included sample files without physical EMG electrodes.

1. Ensure **Data Source** drop-down is set to **"Offline CSV Replay Mode"**.
2. Click **[Select CSV File...]** and navigate to your `mock_data/output/` folder. Choose an included sample file such as:
   * `mock_data/output/TB_REALISTIC_10MIN_raw.csv` (10-minute simulation of a data-entry worker with right-dominant mouse strain).
3. Click **[Start Replay]**.
4. **Observe the GUI Tabs**:
   * **Tab 1 (Live Bilateral Signals)**: Real-time bilateral electrical waveforms (Blue = Bandpass filtered, Red = Moving RMS envelope).
   * **Tab 2 (Analytics & Scoring Dashboard)**: Monitors live Worst-Side Driven Composite Scores (0.0–3.0), cumulative physical load (EIndex), Short-SUMA Cinderella fiber strain penalties, micro-relaxation gap frequencies, and Welch's Spectral Fatigue downward drift slopes (Hz/min).
5. As simulated playback crosses 10-minute epoch boundaries (or reaches the end of the file), check your terminal notifications to confirm that feature rows are written directly into `output_logs/<subject_name>/`!

---

### Mode B: Live Hardware Streaming (Dual Arduino UNO R4s)
Used during hardware monitoring sessions with two Arduino UNO R4 Minimas coupled with Upside Down Labs BioAmp EXG Pills positioned over the Left and Right upper trapezius muscles.

#### Step 1: Start the Dual-Arduino LSL Synchronizer Bridge (Terminal 1)
Because dual USB serial microcontrollers operate on distinct COM ports, we use a synchronized Lab Streaming Layer (LSL) bridge script. Open a terminal, activate `.venv`, and run:
```powershell
python start_lsl_stream.py
```
* **What this does**: Automatically scans your operating system (Windows, macOS, or Linux) to detect connected USB/Serial acquisition devices without requiring hardcoded COM port numbers. It handshakes binary packet streaming without driver buffer locks, and broadcasts a synchronized 500 Hz 2-channel bilateral stream named `'Chords_EMG_Bilateral'`.
* *(Note: If you have more than two serial devices attached or wish to override the order of Left and Right shoulders, you can explicitly pass port names via CLI flags: `python start_lsl_stream.py --left COM3 --right COM4` or `--left /dev/ttyUSB0 --right /dev/ttyUSB1`).*

#### Step 2: Connect to the LSL Stream in the App (Terminal 2)
1. In a second terminal window, launch the application: `python chordspy/tensionbudget_app.py`.
2. Ensure your Subject Profile details are completed in the top bar.
3. Switch the **Data Source** drop-down to **"Live LSL Stream (Uno R4)"**.
4. Click **[Connect LSL Stream]**. The status label will display connection to `'Chords_Bilateral_Trapezius @ 500 Hz'`.

---

### Step 3: Perform the 5-Second Shoulder Shrug Calibration (Live Mode)
Electrical impedance, skin moisture, and electrode placement vary across individual sittings. To scale raw voltages (`mV`) into meaningful percentage ratios (`%RVE`), you must capture a baseline submaximal calibration hold:

1. Click on **Tab 3: Shrug Calibration Hold**.
2. Sit upright in a neutral, relaxed posture.
3. Click **[Start 5-Second Shrug Calibration Hold]**.
4. **Action**: Immediately raise both shoulders into a moderate, isometric shoulder shrug (submaximal hold at ~50% effort) and hold steady for 5 seconds as the progress gauge fills to 100%.
5. When finished, the system calculates the middle 3-second trimmed RMS baseline voltages (`calibration_rms_left` and `calibration_rms_right`), displays them on screen, stores them in `tensionbudget_calibration.json`, and records them into your active session metadata file.

---

## 4. The Automated Local Logging Vault & Session Folders (`output_logs/`)

As you stream live hardware data or run offline replays, the logger records all telemetry inside isolated, self-contained **Session Folders**:
```
output_logs/
  └── Mohit_K/                                      <-- Subject's dedicated research vault
        ├── user_profile.json                       <-- Preserves DOB, Sex, Weight, Height & dynamic BMI
        └── session_20260804_130742/                <-- NEW Dedicated session directory per recording
              ├── session_metadata.json             <-- Keeps timestamps, user snapshot & calibration baselines (mV)
              ├── session_features.csv              <-- Clock 2 (38 columns, 5-minute epoch summary vector)
              └── session_raw.csv                   <-- Clock 1 (500 Hz high-frequency raw telemetry stream)
```

### Why is this folder structured this way?
* **Relational `epoch_index` Binding & Raw Integrity**: The high-frequency raw telemetry file (`_raw.csv`) logs every individual 500 Hz waveform sample with an explicit `epoch_index` column. This renders downstream analysis completely immune to sample-rate jitters or temporary serial packet drops—researchers can effortlessly join dense high-frequency waveforms to sparse 5-minute feature targets in PyTorch or PostgreSQL (`SELECT * FROM raw JOIN features USING (epoch_index)`).
* **Interactive Borg CR-10 Popup Modal**: To prevent forgotten self-reports or stale forward-filling, a non-blocking dialog pops up at every 5-minute epoch mark prompting the user for their perceived muscular strain on the continuous `0.0–10.0` Borg CR-10 scale. Submitting a score logs it strictly into `session_features.csv` ($y$), preserving `session_raw.csv` as an untouched 5-column Stage-1 physical acquisition stream (`sample_index, timestamp_s, epoch_index, raw_adc_left, raw_adc_right`) that joins cleanly back to features via `epoch_index`.
* **Explicit Missingness Flags**: Unrecorded or skipped ratings are recorded with `strain_reported = 0` and empty strings `""` (SQL `NULL`), eliminating `-1.0` sentinel distortion from Bayesian Hierarchical Model priors.
* **Offline Disk-First Storage & Firewall-Immune Cloud Sync**: **Local offline storage in `output_logs/` is the unconditional primary source of truth.** During live data recordings, the application makes zero network calls, ensuring continuous telemetry capture even without Wi-Fi. Upon pressing "Stop Recording," `end_session()` first safely persists all CSVs, Parquet waveform archives, and JSON manifests to disk. Only then does it perform a non-blocking background sync to Supabase over standard **HTTPS (Port 443)** via our Edge Function gateway—bypassing corporate/campus database port blocking (`5432`/`6543`). If network synchronization fails or Wi-Fi is offline, local disk files remain fully intact for batch uploading later via `python -m scripts.ingest_to_postgres --log-dir output_logs`.

---

## 5. Generating Custom Mock Datasets

If you need to generate test files of varying lengths or simulated workload conditions, run the bundled data generators from your terminal:

* **10-Minute Realistic Workstation Scenario** (simulates typing, reaching bursts, and mouse-induced right asymmetry):
  ```powershell
  python mock_data/generate_realistic_mock.py
  ```
  *(Outputs directly to `mock_data/output/TB_REALISTIC_10MIN_raw.csv`)*
* **1-Hour or 2-Hour Prolonged Office Work Replay File**:
  ```powershell
  python mock_data/generate_tb_raw.py
  ```
  *(Outputs a prolonged test session into `mock_data/output/`)*

---

## 6. Troubleshooting & Common FAQs

1. **Error: `ModuleNotFoundError: No module named 'chordspy'`**
   * Make sure you launch the application from the root project directory (`C:\...\Chords-Python-main`) using: `python chordspy/tensionbudget_app.py`. The script automatically configures `sys.path` to recognize the root workspace.
2. **"No LSL stream found" when connecting Live in the App**
   * Verify that your hardware bridge script (`python start_lsl_stream.py`) is actively running without serial exceptions in Terminal 1 before pressing **[Connect LSL Stream]** in Terminal 2.
3. **Serial port access denied or COM port errors in `start_lsl_stream.py`**
   * Open Device Manager (Windows) to identify which COM ports are assigned to your two Arduino UNO R4 Minimas. Ensure no other applications (like Arduino IDE Serial Monitor or BrainVision LSL Viewer) are keeping those ports open. Edit `PORT_L` and `PORT_R` inside `start_lsl_stream.py` to match your confirmed COM port numbers.
4. **Why is my Left channel reading high percentages like $>300\%$ RVE?**
   * This indicates either an uncalibrated session using stale baselines from another person or temporary skin-electrode impedance before natural perspiration settles. Always switch to **Tab 3** after connecting and click **Start 5-Second Shrug Calibration Hold** so your current electrode placement is scaled properly!
5. **Where can I find the raw 500 Hz high-frequency waveforms?**
   * Inside your session folder under `output_logs/<your_name>/session_<timestamp>/session_<timestamp>_raw.csv`! Every single sample is buffered and flushed directly to disk once per second without locking the UI or RAM.
