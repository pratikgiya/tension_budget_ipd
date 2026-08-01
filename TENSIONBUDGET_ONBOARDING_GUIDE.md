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
* **The Two-Clock Principle**: 
  * **Clock 1 (In-Memory Live Engine)**: Processes 500 Hz sEMG streams entirely in RAM to power real-time graphical displays at 50 Hz refresh rates. Raw high-speed signal data is discarded when the window closes to conserve disk and network bandwidth.
  * **Clock 2 (10-Minute Discrete Summary Logger)**: Exactly once every 10 minutes of elapsed working duration (and upon session completion), a 30+ column feature vector is computed and deposited into an automated local CSV directory (`output_logs/`), formatted for automated ingestion into PostgreSQL machine learning models.

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
* **What this does**: Automatically opens **COM5** (Left Channel at 500 Hz, frame terminator `0x01`) and **COM6** (Right Channel at 250 Hz, frame terminator `0x0D`), supersamples the Right stream to match 500 Hz via linear interpolation, and publishes a synchronized 2-channel bilateral stream named `'Chords_Bilateral_Trapezius'`.
* *(Note: If your computers assign different port numbers, open `start_lsl_stream.py` in a text editor and adjust `PORT_L = 'COM5'` and `PORT_R = 'COM6'` to match your Device Manager).*

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

## 4. The Automated Local Logging Vault (`output_logs/`)

As you stream live hardware data or run offline replays, the logger records data inside your workspace root:
```
output_logs/
  └── Mohit_K/                                 <-- Subject's dedicated research folder
        ├── user_profile.json                  <-- Preserves DOB, Sex, Weight, Height & calculated BMI
        ├── session_20260801_1030_metadata.json<-- Keeps start/end timestamps & Shrug Calibration baselines (mV)
        └── session_20260801_1030_features.csv <-- Wide-format feature table logged every 10 minutes!
```

### Why is this folder structured this way?
* **Machine Learning Readiness**: The `_features.csv` table formats data into a 30+ column wide-tabular layout matching our PostgreSQL cloud database specification. Every row is one 10-minute observation. Columns include EIndex exposure ratings, SUMA counts, Amplitude Probability Distribution Function (APDF 10/50/90) levels, laterality asymmetry index, median/mean power frequencies (MDF/MNF), and explicit missingness indicators (`mdf_computed_left/right`).
* **Easy Database Ingestion**: When you are ready to upload local historical experiments to a cloud database, these CSV files can be directly ingested into PostgreSQL tables or Pandas dataframes via simple bulk import commands with zero reformatting required.

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
4. **Why isn't my raw 500 Hz signal being saved into `output_logs/`?**
   * This is deliberate design under our **Two-Clock Principle**. Saving uncompressed 500 Hz raw multichannel arrays would rapidly consume gigabytes of storage during all-day workstation monitoring. The app extracts all diagnostic ergonomic features in memory every 10 minutes and discards the raw electrical waveform when the window closes to keep lightweight file logs.
