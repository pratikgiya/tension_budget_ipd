"""
TensionBudget — Realistic Mock EMG Data Generator
===================================================
Generates a 10-minute bilateral upper-trapezius EMG session that produces
meaningful, dynamic dashboard readings without real hardware.

Physiological scenario simulated:
  - Ergonomics data-entry workstation session
  - Posture phases: neutral → forward lean → recovery → right-hand mouse burst
  - 4 distinct activity zones with different load levels and asymmetry
  - Superimposed low-frequency fatigue drift (MDF declining over time)
  - Rest/recovery micro-gaps scattered throughout
  - Mild right-dominant asymmetry (realistic for mouse-dominant workers)

Output files:
  mock_data/output/TB_REALISTIC_10MIN_raw.csv  — Raw ADC format matching hardware
  tensionbudget_calibration.json               — Pre-built calibration (auto-loaded by app)

Run from repo root:
  $env:PYTHONPATH="."; python mock_data/generate_realistic_mock.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Block optional hardware deps so chordspy can be imported headlessly
for _mod in ['serial', 'serial.tools', 'serial.tools.list_ports',
             'bleak', 'pylsl', 'flask', 'Flask',
             'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore',
             'pyqtgraph', 'pyaudio', 'keyboard', 'pygame']:
    sys.modules.setdefault(_mod, MagicMock())

import numpy as np
from scipy.signal import butter, sosfilt

# ── Parameters ────────────────────────────────────────────────────────

FS          = 500           # Hz  — matches Uno R4 firmware
DURATION_S  = 10 * 60      # 10 minutes
N_SAMPLES   = FS * DURATION_S
ADC_BITS    = 14
ADC_MID     = 2 ** (ADC_BITS - 1)   # 8192 — DC midpoint
ADC_MAX     = 2 ** ADC_BITS - 1      # 16383

# Calibration reference: RMS of a gentle ~40% MVC voluntary shrug.
# For 14-bit Uno R4 recording at 500 Hz after 20-240 Hz bandpass,
# typical sEMG RMS at ~40% MVC sits at roughly 120-180 ADC counts
# after high-pass (DC removed). We set 150 as a realistic shrug reference.
CALIB_REF_RMS = 150.0   # ADC counts (after bandpass, pre-rectification)

np.random.seed(42)


# ── Activity Schedule ─────────────────────────────────────────────────
# Each segment: (start_s, end_s, left_%MVC, right_%MVC, fatigue_start_hz)
# %MVC drives the simulated EMG amplitude. 100% = at-reference contraction.
# We simulate the EMG as broadband noise filtered to the 20-240 Hz band,
# amplitude-modulated by the %MVC envelope.

SEGMENTS = [
    # (start_s, end_s, left_pct_mvc, right_pct_mvc, note)
    (   0,  30, 0.0, 0.0,  "Session start — settling / instrument quiet"),
    (  30,  90, 6.0, 8.0,  "Low-load keyboard typing"),
    (  90, 120, 0.0, 0.0,  "Short posture break — micro-rest"),
    ( 120, 200, 8.0, 12.0, "Moderate load, right dominant (mouse use)"),
    ( 200, 240, 0.0, 0.0,  "Rest gap > 20s — full recovery"),
    ( 240, 340, 5.0, 9.0,  "Resume typing — moderate"),
    ( 340, 360, 14.0,18.0, "HIGH BURST — reaching across desk"),
    ( 360, 420, 0.0, 0.0,  "Recovery pause"),
    ( 420, 490, 7.0, 11.0, "Typing, slightly fatigued"),
    ( 490, 510, 0.0, 0.0,  "Micro-rest"),
    ( 510, 570, 5.0, 8.0,  "Low-load resume"),
    ( 570, 600, 2.0, 3.0,  "Wind-down"),
]


def make_emg_segment(n: int, pct_mvc: float, fs: int,
                     center_hz: float = 100.0, noise_seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic EMG segment of n samples at pct_mvc amplitude.

    Method:
      1. Generate white Gaussian noise (physiologically correct EMG
         is broadband, not sinusoidal).
      2. Bandpass-filter to [20, 240] Hz (matching our recording pipeline).
      3. Scale amplitude so that the RMS of the output roughly tracks
         pct_mvc/100 * CALIB_REF_RMS.
      4. Add a small amount of electrode motion artifact (very low freq).

    If pct_mvc == 0.0, returns noise near the sensor noise floor (~5 ADC counts).
    """
    rng = np.random.default_rng(noise_seed)

    if pct_mvc <= 0.01:
        # Resting floor: pure electrode noise ~3–6 ADC counts RMS
        return rng.normal(0.0, 4.0, n)

    # Target RMS in ADC counts for this %MVC
    target_rms = (pct_mvc / 100.0) * CALIB_REF_RMS

    # 1. White noise base
    raw_noise = rng.normal(0.0, 1.0, n)

    # 2. Bandpass 20-240 Hz
    nyq = 0.5 * fs
    sos = butter(4, [20.0 / nyq, 240.0 / nyq], btype='bandpass', output='sos')
    emg = sosfilt(sos, raw_noise)

    # 3. Scale to target RMS
    current_rms = np.sqrt(np.mean(emg ** 2))
    if current_rms > 1e-10:
        emg = emg * (target_rms / current_rms)

    # 4. Add tiny motion artifact at 0.5-4 Hz (very small, < 5% of signal)
    t = np.arange(n) / fs
    artifact_freq = 0.5 + rng.uniform(0.0, 3.5)
    artifact = rng.normal(0.0, target_rms * 0.04) * np.sin(2 * np.pi * artifact_freq * t)
    emg += artifact

    return emg


def apply_fatigue_drift(emg: np.ndarray, t_start_s: float,
                        duration_total_s: float, fs: int,
                        drift_hz_per_min: float = -0.5) -> np.ndarray:
    """
    Simulate spectral fatigue by progressively high-pass shifting the signal
    over the full session. We do this by applying a very gentle dynamic emphasis
    on lower frequencies (as fatigue accumulates, lower-frequency MUs dominate).

    Implementation: blend low-passed version into signal, with blend factor
    proportional to elapsed session time. This causes MDF to drift downward.
    """
    n = len(emg)
    # Linear blend factor: 0.0 at session start, 0.15 at end (subtle)
    t_end_s = t_start_s + n / fs
    blend_start = min(t_start_s / duration_total_s, 1.0)
    blend_end   = min(t_end_s   / duration_total_s, 1.0)

    # Low-pass < 80 Hz component (lower-frequency MU activity in fatigue)
    nyq = 0.5 * fs
    try:
        sos_lp = butter(4, 80.0 / nyq, btype='low', output='sos')
        emg_low = sosfilt(sos_lp, emg)
    except Exception:
        return emg

    # Ramp blend factor from blend_start to blend_end
    blend = np.linspace(blend_start, blend_end, n) * 0.18
    emg_fatigued = emg * (1 - blend) + emg_low * blend
    return emg_fatigued


def generate_channel(segments, fs, duration_s, fatigue=True, channel_seed=0):
    """Concatenate all segments into one full-session EMG signal."""
    out = np.zeros(duration_s * fs, dtype=np.float64)

    for i, seg in enumerate(segments):
        start_s, end_s, pct_left, pct_right, _ = seg
        n = (end_s - start_s) * fs
        pct = pct_left if channel_seed == 0 else pct_right

        chunk = make_emg_segment(n, pct, fs, noise_seed=channel_seed * 100 + i)

        if fatigue and pct > 0:
            chunk = apply_fatigue_drift(chunk, start_s, duration_s, fs)

        out[start_s * fs : end_s * fs] = chunk

    return out


def to_adc_counts(emg_signal: np.ndarray, adc_mid: int, adc_max: int) -> np.ndarray:
    """Shift EMG (zero-mean, ADC-count amplitude) to 14-bit ADC range centred on mid."""
    adc = emg_signal + adc_mid
    return np.clip(adc, 0, adc_max).astype(int)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TensionBudget — Realistic 10-Min Mock EMG Generator")
    print("=" * 60)

    # Generate both channels
    print("\n[1/4] Generating Left channel (ch0 / A0)...")
    emg_left  = generate_channel(SEGMENTS, FS, DURATION_S, fatigue=True, channel_seed=0)

    print("[2/4] Generating Right channel (ch1 / A1)  [right-dominant worker]...")
    emg_right = generate_channel(SEGMENTS, FS, DURATION_S, fatigue=True, channel_seed=1)

    # Convert to 14-bit ADC format (floating channels filled at ADC_MID)
    adc_left  = to_adc_counts(emg_left,  ADC_MID, ADC_MAX)
    adc_right = to_adc_counts(emg_right, ADC_MID, ADC_MAX)
    adc_float = np.full(N_SAMPLES, ADC_MID, dtype=int)  # ch2-5: unused floating

    # Write raw CSV in hardware format
    out_path = Path("mock_data") / "output" / "TB_REALISTIC_10MIN_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[3/4] Writing raw CSV  (hardware format — same as Uno R4 output)...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        # Metadata header block
        f.write("# TensionBudget Session Recording\n")
        f.write("# Session ID: TB_REALISTIC_10MIN\n")
        f.write("# Board: UNO-R4\n")
        f.write("# Sampling Rate: 500 Hz\n")
        f.write("# Resolution: 14 bits\n")
        f.write("# Duration: 10 min (600 s)\n")
        f.write("# Scenario: Ergonomic data-entry workstation with right-dominant mouse use\n")
        f.write("# Calibration reference RMS (shrug ~40% MVC): "
                f"{CALIB_REF_RMS:.2f} ADC counts\n")
        f.write("#\n")
        f.write("# Activity segments:\n")
        for seg in SEGMENTS:
            f.write(f"#   {seg[0]:3d}s-{seg[1]:3d}s  L={seg[2]:.1f}%MVC "
                    f"R={seg[3]:.1f}%MVC  — {seg[4]}\n")
        f.write("#\n")
        f.write("# Column guide:\n")
        f.write("#   Sample_Index    : sequential sample number (1-based)\n")
        f.write("#   Timestamp_s     : elapsed time in seconds\n")
        f.write("#   Arduino_Counter : 8-bit hardware timer (0-255 wrapping)\n")
        f.write("#   Channel1        : LEFT trapezius EMG (14-bit ADC, 0-16383)\n")
        f.write("#   Channel2        : RIGHT trapezius EMG\n")
        f.write("#   Channel3-6      : unused firmware slots (floating pins ~ 8191)\n")
        f.write("#\n")
        f.write("Sample_Index,Timestamp_s,Arduino_Counter,Channel1,Channel2,"
                "Channel3,Channel4,Channel5,Channel6\n")

        for i in range(N_SAMPLES):
            ts = i / FS
            counter = i % 256
            row = (
                f"{i+1},{ts:.6f},{counter},"
                f"{adc_left[i]},{adc_right[i]},"
                f"{adc_float[i]},{adc_float[i]},{adc_float[i]},{adc_float[i]}\n"
            )
            f.write(row)

    print(f"    Saved: {out_path.resolve()}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Write calibration JSON (auto-loaded by the app on startup)
    import json, time
    calib_path = Path("tensionbudget_calibration.json")
    calib_data = {
        "left":  {"reference_rms": CALIB_REF_RMS},
        "right": {"reference_rms": CALIB_REF_RMS},
        "timestamp": time.time(),
        "note": (
            "Auto-generated calibration for offline mock testing. "
            f"Reference = {CALIB_REF_RMS:.1f} ADC counts, representing "
            "~40% MVC voluntary shrug on 14-bit Uno R4 at 500 Hz. "
            "Replace with real shrug-hold reference when hardware is available."
        )
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=4)

    print(f"[4/4] Calibration JSON written: {calib_path.resolve()}")

    print("\n" + "=" * 60)
    print("  DONE! To replay in the monitor:")
    print("    python -m chordspy.tensionbudget_app")
    print("    → Select CSV Replay Mode")
    print("    → Load:  mock_data/output/TB_REALISTIC_10MIN_raw.csv")
    print("    → Press Start Replay")
    print()
    print("  Expected dashboard behaviour:")
    print("    0-30s   : EIndex ~ -2.0, scores near 0 (resting)")
    print("    30-90s  : EIndex rises, mild right asymmetry")
    print("    200-240s: EIndex drops back to -2 (rest gap)")
    print("    340-360s: EIndex peaks → HIGH BURST event visible")
    print("    Spectral slope: should show negative drift (fatigue)")
    print("=" * 60)


if __name__ == "__main__":
    main()
