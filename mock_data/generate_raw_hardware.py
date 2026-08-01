"""
Generate small, readable mock raw hardware files.
No pipeline, no analysis — just what the Arduino + BioAmp EXG Pill emits.
"""

import csv
import numpy as np
from pathlib import Path

np.random.seed(42)
FS = 500          # Hz
ROWS = 500        # 1 second of data — small enough to open and read
MID_RAIL = 8192   # 14-bit ADC mid-point (Vcc/2), BioAmp EXG Pill sits here

out = Path("mock_data") / "raw_hardware_samples"
out.mkdir(parents=True, exist_ok=True)


def emg_noise(n, amplitude=80):
    """Bandlimited noise simulating an active muscle burst on a 14-bit ADC."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [20.0, 240.0], btype='bandpass', output='sos', fs=FS)
    noise = np.random.normal(0, 1, n)
    return np.clip(np.round(MID_RAIL + sosfilt(sos, noise) * amplitude), 0, 16383).astype(int)


def baseline(n, amplitude=3):
    """Resting baseline — tiny noise on top of mid-rail."""
    return np.clip(np.round(MID_RAIL + np.random.normal(0, amplitude, n)), 0, 16383).astype(int)


# ────────────────────────────────────────────────────────────────────
# FILE 1: 2-channel bilateral recording
#   Channel1 = Left trapezius  (active — moderate contraction)
#   Channel2 = Right trapezius (active — slightly higher exertion)
#   Channel3-6 = unused firmware slots (near-constant mid-rail noise)
# ────────────────────────────────────────────────────────────────────
p2ch = out / "bilateral_2ch_raw.csv"
with open(p2ch, "w", newline="", encoding="utf-8") as f:
    f.write("# Hardware: Arduino UNO R4 Minima + 2x BioAmp EXG Pill\n")
    f.write("# Sampling Rate: 500 Hz  |  ADC: 14-bit (0-16383)  |  Baud: 230400\n")
    f.write("# Electrode plan: 2 bipolar channels, 1 shared neck reference\n")
    f.write("#   Channel1 = LEFT trapezius EMG\n")
    f.write("#   Channel2 = RIGHT trapezius EMG (slightly higher load in this session)\n")
    f.write("#   Channel3-6 = unused firmware slots (firmware always emits 6)\n")
    f.write("# NOTE: Firmware hardcodes NUM_CHANNELS=6 — always 6 columns regardless\n")
    f.write("#       of how many electrodes are physically wired.\n")
    f.write("# Counter = 8-bit hardware timer (wraps 0->255->0->...)\n")
    f.write("#\n")
    w = csv.writer(f)
    w.writerow(["Counter", "Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6"])

    left  = emg_noise(ROWS, amplitude=80)   # Left: moderate load
    right = emg_noise(ROWS, amplitude=100)  # Right: 25% higher load
    ch3   = baseline(ROWS, amplitude=2)
    ch4   = baseline(ROWS, amplitude=2)
    ch5   = baseline(ROWS, amplitude=2)
    ch6   = baseline(ROWS, amplitude=2)

    for i in range(ROWS):
        w.writerow([i % 256, left[i], right[i], ch3[i], ch4[i], ch5[i], ch6[i]])

print(f"Written: {p2ch}  ({p2ch.stat().st_size:,} bytes, {ROWS} rows = {ROWS/FS:.1f}s)")


# ────────────────────────────────────────────────────────────────────
# FILE 2: 1-channel single-side recording
#   Channel1 = one trapezius (L or R — determined after hardware test)
#   Channel2-6 = unused
# ────────────────────────────────────────────────────────────────────
p1ch = out / "single_1ch_raw.csv"
with open(p1ch, "w", newline="", encoding="utf-8") as f:
    f.write("# Hardware: Arduino UNO R4 Minima + 1x BioAmp EXG Pill\n")
    f.write("# Sampling Rate: 500 Hz  |  ADC: 14-bit (0-16383)  |  Baud: 230400\n")
    f.write("# Electrode plan: 1 bipolar channel (3 electrodes: +, -, ref)\n")
    f.write("#   Channel1 = ONE trapezius EMG (L or R — verify with hardware test)\n")
    f.write("#   Channel2-6 = unused firmware slots\n")
    f.write("# NOTE: Firmware hardcodes NUM_CHANNELS=6 — always 6 columns regardless.\n")
    f.write("#\n")
    w = csv.writer(f)
    w.writerow(["Counter", "Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6"])

    trap = emg_noise(ROWS, amplitude=75)
    noise_cols = [baseline(ROWS, amplitude=2) for _ in range(5)]

    for i in range(ROWS):
        w.writerow([i % 256, trap[i]] + [noise_cols[j][i] for j in range(5)])

print(f"Written: {p1ch}  ({p1ch.stat().st_size:,} bytes, {ROWS} rows = {ROWS/FS:.1f}s)")


# ────────────────────────────────────────────────────────────────────
# Show the first 10 rows of each so the user can see the format
# ────────────────────────────────────────────────────────────────────
for path in [p2ch, p1ch]:
    print(f"\n{'─'*60}")
    print(f"  {path.name}  — first 10 data rows:")
    print(f"{'─'*60}")
    lines = path.read_text(encoding="utf-8").splitlines()
    # Print comment headers then first 10 data rows
    for line in lines:
        if line.startswith("#"):
            print(f"  {line}")
    data_lines = [l for l in lines if not l.startswith("#")]
    for line in data_lines[:11]:  # header + 10 rows
        print(f"  {line}")
