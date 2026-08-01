"""
TB Raw Session File Generator
==============================
Generates a TB_mock_session_raw.csv in the exact format that:
  - TBRecorder writes during a real live hardware session
  - process_tb_session() reads and runs through the full pipeline

No chordspy imports needed — completely standalone (only numpy + scipy).

CONFIGURE AND RUN:
    python mock_data/generate_tb_raw.py

OUTPUT FILES:
    <OUTPUT_DIR>/TB_mock_<SESSION_ID>_raw.csv   <- feed this into process_tb_session()
    <OUTPUT_DIR>/TB_mock_<SESSION_ID>_raw.json  <- calibration sidecar (auto-loaded)
"""

import csv
import json
import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit everything in this block
# ════════════════════════════════════════════════════════════════════════════

# ── Session duration ─────────────────────────────────────────────────────────
DURATION_SECONDS = 7200       # How many seconds of data to generate
                             # 60 = 1 min / 300 = 5 min / 3600 = 1 hour

# ── Channel mode ─────────────────────────────────────────────────────────────
# "bilateral"  → 2 real EMG channels (Channel1=Left, Channel2=Right)
# "single"     → 1 real EMG channel  (Channel1 only)
CHANNEL_MODE = "bilateral"

# ── Activity pattern ─────────────────────────────────────────────────────────
# Realistic 2-hour office/desk-work EMG session.
# Covers: sustained typing blocks, micro-rests, stretch breaks, short bursts.
# Works for both 1-hour (3600s) and 2-hour (7200s) — entries beyond
# DURATION_SECONDS are simply ignored by _build_envelope().
ACTIVITY_SCHEDULE = [
    # ── Minute 0-5: settle in, light warm-up ─────────────────────────────
    (0,     10,   "rest"),
    (10,    60,   "low"),

    # ── Minute 1-15: sustained typing block ──────────────────────────────
    (60,    120,  "moderate"),   # 60 s → 1-2 min SUMA bin
    (120,   122,  "rest"),       # micro-rest gap
    (122,   300,  "moderate"),   # 3 min typing → 2-4 min SUMA bin
    (300,   303,  "rest"),       # micro-rest
    (303,   360,  "moderate"),

    # ── Minute 6: short burst (reaching, adjusting monitor) ──────────────
    (360,   363,  "high"),
    (363,   366,  "rest"),
    (366,   369,  "high"),
    (369,   372,  "rest"),

    # ── Minute 6-20: back to typing ───────────────────────────────────────
    (372,   600,  "moderate"),   # ~4 min → 4-8 min SUMA bin
    (600,   603,  "rest"),       # micro-rest
    (603,   900,  "moderate"),   # 5 min block
    (900,   904,  "rest"),       # micro-rest

    # ── Minute 15: coffee break ────────────────────────────────────────────
    (904,   960,  "rest"),       # ~1 min walk/break → resets muscle

    # ── Minute 16-30: second work block ───────────────────────────────────
    (960,   1020, "low"),        # easing back in
    (1020,  1200, "moderate"),
    (1200,  1203, "rest"),       # micro-rest
    (1203,  1500, "moderate"),   # ~5 min → 4-8 min SUMA bin
    (1500,  1505, "rest"),
    (1505,  1800, "moderate"),

    # ── Minute 30: stretch / posture reset ────────────────────────────────
    (1800,  1860, "rest"),

    # ── Minute 31-45: increased load (deadline pressure) ──────────────────
    (1860,  1863, "high"),
    (1863,  1866, "rest"),
    (1866,  1869, "high"),
    (1869,  1872, "rest"),
    (1872,  2100, "moderate"),
    (2100,  2103, "rest"),
    (2103,  2400, "moderate"),   # ~5 min
    (2400,  2403, "rest"),
    (2403,  2700, "moderate"),

    # ── Minute 45: short break ─────────────────────────────────────────────
    (2700,  2760, "rest"),

    # ── Minute 46-60: moderate typing, some short bursts ──────────────────
    (2760,  2820, "low"),
    (2820,  3000, "moderate"),
    (3000,  3003, "high"),       # burst
    (3003,  3006, "rest"),
    (3006,  3009, "high"),
    (3009,  3012, "rest"),
    (3012,  3300, "moderate"),
    (3300,  3303, "rest"),
    (3303,  3600, "moderate"),

    # ────────────── HOUR 1 COMPLETE ─────────────────────────────────────────
    # Entries below only affect 2-hour (7200s) sessions

    # ── Minute 60: lunch break / posture reset ─────────────────────────────
    (3600,  3720, "rest"),       # 2 min full rest

    # ── Minute 62-75: post-lunch low activity ─────────────────────────────
    (3720,  3780, "low"),
    (3780,  4200, "moderate"),
    (4200,  4204, "rest"),
    (4204,  4500, "moderate"),

    # ── Minute 75: stretch ────────────────────────────────────────────────
    (4500,  4560, "rest"),

    # ── Minute 76-90: afternoon typing block ──────────────────────────────
    (4560,  4620, "low"),
    (4620,  4800, "moderate"),
    (4800,  4803, "rest"),
    (4803,  5100, "moderate"),
    (5100,  5103, "high"),       # burst
    (5103,  5106, "rest"),
    (5106,  5109, "high"),
    (5109,  5112, "rest"),
    (5112,  5400, "moderate"),

    # ── Minute 90: coffee break ───────────────────────────────────────────
    (5400,  5460, "rest"),

    # ── Minute 91-105: fatigue accumulating (slightly lower sustained amp) ─
    (5460,  5520, "low"),
    (5520,  5700, "moderate"),
    (5700,  5703, "rest"),
    (5703,  6000, "moderate"),
    (6000,  6003, "rest"),
    (6003,  6300, "moderate"),

    # ── Minute 105: quick break ───────────────────────────────────────────
    (6300,  6360, "rest"),

    # ── Minute 106-119: wrapping up ───────────────────────────────────────
    (6360,  6420, "low"),
    (6420,  6600, "moderate"),
    (6600,  6603, "high"),       # burst
    (6603,  6606, "rest"),
    (6606,  6900, "moderate"),
    (6900,  6903, "rest"),
    (6903,  7100, "moderate"),

    # ── Last 100s: wind down ──────────────────────────────────────────────
    (7100,  7150, "low"),
    (7150,  7200, "rest"),
]

# ── Amplitude table (ADC counts above/below mid-rail) ────────────────────────
# Tune these to change how "loud" each activity state is in the raw data
AMPLITUDE = {
    "rest":     3,    # just ADC floor noise (~8192 ±3 counts)
    "low":     30,    # light muscle activation
    "moderate": 80,   # typical desk-work load
    "high":    140,   # near-peak contraction
}

# ── Calibration reference (simulated shrug hold) ─────────────────────────────
# These go into the sidecar JSON and are loaded automatically by process_tb_session()
# Units: ADC counts (matching the raw CSV values)
# Set to None to omit calibration (pipeline will warn but still run)
CALIB_LEFT_RMS_ADC  = 120.5   # reference shrug RMS for Left channel
CALIB_RIGHT_RMS_ADC = 150.2   # reference shrug RMS for Right channel

# ── Right side exertion ratio ─────────────────────────────────────────────────
# >1.0 = Right works harder (visible as positive Laterality Index)
# <1.0 = Left works harder
# 1.0  = Symmetric
RIGHT_EXERTION_RATIO = 1.25

# ── Hardware metadata (matches your actual device) ────────────────────────────
BOARD           = "UNO-R4"
SAMPLING_RATE   = 500         # Hz — locked hardware value
RESOLUTION_BITS = 14          # ADC bits — locked hardware value
NUM_CHANNELS    = 6           # firmware always emits 6, regardless of wired electrodes
PROTOCOL        = "usb"
BAUD_RATE       = 230400

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("mock_data") / "output"
SESSION_ID  = f"TB_MOCK_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION (don't need to edit below unless experimenting)
# ════════════════════════════════════════════════════════════════════════════

MID_RAIL = (2 ** RESOLUTION_BITS - 1) // 2   # = 8191 for 14-bit


def _build_envelope(n_samples: int, schedule: list, fs: int) -> np.ndarray:
    """Build a sample-wise amplitude envelope from the activity schedule."""
    env = np.full(n_samples, AMPLITUDE["rest"], dtype=float)
    for start_s, end_s, state in schedule:
        i0 = int(start_s * fs)
        i1 = min(int(end_s * fs), n_samples)
        env[i0:i1] = AMPLITUDE.get(state, AMPLITUDE["rest"])
    return env


def _make_emg(n: int, envelope: np.ndarray, fs: int, seed: int) -> np.ndarray:
    """
    Synthesize a 14-bit ADC EMG channel.
    Bandlimited Gaussian noise scaled by the amplitude envelope,
    centred on MID_RAIL (Vcc/2 ≈ 8191 for BioAmp EXG Pill on 3.3 V).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    sos = butter(4, [20.0, 240.0], btype="bandpass", output="sos", fs=fs)
    filtered = sosfilt(sos, noise)
    raw = MID_RAIL + filtered * envelope
    return np.clip(np.round(raw), 0, 2**RESOLUTION_BITS - 1).astype(int)


def _make_unused(n: int, seed: int) -> np.ndarray:
    """Floating ADC pin — sits near MID_RAIL with ±2 count noise."""
    rng = np.random.default_rng(seed)
    return np.clip(
        np.round(MID_RAIL + rng.standard_normal(n) * 2), 0, 2**RESOLUTION_BITS - 1
    ).astype(int)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = DURATION_SECONDS * SAMPLING_RATE

    print(f"Generating {DURATION_SECONDS}s @ {SAMPLING_RATE} Hz = {n:,} samples")
    print(f"Mode: {CHANNEL_MODE}  |  Output: {OUTPUT_DIR}")

    # Build envelopes
    envelope_left  = _build_envelope(n, ACTIVITY_SCHEDULE, SAMPLING_RATE)
    envelope_right = _build_envelope(n, ACTIVITY_SCHEDULE, SAMPLING_RATE) * RIGHT_EXERTION_RATIO

    # Synthesise channels
    ch1 = _make_emg(n, envelope_left,  SAMPLING_RATE, seed=7)    # Left EMG
    ch2 = _make_emg(n, envelope_right, SAMPLING_RATE, seed=99)   # Right EMG
    ch3 = _make_unused(n, seed=1)
    ch4 = _make_unused(n, seed=2)
    ch5 = _make_unused(n, seed=3)
    ch6 = _make_unused(n, seed=4)

    if CHANNEL_MODE == "single":
        # Zero out Channel2 entirely for single-channel mode
        ch2 = _make_unused(n, seed=5)

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_name = f"{SESSION_ID}_raw.csv"
    csv_path = OUTPUT_DIR / csv_name

    channel_map_str = (
        "{'left': 'Channel1', 'right': 'Channel2'}" if CHANNEL_MODE == "bilateral"
        else "{'trapezius': 'Channel1'}"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        # Metadata header — same format TBRecorder writes
        fh.write(f"# TensionBudget Session Recording\n")
        fh.write(f"# Session ID: {SESSION_ID}\n")
        fh.write(f"# Board: {BOARD}\n")
        fh.write(f"# Sampling Rate: {SAMPLING_RATE} Hz\n")
        fh.write(f"# Resolution: {RESOLUTION_BITS} bits\n")
        fh.write(f"# Num Channels: {NUM_CHANNELS}\n")
        fh.write(f"# Protocol: {PROTOCOL} ({BAUD_RATE} baud)\n")
        fh.write(f"# Channel Map: {channel_map_str}\n")
        fh.write(f"# Duration: {DURATION_SECONDS} s\n")
        fh.write(f"# Channel mode: {CHANNEL_MODE}\n")
        fh.write(f"#\n")
        fh.write(f"# Column guide:\n")
        fh.write(f"#   Sample_Index    : sequential sample number (1-based)\n")
        fh.write(f"#   Timestamp_s     : elapsed time in seconds\n")
        fh.write(f"#   Arduino_Counter : 8-bit hardware timer (0-255 wrapping)\n")
        fh.write(f"#   Channel1        : {'LEFT trapezius EMG' if CHANNEL_MODE == 'bilateral' else 'trapezius EMG'} (14-bit ADC, 0-16383)\n")
        fh.write(f"#   Channel2        : {'RIGHT trapezius EMG' if CHANNEL_MODE == 'bilateral' else 'unused (floating pin ~ MID_RAIL)'}\n")
        fh.write(f"#   Channel3-6      : unused firmware slots (floating pins ~ {MID_RAIL})\n")
        fh.write(f"#\n")

        writer = csv.writer(fh)
        writer.writerow([
            "Sample_Index", "Timestamp_s", "Arduino_Counter",
            "Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6",
        ])

        for i in range(n):
            writer.writerow([
                i + 1,
                f"{i / SAMPLING_RATE:.6f}",
                i % 256,            # 8-bit counter wrap
                ch1[i], ch2[i], ch3[i], ch4[i], ch5[i], ch6[i],
            ])

    print(f"  CSV written : {csv_path}  ({csv_path.stat().st_size / 1e6:.2f} MB)")

    # ── Write sidecar calibration JSON ───────────────────────────────────────
    json_name = f"{SESSION_ID}_raw.json"
    json_path = OUTPUT_DIR / json_name

    calib = {}
    if CHANNEL_MODE == "bilateral":
        calib = {
            "left":  {"reference_rms": CALIB_LEFT_RMS_ADC,  "unit": "ADC_counts", "type": "RVE"},
            "right": {"reference_rms": CALIB_RIGHT_RMS_ADC, "unit": "ADC_counts", "type": "RVE"},
        }
    elif CHANNEL_MODE == "single":
        calib = {
            "trapezius": {"reference_rms": CALIB_LEFT_RMS_ADC, "unit": "ADC_counts", "type": "RVE"},
        }

    manifest = {
        "session_id": SESSION_ID,
        "board": BOARD,
        "sampling_rate": SAMPLING_RATE,
        "resolution_bits": RESOLUTION_BITS,
        "channel_mode": CHANNEL_MODE,
        "duration_s": DURATION_SECONDS,
        "calibration": calib,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)
    print(f"  JSON written: {json_path}")

    # ── Preview first 5 data rows ─────────────────────────────────────────────
    print()
    print("  First 5 rows of CSV:")
    print(f"  {'Sample_Index':>12} {'Timestamp_s':>12} {'Arduino_Counter':>15} "
          f"{'Channel1':>10} {'Channel2':>10} {'Channel3':>10} ... (Ch4-6 similar)")
    for i in range(5):
        print(f"  {i+1:>12} {i/SAMPLING_RATE:>12.6f} {i%256:>15} "
              f"{ch1[i]:>10} {ch2[i]:>10} {ch3[i]:>10} ...")

    print()
    print(f"  To run through the pipeline:")
    print(f'    from chordspy.tensionbudget.preprocessing import process_tb_session')
    print(f'    result = process_tb_session("{csv_path}")')


if __name__ == "__main__":
    main()
