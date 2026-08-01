"""
Run generate_tb_raw.py for both 1-hour and 2-hour sessions sequentially.
Run from repo root:
    $env:PYTHONUTF8="1"; python mock_data/run_both_sessions.py
"""

import sys
import datetime
from pathlib import Path

# Add repo root to path so the generator script functions are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.signal import butter, sosfilt
import numpy as np
import csv, json

# ── Copy the core functions from generate_tb_raw inline ──────────────────────
SAMPLING_RATE   = 500
RESOLUTION_BITS = 14
NUM_CHANNELS    = 6
BOARD           = "UNO-R4"
PROTOCOL        = "usb"
BAUD_RATE       = 230400
RIGHT_EXERTION_RATIO = 1.25
CALIB_LEFT_RMS_ADC   = 120.5
CALIB_RIGHT_RMS_ADC  = 150.2
CHANNEL_MODE    = "bilateral"
OUTPUT_DIR      = Path("mock_data") / "output"
MID_RAIL        = (2 ** RESOLUTION_BITS - 1) // 2

AMPLITUDE = {
    "rest":     3,
    "low":     30,
    "moderate": 80,
    "high":    140,
}

ACTIVITY_SCHEDULE = [
    (0,     10,   "rest"),
    (10,    60,   "low"),
    (60,    120,  "moderate"),
    (120,   122,  "rest"),
    (122,   300,  "moderate"),
    (300,   303,  "rest"),
    (303,   360,  "moderate"),
    (360,   363,  "high"),
    (363,   366,  "rest"),
    (366,   369,  "high"),
    (369,   372,  "rest"),
    (372,   600,  "moderate"),
    (600,   603,  "rest"),
    (603,   900,  "moderate"),
    (900,   904,  "rest"),
    (904,   960,  "rest"),
    (960,   1020, "low"),
    (1020,  1200, "moderate"),
    (1200,  1203, "rest"),
    (1203,  1500, "moderate"),
    (1500,  1505, "rest"),
    (1505,  1800, "moderate"),
    (1800,  1860, "rest"),
    (1860,  1863, "high"),
    (1863,  1866, "rest"),
    (1866,  1869, "high"),
    (1869,  1872, "rest"),
    (1872,  2100, "moderate"),
    (2100,  2103, "rest"),
    (2103,  2400, "moderate"),
    (2400,  2403, "rest"),
    (2403,  2700, "moderate"),
    (2700,  2760, "rest"),
    (2760,  2820, "low"),
    (2820,  3000, "moderate"),
    (3000,  3003, "high"),
    (3003,  3006, "rest"),
    (3006,  3009, "high"),
    (3009,  3012, "rest"),
    (3012,  3300, "moderate"),
    (3300,  3303, "rest"),
    (3303,  3600, "moderate"),
    # Hour 2
    (3600,  3720, "rest"),
    (3720,  3780, "low"),
    (3780,  4200, "moderate"),
    (4200,  4204, "rest"),
    (4204,  4500, "moderate"),
    (4500,  4560, "rest"),
    (4560,  4620, "low"),
    (4620,  4800, "moderate"),
    (4800,  4803, "rest"),
    (4803,  5100, "moderate"),
    (5100,  5103, "high"),
    (5103,  5106, "rest"),
    (5106,  5109, "high"),
    (5109,  5112, "rest"),
    (5112,  5400, "moderate"),
    (5400,  5460, "rest"),
    (5460,  5520, "low"),
    (5520,  5700, "moderate"),
    (5700,  5703, "rest"),
    (5703,  6000, "moderate"),
    (6000,  6003, "rest"),
    (6003,  6300, "moderate"),
    (6300,  6360, "rest"),
    (6360,  6420, "low"),
    (6420,  6600, "moderate"),
    (6600,  6603, "high"),
    (6603,  6606, "rest"),
    (6606,  6900, "moderate"),
    (6900,  6903, "rest"),
    (6903,  7100, "moderate"),
    (7100,  7150, "low"),
    (7150,  7200, "rest"),
]


def _build_envelope(n, schedule, fs):
    env = np.full(n, AMPLITUDE["rest"], dtype=float)
    for s, e, state in schedule:
        i0, i1 = int(s * fs), min(int(e * fs), n)
        env[i0:i1] = AMPLITUDE.get(state, AMPLITUDE["rest"])
    return env


def _make_emg(n, envelope, fs, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    sos = butter(4, [20.0, 240.0], btype="bandpass", output="sos", fs=fs)
    raw = MID_RAIL + sosfilt(sos, noise) * envelope
    return np.clip(np.round(raw), 0, 2**RESOLUTION_BITS - 1).astype(int)


def _make_unused(n, seed):
    rng = np.random.default_rng(seed)
    return np.clip(
        np.round(MID_RAIL + rng.standard_normal(n) * 2), 0, 2**RESOLUTION_BITS - 1
    ).astype(int)


def generate(duration_seconds: int, session_id: str):
    n = duration_seconds * SAMPLING_RATE
    print(f"  Samples : {n:,}  ({duration_seconds}s = {duration_seconds/3600:.1f} hr)")

    env_l = _build_envelope(n, ACTIVITY_SCHEDULE, SAMPLING_RATE)
    env_r = env_l * RIGHT_EXERTION_RATIO

    ch1 = _make_emg(n, env_l, SAMPLING_RATE, seed=7)
    ch2 = _make_emg(n, env_r, SAMPLING_RATE, seed=99)
    ch3, ch4, ch5, ch6 = [_make_unused(n, s) for s in [1, 2, 3, 4]]

    csv_path  = OUTPUT_DIR / f"{session_id}_raw.csv"
    json_path = OUTPUT_DIR / f"{session_id}_raw.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# TensionBudget Session Recording\n")
        fh.write(f"# Session ID: {session_id}\n")
        fh.write(f"# Board: {BOARD}\n")
        fh.write(f"# Sampling Rate: {SAMPLING_RATE} Hz\n")
        fh.write(f"# Resolution: {RESOLUTION_BITS} bits\n")
        fh.write(f"# Num Channels: {NUM_CHANNELS}\n")
        fh.write(f"# Protocol: {PROTOCOL} ({BAUD_RATE} baud)\n")
        fh.write(f"# Channel Map: {{'left': 'Channel1', 'right': 'Channel2'}}\n")
        fh.write(f"# Duration: {duration_seconds} s ({duration_seconds/3600:.1f} hr)\n")
        fh.write(f"# Channel1 = LEFT trapezius EMG\n")
        fh.write(f"# Channel2 = RIGHT trapezius EMG\n")
        fh.write(f"# Channel3-6 = unused firmware slots (floating ~{MID_RAIL})\n")
        fh.write(f"#\n")

        w = csv.writer(fh)
        w.writerow(["Sample_Index","Timestamp_s","Arduino_Counter",
                    "Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"])

        CHUNK = 50_000   # write in chunks so we don't hold everything in RAM strings
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            for i in range(start, end):
                w.writerow([
                    i + 1, f"{i/SAMPLING_RATE:.6f}", i % 256,
                    ch1[i], ch2[i], ch3[i], ch4[i], ch5[i], ch6[i],
                ])

    size_mb = csv_path.stat().st_size / 1e6
    print(f"  CSV     : {csv_path.name}  ({size_mb:.1f} MB)")

    manifest = {
        "session_id": session_id,
        "board": BOARD,
        "sampling_rate": SAMPLING_RATE,
        "duration_s": duration_seconds,
        "calibration": {
            "left":  {"reference_rms": CALIB_LEFT_RMS_ADC,  "unit": "ADC_counts", "type": "RVE"},
            "right": {"reference_rms": CALIB_RIGHT_RMS_ADC, "unit": "ADC_counts", "type": "RVE"},
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)
    print(f"  JSON    : {json_path.name}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Generating 1-hour session  (3600 s, ~180 MB)")
    print("=" * 55)
    generate(3600, "TB_MOCK_1HR_SESSION")

    print()
    print("=" * 55)
    print("  Generating 2-hour session  (7200 s, ~360 MB)")
    print("=" * 55)
    generate(7200, "TB_MOCK_2HR_SESSION")

    print()
    print("Both files ready in:", OUTPUT_DIR.absolute())
    print("Feed into pipeline with:")
    print("  from chordspy.tensionbudget.preprocessing import process_tb_session")
    print("  process_tb_session('mock_data/output/TB_MOCK_1HR_SESSION_raw.csv')")
    print("  process_tb_session('mock_data/output/TB_MOCK_2HR_SESSION_raw.csv')")
