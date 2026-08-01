"""
TensionBudget Mock Data Generator
===================================
Synthesizes a realistic 3-minute bilateral trapezius EMG session,
runs it through the full TB preprocessing + analytics pipeline,
and writes all intermediate and final output files.

Run from the repo root:
    python -m mock_data.generate_mock_session
or:
    $env:PYTHONPATH="."; python mock_data/generate_mock_session.py
"""

import sys
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

# ── Block all optional hardware dependencies before chordspy loads ─────
# The main chordspy/__init__.py eagerly imports app.py -> connection.py
# which chains into serial, bleak, pylsl, Flask etc. None of those are
# needed for the TensionBudget analytics subpackage.
_MOCK_MODULES = [
    'serial', 'serial.tools', 'serial.tools.list_ports',
    'bleak', 'pylsl', 'flask', 'Flask',
    'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
    'pyqtgraph', 'pyaudio',
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# Now safe to import from the tensionbudget subpackage
import numpy as np
from scipy.signal import butter, sosfilt

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.preprocessing import process_tb_session
from chordspy.tensionbudget.session_report import generate_session_report


# ── Signal synthesis helpers ──────────────────────────────────────────

def _make_emg_signal(n_samples: int, fs: int = 500,
                     amplitude_scale: float = 1.0,
                     seed: int = 42) -> np.ndarray:
    """
    Synthesize a realistic 14-bit ADC EMG recording.

    Signal design (for a 180-second session):
    ─────────────────────────────────────────
    00 – 15 s  │ Baseline rest (~2 ADC counts RMS envelope)
    15 – 45 s  │ Sustained moderate contraction (~30 s continuous)
               │   → lands in Koch 20–60 s SUMA bin
    45 – 47 s  │ Micro-rest gap (~2 s below 3 % RVE threshold)
               │   → captured by Marker & Maluf gap detector
    47 – 130 s │ Long continuous desk-work tension (~83 s)
               │   → lands in Koch 1–2 min SUMA bin
    130 – 180 s│ Repeated short bursts (3 s active / 1.5 s rest)
               │   → populates short-event 1.5–5 s SUMA bins
               │   → drives the short-SUMA frequency penalty term
    """
    np.random.seed(seed)
    noise = np.random.normal(0, 1, n_samples)

    # Bandpass to physiological EMG spectrum (20-240 Hz)
    sos = butter(4, [20.0, 240.0], btype='bandpass', output='sos', fs=fs)
    emg = sosfilt(sos, noise)

    # Amplitude envelope schedule
    env = np.full(n_samples, 2.0)                          # baseline rest
    env[7_500:22_500] = 35.0 * amplitude_scale             # sustained 30 s
    env[22_500:23_500] = 1.2                               # micro-rest gap
    env[23_500:65_000] = 45.0 * amplitude_scale            # long desk work
    for burst in range(65_000, 85_000, 3_000):             # short bursts
        env[burst:burst + 1_500] = 22.0 * amplitude_scale  # 3 s active
        env[burst + 1_500:burst + 3_000] = 1.2             # 1.5 s rest
    env[85_000:] = 1.5                                     # trailing rest

    # BioAmp EXG Pill sits at Vcc/2 mid-rail ≈ 8192 counts (14-bit, 0–16383)
    adc_counts = 8192 + emg * env
    return np.clip(np.round(adc_counts), 0, 16383).astype(int)


# ── Main generation routine ───────────────────────────────────────────

def main():
    out_dir = Path("mock_data") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    FS = 500
    DURATION_S = 180
    N = FS * DURATION_S

    # ── Step 1: Synthesise bilateral EMG ──────────────────────────────
    print("Step 1/4 — Synthesising bilateral trapezius EMG (180 s @ 500 Hz)…")
    left_raw  = _make_emg_signal(N, fs=FS, amplitude_scale=1.00, seed=7)
    # Right side 25 % higher exertion → visible asymmetry in Laterality Index
    right_raw = _make_emg_signal(N, fs=FS, amplitude_scale=1.25, seed=99)
    unused = np.clip(
        np.round(8192 + np.random.normal(0, 1, (4, N))), 0, 16383
    ).astype(int)

    # ── Step 2: Write the raw TensionBudget CSV ────────────────────────
    csv_path = out_dir / "TB_mock_session_raw.csv"
    print(f"Step 2/4 — Writing raw session CSV → {csv_path.name}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        # TensionBudget metadata header (comment lines)
        for line in [
            "# TensionBudget Session Recording",
            "# Session ID: TB_MOCK_20260730_001",
            "# Board: UNO-R4",
            "# Sampling Rate: 500 Hz",
            "# Resolution: 14 bits",
            "# Num Channels: 6",
            "# Protocol: usb (230400 baud)",
            "# Channel Map: {'left': 'Channel1', 'right': 'Channel2'}",
        ]:
            fh.write(line + '\n')

        w = csv.writer(fh)
        w.writerow([
            "Sample_Index", "Timestamp_s", "Arduino_Counter",
            "Channel1", "Channel2",
            "Channel3", "Channel4", "Channel5", "Channel6",
        ])
        for i in range(N):
            w.writerow([
                i + 1,
                f"{i / FS:.6f}",
                i % 256,           # 8-bit wrapping hardware counter
                left_raw[i],
                right_raw[i],
                unused[0, i], unused[1, i], unused[2, i], unused[3, i],
            ])

    # Sidecar JSON calibration manifest
    json_path = out_dir / "TB_mock_session_raw.json"
    manifest = {
        "session_id": "TB_MOCK_20260730_001",
        "board": "UNO-R4",
        "sampling_rate": 500,
        "calibration": {
            # Simulated 5-s reference shrug: middle-3s RMS (in raw ADC counts)
            "left":  {"reference_rms": 120.5, "unit": "ADC_counts", "type": "RVE"},
            "right": {"reference_rms": 150.2, "unit": "ADC_counts", "type": "RVE"},
        },
    }
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=4)
    print(f"       Sidecar calibration JSON → {json_path.name}")

    # ── Step 3: Preprocessing pipeline ────────────────────────────────
    print("Step 3/4 — Running TensionBudget preprocessing pipeline…")
    print("           (Stage 1→2→3→4→5→6: voltage conversion, 20-240 Hz")
    print("            Butterworth SOS filter, rectification, 100ms RMS,")
    print("            %RVE normalization — both sides independently)")

    TBConfig.CHANNEL_MAP = {"left": 0, "right": 1}

    proc = process_tb_session(
        csv_path=str(csv_path),
        output_path=str(out_dir / "TB_mock_session_processed.csv"),
        config=TBConfig,
        vref=3.3,
    )
    print(f"       Full-rate processed CSV  → {Path(proc['output_path']).name}")
    rms_csv = Path(str(proc['output_path']).replace(
        '_processed.csv', '_processed_rms.csv'))
    print(f"       RMS-rate envelope CSV    → {rms_csv.name}")

    # ── Step 4: Analytics + Scoring ───────────────────────────────────
    print("Step 4/4 — Running analytics & scoring engine…")
    print("           (gaps, SUMA bins, APDF, EIndex, composite score)")

    report_path = out_dir / "TB_mock_session_analytics.json"
    report = generate_session_report(
        processed_csv_path=str(rms_csv),
        output_path=str(report_path),
        config=TBConfig,
    )

    # ── Pretty-print summary to console ───────────────────────────────
    print()
    print("=" * 65)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 65)

    for side, ch in report.get("channels", {}).items():
        print(f"\n  [{side.upper()} TRAPEZIUS]")
        gaps = ch.get("gaps", {})
        print(f"    Relaxation gaps : {gaps.get('count', 'N/A')} events")
        print(f"    Muscular rest % : {gaps.get('rest_pct', 0):.1f} %")
        print(f"    Gap freq/min    : {gaps.get('frequency_per_min', 0):.2f}")

        apdf = ch.get("active_apdf", {})
        print(f"    APDF P10/P50/P90: "
              f"{apdf.get(10, float('nan')):.1f} / "
              f"{apdf.get(50, float('nan')):.1f} / "
              f"{apdf.get(90, float('nan')):.1f}  %%RVE")

        suma = ch.get("suma_bins", {})
        print(f"    SUMA histogram  : {suma}")

    ai = report.get("asymmetry", {}).get("laterality_index", "N/A")
    print(f"\n  [ASYMMETRY]  Laterality Index (AI) = {ai:.4f}" 
          if isinstance(ai, float) else f"\n  [ASYMMETRY]  AI = {ai}")
    print(f"  (AI > 0 → Right dominant; AI < 0 → Left dominant)")
    print()
    print(f"  Output folder: {out_dir.absolute()}")
    print(f"  Files generated:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")
    print("=" * 65)


if __name__ == "__main__":
    main()
