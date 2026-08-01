"""
TensionBudget — Real Chords Web Data Adapter
=============================================
Loads the CSV files recorded from the legacy Chords Web interface and
runs them through the TensionBudget pipeline starting at Stage 2 (bandpass),
since the Chords Web UI already performs the ADC -> Voltage conversion.

FILE FORMAT (what Chords Web exports):
    Counter, Channel3, Channel4    <- 2-channel recording (e.g. L + R trapezius)
    Counter, Channel1              <- 1-channel recording (single side)

    Values are in Volts (float), already ADC-converted.
    No timestamps in the file — we reconstruct from sample index at 500 Hz.

PIPELINE ENTRY POINT:
    Stage 1 (ADC→Voltage)  : SKIPPED  — already done by Chords Web
    Stage 2 (Bandpass)     : ENTRY    — 20-240 Hz Butterworth SOS
    Stage 3 (Notch)        : auto     — only if mains spike detected
    Stage 4 (Rectify)      : np.abs()
    Stage 5 (RMS envelope) : 100ms window, 20ms step (Marker & Maluf 2016)
    Stage 6 (Normalize)    : /reference_rms * 100  -> %RVE
    Analytics              : gaps, SUMA, APDF, EIndex, composite score

CALIBRATION:
    Without a real shrug reference, we estimate reference_rms as the 95th
    percentile of the RMS envelope (a conservative "near-peak" proxy).
    This gives physically plausible %RVE values for exploration.
    Replace with your actual shrug-hold RMS when available.

Run from repo root:
    $env:PYTHONPATH="."; $env:PYTHONUTF8="1"; python mock_data/run_real_chords_data.py
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

# Block optional hardware deps before chordspy loads
for _mod in ['serial', 'serial.tools', 'serial.tools.list_ports',
             'bleak', 'pylsl', 'flask', 'Flask',
             'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui',
             'pyqtgraph', 'pyaudio']:
    sys.modules.setdefault(_mod, MagicMock())

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, welch

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.preprocessing import (
    bandpass_filter, notch_filter, rectify, rms_envelope,
    detect_mains_interference,
)
from chordspy.tensionbudget.analytics import analyze_channel, compute_asymmetry


# ── Configuration ────────────────────────────────────────────────────

DATA_DIR = Path("prev_data_from chords_web-pre-processed")
OUT_DIR  = Path("mock_data") / "real_data_output"

# The 3 recordings
RECORDINGS = [
    {
        "file": "ChordsWeb-20260327-143455.csv",
        "channels": {"Channel3": "trapezius"},   # 1 active EMG channel
        # or for 2-channel: {"Channel3": "left", "Channel4": "right"}
    },
    {
        "file": "ChordsWeb-20260327-143759.csv",
        "channels": {"Channel3": "left", "Channel4": "right"},
    },
    {
        "file": "ChordsWeb-20260327-145039.csv",
        "channels": {"Channel1": "trapezius"},
    },
]

FS = 500  # Hz — locked hardware sampling rate


# ── Core processing for one recording ────────────────────────────────

def process_chords_file(rec_cfg: dict) -> dict:
    fpath = DATA_DIR / rec_cfg["file"]
    channel_label_map = rec_cfg["channels"]  # {col_name: label}

    print(f"\n{'='*60}")
    print(f"  Processing: {fpath.name}")
    print(f"  Channels:   {channel_label_map}")

    # 1. Load CSV
    import csv
    header = None
    rows = []
    with open(fpath, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i == 0:
                header = [c.strip() for c in row]
            else:
                rows.append(row)

    # Build numpy arrays per column
    col_idx = {h: i for i, h in enumerate(header)}
    n_samples = len(rows)
    duration_s = n_samples / FS
    print(f"  Samples:    {n_samples:,}  ({duration_s:.1f} s = {duration_s/60:.1f} min)")

    raw_cols = {}
    for col in channel_label_map.keys():
        if col not in col_idx:
            print(f"  WARNING: column '{col}' not found in {fpath.name} — skipping")
            continue
        raw_cols[col] = np.array([float(r[col_idx[col]]) for r in rows])

    # Reconstruct timestamps from sample index (no real timestamps in Chords Web files)
    timestamps = np.arange(n_samples) / FS

    # 2. For each channel: bandpass → (notch if needed) → rectify → RMS → normalize
    results = {}
    for col, label in channel_label_map.items():
        if col not in raw_cols:
            continue

        voltage = raw_cols[col]  # Already in Volts from Chords Web
        print(f"\n  [{label.upper()} — {col}]")
        print(f"    Input range: [{voltage.min():.5f}, {voltage.max():.5f}] V")

        # Stage 2: 20–240 Hz bandpass
        filtered = bandpass_filter(voltage, fs=FS,
                                       low_hz=TBConfig.BANDPASS_LOW_HZ,
                                       high_hz=TBConfig.BANDPASS_HIGH_HZ,
                                       order=TBConfig.BANDPASS_ORDER)
        print(f"    After bandpass: RMS = {np.sqrt(np.mean(filtered**2)):.6f} V")

        # Stage 3: Notch (auto — only if mains interference detected)
        has_mains = detect_mains_interference(filtered, fs=FS,
                                                    freq_hz=TBConfig.NOTCH_FREQ_HZ)
        if has_mains:
            filtered = notch_filter(filtered, fs=FS,
                                        freq_hz=TBConfig.NOTCH_FREQ_HZ,
                                        quality_factor=TBConfig.NOTCH_QUALITY_FACTOR)
            print(f"    Mains spike detected → 50 Hz notch applied")
        else:
            print(f"    Mains check: no significant 50 Hz spike — notch skipped")

        # Stage 4: Full-wave rectification
        rect = rectify(filtered)

        # Stage 5: RMS envelope (100ms window, 20ms step)
        rms_vals, rms_idx = rms_envelope(rect,
                                             window_samples=TBConfig.rms_window_samples(),
                                             step_samples=TBConfig.rms_step_samples())
        rms_timestamps = timestamps[rms_idx]
        print(f"    RMS envelope: {len(rms_vals)} frames "
              f"(min={rms_vals.min():.6f}, max={rms_vals.max():.6f}, "
              f"mean={rms_vals.mean():.6f}) V")

        # Stage 6: Normalize to %RVE
        # Without a real calibration shrug, use 95th-percentile RMS as proxy
        reference_rms = np.percentile(rms_vals, 95)
        print(f"    Calibration (proxy 95th pct): {reference_rms:.6f} V")
        normalized = (rms_vals / reference_rms) * 100.0  # %RVE
        print(f"    Normalized %RVE: mean={normalized.mean():.1f}%, "
              f"max={normalized.max():.1f}%, "
              f"p50={np.percentile(normalized, 50):.1f}%")

        results[label] = {
            "voltage": voltage,
            "filtered": filtered,
            "rectified": rect,
            "rms": rms_vals,
            "rms_timestamps": rms_timestamps,
            "normalized": normalized,
            "reference_rms_v": float(reference_rms),
            "n_samples": n_samples,
            "duration_s": duration_s,
        }

    # 3. Analytics engine
    print(f"\n  [ANALYTICS]")
    analytics = {}
    for label, ch in results.items():
        norm = ch["normalized"]
        ch_report = analyze_channel(norm, config=TBConfig, is_live_buffer=False)
        analytics[label] = ch_report

        gaps = ch_report.get("gaps", {})
        apdf = ch_report.get("active_apdf", {})
        suma = ch_report.get("suma_bins", {})
        print(f"    {label}:")
        print(f"      Relaxation gaps : {gaps.get('count', 'N/A')} events  |  "
              f"rest% = {gaps.get('rest_pct', 0):.1f}%  |  "
              f"{gaps.get('frequency_per_min', 0):.2f}/min")
        print(f"      APDF P10/P50/P90: "
              f"{apdf.get(10, float('nan')):.1f} / "
              f"{apdf.get(50, float('nan')):.1f} / "
              f"{apdf.get(90, float('nan')):.1f}  %%RVE")
        active_short = suma.get('1.5-5s', 0) + suma.get('5-10s', 0)
        print(f"      SUMA short events (1.5-5s + 5-10s): {active_short}")
        print(f"      SUMA full: {suma}")

    # Asymmetry (only if bilateral)
    ai_info = {}
    labels = list(analytics.keys())
    if "left" in analytics and "right" in analytics:
        p50_l = analytics["left"]["active_apdf"].get(50, float("nan"))
        p50_r = analytics["right"]["active_apdf"].get(50, float("nan"))
        ai = compute_asymmetry(p50_l, p50_r)
        ai_info = {
            "laterality_index": float(ai),
            "left_p50_pct_rve": float(p50_l),
            "right_p50_pct_rve": float(p50_r),
            "interpretation": ("Right dominant" if ai > 0.05
                               else "Left dominant" if ai < -0.05
                               else "Symmetric (|AI| < 0.05)"),
        }
        print(f"\n    Laterality Index (AI) = {ai:.4f}  → {ai_info['interpretation']}")
    elif len(labels) == 1:
        print(f"\n    Single-channel recording — asymmetry not applicable.")

    # 4. Save outputs
    out_stem = fpath.stem
    out_dir_rec = OUT_DIR / out_stem
    out_dir_rec.mkdir(parents=True, exist_ok=True)

    # Write RMS + normalized CSV
    rms_csv = out_dir_rec / f"{out_stem}_rms.csv"
    import csv as _csv
    with open(rms_csv, 'w', newline='', encoding='utf-8') as fh:
        fh.write(f"# Source: {fpath.name}\n")
        fh.write(f"# Sampling Rate: {FS} Hz\n")
        fh.write(f"# Pipeline entry: Stage 2 (voltage already converted by Chords Web)\n")
        fh.write(f"# Bandpass: {TBConfig.BANDPASS_LOW_HZ}-{TBConfig.BANDPASS_HIGH_HZ} Hz\n")
        fh.write(f"# RMS window: {TBConfig.RMS_WINDOW_MS}ms, step: {TBConfig.RMS_STEP_MS}ms\n")
        fh.write(f"# Normalization: 95th-percentile proxy (replace with real shrug RMS)\n")
        wr = _csv.writer(fh)
        header_cols = ["RMS_Frame", "Timestamp_s"]
        for lbl in results:
            header_cols += [f"{lbl}_rms_v", f"{lbl}_normalized_pct_rve"]
        wr.writerow(header_cols)
        n_frames = len(list(results.values())[0]["rms"])
        ref_ts = list(results.values())[0]["rms_timestamps"]
        for i in range(n_frames):
            row = [i + 1, f"{ref_ts[i]:.4f}"]
            for lbl in results:
                row.append(f"{results[lbl]['rms'][i]:.8f}")
                row.append(f"{results[lbl]['normalized'][i]:.4f}")
            wr.writerow(row)

    # Write analytics JSON
    report = {
        "source_file": fpath.name,
        "duration_s": duration_s,
        "n_samples": n_samples,
        "sampling_rate_hz": FS,
        "pipeline_entry": "Stage2_bandpass",
        "calibration_note": "95th-percentile proxy — replace with real shrug RMS",
        "channels": {lbl: analytics[lbl] for lbl in analytics},
        "asymmetry": ai_info,
    }
    json_out = out_dir_rec / f"{out_stem}_analytics.json"
    def _serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(json_out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=4, default=_serial)

    print(f"\n  Outputs saved to: {out_dir_rec}")
    print(f"    {rms_csv.name}   ({rms_csv.stat().st_size:,} bytes)")
    print(f"    {json_out.name}  ({json_out.stat().st_size:,} bytes)")

    return report


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check/configure pipeline
    TBConfig.CHANNEL_MAP = None  # handled per-recording above

    all_reports = {}
    for rec in RECORDINGS:
        fpath = DATA_DIR / rec["file"]
        if not fpath.exists():
            print(f"Skipping (not found): {fpath}")
            continue
        report = process_chords_file(rec)
        all_reports[rec["file"]] = report

    print(f"\n\n{'='*60}")
    print("  ALL RECORDINGS PROCESSED")
    print(f"  Output root: {OUT_DIR.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
