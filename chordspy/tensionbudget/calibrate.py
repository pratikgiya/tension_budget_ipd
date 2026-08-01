"""
TensionBudget — Phase 4 Calibration CLI

Command-line tool to perform a Reference Voluntary Exertion (RVE) calibration.
Prompts the user to perform a 5-second contraction for the left trapezius,
then the right trapezius. Computes the reference RMS for each side and
saves it to tensionbudget_calibration.json.

Usage:
    python -m chordspy.tensionbudget.calibrate
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

from chordspy.chords_serial import Chords_USB
from chordspy.connection import Connection
from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.calibration import compute_reference_rms


def _record_side(conn, side_name, channel_index, duration_s=5.0, fs=500):
    """Records raw data for a specific side for a set duration."""
    print(f"\n--- Calibrating {side_name.upper()} Trapezius ---")
    print(f"When ready, you will perform a {duration_s}-second reference shrug.")
    print("Instruction: RAMP UP to your reference effort, HOLD steady, then RELAX.")
    input("Press ENTER to start recording...")

    # Data collection buffer
    expected_samples = int(duration_s * fs)
    raw_data = []

    def sample_callback(packet):
        # packet structure: (timestamp_ms, [arduino_counter, ch1, ch2, ch3, ch4, ch5, ch6])
        # ch1 is index 1 in the data list, so if channel_index is 0, it's packet[1][1]
        data = packet[1]
        raw_val = data[channel_index + 1] # +1 because arduino_counter is index 0
        raw_data.append(raw_val)

    # Register callback and start recording
    print("\nRecording... (RAMP UP -> HOLD -> RELAX)")
    conn.register_sample_callback(sample_callback)
    
    start_time = time.time()
    
    # Progress bar loop
    while (time.time() - start_time) < duration_s:
        elapsed = time.time() - start_time
        progress = int((elapsed / duration_s) * 20)
        bar = "[" + "=" * progress + " " * (20 - progress) + "]"
        print(f"\r{bar} {elapsed:.1f}/{duration_s:.1f}s", end="")
        time.sleep(0.1)

    # Clean up
    conn.remove_sample_callback(sample_callback)
    print("\nDone recording.")

    # We might have slightly fewer or more samples due to timing, that's fine.
    return np.array(raw_data)


def main():
    print("=" * 50)
    print(" TensionBudget Calibration (RVE)")
    print("=" * 50)

    # 1. Resolve channel map
    channel_map = TBConfig.CHANNEL_MAP
    if channel_map is None:
        print("Error: TBConfig.CHANNEL_MAP is not set.")
        print("You must physically test and assign the correct raw channels to 'left' and 'right' in config.py before calibrating.")
        sys.exit(1)

    if "left" not in channel_map or "right" not in channel_map:
        print("Error: CHANNEL_MAP must contain 'left' and 'right' keys.")
        sys.exit(1)

    fs = TBConfig.TARGET_SAMPLING_RATE

    # 2. Connect to device
    print("\nConnecting to Chords BioAmp device...")
    try:
        usb = Chords_USB()
        conn = Connection(usb)
        
        # Give it a moment to stabilize
        time.sleep(1.0)
        
        if not conn.is_connected():
            print("Failed to connect to device. Is it plugged in?")
            sys.exit(1)
            
        print("Connected!")
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    calibration_data = {}

    try:
        # 3. Record and process Left
        left_raw = _record_side(conn, "left", channel_map["left"], duration_s=5.0, fs=fs)
        left_rms = compute_reference_rms(left_raw, fs=fs, config=TBConfig)
        
        print(f"-> Left Reference RMS calculated: {left_rms:.2f}")
        
        calibration_data["left"] = {
            "reference_rms": left_rms,
            "reference_type": "RVE"
        }

        # 4. Record and process Right
        right_raw = _record_side(conn, "right", channel_map["right"], duration_s=5.0, fs=fs)
        right_rms = compute_reference_rms(right_raw, fs=fs, config=TBConfig)
        
        print(f"-> Right Reference RMS calculated: {right_rms:.2f}")

        calibration_data["right"] = {
            "reference_rms": right_rms,
            "reference_type": "RVE"
        }

    finally:
        # Disconnect safely
        conn.stop()

    # 5. Save calibration data
    cal_file = Path.cwd() / "tensionbudget_calibration.json"
    with open(cal_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print("\n=" * 50)
    print("Calibration Complete!")
    print(f"Saved to: {cal_file.name}")
    print("=" * 50)


if __name__ == "__main__":
    main()
