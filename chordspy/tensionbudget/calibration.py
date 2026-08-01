"""
TensionBudget — Phase 4 Calibration Layer

Provides functions to compute reference RMS from a calibration recording.
Supports Reference Voluntary Exertion (RVE) or Maximum Voluntary Contraction (MVC).
"""

import numpy as np

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.preprocessing import preprocess_channel


def compute_reference_rms(raw_signal, fs=None, config=None, vref=None):
    """
    Computes the reference RMS value from a calibration recording.

    The recording is assumed to be a standard 5-second shrug (ramp up, 
    hold steady, relax). This function extracts the middle 3-second 
    "stable" window from the RMS envelope and returns the mean RMS 
    over that window.

    NOTE: The 5-second recording duration and 3-second extraction window 
    are project-specific engineering choices for robust baseline capture, 
    not direct derivatives of the core cited literature.

    Args:
        raw_signal: 1-D numpy array of raw ADC values for one channel.
        fs: Sampling rate in Hz. Defaults from config.
        config: TBConfig class (or compatible). Defaults to TBConfig.
        vref: ADC reference voltage.

    Returns:
        float: The mean RMS value of the middle 3-second window.
    """
    cfg = config or TBConfig
    fs = fs or cfg.TARGET_SAMPLING_RATE

    # Total expected duration is 5 seconds. We want the middle 3 seconds.
    # That means we drop the first 1 second and the last 1 second.
    skip_seconds = 1.0

    # 1. Run the standard preprocessing pipeline to get the RMS envelope
    result = preprocess_channel(raw_signal, fs=fs, config=cfg, vref=vref)
    rms_values = result['rms']
    rms_indices = result['rms_indices']

    if len(rms_values) == 0:
        return np.nan

    # 2. Convert sample indices back to time to slice the middle window
    # rms_indices represent the center sample of each RMS window
    times_s = rms_indices / fs

    total_duration_s = len(raw_signal) / fs

    # If the recording is shorter than 3 seconds, we can't reliably extract
    # a middle 3-second window. Fall back to using the middle 60% of whatever we have.
    if total_duration_s <= 3.0:
        skip_seconds = total_duration_s * 0.2

    start_time = skip_seconds
    end_time = total_duration_s - skip_seconds

    # Create a mask for the stable window
    mask = (times_s >= start_time) & (times_s <= end_time)
    
    stable_rms = rms_values[mask]

    if len(stable_rms) == 0:
        # Fallback if the mask somehow excluded everything
        return float(np.mean(rms_values))

    # 3. Compute the mean of the stable RMS window
    reference_rms = float(np.mean(stable_rms))

    return reference_rms
