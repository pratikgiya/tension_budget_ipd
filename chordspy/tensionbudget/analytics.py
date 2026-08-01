"""
TensionBudget — Stage 3 Core Analytics Engine.

Computes rule-based physiological metrics (gaps, sustained activity, APDF, and asymmetry)
from normalized %RVE envelopes. This is designed to operate as pure functions that can
run against a full session array (offline) or a rolling buffer (live).
"""

import numpy as np

from chordspy.tensionbudget.config import TBConfig


def compute_active_mask(normalized_rms, config=None):
    """
    Generate boolean mask where True = active tension, False = resting.

    Args:
        normalized_rms: 1-D numpy array of %RVE values.
        config: TBConfig class.

    Returns:
        1-D boolean numpy array.
    """
    cfg = config or TBConfig
    return normalized_rms >= cfg.GAP_REST_THRESHOLD_PCT


def detect_gaps(normalized_rms, config=None):
    """
    Detect relaxation gaps (micro-rests).
    A gap is any period where tension is < REST_THRESHOLD_PCT for at least MIN_GAP_DURATION_S.

    Args:
        normalized_rms: 1-D numpy array of %RVE values.
        config: TBConfig class.

    Returns:
        List of gap dictionaries: [{'start_idx': i, 'end_idx': j, 'duration_s': d}, ...]
    """
    cfg = config or TBConfig
    
    # Below threshold = True (resting)
    rest_mask = normalized_rms < cfg.GAP_REST_THRESHOLD_PCT
    
    # Find transitions
    # prepend 0 and append 0 to catch edges
    padded = np.insert(rest_mask, 0, False)
    padded = np.append(padded, False)
    diff = np.diff(padded.astype(int))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    min_samples = cfg.min_gap_samples()
    step_s = cfg.RMS_STEP_MS / 1000.0
    
    gaps = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= min_samples:
            gaps.append({
                'start_idx': int(s),
                'end_idx': int(e),
                'duration_s': float(length * step_s)
            })
            
    return gaps


def detect_sustained_activity(normalized_rms, config=None):
    """
    Detect sustained activity (SUMA-style Cinderella-fiber events).
    Tension > SUSTAINED_ACTIVITY_THRESHOLD_PCT for at least MIN_SUSTAINED_DURATION_S.
    
    Bins the events into specific Koch 2024 duration classes.

    Args:
        normalized_rms: 1-D numpy array of %RVE values.
        config: TBConfig class.

    Returns:
        Tuple of (events, bin_counts):
            events: List of dicts [{'start_idx': i, 'end_idx': j, 'duration_s': d}, ...]
            bin_counts: Dict with counts for each duration bin.
    """
    cfg = config or TBConfig
    
    # Above threshold = True (active)
    active_mask = normalized_rms > cfg.SUSTAINED_ACTIVITY_THRESHOLD_PCT
    
    # Find transitions
    padded = np.insert(active_mask, 0, False)
    padded = np.append(padded, False)
    diff = np.diff(padded.astype(int))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    min_samples = cfg.sustained_min_samples()
    step_s = cfg.RMS_STEP_MS / 1000.0
    
    events = []
    
    # Bins from Koch 2024
    bin_counts = {
        "1.5-5s": 0,
        "5-10s": 0,
        "10-20s": 0,
        "20-60s": 0,
        "1-2min": 0,
        "2-4min": 0,
        "4-8min": 0,
        "8-10min": 0,
        "10-20min": 0,
        ">20min": 0
    }
    
    def get_bin(dur_s):
        if dur_s < 5: return "1.5-5s"
        if dur_s < 10: return "5-10s"
        if dur_s < 20: return "10-20s"
        if dur_s < 60: return "20-60s"
        if dur_s < 120: return "1-2min"
        if dur_s < 240: return "2-4min"
        if dur_s < 480: return "4-8min"
        if dur_s < 600: return "8-10min"
        if dur_s < 1200: return "10-20min"
        return ">20min"
    
    for s, e in zip(starts, ends):
        length = e - s
        if length >= min_samples:
            dur_s = length * step_s
            events.append({
                'start_idx': int(s),
                'end_idx': int(e),
                'duration_s': float(dur_s)
            })
            b = get_bin(dur_s)
            bin_counts[b] += 1
            
    return events, bin_counts


def compute_apdf(normalized_rms, config=None, active_only=False):
    """
    Compute Amplitude Probability Distribution Function (APDF) percentiles.
    Usually 10th (static level), 50th (median), and 90th (peak level).

    Args:
        normalized_rms: 1-D numpy array of %RVE values.
        config: TBConfig class.
        active_only: If True, computes percentiles ONLY on samples >= REST_THRESHOLD_PCT.
                     This strips out rest periods to compute the true 'Active APDF'.

    Returns:
        Dict mapping percentile -> value (e.g., {10: 2.1, 50: 8.5, 90: 22.4})
    """
    cfg = config or TBConfig
    
    valid_data = normalized_rms[~np.isnan(normalized_rms)]
    
    if active_only:
        valid_data = valid_data[valid_data >= cfg.GAP_REST_THRESHOLD_PCT]
        
    if len(valid_data) == 0:
        return {p: np.nan for p in cfg.APDF_PERCENTILES}
        
    percentiles = np.percentile(valid_data, cfg.APDF_PERCENTILES)
    return {p: float(v) for p, v in zip(cfg.APDF_PERCENTILES, percentiles)}


def compute_asymmetry(apdf50_left, apdf50_right):
    """
    Compute Side-to-Side Asymmetry using the Laterality Index.
    
    AI = (Right - Left) / (Right + Left)
    Bounded between -1.0 (fully left) and +1.0 (fully right).
    0.0 means perfectly symmetric.
    
    NOTE: This is a project-specific addition, not a direct literature derivative.

    Args:
        apdf50_left: Median %RVE of left trapezius.
        apdf50_right: Median %RVE of right trapezius.

    Returns:
        float: Laterality index [-1.0 to 1.0], or NaN if invalid.
    """
    if np.isnan(apdf50_left) or np.isnan(apdf50_right):
        return np.nan
        
    denom = apdf50_right + apdf50_left
    if denom == 0:
        return 0.0  # Both sides perfectly at 0 tension
        
    return float((apdf50_right - apdf50_left) / denom)


def analyze_channel(normalized_rms, config=None, is_live_buffer=False):
    """
    Run full analytics suite on a single channel's normalized %RVE stream.

    Args:
        normalized_rms: 1-D numpy array of %RVE values.
        config: TBConfig class.
        is_live_buffer: If True, indicates this is a rolling buffer (e.g. 10 mins).
                        If False, it's a full session.

    Returns:
        Dict of computed analytics.
    """
    cfg = config or TBConfig
    step_s = cfg.RMS_STEP_MS / 1000.0
    
    gaps = detect_gaps(normalized_rms, config=cfg)
    suma_events, suma_bins = detect_sustained_activity(normalized_rms, config=cfg)
    session_apdf = compute_apdf(normalized_rms, config=cfg, active_only=False)
    active_apdf = compute_apdf(normalized_rms, config=cfg, active_only=True)
    
    mask = compute_active_mask(normalized_rms, config=cfg)
    active_samples = np.sum(mask)
    total_active_time_s = active_samples * step_s
    
    total_gap_time_s = sum(g['duration_s'] for g in gaps)
    
    return {
        "session_apdf": session_apdf,
        "active_apdf": active_apdf,
        "gaps_count": len(gaps),
        "gaps_total_time_s": total_gap_time_s,
        "gaps_events": gaps,
        "suma_count": len(suma_events),
        "suma_bins": suma_bins,
        "suma_events": suma_events,
        "total_active_time_s": total_active_time_s,
    }
