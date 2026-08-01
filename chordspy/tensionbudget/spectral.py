"""
TensionBudget — Stage 10: Conditional MDF/MNF Spectral Fatigue.

Estimates median power frequency (MDF) and mean power frequency (MNF) of
upper-trapezius sEMG over time using Welch's method, then fits an OLS linear
slope to detect systematic fatigue-related frequency downshift.

CRITICAL IMPLEMENTATION NOTES
──────────────────────────────
1. Pre-rectification input required.
   This module must receive the Stage 2 bandpass-filtered signal BEFORE
   rectification (Stage 4). Rectification folds negative half-cycles into
   positive ones, creating strong harmonic distortion at DC and 2f, 3f, ...
   which completely invalidates spectral frequency estimates.
   Pipeline tap: output of bandpass_filter(), before rectify().

2. Passband ceiling caveat (240 Hz, not 450 Hz).
   Our locked hardware sampling rate fs=500 Hz → Nyquist=250 Hz → bandpass
   ceiling 240 Hz (margin for filter roll-off). De Luca (1997) and Farina (2002)
   assume fs≥1000 Hz with a 20–450 Hz passband. Consequently:
     - Our absolute MDF/MNF values will be systematically LOWER than published
       norms (signal power above 240 Hz is missing from the integral).
     - MDF/MNF must be interpreted as INTERNAL SESSION TREND INDICATORS ONLY,
       not compared against published absolute baselines.
   This caveat is embedded in all output dictionaries under the key
   'passband_ceiling_caveat'.

3. Active-epoch gating (project heuristic, not literature-sourced).
   Epochs where <50% of samples are above GAP_REST_THRESHOLD_PCT (3.0% RVE)
   are skipped. Computing frequency peaks on electrical resting-floor noise
   is meaningless and would corrupt the slope estimate.

4. Bilateral independence.
   Left and right channels are processed completely independently at every
   step. No cross-channel mixing occurs before the final output dict.

References:
    De Luca, C.J. (1997). The use of surface electromyography in biomechanics.
        J. Appl. Biomech., 13(2), 135–163.
    Farina, D., Merletti, R., & Enoka, R.M. (2004). The extraction of neural
        strategies from the surface EMG. J. Appl. Physiol., 96(4), 1486–1495.
    Farina, D., et al. (2002). Comparison of algorithms for estimation of EMG
        variables during voluntary isometric contractions. J. Electromyogr.
        Kinesiol., 12(5), 337–349.
"""

import numpy as np
from scipy.signal import welch

from chordspy.tensionbudget.config import TBConfig

# Caveat string embedded in every output dict (see note 2 above)
_PASSBAND_CAVEAT = (
    "MDF/MNF computed on 20-240 Hz band (Nyquist ceiling at fs=500 Hz). "
    "De Luca/Farina standard uses 20-450 Hz (fs>=1000 Hz). "
    "Absolute MDF/MNF values will be lower than published norms. "
    "Interpret as internal session trend/slope only — "
    "do NOT compare against published absolute baselines."
)


# ── 1. Welch PSD ─────────────────────────────────────────────────────────────

def compute_welch_psd(signal, fs=None, window_sec=None, overlap_sec=None,
                      band=None, config=None):
    """
    Estimate power spectral density of a single-channel EMG signal.

    Uses Welch's method with a Hanning window. Output is sliced to the
    valid passband [SPECTRAL_BAND_LOW_HZ, SPECTRAL_BAND_HIGH_HZ] before
    returning, so callers never see out-of-band artefacts.

    Args:
        signal:      1-D numpy array. Must be the PRE-RECTIFICATION bandpass-
                     filtered signal (Stage 2 output). See module docstring.
        fs:          Sampling rate (Hz). Defaults to TBConfig.TARGET_SAMPLING_RATE.
        window_sec:  Welch epoch length (s). Defaults to TBConfig.SPECTRAL_WINDOW_S.
        overlap_sec: Overlap between epochs (s). Defaults to TBConfig.SPECTRAL_OVERLAP_S.
        band:        (low_hz, high_hz) tuple. Defaults to config band constants.
        config:      TBConfig class.

    Returns:
        Tuple (freqs, psd):
            freqs: 1-D numpy array of frequency bins (Hz) within [low, high].
            psd:   1-D numpy array of power spectral density values (V²/Hz).
            Both are empty arrays if the signal has zero power.
    """
    cfg = config or TBConfig
    fs          = fs          or cfg.TARGET_SAMPLING_RATE
    window_sec  = window_sec  or cfg.SPECTRAL_WINDOW_S
    overlap_sec = overlap_sec or cfg.SPECTRAL_OVERLAP_S
    low_hz, high_hz = band or (cfg.SPECTRAL_BAND_LOW_HZ, cfg.SPECTRAL_BAND_HIGH_HZ)

    n_per_seg = int(window_sec  * fs)
    n_overlap  = int(overlap_sec * fs)

    # Guard: signal shorter than one window
    if len(signal) < n_per_seg:
        return np.array([]), np.array([])

    # Guard: zero-power signal (e.g. all-zeros, silence)
    if np.allclose(signal, 0.0):
        return np.array([]), np.array([])

    freqs, psd = welch(
        signal,
        fs=fs,
        window='hann',
        nperseg=n_per_seg,
        noverlap=n_overlap,
        detrend='constant',   # remove DC offset within each epoch
        scaling='density',
    )

    # Slice to valid passband
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    return freqs[band_mask], psd[band_mask]


# ── 2. MDF and MNF ───────────────────────────────────────────────────────────

def compute_mdf_mnf(freqs, psd):
    """
    Compute Median Power Frequency (MDF) and Mean Power Frequency (MNF).

    MDF: frequency at which cumulative PSD reaches 50% of total power.
         Uses linear interpolation between frequency bins for sub-bin accuracy.

    MNF: power-weighted centroid of the spectrum.
         MNF = Σ(f_i × P_i) / Σ(P_i)

    Args:
        freqs: 1-D numpy array of frequency bins (Hz) — from compute_welch_psd.
        psd:   1-D numpy array of PSD values — from compute_welch_psd.

    Returns:
        Tuple (mdf_hz, mnf_hz):
            Both are np.nan if the inputs are empty or have zero total power.
    """
    if len(freqs) == 0 or len(psd) == 0:
        return np.nan, np.nan

    total_power = np.trapezoid(psd, freqs)
    if total_power <= 0.0:
        return np.nan, np.nan

    # MDF: find frequency where cumulative power crosses 50%
    bin_width = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    cumulative = np.cumsum(psd) * bin_width
    half_power = total_power * 0.5

    # Linear interpolation for sub-bin precision
    idx = np.searchsorted(cumulative, half_power)
    if idx == 0:
        mdf_hz = float(freqs[0])
    elif idx >= len(freqs):
        mdf_hz = float(freqs[-1])
    else:
        # Lerp between (freqs[idx-1], cumul[idx-1]) and (freqs[idx], cumul[idx])
        t = (half_power - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1])
        mdf_hz = float(freqs[idx - 1] + t * (freqs[idx] - freqs[idx - 1]))

    # MNF: weighted centroid
    mnf_hz = float(np.sum(freqs * psd) / np.sum(psd))

    return mdf_hz, mnf_hz


# ── 3. Active spectral series extraction ─────────────────────────────────────

def extract_active_spectral_series(pre_rect_signal, active_mask=None,
                                   fs=None, config=None):
    """
    Extract a time series of (t, MDF, MNF) from a full-session signal.

    Slides a Welch window across the signal and computes MDF/MNF for each
    epoch. Epochs where active samples < SPECTRAL_MIN_ACTIVE_RATIO are
    skipped (gating: avoids fitting frequency peaks to resting-floor noise).

    Args:
        pre_rect_signal: 1-D numpy array. PRE-RECTIFICATION Stage 2 output.
        active_mask:     Optional 1-D boolean array (same length as signal).
                         True = sample is above GAP_REST_THRESHOLD_PCT.
                         If None, all samples are treated as active (no gating).
        fs:              Sampling rate (Hz).
        config:          TBConfig class.

    Returns:
        Dict with keys:
            'times_sec'  : 1-D numpy array of epoch centre times (s).
            'mdf_hz'     : 1-D numpy array of MDF values (Hz). May contain NaN.
            'mnf_hz'     : 1-D numpy array of MNF values (Hz). May contain NaN.
            'n_total'    : total number of candidate epochs.
            'n_active'   : number of epochs that passed the active-ratio gate.
            'n_skipped'  : epochs skipped due to insufficient active ratio.
    """
    cfg         = config or TBConfig
    fs          = fs or cfg.TARGET_SAMPLING_RATE
    win_samples = int(cfg.SPECTRAL_WINDOW_S  * fs)
    step        = int((cfg.SPECTRAL_WINDOW_S - cfg.SPECTRAL_OVERLAP_S) * fs)
    min_ratio   = cfg.SPECTRAL_MIN_ACTIVE_RATIO

    n = len(pre_rect_signal)

    times, mdf_vals, mnf_vals = [], [], []
    n_total = n_active = n_skipped = 0

    for start in range(0, n - win_samples + 1, step):
        end = start + win_samples
        epoch = pre_rect_signal[start:end]
        centre_time = (start + win_samples / 2) / fs
        n_total += 1

        # Active-ratio gating
        if active_mask is not None:
            epoch_mask = active_mask[start:end]
            active_ratio = np.mean(epoch_mask)
        else:
            active_ratio = 1.0

        if active_ratio < min_ratio:
            n_skipped += 1
            continue

        # Compute PSD and derive MDF/MNF
        freqs, psd = compute_welch_psd(epoch, fs=fs, config=cfg)
        mdf, mnf = compute_mdf_mnf(freqs, psd)

        times.append(centre_time)
        mdf_vals.append(mdf)
        mnf_vals.append(mnf)
        n_active += 1

    return {
        'times_sec': np.array(times),
        'mdf_hz':    np.array(mdf_vals),
        'mnf_hz':    np.array(mnf_vals),
        'n_total':   n_total,
        'n_active':  n_active,
        'n_skipped': n_skipped,
    }


# ── 4. OLS fatigue slope ──────────────────────────────────────────────────────

def compute_spectral_fatigue_slope(times_sec, mdf_values, config=None):
    """
    Fit an OLS linear trend to the MDF time series.

    Reports slope in Hz/min (not Hz/s) to match clinically intuitive units.
    Applies fatigue flag thresholds from TBConfig — NOT hardcoded — so
    changing FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN or FATIGUE_MIN_R_SQUARED
    in config.py automatically propagates without touching this function.

    Args:
        times_sec:   1-D numpy array of epoch centre times in seconds.
        mdf_values:  1-D numpy array of MDF values in Hz (may contain NaN).
        config:      TBConfig class.

    Returns:
        Dict with keys:
            'slope_hz_per_min'  : OLS slope (Hz/min). Negative = downshift.
            'intercept_hz'      : OLS intercept (Hz at t=0).
            'r_squared'         : coefficient of determination R².
            'n_windows'         : number of valid (non-NaN) windows used.
            'is_fatiguing'      : True if slope < FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN
                                  AND r_squared >= FATIGUE_MIN_R_SQUARED.
            'slope_threshold_used': the threshold value read from config (not hardcoded).
            'r_squared_threshold_used': the R² threshold read from config.
            'sufficient_data'   : False if n_windows < SPECTRAL_MIN_VALID_WINDOWS.
            'note'              : human-readable explanation of the result.
    """
    cfg = config or TBConfig
    slope_thresh = cfg.FATIGUE_SLOPE_THRESHOLD_HZ_PER_MIN
    r2_thresh    = cfg.FATIGUE_MIN_R_SQUARED
    min_windows  = cfg.SPECTRAL_MIN_VALID_WINDOWS

    # Filter out NaN pairs
    valid = ~(np.isnan(times_sec) | np.isnan(mdf_values))
    t_valid = times_sec[valid]
    m_valid = mdf_values[valid]
    n_valid = int(np.sum(valid))

    # Insufficient data guard
    if n_valid < min_windows:
        return {
            'slope_hz_per_min': np.nan,
            'intercept_hz': np.nan,
            'r_squared': np.nan,
            'n_windows': n_valid,
            'is_fatiguing': False,
            'slope_threshold_used': slope_thresh,
            'r_squared_threshold_used': r2_thresh,
            'sufficient_data': False,
            'note': (f"Insufficient valid windows ({n_valid} < {min_windows}). "
                     f"Session too short or too much resting time for slope estimate."),
        }

    # OLS via numpy polyfit (degree 1)
    t_min = t_valid / 60.0   # convert seconds → minutes for Hz/min slope
    coeffs = np.polyfit(t_min, m_valid, deg=1)
    slope_hz_per_min = float(coeffs[0])
    intercept_hz     = float(coeffs[1])

    # R² — proportion of MDF variance explained by the linear time trend
    mdf_pred  = np.polyval(coeffs, t_min)
    ss_res    = np.sum((m_valid - mdf_pred) ** 2)
    ss_tot    = np.sum((m_valid - np.mean(m_valid)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Fatigue flag: threshold from config, not hardcoded
    is_fatiguing = (slope_hz_per_min < slope_thresh) and (r_squared >= r2_thresh)

    if is_fatiguing:
        note = (f"Fatigue trend detected: MDF declining at {slope_hz_per_min:.3f} Hz/min "
                f"(threshold: {slope_thresh} Hz/min), R²={r_squared:.3f} "
                f"(threshold: {r2_thresh}). "
                "Interpret as internal session trend only — see passband caveat.")
    elif slope_hz_per_min < slope_thresh:
        note = (f"Slope ({slope_hz_per_min:.3f} Hz/min) below threshold but "
                f"R²={r_squared:.3f} < {r2_thresh} — trend too noisy to confirm.")
    else:
        note = (f"No significant fatigue trend: slope={slope_hz_per_min:.3f} Hz/min "
                f"(threshold: {slope_thresh} Hz/min), R²={r_squared:.3f}.")

    return {
        'slope_hz_per_min': slope_hz_per_min,
        'intercept_hz':     intercept_hz,
        'r_squared':        r_squared,
        'n_windows':        n_valid,
        'is_fatiguing':     is_fatiguing,
        'slope_threshold_used':    slope_thresh,
        'r_squared_threshold_used': r2_thresh,
        'sufficient_data':  True,
        'note': note,
    }


# ── 5. Bilateral entry point ──────────────────────────────────────────────────

def analyze_bilateral_spectral_fatigue(pre_rect_left, pre_rect_right,
                                       active_mask_left=None,
                                       active_mask_right=None,
                                       fs=None, config=None):
    """
    Compute spectral fatigue analysis for both trapezius channels independently.

    Left and Right are processed with zero cross-channel mixing. The final
    output dict embeds the mandatory passband ceiling caveat.

    Args:
        pre_rect_left:   1-D numpy array — Stage 2 bandpass output, LEFT channel,
                         PRE-RECTIFICATION.
        pre_rect_right:  1-D numpy array — Stage 2 bandpass output, RIGHT channel,
                         PRE-RECTIFICATION.
        active_mask_left:  Optional boolean mask for left channel gating.
        active_mask_right: Optional boolean mask for right channel gating.
        fs:              Sampling rate (Hz).
        config:          TBConfig class.

    Returns:
        Dict with keys:
            'left'  : dict from compute_spectral_fatigue_slope for left channel,
                      plus 'spectral_series' sub-dict from extract_active_spectral_series.
            'right' : same for right channel.
            'passband_ceiling_caveat': mandatory warning string (always present).
    """
    cfg = config or TBConfig
    fs  = fs or cfg.TARGET_SAMPLING_RATE

    results = {}
    for side, signal, mask in [
        ('left',  pre_rect_left,  active_mask_left),
        ('right', pre_rect_right, active_mask_right),
    ]:
        series = extract_active_spectral_series(
            signal, active_mask=mask, fs=fs, config=cfg
        )
        slope_report = compute_spectral_fatigue_slope(
            series['times_sec'], series['mdf_hz'], config=cfg
        )
        results[side] = {
            **slope_report,
            'spectral_series': {
                'times_sec': series['times_sec'].tolist(),
                'mdf_hz':    series['mdf_hz'].tolist(),
                'mnf_hz':    series['mnf_hz'].tolist(),
                'n_total':   series['n_total'],
                'n_active':  series['n_active'],
                'n_skipped': series['n_skipped'],
            },
        }

    results['passband_ceiling_caveat'] = _PASSBAND_CAVEAT
    return results
