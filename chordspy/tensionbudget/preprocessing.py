"""
TensionBudget — Stage 2 Preprocessing Pipeline.

Reproducible signal processing pipeline for upper-trapezius sEMG data.
Works both offline (on previously recorded TB CSVs) and live (via
per-sample incremental processing).

Pipeline stages (per channel, independently):
    1. Raw-to-voltage conversion (optional, configurable)
    2. Bandpass filter — 20–240 Hz, 4th-order Butterworth, zero-phase (filtfilt)
    3. Notch filter — 50 Hz mains rejection (optional, configurable)
    4. Full-wave rectification — np.abs()
    5. RMS envelope — 100 ms window, 20 ms step (Marker & Maluf 2016)
    6. Normalized output — placeholder for %MVC/%RVE (Phase 4 calibration)

All parameters come from TBConfig — no magic numbers in this module.

References:
    De Luca, C.J. (1997). The use of surface electromyography in biomechanics.
    Farina, D. et al. (2002). Comparison of algorithms for estimation of EMG
        variables during voluntary isometric contractions.
    Marker, R.J. & Maluf, K.S. (2016). Upper trapezius motor unit firing
        patterns during repetitive low-force work.

Usage (offline):
    from chordspy.tensionbudget.preprocessing import process_tb_session
    result = process_tb_session("TB_20260708_193000_a1b2c3d4.csv")

Usage (per-channel):
    from chordspy.tensionbudget.preprocessing import preprocess_channel
    result = preprocess_channel(raw_signal, fs=500)
"""

import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.signal import (
    butter, filtfilt, iirnotch, welch, 
    sosfiltfilt, tf2sos, sosfilt, sosfilt_zi
)

from chordspy.tensionbudget.config import TBConfig


# ── 1. Raw-to-voltage conversion ─────────────────────────────────────

def raw_to_voltage(signal, resolution_bits=None, vref=None):
    """
    Convert raw ADC counts to voltage (optional layer).

    If the ADC reference voltage and resolution are known, this converts
    raw integer counts to millivolts. If either is None, returns the
    signal unchanged — raw counts are preserved as-is.

    The BioAmp EXG Pill has its own analog gain stage before the Arduino
    ADC, so the resulting voltage is at the ADC input, not at the skin.
    For relative measures (RMS ratios, %MVC), raw counts work fine and
    voltage conversion is not required.

    Args:
        signal: 1-D numpy array of raw ADC values.
        resolution_bits: ADC resolution (e.g. 14 for UNO-R4). If None,
            uses TBConfig.TARGET_RESOLUTION_BITS.
        vref: ADC reference voltage in volts. If None, conversion is
            skipped (returns signal unchanged).

    Returns:
        1-D numpy array. Voltage in mV if vref is provided, otherwise
        raw counts unchanged.
    """
    if vref is None:
        return signal.copy()

    res = resolution_bits or TBConfig.TARGET_RESOLUTION_BITS
    adc_max = (2 ** res) - 1
    voltage_mv = (signal / adc_max) * vref * 1000.0  # Convert to mV
    return voltage_mv


# ── 2. Bandpass filter ────────────────────────────────────────────────

def bandpass_filter(signal, fs=None, low_hz=None, high_hz=None, order=None):
    """
    Apply zero-phase Butterworth bandpass filter.

    Default: 20–240 Hz, 4th-order, applied via scipy.signal.filtfilt
    for zero-phase distortion. This is the literature-correct band
    (De Luca 1997 / Farina 2002) truncated to fit under the 250 Hz
    Nyquist limit at 500 Hz sampling.

    Left and right channels must be filtered independently (separate
    calls) — do not pass multi-channel data to this function.

    Args:
        signal: 1-D numpy array (single channel).
        fs: Sampling rate in Hz. Defaults to TBConfig.TARGET_SAMPLING_RATE.
        low_hz: Low cutoff frequency. Defaults to TBConfig.BANDPASS_LOW_HZ.
        high_hz: High cutoff frequency. Defaults to TBConfig.BANDPASS_HIGH_HZ.
        order: Filter order. Defaults to TBConfig.BANDPASS_ORDER.

    Returns:
        1-D numpy array of filtered signal (same length as input).

    Raises:
        ValueError: If high_hz >= fs/2 (Nyquist violation).
    """
    fs = fs or TBConfig.TARGET_SAMPLING_RATE
    low = low_hz or TBConfig.BANDPASS_LOW_HZ
    high = high_hz or TBConfig.BANDPASS_HIGH_HZ
    n = order or TBConfig.BANDPASS_ORDER

    nyq = 0.5 * fs
    if high >= nyq:
        raise ValueError(
            f"Bandpass high cutoff ({high} Hz) must be below Nyquist "
            f"({nyq} Hz) at {fs} Hz sampling rate."
        )

    sos = butter(n, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


# ── 3. Notch filter ──────────────────────────────────────────────────

def notch_filter(signal, fs=None, freq_hz=None, quality_factor=None):
    """
    Apply zero-phase notch filter for mains interference rejection.

    Default: 50 Hz (India/EU). Set TBConfig.NOTCH_FREQ_HZ to 60 for US.

    Args:
        signal: 1-D numpy array (single channel).
        fs: Sampling rate in Hz.
        freq_hz: Notch center frequency. Defaults to TBConfig.NOTCH_FREQ_HZ.
        quality_factor: Q factor. Defaults to TBConfig.NOTCH_QUALITY_FACTOR.

    Returns:
        1-D numpy array of notch-filtered signal (same length as input).
    """
    fs = fs or TBConfig.TARGET_SAMPLING_RATE
    freq = freq_hz or TBConfig.NOTCH_FREQ_HZ
    Q = quality_factor or TBConfig.NOTCH_QUALITY_FACTOR

    b, a = iirnotch(freq, Q, fs)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal)


def detect_mains_interference(signal, fs=None, freq_hz=None, threshold_ratio=5.0):
    """
    Detect if there is a strong mains interference spike in the signal.

    Uses Welch's method to estimate the power spectral density. If the power
    at the mains frequency is significantly higher than the median power in
    the surrounding band (e.g., 20-240 Hz), it returns True.

    Args:
        signal: 1-D numpy array (single channel).
        fs: Sampling rate in Hz.
        freq_hz: Expected mains frequency (e.g., 50.0). Defaults from config.
        threshold_ratio: How many times larger the peak must be compared to
            the median background power to be considered interference.

    Returns:
        bool: True if strong interference detected, False otherwise.
    """
    fs = fs or TBConfig.TARGET_SAMPLING_RATE
    freq = freq_hz or TBConfig.NOTCH_FREQ_HZ

    # Compute Power Spectral Density
    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))

    # Find the frequency bin closest to the mains frequency
    idx_mains = np.argmin(np.abs(f - freq))
    
    # Check surrounding band (e.g., 20-240 Hz) for baseline
    band_mask = (f >= TBConfig.BANDPASS_LOW_HZ) & (f <= TBConfig.BANDPASS_HIGH_HZ)
    if not np.any(band_mask):
        return False # Should not happen unless bandpass config is invalid
        
    median_power = np.median(Pxx[band_mask])
    peak_power = Pxx[idx_mains]

    # If peak is a sharp spike compared to median muscle activity, it's noise
    return peak_power > (median_power * threshold_ratio)


# ── 4. Rectification ─────────────────────────────────────────────────

def rectify(signal):
    """
    Full-wave rectification.

    Takes the absolute value of the signal. This is a distinct,
    explicit preprocessing step — not folded into another function.

    Args:
        signal: 1-D numpy array.

    Returns:
        1-D numpy array of rectified signal (all values >= 0).
    """
    return np.abs(signal)


# ── 5. RMS envelope ──────────────────────────────────────────────────

def rms_envelope(signal, window_samples=None, step_samples=None):
    """
    Compute RMS envelope with sliding window.

    Default: 100 ms window (50 samples at 500 Hz), 20 ms step
    (10 samples) — Marker & Maluf (2016) specification.

    The output is shorter than the input because the window cannot
    extend past the signal edges. Each RMS value corresponds to the
    center of its window.

    Args:
        signal: 1-D numpy array (should be rectified, but works on any).
        window_samples: Window length in samples. Defaults from TBConfig.
        step_samples: Step size in samples. Defaults from TBConfig.

    Returns:
        Tuple of (rms_values, center_indices):
            - rms_values: 1-D numpy array of RMS values.
            - center_indices: 1-D numpy array of sample indices
              corresponding to the center of each window.
    """
    win = window_samples or TBConfig.rms_window_samples()
    step = step_samples or TBConfig.rms_step_samples()

    n = len(signal)
    if n < win:
        # Signal shorter than one window — return single RMS of entire signal
        return np.array([np.sqrt(np.mean(signal ** 2))]), np.array([n // 2])

    rms_values = []
    center_indices = []

    for start in range(0, n - win + 1, step):
        window = signal[start:start + win]
        rms_val = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms_val)
        center_indices.append(start + win // 2)

    return np.array(rms_values), np.array(center_indices)


# ── 6. Per-channel pipeline ──────────────────────────────────────────

def preprocess_channel(raw_signal, fs=None, config=None, vref=None, reference_rms=None):
    """
    Full preprocessing pipeline for a single EMG channel.

    Runs all stages in order using zero-phase offline filters (sosfiltfilt):
        raw → (voltage) → bandpass → (notch) → rectify → RMS → (normalized)

    Left and right channels must be processed with SEPARATE calls to
    this function — they do not share filter state.

    Args:
        raw_signal: 1-D numpy array of raw ADC values for one channel.
        fs: Sampling rate in Hz. Defaults from config.
        config: TBConfig class (or compatible). Defaults to TBConfig.
        vref: ADC reference voltage for raw-to-voltage conversion.
            If None, voltage conversion is skipped (raw counts preserved).
        reference_rms: Reference RMS denominator (e.g., from RVE calibration).
            If provided, `normalized` will contain %RVE. If None, `normalized`
            will be NaN.

    Returns:
        dict with keys:
            'raw'           — original raw signal (numpy array, len N)
            'voltage'       — voltage-converted signal, or raw if vref=None (len N)
            'filtered'      — bandpass-filtered signal (len N)
            'rectified'     — full-wave rectified signal (len N)
            'rms'           — RMS envelope values (len M, M < N)
            'rms_indices':   sample indices at center of each RMS window (len M)
            'normalized':    normalized %RVE array (if reference_rms provided) or NaN
            'fs'            — sampling rate used
            'config_snapshot' — dict of filter parameters used
    """
    cfg = config or TBConfig
    fs = fs or cfg.TARGET_SAMPLING_RATE

    # Stage 1: Raw-to-voltage (optional)
    voltage = raw_to_voltage(raw_signal, vref=vref)

    # Stage 2: Bandpass filter
    filtered = bandpass_filter(
        voltage, fs=fs,
        low_hz=cfg.BANDPASS_LOW_HZ,
        high_hz=cfg.BANDPASS_HIGH_HZ,
        order=cfg.BANDPASS_ORDER,
    )

    # Stage 3: Notch filter (optional or auto)
    notch_applied = False
    if cfg.NOTCH_ENABLED is True:
        notch_applied = True
    elif cfg.NOTCH_ENABLED == "auto":
        notch_applied = detect_mains_interference(filtered, fs=fs, freq_hz=cfg.NOTCH_FREQ_HZ)

    if notch_applied:
        filtered = notch_filter(
            filtered, fs=fs,
            freq_hz=cfg.NOTCH_FREQ_HZ,
            quality_factor=cfg.NOTCH_QUALITY_FACTOR,
        )

    # Stage 4: Rectification
    rectified = rectify(filtered)

    # Stage 5: RMS envelope
    rms, rms_idx = rms_envelope(
        rectified,
        window_samples=cfg.rms_window_samples(),
        step_samples=cfg.rms_step_samples(),
    )

    # Stage 6: Normalized output
    if reference_rms is not None and reference_rms > 0:
        normalized = (rms / reference_rms) * 100.0
    else:
        normalized = np.full_like(rms, np.nan)

    return {
        'raw': raw_signal,
        'voltage': voltage,
        'filtered': filtered,
        'rectified': rectified,
        'rms': rms,
        'rms_indices': rms_idx,
        'normalized': normalized,
        'fs': fs,
        'config_snapshot': {
            'bandpass_hz': (cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ),
            'bandpass_order': cfg.BANDPASS_ORDER,
            'notch_enabled': cfg.NOTCH_ENABLED,
            'notch_applied': notch_applied,
            'notch_hz': cfg.NOTCH_FREQ_HZ,
            'rms_window_ms': cfg.RMS_WINDOW_MS,
            'rms_step_ms': cfg.RMS_STEP_MS,
            'vref': vref,
            'reference_rms': reference_rms,
        },
    }

class StreamingChannelProcessor:
    """
    Live processing pipeline for a single EMG channel using causal stateful filters.
    
    Wraps scipy.signal.sosfilt with persistent state (zi) to process incoming 
    blocks of data chunk-by-chunk without edge artifacts.
    Note: Causal filtering introduces phase delay unlike offline zero-phase filtering.
    """
    
    def __init__(self, fs=None, config=None, vref=None, reference_rms=None):
        self.cfg = config or TBConfig
        self.fs = fs or self.cfg.TARGET_SAMPLING_RATE
        self.vref = vref
        self.reference_rms = reference_rms
        
        # Bandpass state
        nyq = 0.5 * self.fs
        self.bp_sos = butter(self.cfg.BANDPASS_ORDER, 
                             [self.cfg.BANDPASS_LOW_HZ / nyq, self.cfg.BANDPASS_HIGH_HZ / nyq], 
                             btype='band', output='sos')
        self.bp_zi = sosfilt_zi(self.bp_sos)
        
        # Notch state
        self.notch_enabled = self.cfg.NOTCH_ENABLED
        if self.notch_enabled is True:
            b, a = iirnotch(self.cfg.NOTCH_FREQ_HZ, self.cfg.NOTCH_QUALITY_FACTOR, self.fs)
            self.notch_sos = tf2sos(b, a)
            self.notch_zi = sosfilt_zi(self.notch_sos)
        else:
            # For 'auto' we don't have enough lookahead in live mode to do Welch easily on the fly
            # safely assuming False for auto until specifically activated
            self.notch_sos = None
            self.notch_zi = None
            
        # RMS rolling state
        self.win = self.cfg.rms_window_samples()
        self.step = self.cfg.rms_step_samples()
        self.rms_buffer = np.array([])
        self.global_sample_count = 0
        
    def process_chunk(self, raw_chunk):
        """
        Process a new chunk of raw samples.
        
        Args:
            raw_chunk: 1-D numpy array of new raw samples.
            
        Returns:
            dict containing raw, voltage, filtered, rectified, rms, normalized.
        """
        n = len(raw_chunk)
        if n == 0:
            return {}
            
        # 1. Voltage
        voltage = raw_to_voltage(raw_chunk, vref=self.vref)
        
        # 2. Bandpass (Causal, stateful)
        if self.global_sample_count == 0 and n > 0:
            self.bp_zi = self.bp_zi * voltage[0]
            if self.notch_zi is not None:
                self.notch_zi = self.notch_zi * voltage[0]
                
        filtered, self.bp_zi = sosfilt(self.bp_sos, voltage, zi=self.bp_zi)
        
        # 3. Notch
        if self.notch_sos is not None:
            filtered, self.notch_zi = sosfilt(self.notch_sos, filtered, zi=self.notch_zi)
            
        # 4. Rectify
        rectified = rectify(filtered)
        
        # 5. RMS (Rolling overlap-add strategy)
        self.rms_buffer = np.concatenate([self.rms_buffer, rectified])
        
        rms_values = []
        # Calculate RMS for every complete window
        while len(self.rms_buffer) >= self.win:
            window = self.rms_buffer[:self.win]
            rms_val = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms_val)
            # Advance buffer by step size
            self.rms_buffer = self.rms_buffer[self.step:]
            
        rms_arr = np.array(rms_values)
        
        # 6. Normalize
        if self.reference_rms is not None and self.reference_rms > 0:
            normalized = (rms_arr / self.reference_rms) * 100.0
        else:
            normalized = np.full_like(rms_arr, np.nan)
            
        self.global_sample_count += n
            
        return {
            'raw': raw_chunk,
            'voltage': voltage,
            'filtered': filtered,
            'rectified': rectified,
            'rms': rms_arr,
            'normalized': normalized
        }


# ── 7. TB CSV loader ─────────────────────────────────────────────────

def load_tb_csv(csv_path):
    """
    Load a TensionBudget-format CSV, skipping metadata comment lines.

    Parses lines starting with '#' as metadata key-value pairs and
    reads the remaining data as a structured array.

    Args:
        csv_path: Path to the TB CSV file.

    Returns:
        Tuple of (data, metadata):
            - data: dict of column_name → numpy array
            - metadata: dict of parsed metadata from comment lines
    """
    csv_path = Path(csv_path)
    metadata = {}
    data_lines = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(TBConfig.CSV_METADATA_COMMENT_CHAR):
                # Parse metadata: "# Key: Value"
                content = stripped.lstrip(TBConfig.CSV_METADATA_COMMENT_CHAR).strip()
                if ':' in content:
                    key, _, value = content.partition(':')
                    metadata[key.strip()] = value.strip()
            elif stripped:
                data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in {csv_path}")

    # First data line is the header
    header = data_lines[0].split(',')
    header = [h.strip() for h in header]

    # Parse data rows
    columns = {h: [] for h in header}
    for row_str in data_lines[1:]:
        values = row_str.split(',')
        for h, v in zip(header, values):
            v = v.strip()
            try:
                columns[h].append(float(v))
            except ValueError:
                columns[h].append(np.nan)

    # Convert to numpy arrays
    data = {h: np.array(vals) for h, vals in columns.items()}

    return data, metadata


def _resolve_channel_columns(data, config=None):
    """
    Resolve which data columns correspond to active EMG channels.

    If CHANNEL_MAP is set (e.g. {"left": 0, "right": 1}), maps to
    named channels. Otherwise, treats all Channel* columns generically.

    Args:
        data: dict from load_tb_csv().
        config: TBConfig class.

    Returns:
        dict mapping label → column_name, e.g.:
            {"left": "Channel1", "right": "Channel2"}
        or if no CHANNEL_MAP:
            {"Ch1": "Channel1", "Ch2": "Channel2", ..., "Ch6": "Channel6"}
    """
    cfg = config or TBConfig
    channel_cols = [k for k in data.keys() if k.startswith("Channel")]
    channel_cols.sort(key=lambda c: int(c.replace("Channel", "")))

    if cfg.CHANNEL_MAP:
        mapping = {}
        for label, idx in cfg.CHANNEL_MAP.items():
            col_name = f"Channel{idx + 1}"
            if col_name in data:
                mapping[label] = col_name
            else:
                print(f"Warning: CHANNEL_MAP['{label}'] = {idx} "
                      f"but {col_name} not in CSV")
        return mapping
    else:
        # Generic: all channels
        return {f"Ch{i+1}": col for i, col in enumerate(channel_cols)}


# ── 8. Offline session processor ──────────────────────────────────────

def process_tb_session(csv_path, output_path=None, config=None, vref=None):
    """
    Process a full TensionBudget recording offline.

    Reads a TB CSV, processes each active channel through the full
    pipeline, and writes a processed CSV with all intermediate stages.
    If a sidecar JSON manifest exists with calibration data, it applies
    the reference RMS for normalization.

    Args:
        csv_path: Path to the raw TB CSV file.
        output_path: Path for the processed CSV. If None, auto-generates
            from input path (e.g. TB_*_processed.csv).
        config: TBConfig class. Defaults to TBConfig.
        vref: ADC reference voltage for voltage conversion. None = skip.

    Returns:
        dict with keys:
            'output_path'   — Path to the processed CSV
            'channels'      — dict of label → preprocess_channel() result
            'metadata'      — metadata dict from the raw CSV
            'config_snapshot' — processing config used
            'sample_rate_hz' — sampling rate
    """
    cfg = config or TBConfig

    # Load raw data
    data, metadata = load_tb_csv(csv_path)

    # Determine sampling rate from metadata or config
    fs = cfg.TARGET_SAMPLING_RATE
    if 'Sampling Rate' in metadata:
        try:
            fs = int(metadata['Sampling Rate'].split()[0])
        except (ValueError, IndexError):
            pass

    # Resolve channel mapping
    channel_map = _resolve_channel_columns(data, cfg)

    if not channel_map:
        raise ValueError("No active channels found in the CSV data.")

    # Try to load sidecar JSON manifest for calibration data
    manifest_path = Path(csv_path).with_suffix('.json')
    calibration_data = {}
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                calibration_data = manifest.get('calibration', {}) or {}
        except Exception as e:
            print(f"Warning: Failed to load manifest {manifest_path.name}: {e}")

    # Process each channel independently
    results = {}
    for label, col_name in channel_map.items():
        raw = data[col_name]
        # Extract reference_rms if available for this specific label (e.g., 'left')
        ch_calib = calibration_data.get(label, {})
        ref_rms = ch_calib.get('reference_rms', None)
        
        results[label] = preprocess_channel(
            raw, fs=fs, config=cfg, vref=vref, reference_rms=ref_rms
        )

    # Generate output path
    if output_path is None:
        in_path = Path(csv_path)
        output_path = in_path.with_name(
            in_path.stem + "_processed" + in_path.suffix
        )
    output_path = Path(output_path)

    # Write processed CSV
    _write_processed_csv(output_path, data, results, metadata, cfg)

    return {
        'output_path': output_path,
        'channels': results,
        'metadata': metadata,
        'config_snapshot': {
            'bandpass_hz': (cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ),
            'bandpass_order': cfg.BANDPASS_ORDER,
            'notch_enabled': cfg.NOTCH_ENABLED,
            'notch_hz': cfg.NOTCH_FREQ_HZ,
            'rms_window_ms': cfg.RMS_WINDOW_MS,
            'rms_step_ms': cfg.RMS_STEP_MS,
            'vref': vref,
        },
        'sample_rate_hz': fs,
    }


def _write_processed_csv(output_path, raw_data, results, metadata, config):
    """
    Write the processed CSV with all intermediate stages.

    Two sections are written:
    1. Full-rate data (N rows): Sample_Index, Timestamp_s, per-channel
       raw, filtered, rectified columns.
    2. RMS-rate data (M rows): RMS_Index, RMS_Timestamp_s, per-channel
       rms, normalized columns (at the lower RMS update rate).

    The two sections are in separate files:
    - *_processed.csv — full-rate filtered/rectified data
    - *_processed_rms.csv — RMS-rate envelope data

    Args:
        output_path: Path for the full-rate processed CSV.
        raw_data: dict from load_tb_csv().
        results: dict of label → preprocess_channel() result.
        metadata: metadata dict from the raw CSV.
        config: TBConfig class.
    """
    c = TBConfig.CSV_METADATA_COMMENT_CHAR

    # ── Full-rate CSV (Sample_Index, Timestamp_s, per-channel stages) ──
    n_samples = len(raw_data.get('Sample_Index', []))
    timestamps = raw_data.get('Timestamp_s', np.arange(n_samples) / config.TARGET_SAMPLING_RATE)
    sample_indices = raw_data.get('Sample_Index', np.arange(1, n_samples + 1))

    labels = sorted(results.keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Metadata header
        f.write(f"{c} TensionBudget Processed Data (Full-Rate)\n")
        f.write(f"{c} Source: {metadata.get('Session ID', 'unknown')}\n")
        f.write(f"{c} Bandpass: {config.BANDPASS_LOW_HZ}-{config.BANDPASS_HIGH_HZ} Hz, "
                f"order {config.BANDPASS_ORDER}\n")
        # Determine if notch was applied to ANY channel
        notch_status = "disabled"
        if config.NOTCH_ENABLED is True:
            notch_status = f"enabled at {config.NOTCH_FREQ_HZ} Hz"
        elif config.NOTCH_ENABLED == "auto":
            any_applied = any(ch.get('config_snapshot', {}).get('notch_applied', False) 
                              for ch in results.values())
            notch_status = f"auto (applied={any_applied}) at {config.NOTCH_FREQ_HZ} Hz"

        f.write(f"{c} Notch: {notch_status}\n")
        f.write(f"{c} Channel Map: {config.CHANNEL_MAP or 'not configured'}\n")

        writer = csv.writer(f)

        # Header row
        header = ['Sample_Index', 'Timestamp_s']
        for label in labels:
            header.extend([
                f"{label}_raw",
                f"{label}_filtered",
                f"{label}_rectified",
            ])
        writer.writerow(header)

        # Data rows
        for i in range(n_samples):
            row = [int(sample_indices[i]), f"{timestamps[i]:.6f}"]
            for label in labels:
                ch = results[label]
                row.append(f"{ch['raw'][i]:.2f}")
                row.append(f"{ch['filtered'][i]:.6f}")
                row.append(f"{ch['rectified'][i]:.6f}")
            writer.writerow(row)

    # ── RMS-rate CSV (separate file) ──
    rms_path = output_path.with_name(
        output_path.stem.replace('_processed', '_processed_rms') + output_path.suffix
    )

    # All channels should have the same RMS length (same window/step)
    rms_len = len(results[labels[0]]['rms'])
    rms_indices = results[labels[0]]['rms_indices']
    rms_timestamps = timestamps[rms_indices] if len(timestamps) > max(rms_indices) else (
        rms_indices / config.TARGET_SAMPLING_RATE
    )

    with open(rms_path, 'w', newline='', encoding='utf-8') as f:
        f.write(f"{c} TensionBudget Processed Data (RMS-Rate)\n")
        f.write(f"{c} Source: {metadata.get('Session ID', 'unknown')}\n")
        f.write(f"{c} RMS Window: {config.RMS_WINDOW_MS} ms, "
                f"Step: {config.RMS_STEP_MS} ms\n")
        f.write(f"{c} Channel Map: {config.CHANNEL_MAP or 'not configured'}\n")

        writer = csv.writer(f)

        # Header row
        header = ['RMS_Index', 'RMS_Timestamp_s']
        for label in labels:
            header.extend([
                f"{label}_rms",
                f"{label}_normalized",
            ])
        writer.writerow(header)

        # Data rows
        for i in range(rms_len):
            row = [i + 1, f"{rms_timestamps[i]:.6f}"]
            for label in labels:
                ch = results[label]
                row.append(f"{ch['rms'][i]:.6f}")
                norm_val = ch['normalized'][i]
                row.append(f"{norm_val:.6f}" if not np.isnan(norm_val) else "")
            writer.writerow(row)

    print(f"Processed CSV (full-rate): {output_path}")
    print(f"Processed CSV (RMS-rate):  {rms_path}")
