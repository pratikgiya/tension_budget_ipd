"""
TensionBudget — Upper-trapezius sEMG monitoring pipeline.

A structured acquisition, preprocessing, and analysis pipeline built on top of
Chords-Python for low-cost upper-trapezius sEMG monitoring in desk-work conditions.

Submodules:
    config         — Central configuration (all thresholds, windows, weights).
    session        — Session identity, metadata, and manifest writing.
    recorder       — Stage 1 CSV recording with timestamps and drop detection.
    preprocessing  — Stage 2 signal processing (filter, rectify, RMS, offline).
    analytics      — Stage 3 analytics (gaps, SUMA, APDF, asymmetry).
    scoring        — Phase 6 composite score (EIndex, frequency penalty, fusion).
    spectral       — Stage 10 spectral fatigue (MDF/MNF, OLS slope, bilateral).
"""

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.session import TBSession
from chordspy.tensionbudget.recorder import TBRecorder
from chordspy.tensionbudget.spectral import (
    compute_welch_psd,
    compute_mdf_mnf,
    extract_active_spectral_series,
    compute_spectral_fatigue_slope,
    analyze_bilateral_spectral_fatigue,
)

__all__ = [
    "TBConfig", "TBSession", "TBRecorder",
    "compute_welch_psd", "compute_mdf_mnf",
    "extract_active_spectral_series",
    "compute_spectral_fatigue_slope",
    "analyze_bilateral_spectral_fatigue",
]
