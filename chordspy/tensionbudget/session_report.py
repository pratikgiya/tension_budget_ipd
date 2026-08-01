"""
TensionBudget — Stage 3 Offline Session Report.

Reads an offline processed CSV (with normalized %RVE columns), runs the
Phase 5 analytics engine on the left and right channels, and outputs a 
structured JSON summary including the side-to-side asymmetry index.
"""

import json
from pathlib import Path

import numpy as np

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.preprocessing import load_tb_csv
from chordspy.tensionbudget.analytics import analyze_channel, compute_asymmetry


def generate_session_report(processed_csv_path, output_path=None, config=None):
    """
    Generate an offline JSON analytics report from a processed TB CSV.
    
    Args:
        processed_csv_path: Path to the *_processed.csv file.
        output_path: Path for the JSON output. Auto-generated if None.
        config: TBConfig class.
        
    Returns:
        dict: The final session summary report.
    """
    cfg = config or TBConfig
    
    # 1. Load the processed CSV
    data, metadata = load_tb_csv(processed_csv_path)
    
    # The processed CSV has columns like 'left_normalized', 'right_normalized'
    channel_map = cfg.CHANNEL_MAP
    if not channel_map:
        raise ValueError("Cannot run session report without a configured CHANNEL_MAP (e.g. left=0, right=1)")
        
    report = {
        "session_id": metadata.get("Session ID", "unknown"),
        "board": metadata.get("Board", "unknown"),
        "channels": {},
        "asymmetry": {}
    }
    
    # 2. Analyze each mapped channel
    for label in channel_map.keys():
        norm_col = f"{label}_normalized"
        if norm_col not in data:
            print(f"Warning: {norm_col} not found in CSV. Did you run process_tb_session with calibration?")
            continue
            
        norm_array = data[norm_col]
        
        # Analyze full offline array (is_live_buffer=False ensures both Active and Session APDF are conceptualized, 
        # though currently compute_apdf just runs on the array provided)
        ch_report = analyze_channel(norm_array, config=cfg, is_live_buffer=False)
        report["channels"][label] = ch_report

    # 3. Compute Side-to-Side Asymmetry (Laterality Index)
    # Requires both left and right to be present
    if "left" in report["channels"] and "right" in report["channels"]:
        apdf50_L = report["channels"]["left"]["active_apdf"][50]
        apdf50_R = report["channels"]["right"]["active_apdf"][50]
        
        ai = compute_asymmetry(apdf50_L, apdf50_R)
        
        report["asymmetry"] = {
            "laterality_index": ai,
            "interpretation": "Negative = Left dominant, Positive = Right dominant, 0 = Symmetric"
        }
        
    # 4. Save JSON report
    if output_path is None:
        in_path = Path(processed_csv_path)
        output_path = in_path.with_name(in_path.stem + "_analytics.json")
        
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert NumPy types to native Python for JSON
        def default_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        json.dump(report, f, indent=4, default=default_serializer)
        
    print(f"Session analytics report generated: {output_path.name}")
    return report

