"""
export_raw_to_excel.py — CLI tool for converting compressed Parquet or CSV raw 500 Hz telemetry archives into organized multi-sheet Microsoft Excel (.xlsx) workbooks.

Because full-day 500 Hz recordings easily surpass Microsoft Excel's ~1,048,576 row limit, this tool splits data across multiple worksheet tabs automatically.

Usage Example:
    python -m scripts.export_raw_to_excel --file output_logs/User/session_TIMESTAMP/session_TIMESTAMP_raw.parquet --rows-per-tab 500000
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' library is required. Please run: pip install pandas")
    sys.exit(1)


def convert_raw_to_multitab_excel(file_path: Path, output_path: Path = None, rows_per_tab: int = 500000):
    if not file_path.exists():
        print(f"Error: Target file '{file_path}' does not exist.")
        return False

    if output_path is None:
        output_path = file_path.with_suffix(".xlsx")

    print("==================================================================")
    print(" TensionBudget Raw Telemetry -> Multi-Tab Excel (.xlsx) Converter")
    print("==================================================================")
    print(f"Source Raw Archive : {file_path.resolve()}")
    print(f"Target Excel File  : {output_path.resolve()}")
    print(f"Rows Per Tab Limit : {rows_per_tab:,} (Excel hardware limit: ~1,048,576)\n")

    print("Loading raw waveform dataset into memory...")
    try:
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            print(f"Error: Unsupported file format '{file_path.suffix}'. Supported formats are .parquet and .csv.")
            return False
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

    total_rows = len(df)
    cols = list(df.columns)
    print(f"Dataset Successfully Loaded -> Rows: {total_rows:,} | Columns: {cols}\n")

    if total_rows == 0:
        print("Warning: Dataset is empty. Creating empty Excel workbook.")

    print(f"Exporting to multi-sheet Excel file at: {output_path.name} ...")
    try:
        with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
            num_chunks = max(1, (total_rows + rows_per_tab - 1) // rows_per_tab)
            for i in range(num_chunks):
                start_idx = i * rows_per_tab
                end_idx = min(total_rows, (i + 1) * rows_per_tab)
                chunk = df.iloc[start_idx:end_idx]
                
                # Format sheet name safely within Excel's 31-character naming limit
                s_label = f"{start_idx // 1000}k" if start_idx >= 1000 else str(start_idx)
                e_label = f"{end_idx // 1000}k" if end_idx >= 1000 else str(end_idx)
                sheet_name = f"Part_{i+1}_{s_label}_{e_label}"
                
                print(f"  -> Writing Worksheet Tab [{sheet_name}] (Rows {start_idx:,} to {end_idx:,})...")
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                
        out_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[Success] Multi-tab Excel workbook generated cleanly: {output_path.resolve()} ({out_size_mb:.2f} MB)")
        return True

    except Exception as e:
        print(f"\nError creating Excel workbook: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet/CSV raw 500 Hz telemetry archives into multi-tab Microsoft Excel workbooks.")
    parser.add_argument("--file", required=True, help="Path to input raw telemetry (.parquet or .csv) file.")
    parser.add_argument("--out", dest="output", default=None, help="Optional custom path for output .xlsx workbook.")
    parser.add_argument("--rows-per-tab", type=int, default=500000, help="Number of sample rows to populate per Excel tab (default: 500,000).")
    args = parser.parse_args()

    input_file = Path(args.file)
    output_file = Path(args.output) if args.output else None

    success = convert_raw_to_multitab_excel(input_file, output_file, args.rows_per_tab)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
