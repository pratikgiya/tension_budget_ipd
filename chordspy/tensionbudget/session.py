"""
TensionBudget — Session management.

Handles session identity, metadata collection, and manifest writing.
Each recording session gets a unique ID and a sidecar JSON manifest.
"""

import uuid
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from chordspy.tensionbudget.config import TBConfig


class TBSession:
    """
    Manages session-level metadata for a TensionBudget recording.

    A session starts when recording begins and ends when recording stops.
    The manifest is written as a sidecar JSON alongside the CSV file.
    """

    def __init__(self):
        self.session_id = None
        self.start_time_utc = None
        self.start_time_local = None
        self.start_perf_counter = None
        self.end_time_utc = None
        self.board = None
        self.sampling_rate = None
        self.resolution = None
        self.num_channels = None
        self.baud_rate = None
        self.protocol = None
        self.total_samples = 0
        self.dropped_samples = 0
        self.csv_filename = None
        self.manifest_filename = None
        self.calibration_data = None

    def begin(self, board, sampling_rate, resolution, num_channels,
              protocol="usb", baud_rate=None):
        """
        Start a new session and capture all metadata.

        Args:
            board: Detected board name (e.g., "UNO-R4")
            sampling_rate: Actual sampling rate in Hz
            resolution: ADC resolution in bits
            num_channels: Number of data channels
            protocol: Connection protocol ("usb", "wifi", "ble")
            baud_rate: Baud rate (for USB connections)
        """
        self.session_id = uuid.uuid4().hex[:12]
        self.start_time_utc = datetime.now(timezone.utc)
        self.start_time_local = datetime.now()
        self.start_perf_counter = time.perf_counter()
        self.board = board
        self.sampling_rate = sampling_rate
        self.resolution = resolution
        self.num_channels = num_channels
        self.baud_rate = baud_rate or TBConfig.TARGET_BAUD_RATE
        self.protocol = protocol
        self.total_samples = 0
        self.dropped_samples = 0

    def elapsed_seconds(self):
        """Return seconds elapsed since session start (high-resolution)."""
        if self.start_perf_counter is None:
            return 0.0
        return time.perf_counter() - self.start_perf_counter

    def generate_csv_filename(self, output_dir=None):
        """
        Generate a timestamped CSV filename.

        Returns:
            Path object for the CSV file.
        """
        timestamp = self.start_time_local.strftime(TBConfig.CSV_TIMESTAMP_FORMAT)
        filename = f"{TBConfig.CSV_FILENAME_PREFIX}_{timestamp}_{self.session_id}.csv"
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path / filename
        return Path(filename)

    def generate_manifest_filename(self, csv_path):
        """
        Generate the manifest filename matching the CSV file.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Path object for the manifest JSON file.
        """
        csv_path = Path(csv_path)
        return csv_path.with_suffix(".json")

    def csv_metadata_header(self):
        """
        Generate metadata lines to write at the top of the CSV file.
        Each line is prefixed with the comment character.

        Returns:
            List of metadata header strings.
        """
        c = TBConfig.CSV_METADATA_COMMENT_CHAR
        lines = [
            f"{c} TensionBudget Session Recording",
            f"{c} Session ID: {self.session_id}",
            f"{c} Board: {self.board}",
            f"{c} Protocol: {self.protocol}",
            f"{c} Baud Rate: {self.baud_rate}",
            f"{c} Sampling Rate: {self.sampling_rate} Hz",
            f"{c} Resolution: {self.resolution} bits",
            f"{c} Channels: {self.num_channels}",
            f"{c} Channel Map: {TBConfig.CHANNEL_MAP or 'not configured'}",
            f"{c} Unused Channels: {TBConfig.unused_channel_indices() or 'none'}",
            f"{c} Recording Start (UTC): {self.start_time_utc.isoformat()}",
            f"{c} Recording Start (Local): {self.start_time_local.isoformat()}",
        ]
        return lines

    def csv_column_headers(self):
        """
        Generate the CSV column header row.

        Returns:
            List of column name strings.
        """
        headers = ["Sample_Index", "Timestamp_s", "Arduino_Counter"]
        for i in range(self.num_channels):
            headers.append(f"Channel{i + 1}")
        return headers

    def end(self):
        """Mark the session as ended and record end time."""
        self.end_time_utc = datetime.now(timezone.utc)

    def to_manifest_dict(self):
        """
        Build a manifest dictionary with all session metadata.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        duration_s = None
        if self.start_time_utc and self.end_time_utc:
            duration_s = (self.end_time_utc - self.start_time_utc).total_seconds()

        expected_samples = None
        if duration_s is not None and self.sampling_rate:
            expected_samples = int(duration_s * self.sampling_rate)

        return {
            "tensionbudget_version": "0.1.0",
            "session_id": self.session_id,
            "board": self.board,
            "protocol": self.protocol,
            "baud_rate": self.baud_rate,
            "sampling_rate_hz": self.sampling_rate,
            "resolution_bits": self.resolution,
            "num_channels": self.num_channels,
            "recording_start_utc": (
                self.start_time_utc.isoformat() if self.start_time_utc else None
            ),
            "recording_start_local": (
                self.start_time_local.isoformat() if self.start_time_local else None
            ),
            "recording_end_utc": (
                self.end_time_utc.isoformat() if self.end_time_utc else None
            ),
            "duration_seconds": duration_s,
            "total_samples_recorded": self.total_samples,
            "dropped_samples_detected": self.dropped_samples,
            "expected_samples": expected_samples,
            "csv_filename": str(self.csv_filename) if self.csv_filename else None,
            "csv_schema": self.csv_column_headers(),
            "channel_map": TBConfig.CHANNEL_MAP,
            "unused_channels": TBConfig.unused_channel_indices(),
            "calibration": self.calibration_data,
        }

    def write_manifest(self, manifest_path=None):
        """
        Write the session manifest to a JSON file.

        Args:
            manifest_path: Path for the manifest file. If None, derives from csv_filename.

        Returns:
            Path to the written manifest file.
        """
        if manifest_path is None:
            if self.csv_filename:
                manifest_path = self.generate_manifest_filename(self.csv_filename)
            else:
                raise ValueError("No CSV filename set and no manifest path provided.")

        manifest_path = Path(manifest_path)
        self.manifest_filename = manifest_path

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_manifest_dict(), f, indent=2, ensure_ascii=False)

        print(f"Session manifest written: {manifest_path}")
        return manifest_path
