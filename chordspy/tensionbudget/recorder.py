"""
TensionBudget — Stage 1 CSV Recorder.

Records raw acquisition data to TensionBudget-format CSV files with:
    - Metadata header comments
    - Enhanced schema: Sample_Index, Timestamp_s, Arduino_Counter, Channel1..Channel6
    - Sidecar JSON session manifest
    - Sample-drop detection via Arduino counter gaps

This is a *separate* recording mode from Chords-Python's built-in CSV recorder.
The existing Chords recorder (Counter, Channel1..N) is left untouched.

Usage:
    tb_recorder = TBRecorder()
    tb_recorder.start(connection_manager)    # registers as sample callback
    ...
    tb_recorder.stop()                       # writes manifest, closes file
"""

import csv
import json
import time
from pathlib import Path

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.session import TBSession


class TBRecorder:
    """
    TensionBudget-specific CSV recorder.

    Registers as a sample callback on a Connection object to receive every
    sample with (channel_data, timestamp, arduino_counter). Writes each
    sample to an enhanced CSV with metadata headers, timestamps, and the
    Arduino counter for drop detection.

    Lifecycle:
        1. Create: ``recorder = TBRecorder(output_dir="recordings/")``
        2. Start:  ``recorder.start(connection_manager)``
        3. (samples flow automatically via callback)
        4. Stop:   ``recorder.stop()``  → writes manifest, closes CSV

    The recorder does NOT touch the existing Chords CSV recording system.
    Both can run simultaneously if desired.
    """

    def __init__(self, output_dir=None):
        """
        Args:
            output_dir: Directory for output files. If None, writes to CWD.
        """
        self.output_dir = output_dir
        self.session = None
        self._csv_file = None
        self._csv_writer = None
        self._connection = None
        self._recording = False
        self._sample_index = 0
        self._start_perf_time = None      # time.perf_counter() at first sample (for Timestamp_s)
        self._last_arduino_counter = None  # For drop detection
        self._drop_events = []            # List of (sample_index, expected, got, dropped) tuples

    @property
    def recording(self):
        """True while actively recording."""
        return self._recording

    def start(self, connection, board=None, sampling_rate=None,
              resolution=None, num_channels=None, protocol="usb"):
        """
        Start TensionBudget recording.

        Begins a new session, opens the CSV file with metadata headers,
        and registers as a sample callback on the Connection object.

        Args:
            connection: A chordspy.connection.Connection instance (must
                have an active stream with stream_active=True).
            board: Board name override. If None, reads from connection.
            sampling_rate: Sampling rate override. If None, reads from connection.
            resolution: ADC resolution override. If None, reads from connection.
            num_channels: Channel count override. If None, reads from connection.
            protocol: Connection protocol string.

        Returns:
            Path to the CSV file being written.

        Raises:
            RuntimeError: If already recording or no active stream.
        """
        if self._recording:
            raise RuntimeError("TBRecorder is already recording.")

        if not connection.stream_active:
            raise RuntimeError(
                "Cannot start TB recording: no active stream on Connection."
            )

        # Resolve hardware parameters from the connection or overrides
        _board = board or getattr(connection, '_detected_board', None)
        if _board is None and hasattr(connection, 'usb_connection') and connection.usb_connection:
            _board = getattr(connection.usb_connection, 'board', TBConfig.TARGET_BOARD)
        _board = _board or TBConfig.TARGET_BOARD

        _sr = sampling_rate or connection.sampling_rate or TBConfig.TARGET_SAMPLING_RATE
        _res = resolution or getattr(connection, 'resolution', TBConfig.TARGET_RESOLUTION_BITS)
        _nch = num_channels or connection.num_channels or TBConfig.TARGET_NUM_RAW_CHANNELS
        _baud = TBConfig.TARGET_BAUD_RATE

        # If connected via USB, try to get actual baud rate
        if connection.usb_connection and hasattr(connection.usb_connection, 'ser'):
            ser = connection.usb_connection.ser
            if ser and ser.is_open:
                _baud = ser.baudrate

        # Create session
        self.session = TBSession()
        self.session.begin(
            board=_board,
            sampling_rate=_sr,
            resolution=_res,
            num_channels=_nch,
            protocol=protocol,
            baud_rate=_baud,
        )

        # Generate file paths
        csv_path = self.session.generate_csv_filename(output_dir=self.output_dir)
        self.session.csv_filename = csv_path

        # Try to load calibration data
        cal_file = Path.cwd() / "tensionbudget_calibration.json"
        if cal_file.exists():
            try:
                with open(cal_file, 'r') as f:
                    self.session.calibration_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load calibration file {cal_file.name}: {e}")

        # Open CSV and write metadata + headers
        self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)

        # Write metadata header comments
        for line in self.session.csv_metadata_header():
            self._csv_file.write(line + '\n')

        # Write channel mapping info if configured
        cmap = TBConfig.CHANNEL_MAP
        c = TBConfig.CSV_METADATA_COMMENT_CHAR
        if cmap:
            self._csv_file.write(
                f"{c} Channel Map: {cmap}\n"
            )
            unused = TBConfig.unused_channel_indices()
            if unused:
                self._csv_file.write(
                    f"{c} Unused/Floating Channels: {unused}\n"
                )
        else:
            self._csv_file.write(
                f"{c} Channel Map: not configured (all channels recorded)\n"
            )

        # Write column headers
        self._csv_writer.writerow(self.session.csv_column_headers())

        # Reset state
        self._sample_index = 0
        self._start_perf_time = None
        self._last_arduino_counter = None
        self._drop_events = []

        # Register as sample callback
        self._connection = connection
        connection.register_sample_callback(self._sample_callback)
        self._recording = True

        print(f"TB recording started: {csv_path}")
        return csv_path

    def _sample_callback(self, channel_data, timestamp, arduino_counter):
        """
        Callback invoked by Connection for every acquired sample.

        Timestamps use time.perf_counter() captured at callback invocation,
        not the LSL local_clock() value passed by the data handler. This gives
        TensionBudget its own independent high-resolution monotonic timing,
        decoupled from LSL's internal scheduling.

        Args:
            channel_data: List of float channel values.
            timestamp: LSL local_clock() timestamp (not used for CSV; kept for
                       compatibility with the callback signature).
            arduino_counter: Arduino's 1-byte rolling counter (0-255) or None.
        """
        if not self._recording:
            return

        # Capture our own high-resolution timestamp (time.perf_counter)
        now = time.perf_counter()
        if self._start_perf_time is None:
            self._start_perf_time = now

        self._sample_index += 1
        relative_time = now - self._start_perf_time

        # Drop detection via Arduino counter (1-byte, wraps 255 → 0).
        # Python's modulo (% 256) always returns non-negative for a positive
        # divisor, so wraparound from 255 → 0 is handled correctly:
        #   expected = (255 + 1) % 256 = 0  →  no false drop on normal rollover
        #   dropped  = (next - prev - 1) % 256  →  correct gap even across wrap
        if arduino_counter is not None:
            if self._last_arduino_counter is not None:
                expected = (self._last_arduino_counter + 1) % 256
                if arduino_counter != expected:
                    dropped = (arduino_counter - self._last_arduino_counter - 1) % 256
                    self.session.dropped_samples += dropped
                    self._drop_events.append((
                        self._sample_index,
                        expected,
                        arduino_counter,
                        dropped,
                    ))
            self._last_arduino_counter = arduino_counter

        # Build and write CSV row
        counter_val = arduino_counter if arduino_counter is not None else ""
        row = [
            self._sample_index,
            f"{relative_time:.6f}",
            counter_val,
        ] + channel_data

        try:
            self._csv_writer.writerow(row)
        except Exception as e:
            print(f"TB CSV write error: {e}")
            self.stop()

        self.session.total_samples = self._sample_index

    def stop(self):
        """
        Stop TensionBudget recording.

        Closes the CSV file, ends the session, writes the manifest JSON,
        and unregisters from the Connection's sample callbacks.

        Returns:
            Path to the session manifest JSON, or None if not recording.
        """
        if not self._recording:
            return None

        self._recording = False

        # Unregister callback
        if self._connection and self._sample_callback in self._connection.sample_callbacks:
            self._connection.sample_callbacks.remove(self._sample_callback)
        self._connection = None

        # Close CSV
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception as e:
                print(f"Error closing TB CSV: {e}")
            self._csv_file = None
            self._csv_writer = None

        # End session and write manifest
        manifest_path = None
        if self.session:
            self.session.end()

            # Add drop events to manifest
            try:
                manifest_path = self.session.write_manifest()
                # Append drop event details to manifest
                if self._drop_events:
                    self._write_drop_report(manifest_path)
            except Exception as e:
                print(f"Error writing TB manifest: {e}")

            total = self.session.total_samples
            drops = self.session.dropped_samples
            csv_name = self.session.csv_filename
            print(f"TB recording stopped: {total} samples, {drops} drops detected")
            print(f"  CSV: {csv_name}")
            if manifest_path:
                print(f"  Manifest: {manifest_path}")

        return manifest_path

    def _write_drop_report(self, manifest_path):
        """
        Append drop event details to the manifest JSON.

        Re-reads the manifest, adds drop_events list, and writes it back.
        """
        import json
        manifest_path = Path(manifest_path)
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            manifest["drop_events"] = [
                {
                    "at_sample_index": ev[0],
                    "expected_counter": ev[1],
                    "received_counter": ev[2],
                    "estimated_dropped": ev[3],
                }
                for ev in self._drop_events
            ]

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error appending drop report to manifest: {e}")

    def get_status(self):
        """
        Return a status dict for UI / diagnostics.

        Returns:
            dict with recording state, sample count, drop count, elapsed time.
        """
        if not self.session:
            return {"recording": False}

        return {
            "recording": self._recording,
            "session_id": self.session.session_id,
            "samples_recorded": self._sample_index,
            "drops_detected": self.session.dropped_samples,
            "elapsed_seconds": round(self.session.elapsed_seconds(), 2),
            "csv_filename": str(self.session.csv_filename) if self.session.csv_filename else None,
        }
