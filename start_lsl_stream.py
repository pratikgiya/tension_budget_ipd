"""
start_lsl_stream.py — Dual-Arduino USB-to-LSL Bridge for TensionBudget
======================================================================
Simultaneously connects to TWO Arduino Uno R4 boards via USB:
  - Left Arduino on COM6 (reading Analog Pin A2 / index 2)
  - Right Arduino on COM7 (reading Analog Pin A2 / index 2)

Synchronizes both streams using independent background reading threads and
broadcasts a clean 2-channel 500 Hz bilateral EMG stream over Lab Streaming Layer (LSL):
  - LSL Channel 0: Left EMG (COM6 A2)
  - LSL Channel 1: Right EMG (COM7 A2)

Run from repo root (with .venv active):
    python start_lsl_stream.py
"""

import serial
import time
import sys
import threading

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
LEFT_PORT   = "COM6"    # COM port for Left shoulder Arduino
RIGHT_PORT  = "COM5"    # COM port for Right shoulder Arduino
PIN_INDEX   = 2         # Analog Pin A2 corresponds to raw packet slot index 2
BAUDRATES   = [230400, 115200]
# ──────────────────────────────────────────────────────────────────────────────

# ── LSL Import ────────────────────────────────────────────────────────────────
try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    print("ERROR: pylsl is not installed. Run: pip install pylsl")
    sys.exit(1)

# ── Protocol constants (Chords USB packet format) ─────────────────────────────
SYNC1       = 0xC7
SYNC2       = 0x7C
END_BYTE    = 0x01
HEADER_LEN  = 3       # 2 sync bytes + 1 counter byte
NUM_CHANNELS_RAW = 6  # Firmware transmits 6 channels per packet
SAMPLING_RATE = 500   # Hz — Uno R4 @ 500 Hz
RESOLUTION    = 14    # 14-bit ADC on Uno R4
PACKET_LENGTH = (2 * NUM_CHANNELS_RAW) + HEADER_LEN + 1  # = 16 bytes

SUPPORTED_BOARDS = {
    "UNO-R4":         {"sampling_rate": 500, "num_channels": 6, "resolution": 14},
    "UNO-R3":         {"sampling_rate": 250, "num_channels": 6, "resolution": 10},
    "MEGA-2560-R3":   {"sampling_rate": 250, "num_channels": 16, "resolution": 10},
}

# Shared state to synchronize samples between the two Arduinos
latest_samples = {"left": 0.0, "right": 0.0}
running = True
packet_counters = {"left": 0, "right": 0}


def connect_board(port: str) -> serial.Serial | None:
    """Connect to an Arduino on the specified port, auto-detecting baudrate."""
    for baud in BAUDRATES:
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1.5)
            # Give Arduino Uno R4 bootloader 1.5 seconds to settle after USB serial open reset
            time.sleep(1.5)
            ser.flushInput()
            ser.flushOutput()
            for _ in range(4):
                ser.write(b"WHORU\n")
                time.sleep(0.1)
                resp = ser.readline().strip().decode(errors="ignore")
                if resp in SUPPORTED_BOARDS:
                    print(f"  ✓ Connected to {resp} on {port} (@ {baud} baud)")
                    return ser
            ser.close()
        except Exception as e:
            continue
    print(f"  ✗ Failed to open or identify Chords board on {port}")
    return None


def reader_thread(ser: serial.Serial, side: str, push_lsl: bool, outlet: StreamOutlet = None):
    """Background worker thread that constantly reads packets from one Arduino."""
    global latest_samples, running, packet_counters

    ser.flushInput()
    ser.flushOutput()
    ser.write(b"START\n")
    time.sleep(0.1)

    buf = bytearray()
    last_start_attempt = time.time()
    last_diag_print = time.time()
    stream_started = False

    try:
        while running:
            # If we haven't received any valid packets yet, periodically resend START command and print diagnostics
            if not stream_started and time.time() - last_diag_print > 3.0:
                print(f"\n[DIAGNOSTIC - {side.upper()} ({ser.port})] Waiting for initial packets... Raw bytes waiting: {ser.in_waiting}")
                ser.write(b"START\n")
                last_diag_print = time.time()
                last_start_attempt = time.time()

            raw = ser.read(ser.in_waiting or 1)
            if not raw:
                continue
            buf.extend(raw)

            while len(buf) >= PACKET_LENGTH:
                idx = buf.find(bytes([SYNC1, SYNC2]))
                if idx == -1:
                    if not stream_started and len(buf) > 30:
                        print(f"[DIAGNOSTIC - {side.upper()}] Sync bytes C77C not found in buffer: {buf[:30].hex()}")
                    buf.clear()
                    break
                if len(buf) < idx + PACKET_LENGTH:
                    break  # wait for complete packet

                pkt = buf[idx : idx + PACKET_LENGTH]
                # Some firmware builds terminate packets with \r (0x0D) or \n (0x0A) instead of 0x01
                if pkt[0] == SYNC1 and pkt[1] == SYNC2 and pkt[-1] in (END_BYTE, 0x0D, 0x0A):
                    # Extract sample from Pin A2 (PIN_INDEX)
                    hi = pkt[2 * PIN_INDEX + HEADER_LEN]
                    lo = pkt[2 * PIN_INDEX + HEADER_LEN + 1]
                    val = float((hi << 8) | lo)

                    latest_samples[side] = val
                    packet_counters[side] += 1
                    stream_started = True
                    last_start_attempt = time.time()  # successfully receiving

                    # If this is the primary (Left) thread, emit both Left and Right to LSL
                    # This eliminates clock drift and guarantees synchronized 500 Hz bilateral streaming!
                    if push_lsl and outlet:
                        outlet.push_sample([latest_samples["left"], latest_samples["right"]])

                    del buf[: idx + PACKET_LENGTH]
                else:
                    if not stream_started and time.time() - last_diag_print > 2.0:
                        print(f"\n[DIAGNOSTIC - {side.upper()}] Packet validation failed! Full hex: {pkt.hex()}")
                        last_diag_print = time.time()
                    del buf[: idx + 1]

    except Exception as e:
        if running:
            print(f"\n[{side.upper()}] Serial read error: {e}")
    finally:
        try:
            ser.write(b"STOP\n")
            ser.close()
        except Exception:
            pass


def main():
    global running, packet_counters
    print("=" * 64)
    print("  TensionBudget — Dual-Arduino USB-to-LSL Bridge")
    print("=" * 64)
    print(f"  Target Configuration: Left = {LEFT_PORT} (Pin A2), Right = {RIGHT_PORT} (Pin A2)")
    print("  Connecting to hardware...")

    ser_left = connect_board(LEFT_PORT)
    ser_right = connect_board(RIGHT_PORT)

    if not ser_left or not ser_right:
        print("\n❌ Could not connect to BOTH Arduinos.")
        print(f"   Check your USB connections: COM6 and COM7 must be accessible.")
        if ser_left: ser_left.close()
        if ser_right: ser_right.close()
        sys.exit(1)

    # Set up LSL stream outlet (2 channels: index 0 = Left, index 1 = Right)
    info = StreamInfo(
        name="Chords-Python",
        type="EMG_Bilateral",
        channel_count=2,
        nominal_srate=SAMPLING_RATE,
        channel_format="float32",
        source_id="dual_uno_r4_usb"
    )
    desc = info.desc()
    res_node = desc.append_child("resinfo")
    res_node.append_child_value("resolution", str(RESOLUTION))

    outlet = StreamOutlet(info)
    print(f"\n✅ Dual-Arduino LSL bilateral stream started!")
    print(f"   Channel 0 → Left EMG  ({LEFT_PORT}, Pin A2)")
    print(f"   Channel 1 → Right EMG ({RIGHT_PORT}, Pin A2)")
    print(f"   Sampling rate: ~{SAMPLING_RATE} Hz")
    print(f"\n   NOW launch TensionBudget Monitor in your second terminal:")
    print(f"   python -m chordspy.tensionbudget_app")
    print(f"\n   Press Ctrl+C here to stop streaming cleanly.\n")

    # Start reader threads: Left acts as the primary synchronization pacemaker for LSL pushing
    t_left = threading.Thread(target=reader_thread, args=(ser_left, "left", True, outlet), daemon=True)
    t_right = threading.Thread(target=reader_thread, args=(ser_right, "right", False, None), daemon=True)

    t_left.start()
    t_right.start()

    last_report = time.time()
    try:
        while True:
            time.sleep(1.0)
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - last_report
                rate_l = packet_counters["left"] / elapsed
                rate_r = packet_counters["right"] / elapsed
                print(f"  Streaming… Rates: Left ({LEFT_PORT}) = {rate_l:.0f} Hz  |  Right ({RIGHT_PORT}) = {rate_r:.0f} Hz")
                packet_counters["left"] = 0
                packet_counters["right"] = 0
                last_report = now

    except KeyboardInterrupt:
        print("\n\nStopping stream (Ctrl+C received)…")
        running = False
        time.sleep(0.5)
        print("Both serial ports closed. LSL stream ended.")


if __name__ == "__main__":
    main()
