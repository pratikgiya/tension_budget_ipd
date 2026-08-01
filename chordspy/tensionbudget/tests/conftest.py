"""
conftest.py — TensionBudget test isolation.

Stubs out hardware-dependent modules that are imported transitively via
chordspy/__init__.py → app.py → connection.py → chords_serial.py → serial.

pyserial is not installed in the dev environment (confirmed per project
context doc: important_context.md §4). The tensionbudget scoring/analytics
modules have zero runtime dependency on serial — this conftest blocks the
import chain before the broken imports are reached.

Strategy: register stub modules for pyserial and the Chords hardware
interface layer BEFORE pytest loads the test file. The stubs must be
registered in sys.modules before any import statement runs, so conftest.py
(loaded first by pytest) is the right place. We do NOT stub chordspy or
chordspy.tensionbudget — those must remain real importable packages.
"""
import sys
import types


def _make_stub(name):
    """Create and register a minimal stub module."""
    mod = types.ModuleType(name)
    mod.__path__ = []   # Marks it as a package-compatible stub
    sys.modules[name] = mod
    return mod


# ── pyserial stubs ────────────────────────────────────────────────────
# Must be registered before chords_serial.py tries `import serial`
_serial = _make_stub("serial")
_serial_tools = _make_stub("serial.tools")
_serial.tools = _serial_tools
_serial_tools_lp = _make_stub("serial.tools.list_ports")
_serial_tools.list_ports = _serial_tools_lp

# Stub a minimal Serial class so any accidental direct use fails clearly
class _FakeSerial:
    def __init__(self, *a, **kw):
        raise RuntimeError("pyserial not installed — hardware not available in dev environment")
_serial.Serial = _FakeSerial
_serial_tools_lp.comports = lambda: []

# ── Chords hardware-interface module stubs ────────────────────────────
# These are the modules that import serial at module-load time.
# We stub them as empty modules so chordspy/__init__.py can load
# without triggering the serial import chain.
# IMPORTANT: we do NOT stub chordspy or chordspy.tensionbudget.

_chords_serial = _make_stub("chordspy.chords_serial")
_chords_serial.Chords_USB = object

_chords_wifi = _make_stub("chordspy.chords_wifi")
_chords_wifi.Chords_WIFI = object

_chords_ble = _make_stub("chordspy.chords_ble")
_chords_ble.Chords_BLE = object

_connection = _make_stub("chordspy.connection")
_connection.Connection = object

_app = _make_stub("chordspy.app")
_app.main = lambda: None

