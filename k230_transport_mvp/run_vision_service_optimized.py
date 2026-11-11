"""Opt-in block tracking and UART service; legacy entries are untouched."""
import sys

if sys.implementation.name != 'micropython':
    raise SystemExit('Use CanMV on K230; this entry does not run on host Python.')
if any(name == 'k230' or name.startswith('k230.') for name in sys.modules):
    raise SystemExit('K230 modules already loaded; use a fresh interpreter to avoid mixing installations.')
root = '/sdcard/transport_mvp_optimized'
if root not in sys.path:
    sys.path.insert(0, root)

from k230.block_profile import make_config
from k230.main_vision_optimized import main

main(make_config())
