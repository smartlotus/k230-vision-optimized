"""Separate opt-in entry; install the matching tree at transport_mvp_optimized."""
import sys

if sys.implementation.name != 'micropython':
    raise SystemExit('Use CanMV on K230; this entry does not run on host Python.')

# A separate root preserves the legacy /sdcard/transport_mvp installation.
root = '/sdcard/transport_mvp_optimized'
if any(name == 'k230' or name.startswith('k230.') for name in sys.modules):
    raise SystemExit('K230 modules already loaded; use a fresh interpreter to avoid mixing installations.')
if root not in sys.path:
    sys.path.insert(0, root)

from k230.block_profile import make_config
from k230.main_optimized import main

main(make_config())
