"""Opt-in fresh recognition with preview rendering every third frame."""
import sys
if sys.implementation.name != 'micropython':
    raise SystemExit('Run explicitly on K230 using CanMV.')
if any(name == 'k230' or name.startswith('k230.') for name in sys.modules):
    raise SystemExit('Use a fresh interpreter to avoid mixing K230 installations.')
root = '/sdcard/transport_mvp_optimized'
if root not in sys.path:
    sys.path.insert(0, root)
import os, binascii
from k230.tuning.live_optimized import main
main(binascii.hexlify(os.urandom(16)).decode(), preview_every=3)
