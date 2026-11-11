"""Opt-in PC live workbench. Does not replace main.py or enable UART."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Run explicitly on K230 using CanMV.')
if '/sdcard/transport_mvp' not in sys.path:sys.path.insert(0,'/sdcard/transport_mvp')
import os,binascii
from k230.tuning.live import main
main(binascii.hexlify(os.urandom(16)).decode())
