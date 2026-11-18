"""Explicit production-code TOP entry; does not replace board startup files."""
import binascii
import os
import sys

if sys.implementation.name!='micropython':raise SystemExit('Run on K230')
root = '/sdcard/tfp1116'
sys.path.insert(0, root)
from k230.topface_service import run

report_dir = '/data/logs'
try:
    os.stat(report_dir)
except OSError:
    os.mkdir(report_dir)

report_name = 'topface-' + binascii.hexlify(os.urandom(8)).decode() + '.json'
run(root + '/model', frames=18000, display=True, ide_preview=True,
    report_path=report_dir + '/' + report_name)
