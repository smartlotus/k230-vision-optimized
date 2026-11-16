"""Explicit production-code TOP entry; does not replace board startup files."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Run on K230')
root='/sdcard/tfp1116'
sys.path.insert(0,root)
from k230.topface_service import run
run(root+'/model',frames=18000,display=True,ide_preview=True,
    report_path=root+'/production-session01.json')
