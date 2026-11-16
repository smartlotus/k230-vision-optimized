import sys,json
sys.path.insert(0,'/sdcard/topface_formal_a01_20261116')
sys.path.insert(0,'/sdcard/cdiag1116')
from topface_trial_gc import run
report=run(root='/sdcard/topface_formal_a01_20261116',frames=18000,display=True,ide_preview=True)
with open('/sdcard/cdiag1116/manual-gc18000-r2.json','w') as f:json.dump(report,f)
print('MANUAL_GC18000_R2_PERSISTED')

