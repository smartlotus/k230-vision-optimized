import sys,json,gc
sys.path.insert(0,'/sdcard/kopt_20261112b')
from tools.native_fps_trial import trial_state
from k230.tuning.live_optimized import main
from k230.tuning.resources import snapshot
with open('/sdcard/kopt_20261112b/protected-state.json') as f:s=trial_state(json.load(f),'/sdcard/kopt_20261112b','TOP',2)
print('FPS02_START:'+json.dumps(snapshot()))
try:
 main('c2222222222222222222222222222222',root=s['profile_root'],mailbox='/sdcard/kopt_20261112b/fps02',max_frames=200,max_duration_ms=15000,max_errors=3,initial_state=s,vector_boundary=True,annotated_preview=True,preview_every=1)
finally:
 gc.collect();print('FPS02_END:'+json.dumps(snapshot()))

