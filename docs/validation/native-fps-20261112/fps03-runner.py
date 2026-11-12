import sys,json,gc
sys.path.insert(0,'/sdcard/kopt_20261112c')
from tools.native_fps_trial import trial_state
from k230.tuning.live_optimized import main
from k230.tuning.resources import snapshot
with open('/sdcard/kopt_20261112c/protected-state.json') as f:s=trial_state(json.load(f),'/sdcard/kopt_20261112c','TOP',2)
print('FPS03_START:'+json.dumps(snapshot()))
try:
 main('c3333333333333333333333333333333',root=s['profile_root'],mailbox='/sdcard/kopt_20261112c/fps03',max_frames=240,max_duration_ms=15000,max_errors=3,initial_state=s,vector_boundary=True,annotated_preview=True,preview_every=1,preview_quality=40)
finally:
 gc.collect();print('FPS03_END:'+json.dumps(snapshot()))

