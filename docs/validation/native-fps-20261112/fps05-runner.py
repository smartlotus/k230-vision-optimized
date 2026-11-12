import sys,json,gc
sys.path.insert(0,'/sdcard/kopt_20261112d')
from tools.native_fps_trial import trial_state
from k230.tuning.live_optimized import main
from k230.tuning.resources import snapshot
with open('/sdcard/kopt_20261112d/protected-state.json') as f:s=trial_state(json.load(f),'/sdcard/kopt_20261112d','TOP',2)
print('FPS05_START:'+json.dumps(snapshot()))
try:
 main('c5555555555555555555555555555555',root=s['profile_root'],mailbox='/sdcard/kopt_20261112d/fps05',max_frames=300,max_duration_ms=15000,max_errors=3,initial_state=s,vector_boundary=True,annotated_preview=True,preview_every=1,preview_quality=40,preview_scale=.25)
finally:
 gc.collect();print('FPS05_END:'+json.dumps(snapshot()))

