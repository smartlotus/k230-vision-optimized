"""Explicit bounded native test; profile cloning never saves production state."""
import json
import gc
from k230.tuning.schema import clone
from k230.tuning.resources import snapshot


def trial_state(protected, root, mode=None, color=None):
    value=clone(protected)
    value['profile_root']=root+'/profiles'
    value['render_mode']='SELECTED'
    if mode is not None:
        if mode not in ('TOP','RING'):raise ValueError('Trial mode must be TOP or RING')
        value['mode']=mode;value['station']='BUFFER' if mode=='TOP' else 'PROCESS'
    if color is not None:
        if type(color) is not int or not 1<=color<=6:raise ValueError('Known color required')
        value['color']=color
    return value


def run(root, trial_id, variant='optimized', frames=30, duration_ms=15000,
        mode=None,color=None,vector=True,preview_every=3):
    if variant not in ('legacy','optimized') or type(frames) is not int or not 1<=frames<=300:
        raise ValueError('Bounded explicit variant required')
    if type(duration_ms) is not int or not 1000<=duration_ms<=60000:
        raise ValueError('Trial duration must be1..60 seconds')
    if len(trial_id)!=32 or any(c not in '0123456789abcdef' for c in trial_id):
        raise ValueError('Trial token required')
    with open(root+'/protected-state.json') as stream:protected=json.load(stream)
    state=trial_state(protected,root,mode,color)
    from k230.camera_roles import MODE_CAMERA
    gc.collect()
    print('NATIVE_TRIAL_START:'+json.dumps({'trial':trial_id,'variant':variant,
        'frames_limit':frames,'duration_limit_ms':duration_ms,'mode':state['mode'],'color':state['color'],
        'active_revision':state['active_profile']['revision'],'draft_revision':state['profile']['revision'],
        'dirty':state['dirty'],'camera_mapping':MODE_CAMERA,'vector':vector,'preview_every':preview_every,
        'resource_before':snapshot(),'uart_enabled':False,'motors_enabled':False}))
    if variant=='optimized':
        from k230.tuning.live_optimized import main
    else:
        from k230.tuning.live import main
    colors=protected.get('diagnostic_material_colors') if state['mode']=='RING' else None
    kwargs={'root':root+'/profiles','mailbox':root+'/mailbox-'+trial_id[:8],
        'max_frames':frames,'initial_state':state,'diagnostic_material_colors':colors,
        'diagnostic_material_color':protected.get('diagnostic_material_color') if state['mode']=='RING' else None,
        'vector_boundary':vector,'annotated_preview':False}
    if variant=='optimized':kwargs.update(preview_every=preview_every,max_duration_ms=duration_ms,max_errors=3)
    try:main(trial_id,**kwargs)
    finally:
        gc.collect()
        print('NATIVE_TRIAL_END:'+json.dumps({'trial':trial_id,'variant':variant,
            'resource_after':snapshot(),'uart_enabled':False,'motors_enabled':False}))
