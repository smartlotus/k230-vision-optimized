"""Bounded static-scene continuity and pixel jitter; no ground-truth accuracy claim."""


class Moments:
    def __init__(self):
        self.n=0;self.mean=[0.0,0.0];self.m2=[0.0,0.0]
        self.low=self.high=None

    def add(self,point):
        self.n+=1
        if self.low is None:self.low=list(point);self.high=list(point)
        for axis in (0,1):
            delta=point[axis]-self.mean[axis]
            self.mean[axis]+=delta/self.n
            self.m2[axis]+=delta*(point[axis]-self.mean[axis])
            self.low[axis]=min(self.low[axis],point[axis]);self.high[axis]=max(self.high[axis],point[axis])

    def report(self):
        return {'samples':self.n,'mean_px':list(self.mean) if self.n else None,
            'population_std_px':[(max(0,v/self.n))**0.5 for v in self.m2] if self.n else None,
            'range_px':[self.high[i]-self.low[i] for i in (0,1)] if self.n else None}


class BlockMetrics:
    def __init__(self):
        self.frames=0;self.replayed_frames=0;self.last_frame=None
        self.all_unique=self.all_stable=0
        self.rows={c:{'visible_frames':0,'unique_eligible_frames':0,'unique_stable_frames':0,
            'multiple_candidate_frames':0,'multiple_eligible_frames':0,'missing_run':0,'longest_missing_run':0,
            'unstable_run':0,'longest_unstable_run':0,'track_changes':0,
            'last_track':None,'last_center':None,'last_center_frame':None,'max_consecutive_step_px':0.0,
            'eligible_centers':Moments(),'stable_centers':Moments()} for c in range(1,7)}

    def update(self,candidates,frame_id):
        if type(frame_id) is not int or frame_id<1:raise ValueError('Fresh positive frame ID required')
        if self.last_frame is not None and frame_id<=self.last_frame:
            self.replayed_frames+=1;return False
        self.frames+=1;self.last_frame=frame_id
        unique_count=stable_count=0
        for color,row in self.rows.items():
            found=[c for c in candidates if c['target_id']==color]
            eligible=[c for c in found if c['eligible']]
            row['visible_frames']+=bool(found)
            row['multiple_candidate_frames']+=len(found)>1
            row['multiple_eligible_frames']+=len(eligible)>1
            unique=len(eligible)==1
            stable=unique and eligible[0]['stable']
            row['unique_eligible_frames']+=unique;row['unique_stable_frames']+=stable
            unique_count+=unique;stable_count+=stable
            row['missing_run']=0 if unique else row['missing_run']+1
            row['unstable_run']=0 if stable else row['unstable_run']+1
            row['longest_missing_run']=max(row['longest_missing_run'],row['missing_run'])
            row['longest_unstable_run']=max(row['longest_unstable_run'],row['unstable_run'])
            if not unique:continue
            item=eligible[0];center=item['center'];track=item['track_id']
            row['eligible_centers'].add(center)
            if stable:row['stable_centers'].add(center)
            if row['last_track'] is not None and row['last_track']!=track:row['track_changes']+=1
            if row['last_track']==track and row['last_center_frame']==frame_id-1:
                step=sum((center[i]-row['last_center'][i])**2 for i in (0,1))**0.5
                row['max_consecutive_step_px']=max(row['max_consecutive_step_px'],step)
            row['last_track']=track;row['last_center']=center;row['last_center_frame']=frame_id
        self.all_unique+=unique_count==6;self.all_stable+=stable_count==6
        return True

    def report(self):
        return {'frames':self.frames,'replayed_frames_ignored':self.replayed_frames,
            'all_six_unique_eligible_frames':self.all_unique,'all_six_unique_stable_frames':self.all_stable,
            'per_color':{str(c):{k:(v.report() if isinstance(v,Moments) else v)
                for k,v in row.items() if not k.startswith('last_') and k not in ('missing_run','unstable_run')}
                for c,row in self.rows.items()},
            'scope':'Unlabeled fixed-scene candidate continuity and center dispersion; not accuracy or physical motion',
            'jitter_sampling':'All unique eligible candidates, including unstable ones; stable subset reported separately',
            'longest_runs_unit':'processed frames, not measured physical absence duration'}


def main():
    import sys
    if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
    sys.path.insert(0,'/sdcard/transport_mvp')
    import json,time
    from vision.block_detector import BlockDetector
    from run_block_preview import main as preview
    from k230.camera_roles import OBJECT_CAMERA_ID
    original=BlockDetector.process
    metrics=BlockMetrics();overhead_us=0

    def measured(detector,image,frame_id,timestamp_ms):
        nonlocal overhead_us
        candidates=original(detector,image,frame_id,timestamp_ms)
        start=time.ticks_us();metrics.update(candidates,frame_id)
        overhead_us+=time.ticks_diff(time.ticks_us(),start)
        return candidates

    BlockDetector.process=measured
    try:
        with open('/sdcard/transport_mvp/confirmed_task_receipt.json') as stream:task=json.load(stream)
    except OSError:task=None
    try:
        result=preview(max_frames=1800,inspection_mode='COLOR',inspection_target=1,inspection_station='RAW',
            confirmed_task=task,report_path='/sdcard/transport_mvp/block_stability_csi%d_raw.json' % OBJECT_CAMERA_ID)
    finally:BlockDetector.process=original
    report=metrics.report()
    report.update(test='REAL_OBJECT_CAMERA_STATIC_CANDIDATE_STABILITY',source='REAL_CAMERA',camera_id=OBJECT_CAMERA_ID,profile_id=result['profile_id'],
        elapsed_ms=result['elapsed_ms'],detections_per_second=result['detections_per_second'],
        capture_errors=result['errors'],fatal=result['fatal'],measurement_overhead_us=overhead_us,
        motors_enabled=False,metric_coordinates_available=False,
        check_completed=result['frames']==1800 and metrics.frames==1800 and result['fatal'] is None)
    with open('/sdcard/transport_mvp/block_stability_csi%d_1800.json' % OBJECT_CAMERA_ID,'w') as stream:stream.write(json.dumps(report))
    print('BLOCK_STABILITY:'+json.dumps(report))
    return report


if __name__=='__main__':main()
