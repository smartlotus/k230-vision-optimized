"""Host synthetic lane geometry/rejection matrix; no camera acceptance."""
import argparse,hashlib,json
from pathlib import Path
from k230.vision_adapter import VisionAdapter

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('vision/lane.py','vision/common.py','k230/vision_adapter.py','tools/evaluate_lane_sequences.py')


class LaneImage:
    def __init__(self,bands,inconsistent=False):self.bands=bands;self.inconsistent=inconsistent
    def width(self):return 320
    def height(self):return 240
    def to_grayscale(self):return self
    def get_pixel(self,x,y):
        if not 0<=x<320 or not 0<=y<240:raise ValueError('Pixel outside image')
        bands=([(48,112)] if y<120 else [(208,272)]) if self.inconsistent else self.bands
        return 128 if any(a<=x<b for a,b in bands) else 0


def evaluate():
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    cases=[]
    for center in (80,160,240):
        for step in (1,2,4):
            cases.append({'name':'center%d-step%d' % (center,step),'bands':[(center-32,center+32)],
                'step':step,'expected_offset':center-160.5,'tolerance':step/2,'category':'offset'})
    for name,bands in (('absent',[]),('all-gray',[(0,320)]),('left-edge',[(0,64)]),
            ('right-edge',[(256,320)]),('two-bands',[(40,104),(200,264)]),
            ('too-narrow',[(144,176)]),('too-wide',[(80,240)]),
            ('inconsistent-rows',[(48,112)]),('one-unique-row',[(128,192)])):
        cases.append({'name':name,'bands':bands,'step':2,'category':'rejection'})
    rows=[]
    for i,c in enumerate(cases,1):
        roi=(0,40,320,1 if c['name']=='one-unique-row' else 160)
        cfg={'CAMERA_IDS':(2,),'MODE_CAMERA':{'LANE':2},'DESKTOP_SINGLE_CAMERA':False,
            'LANE_ROI':roi,'LANE_THRESHOLD':(90,160),'LANE_MIN_WIDTH_PX':48,'LANE_MAX_WIDTH_PX':112,
            'LANE_SCAN_ROWS':3,'LANE_SAMPLE_STEP':c['step'],'LANE_MAX_CENTER_SPREAD_PX':20,'CALIBRATIONS':{}}
        packet={'image':LaneImage(c['bands'],c['name']=='inconsistent-rows'),'camera_id':2,'frame_id':i,
            'timestamp_ms':i*40,'is_simulated':True,'image_profile':{'width':320,'height':240,'crop':False,
                'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}}
        result=VisionAdapter(cfg).detect('LANE',packet)
        boundary=(result['is_simulated'] is True and result['camera_id']==2 and result['x_mm'] is None
            and result['y_mm'] is None and result['calibration_id'] is None and result['target_pose_valid'] is False)
        row={'input':c,'status':result['status'],'reason':result['reason'],'offset_px':result.get('offset_px'),
            'unique_sample_rows':len(set(roi[1]+int((j+1)*roi[3]/4) for j in range(3))),'boundary_passed':boundary}
        if c['category']=='offset':
            error=abs(result['offset_px']-c['expected_offset']) if result.get('offset_px') is not None else None
            row.update(error_px=error,behavior_passed=result['status']=='OK' and result.get('auxiliary_only') is True
                and error is not None and error<=c['tolerance'])
        else:
            row['behavior_passed']=(result['status'] in ('NOT_FOUND','ERROR','UNCALIBRATED') and
                all(result.get(k) is None for k in ('u_px','v_px','offset_px')))
        row['passed']=row['behavior_passed'] and boundary;rows.append(row)
    offset=[r for r in rows if r['input']['category']=='offset'];negative=rows[9:]
    unchanged=hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    return {'status':'PASS' if unchanged and all(r['passed'] for r in rows) else 'FAIL',
        'scope':'HOST_SYNTHETIC_LANE_CONTRACT_ONLY','source':'GENERATED_FAKE_GRAYSCALE_NOT_CAMERA',
        'is_simulated':True,'physical_camera_accessed':False,'live_CSI2_lane_acceptance':False,'motors_enabled':False,
        'source_hashes':hashes,'source_files_unchanged':unchanged,'denominator_unit':'independent_synthetic_case',
        'offset':{'expected':len(offset),'passed':sum(r['behavior_passed'] for r in offset),
            'errors_px':[r['error_px'] for r in offset]},
        'rejection':{'expected':len(negative),'passed':sum(r['behavior_passed'] for r in negative)},
        'boundary':{'expected':len(rows),'passed':sum(r['boundary_passed'] for r in rows)},'cases':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=evaluate()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    raise SystemExit(0 if report['status']=='PASS' else 1)
