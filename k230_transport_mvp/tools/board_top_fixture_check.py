"""Native six-color round-face fixture test, explicitly simulated, never a live TOP claim."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
sys.path.insert(0,'/sdcard/transport_mvp')
import image,json
from k230.visual_inspector import VisualInspector
from ports.protocol import encode_observation,StreamParser

ROOT='/sdcard/transport_mvp'
layout=json.load(open(ROOT+'/fixtures/synthetic-top-layout.json'))
if layout.get('is_simulated') is not True:raise ValueError('Expected a simulated fixture')
profile={'width':320,'height':240,'crop':False,'hmirror':False,'vflip':False,'rotation':0,
         'pose_id':'GENERATED_TOP_FILE_ONLY'}
thresholds={}
img=image.Image(ROOT+'/fixtures/synthetic-top-faces.png').to_rgb565()
for key,center in layout['centers'].items():
    x,y=center
    stat=img.get_statistics(roi=(x-8,y-8,16,16))
    l,a,b=stat.l_median(),stat.a_median(),stat.b_median()
    thresholds[int(key)]=(max(0,l-6),min(100,l+6),max(-128,a-10),min(127,a+10),
                          max(-128,b-10),min(127,b+10))
pixel_profile={'schema_version':1,'provenance':{'source':'SYNTHETIC_TOP_FILE_ONLY'},
    'TOP_CONFIG':{'BUFFER':{'camera_id':2,'image_profile':profile,'thresholds':thresholds,
        'roi':(1,1,318,238),'reference':(160,120),'min_area':800,'max_area':6000,
        'surface_shape':{'min_axis_ratio':0.8,'max_axis_ratio':1.25,'min_density':0.65}}}}
inspector=VisualInspector('TOP',1,'BUFFER',pixel_profile,object_camera_id=2,front_camera_id=1)
results=[]
frame_id=0
for color in range(1,7):
    inspector.apply({'seq':color,'mode':'TOP','target_id':color,'station':'BUFFER'})
    for _ in range(3):
        frame_id+=1
        obs=inspector.process({'image':img,'camera_id':2,'frame_id':frame_id,
            'timestamp_ms':frame_id*40,'is_simulated':True,'width':320,'height':240,
            'image_profile':dict(profile)})
    expected=layout['centers'][str(color)]
    error=None if obs['u_px'] is None else ((obs['u_px']-expected[0])**2+(obs['v_px']-expected[1])**2)**0.5
    wire=StreamParser().feed(encode_observation(color,obs))
    results.append({'color_id':color,'observation':obs,'center_error_px':error,'wire_decoded':len(wire)==1,
                    'passed':error is not None and error<=1 and obs['stable'] and obs['x_mm'] is None
                    and obs['y_mm'] is None and not obs['target_pose_valid'] and obs['is_simulated']})
# A camera/pose profile mismatch must not reuse even a valid fixture center.
frame_id+=1
mismatch=inspector.process({'image':img,'camera_id':2,'frame_id':frame_id,'timestamp_ms':frame_id*40,
    'is_simulated':True,'image_profile':dict(profile,pose_id='WRONG_POSE')})
report={'test':'NATIVE_TOP_ROUND_FACE_FIXTURE','source':'GENERATED_FILE_FIXTURE_NOT_CAMERA',
    'is_simulated':True,'thresholds_sampled_from_same_fixture':True,'fixture_layout':layout,
    'results':results,'passed':sum(r['passed'] for r in results),'expected':6,
    'mismatch_rejected':mismatch['reason']=='TOP_PROFILE_BINDING_MISMATCH' and mismatch['u_px'] is None,
    'motors_enabled':False,'live_TOP_acceptance':'PENDING_REAL_TOP_FACE_SAMPLES'}
with open(ROOT+'/top_fixture_check.json','w') as stream:stream.write(json.dumps(report))
print('TOP_FIXTURE_CHECK:'+json.dumps(report))
