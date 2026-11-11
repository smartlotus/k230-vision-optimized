"""Strict manual profiles and a single export into existing vision algorithms."""
import json
from vision.common import finite
from k230.camera_roles import MODE_CAMERA

CONTEXTS=(('COLOR','RAW'),('COLOR','PROCESS'),('COLOR','BUFFER'),('TOP','BUFFER'),
          ('RING','PROCESS'),('RING','BUFFER'),('QR','FRONT'),('LANE','FRONT'))
SIZES=((640,360),(320,180),(1280,720),(1920,1080))


def clone(value):
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if isinstance(value,list):return [clone(v) for v in value]
    if isinstance(value,tuple):return tuple(clone(v) for v in value)
    return value
def context_key(mode,station):return mode+'_'+station


def defaults():
    contexts={}
    for mode,station in CONTEXTS:
        contexts[context_key(mode,station)]={'roi':[0,0,640,360],'reference':[320,180],
            'thresholds':{str(c):None for c in range(1,7)},'min_area':120,'max_area':60000,
            'min_pixels':100,'min_density':0.5,'min_aspect':0.25,'max_aspect':1.5,'edge_margin':2,
            'r_min':6,'r_max':70,'circle_threshold':2000,'r_step':2,'center_tolerance':8,
            'gray_min':60,'gray_max':180,'lane_min_width':30,'lane_max_width':400,
            'scan_rows':3,'sample_step':2,'center_spread':20,
            'id_rois':{'1':None,'2':None,'3':None},'ring_id_reviewed':False,
            'sampling_status':'UNSET'}
    return {'schema_version':1,'profile_id':'manual-panel-v1','revision':0,
        'binding':{'width':640,'height':360,'input_width':1920,'input_height':1080,
            'crop':False,'hmirror':False,'vflip':False,'rotation':0,'pose_id':None,'pixel_format':'RGB565'},
        'contexts':contexts,'stability':{'frames':3,'motion_px':4,'gap_ms':300},
        'qr':{'confirmations':3,'timeout_ms':5000,'gap_ms':500},
        'camera':{'fps':30,'auto_exposure':True,'exposure_us':10000,'auto_white_balance':True},
        'display':{'mode':'NATIVE','pan_x':0,'pan_y':0},'locked':False,
        'metadata':{'source':'MANUAL_UNVERIFIED','geometry_verified':False,'color_verified':False}}


def number(value,low,high,label,integer=False):
    if not finite(value) or (integer and type(value) is not int) or not low<=value<=high:
        raise ValueError(label+' outside range')


def check_roi(roi,width,height):
    if not isinstance(roi,(list,tuple)) or len(roi)!=4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI must contain four integers')
    x,y,w,h=roi
    if x<0 or y<0 or w<8 or h<8 or x+w>width or y+h>height:raise ValueError('ROI outside image')


def validate(profile):
    template=defaults()
    if not isinstance(profile,dict) or set(profile)!=set(template) or profile['schema_version']!=1:
        raise ValueError('Manual profile schema mismatch')
    if not isinstance(profile['profile_id'],str) or not 1<=len(profile['profile_id'])<=40:raise ValueError('Invalid profile ID')
    number(profile['revision'],0,2147483647,'revision',True)
    b=profile['binding']
    if not isinstance(b,dict) or set(b)!=set(template['binding']):raise ValueError('Binding schema mismatch')
    width,height=b['width'],b['height']
    if (width,height) not in SIZES or type(width) is not int or type(height) is not int:raise ValueError('Only16:9 image modes supported')
    if (any(type(b[k]) is not int for k in ('input_width','input_height'))
            or (b['input_width'],b['input_height'])!=(1920,1080) or b['pixel_format']!='RGB565'):raise ValueError('Unsupported input mode')
    if b['crop'] is not False or type(b['rotation']) is not int or b['rotation']!=0:raise ValueError('Explicit unsupported geometry change')
    if any(type(b[k]) is not bool for k in ('hmirror','vflip')):raise ValueError('Invalid mirror flag')
    if b['pose_id'] is not None and (not isinstance(b['pose_id'],str) or not 1<=len(b['pose_id'])<=24):raise ValueError('Invalid pose ID')
    if type(profile['locked']) is not bool:raise ValueError('Invalid lock state')
    if not isinstance(profile['contexts'],dict) or set(profile['contexts'])!=set(template['contexts']):raise ValueError('Context mismatch')
    for key,c in profile['contexts'].items():
        if not isinstance(c,dict) or set(c)!=set(template['contexts'][key]):raise ValueError('Context fields mismatch')
        check_roi(c['roi'],width,height)
        if not isinstance(c['reference'],(list,tuple)) or len(c['reference'])!=2:raise ValueError('Reference required')
        number(c['reference'][0],0,width-1,'reference x');number(c['reference'][1],0,height-1,'reference y')
        if not isinstance(c['thresholds'],dict) or set(c['thresholds'])!={'1','2','3','4','5','6'}:raise ValueError('Fixed six color IDs required')
        for threshold in c['thresholds'].values():
            if threshold is None:continue
            if not isinstance(threshold,(tuple,list)) or len(threshold)!=6:raise ValueError('Six LAB bounds required')
            for axis,(low,high) in enumerate(((0,100),(-128,127),(-128,127))):
                a,z=threshold[2*axis:2*axis+2]
                number(a,low,high,'LAB',True);number(z,low,high,'LAB',True)
                if a>z:raise ValueError('LAB lower exceeds upper')
        for name,low,high in (('min_area',1,width*height),('max_area',1,width*height),
                ('min_pixels',1,width*height),('edge_margin',0,32),('r_min',1,180),('r_max',1,180),
                ('circle_threshold',1,100000),('r_step',1,20),('center_tolerance',1,50),
                ('gray_min',0,255),('gray_max',0,255),('lane_min_width',1,width),('lane_max_width',1,width),
                ('scan_rows',3,16),('sample_step',1,8),('center_spread',1,100)):
            number(c[name],low,high,name,True)
        for name,low,high in (('min_density',0.01,1),('min_aspect',0.05,10),('max_aspect',0.05,10)):
            number(c[name],low,high,name)
        if any(c[a]>c[z] for a,z in (('min_area','max_area'),('min_aspect','max_aspect'),
                ('r_min','r_max'),('gray_min','gray_max'),('lane_min_width','lane_max_width'))):raise ValueError('Lower exceeds upper')
        if type(c['ring_id_reviewed']) is not bool:raise ValueError('Invalid ring review')
        if c['sampling_status'] not in ('UNSET','MANUAL_UNVERIFIED'):raise ValueError('UI cannot assert hardware acceptance')
        if not isinstance(c['id_rois'],dict) or set(c['id_rois'])!={'1','2','3'}:raise ValueError('Fixed ring IDs required')
        rois=[]
        for roi in c['id_rois'].values():
            if roi is not None:check_roi(roi,width,height);rois.append(roi)
        if c['ring_id_reviewed']:
            if len(rois)!=3 or b['pose_id'] is None:raise ValueError('Three numbered ROIs and fixed pose required')
            for i,a in enumerate(rois):
                for z in rois[i+1:]:
                    if max(a[0],z[0])<min(a[0]+a[2],z[0]+z[2]) and max(a[1],z[1])<min(a[1]+a[3],z[1]+z[3]):raise ValueError('Numbered ring regions overlap')
    for group in ('stability','qr','camera','display','metadata'):
        if not isinstance(profile[group],dict) or set(profile[group])!=set(template[group]):raise ValueError(group+' schema mismatch')
    s=profile['stability'];q=profile['qr'];cam=profile['camera'];d=profile['display']
    for value,low,high,name in ((s['frames'],3,20,'stable frames'),(s['motion_px'],1,30,'motion'),
            (s['gap_ms'],50,1000,'stable gap'),(q['confirmations'],3,10,'QR confirmations'),
            (q['timeout_ms'],500,20000,'QR timeout'),(q['gap_ms'],50,1000,'QR gap'),
            (cam['exposure_us'],100,33333,'exposure'),(d['pan_x'],0,max(0,width-1),'pan x'),(d['pan_y'],0,max(0,height-1),'pan y')):
        number(value,low,high,name,True)
    if type(cam['fps']) is not int or cam['fps'] not in (30,60):raise ValueError('FPS30/60 only')
    if cam['exposure_us']>1000000//cam['fps'] and not cam['auto_exposure']:raise ValueError('Exposure exceeds frame period')
    if any(type(cam[k]) is not bool for k in ('auto_exposure','auto_white_balance')):raise ValueError('Invalid camera flag')
    if cam['auto_white_balance'] is not True:raise ValueError('Firmware manual white balance unavailable')
    if d['mode'] not in ('NATIVE','FIT'):raise ValueError('Invalid display mode')
    if profile['metadata']!={'source':'MANUAL_UNVERIFIED','geometry_verified':False,'color_verified':False}:
        raise ValueError('Tuning cannot create calibration or scientific acceptance')
    return profile


def image_profile(profile):
    b=profile['binding']
    return {k:b[k] for k in ('width','height','crop','hmirror','vflip','rotation','pose_id')}


def compile_config(profile):
    """Deep, independent stations; same adapter, no alternate detector or motion path."""
    validate(profile)
    from k230.block_profile import make_config
    cfg=clone(make_config());b=profile['binding'];s=profile['stability'];q=profile['qr']
    cfg.update(WIDTH=b['width'],HEIGHT=b['height'],SENSOR_FPS=profile['camera']['fps'],
        CROP=b['crop'],HMIRROR=b['hmirror'],VFLIP=b['vflip'],POSE_ID=b['pose_id'],
        STABLE_FRAMES=s['frames'],STABLE_MAX_MOTION_PX=s['motion_px'],STABLE_MAX_GAP_MS=s['gap_ms'],
        QR_CONFIRMATIONS=q['confirmations'],QR_TIMEOUT_MS=q['timeout_ms'],QR_MAX_GAP_MS=q['gap_ms'],
        CALIBRATIONS={},UART_ENABLED=False,WORKSPACE_MM=None)
    cfg['COLOR_CONFIG']={};cfg['TOP_CONFIG']={};cfg['RING_CONFIG']={}
    for mode,station in CONTEXTS:
        c=clone(profile['contexts'][context_key(mode,station)])
        if mode=='COLOR':
            parameters={'profile_id':profile['profile_id']+'-r'+str(profile['revision'])+'-'+station,
                'camera_id':MODE_CAMERA[mode],'width':b['width'],'height':b['height'],
                'thresholds':{int(k):v for k,v in c['thresholds'].items()},'roi':c['roi'],
                'min_pixels':c['min_pixels'],'min_area':c['min_area'],'max_area':c['max_area'],
                'min_density':c['min_density'],'min_aspect':c['min_aspect'],'max_aspect':c['max_aspect'],
                'edge_margin':c['edge_margin'],'max_candidates':12,'stable_frames':s['frames'],
                'stable_motion_px':s['motion_px'],'track_distance_px':30,'max_gap_ms':s['gap_ms'],
                'sampling_verified':False,'measurement_camera_id':None}
            cfg['COLOR_CONFIG'][station]={'block_parameters':parameters,'roi':c['roi'],
                'reference':c['reference'],'min_area':c['min_area'],'max_area':c['max_area']}
        elif mode=='TOP':
            cfg['TOP_CONFIG'][station]={'camera_id':MODE_CAMERA[mode],'image_profile':image_profile(profile),
                'thresholds':{int(k):v for k,v in c['thresholds'].items()},'roi':c['roi'],
                'reference':c['reference'],'min_area':c['min_area'],'max_area':c['max_area'],
                'surface_shape':{'min_axis_ratio':c['min_aspect'],'max_axis_ratio':c['max_aspect'],'min_density':c['min_density']}}
        elif mode=='RING':
            cfg['RING_CONFIG'][station]={'camera_id':MODE_CAMERA[mode],'image_profile':image_profile(profile),
                'roi':c['roi'],'verified':c['ring_id_reviewed'],
                'center_tolerance_px':c['center_tolerance'],
                'id_rois':{int(k):v for k,v in c['id_rois'].items() if v is not None},
                'parameters':{'threshold':c['circle_threshold'],'r_min':c['r_min'],'r_max':c['r_max'],'r_step':c['r_step']}}
            cfg['RING_CENTER_TOLERANCE_PX']=c['center_tolerance']
        elif mode=='QR':cfg['QR_ROI']=c['roi']
        else:
            cfg.update(LANE_ROI=c['roi'],LANE_THRESHOLD=(c['gray_min'],c['gray_max']),
                LANE_MIN_WIDTH_PX=c['lane_min_width'],LANE_MAX_WIDTH_PX=c['lane_max_width'],
                LANE_SCAN_ROWS=c['scan_rows'],LANE_SAMPLE_STEP=c['sample_step'],LANE_MAX_CENTER_SPREAD_PX=c['center_spread'])
    return cfg
