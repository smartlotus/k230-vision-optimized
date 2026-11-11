"""Five-mode pixel inspection with reversible file controls and no motion API."""
import json
from k230.block_profile import make_config
from k230.vision_adapter import VisionAdapter


class VisualInspector:
    MODES=('COLOR','QR','RING','TOP','LANE','IDLE')

    def __init__(self,mode='COLOR',target_id=1,station='RAW',pixel_profile=None,object_camera_id=None,front_camera_id=None):
        self.config=make_config(object_camera_id,front_camera_id)
        self.config['CALIBRATIONS']={}
        profile={'width':320,'height':240,'crop':False,'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}
        # Algorithm-only diagnostic defaults: IDs stay unknown without a bound reference.
        self.config['RING_CONFIG']={site:{'roi':(0,0,320,240),'camera_id':self.config['MODE_CAMERA']['RING'],'image_profile':dict(profile),
            'verified':False,'parameters':{'threshold':2000,'r_min':6,'r_max':55,'r_step':3}}
            for site in ('PROCESS','BUFFER')}
        self.profile_error=None
        if pixel_profile is not None:self.load_profile(pixel_profile)
        self.adapter=VisionAdapter(self.config)
        self.seq=-1
        self.mode='COLOR';self.target_id=1;self.station='RAW';self.last_error=None
        if not self.apply({'seq':0,'mode':mode,'target_id':target_id,'station':station}):
            raise ValueError(self.last_error)

    def load_profile(self,profile):
        if not isinstance(profile,dict) or profile.get('schema_version')!=1:
            raise ValueError('Pixel profile schema_version=1 required')
        allowed={'schema_version','provenance','TOP_CONFIG','RING_CONFIG','LANE_ROI','LANE_THRESHOLD',
                 'LANE_MIN_WIDTH_PX','LANE_MAX_WIDTH_PX','LANE_MAX_CENTER_SPREAD_PX'}
        if set(profile)-allowed:raise ValueError('Pixel inspector cannot import H, motion, UART or camera changes')
        changes={}
        for key,value in profile.items():
            if key in ('schema_version','provenance'):continue
            copied=json.loads(json.dumps(value))
            if key in ('TOP_CONFIG','RING_CONFIG'):
                if not isinstance(copied,dict):raise ValueError('Per-station profile required')
                for cfg in copied.values():
                    if not isinstance(cfg,dict):raise ValueError('Station profile must be an object')
                    if key=='TOP_CONFIG':
                        permitted={'camera_id','image_profile','thresholds','roi','reference','min_area',
                                   'max_area','surface_shape','profile_id','provenance',
                                   'circle_refine_ids','circle_parameters','circle_center_tolerance_px','circle_boundary'}
                        if set(cfg)-permitted or cfg.get('surface_shape') is None:
                            raise ValueError('TOP requires a separate bound surface-shape profile')
                        from vision.top import validate_circle_config
                        validate_circle_config(cfg)
                    if 'thresholds' in cfg:cfg['thresholds']={int(k):v for k,v in cfg['thresholds'].items()}
                    if 'id_rois' in cfg:cfg['id_rois']={int(k):v for k,v in cfg['id_rois'].items()}
                    if 'center_gates' in cfg:cfg['center_gates']={int(k):v for k,v in cfg['center_gates'].items()}
            changes[key]=copied
        self.config.update(changes)
        self.config['CALIBRATIONS']={}

    def apply(self,command):
        try:
            if not isinstance(command,dict):raise ValueError('Control must be an object')
            if set(command)-{'seq','mode','target_id','station'}:raise ValueError('Unsupported inspection control')
            seq=command.get('seq')
            if type(seq) is not int or not 0<=seq<=65535:raise ValueError('Invalid inspection sequence')
            if seq<=self.seq:return False
            mode=command.get('mode')
            if mode not in self.MODES:raise ValueError('Unknown inspection mode')
            target=command.get('target_id');station=command.get('station')
            if mode in ('COLOR','TOP'):
                if type(target) is not int or target not in range(1,7):raise ValueError('Color target 1..6 required')
                if station not in (('BUFFER',) if mode=='TOP' else ('RAW','PROCESS','BUFFER')):
                    raise ValueError('Invalid color/top station')
            elif mode=='RING':
                if type(target) is not int or target not in (1,2,3):raise ValueError('Ring target 1..3 required')
                if station not in ('PROCESS','BUFFER'):raise ValueError('Ring station required')
            else:
                if target is not None:raise ValueError('This mode has no target ID')
                if station not in (None,'QR'):raise ValueError('Invalid QR/lane station')
            self.mode,self.target_id,self.station,self.seq=mode,target,station,seq
            self.last_error=None
            if hasattr(self,'adapter'):
                self.capture_failed()
            return True
        except (ValueError,TypeError) as error:
            self.last_error=str(error)
            return False

    def poll_file(self,path):
        try:
            with open(path) as stream:data=stream.read(1025)
        except OSError:return False
        if len(data)>1024:
            self.last_error='Inspection control exceeds 1024 bytes';return False
        try:return self.apply(json.loads(data))
        except (ValueError,TypeError) as error:
            self.last_error=str(error);return False

    def camera_id(self):
        return self.config['MODE_CAMERA'].get(self.mode,self.config['MODE_CAMERA']['COLOR'])

    def suspend(self,reason):
        self.mode='IDLE';self.target_id=None;self.station=None;self.last_error=reason
        self.capture_failed()

    def capture_failed(self):
        self.adapter.tracker.reset();self.adapter.last_block_candidates=[]
        self.adapter.qr.reset_scan();self.adapter.context=None;self.adapter.blocks={}

    def process(self,frame):
        if self.mode=='IDLE':return None
        result=self.adapter.detect(self.mode,frame,self.target_id,self.station)
        if self.mode=='QR' and result.get('reason')=='QR_TIMEOUT':
            # A scan window expires; the session deduplication remains in force.
            self.adapter.qr.reset_scan()
        return result

    def status(self):
        return {'mode':self.mode,'camera_id':self.camera_id(),'target_id':self.target_id,
                'station':self.station,'control_seq':self.seq,'control_error':self.last_error,
                'profile_error':self.profile_error,
                'millimeter_calibration_loaded':False,'motors_enabled':False}
