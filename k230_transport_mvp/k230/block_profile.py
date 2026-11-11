"""Measured CSI2 side-view bench profile; no metric or motion calibration."""
PROFILE_ID = 'csi2-block-lab-20261101-v10'
CAMERA_ID = 2
WIDTH, HEIGHT = 320, 240
COLOR_NAMES = {1:'RED',2:'YELLOW',3:'BLUE',4:'GREEN',5:'BLACK',6:'CYAN'}
DRAW_COLORS = {1:(255,60,60),2:(255,255,0),3:(70,110,255),4:(0,255,80),5:(255,255,255),6:(0,255,255)}
# Native GC2093 RGB565 LAB, 5 successive frames, hand-selected interior ROIs.
# All six colors were sampled on the real CSI2; only this bench lighting is covered.
THRESHOLDS = {1:(5,95,24,65,7,55),2:(15,100,-35,5,35,100),
              3:(3,90,9,55,-85,-30),4:(5,95,-65,-19,1,50),
              5:(0,26,-12,12,-12,12),6:(10,95,-30,6,-55,-24)}
# Distance changes AE/AWB. Chroma ranges originate from live samples; wider L
# bounds are an algorithm choice, not an additional measured lighting domain.
PARAMETERS = {'profile_id':PROFILE_ID,'camera_id':CAMERA_ID,'width':WIDTH,'height':HEIGHT,
              # Measured image band containing the current side-view bench objects.
              # This is an observation window, not a verified robotic grasp workspace.
              'thresholds':THRESHOLDS,'roi':(0,35,WIDTH,150),'min_pixels':250,
              'min_area':400,'max_area':24000,'min_density':0.60,
              'min_aspect':0.25,'max_aspect':1.3,'edge_margin':2,'max_candidates':12,
              # Upright side-view bench prior; TOP uses a separate unsampled profile.
              # Current bench background shadow: 397-423 pixels; real black body >2100.
              # This size floor is specific to this viewing distance, not a metric estimate.
              'class_filters':{5:{'min_pixels':600,'min_density':0.62,'min_aspect':0.4,'max_aspect':1.15}},
              'stable_frames':3,'stable_motion_px':4,'track_distance_px':30,'max_gap_ms':250}
NOMINAL_DIMENSIONS_MM = {'top_diameter':30,'bottom_diameter':50,'height':60,'end_height':8}
DIMENSION_SOURCE = 'things/mmexport1793457770940.jpg; nominal drawing, not measured calibration'
MEASUREMENT_SOURCE = 'evidence/board/block-lab-six-v2-probe.json'
LIVE_UNVERIFIED_COLORS = ()  # Six-color sampling complete; robustness acceptance tracked separately.


def parameters_for_camera(camera_id):
    """Retain measured CSI2 provenance when providing provisional CSI1 pixels."""
    if type(camera_id) is not int or camera_id not in (1,2):raise ValueError('Only healthy CSI1/2 roles are supported')
    result=dict(PARAMETERS)
    result['camera_id']=camera_id
    result['measurement_camera_id']=CAMERA_ID
    result['sampling_verified']=camera_id==CAMERA_ID
    if camera_id!=CAMERA_ID:
        result['profile_id']='csi1-provisional-from-csi2-v10'
        result['roi']=(0,0,WIDTH,HEIGHT)
    return result


def make_config(object_camera_id=None,front_camera_id=None):
    """Opt-in dual-camera observation configuration with measured color profile."""
    from . import config as defaults
    from .camera_roles import OBJECT_CAMERA_ID,FRONT_CAMERA_ID
    object_camera_id=OBJECT_CAMERA_ID if object_camera_id is None else object_camera_id
    front_camera_id=FRONT_CAMERA_ID if front_camera_id is None else front_camera_id
    if type(object_camera_id) is not int or type(front_camera_id) is not int or {object_camera_id,front_camera_id}!={1,2}:
        raise ValueError('Distinct healthy CSI1/2 roles required')
    parameters=parameters_for_camera(object_camera_id)
    result={name:getattr(defaults,name) for name in dir(defaults) if name.isupper()}
    result.update(CAMERA_IDS=(object_camera_id,front_camera_id),CAMERA_PRIORITY=(object_camera_id,front_camera_id,0),
        MODE_CAMERA={'QR':front_camera_id,'LANE':front_camera_id,'COLOR':object_camera_id,'RING':object_camera_id,'TOP':object_camera_id},
        DESKTOP_SINGLE_CAMERA=False,WIDTH=WIDTH,HEIGHT=HEIGHT,POSE_ID=None,
        DEFAULT_MODE='COLOR',DEFAULT_TARGET_ID=1,DEFAULT_STATION='RAW',LOOP_YIELD_MS=0,
        COLOR_CONFIG={station:{'block_parameters':dict(parameters),'roi':parameters['roi'],
            'reference':(WIDTH/2,HEIGHT/2),'min_area':parameters['min_area'],
            'max_area':parameters['max_area']} for station in ('RAW','PROCESS','BUFFER')})
    return result
