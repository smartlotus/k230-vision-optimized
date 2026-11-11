"""Same-frame pixel alignment diagnostics, without metric or motion authority."""
from math import sqrt
from .common import finite,elapsed_ms
from .placement_score import alignment_quality,score_placement_receipt
from k230.camera_roles import OBJECT_CAMERA_ID


def measure_alignment(material, target, *, now_ms, tolerance_px=10, max_age_ms=250):
    """Compare explicitly selected observations; never choose a target by distance."""
    if not finite(tolerance_px) or tolerance_px<=0:raise ValueError('positive finite tolerance required')
    if type(now_ms) is not int or type(max_age_ms) is not int or not 1<=max_age_ms<=5000:
        raise ValueError('integer clock and bounded max age required')
    result={'status':'UNKNOWN','reason':'OBSERVATIONS_REQUIRED','dx_px':None,'dy_px':None,
        'error_px':None,'quality':None,'coord_frame':'IMAGE_PX','x_mm':None,'y_mm':None,
        'target_pose_valid':False,'official_points':None}
    if not isinstance(material,dict) or not isinstance(target,dict):return result
    if any(v.get('status') not in ('OK','UNCALIBRATED','GEOMETRY_ONLY') for v in (material,target)):
        result['reason']='TARGET_GEOMETRY_REQUIRED';return result
    for value in (material,target):
        if (type(value.get('frame_id')) is not int or value['frame_id']<1
                or type(value.get('camera_id')) is not int
                or type(value.get('timestamp_ms')) is not int
                or type(value.get('is_simulated')) is not bool
                or not isinstance(value.get('image_profile'),dict)):
            result['reason']='FRAME_SOURCE_REQUIRED';return result
    if any(material[k]!=target[k] for k in ('camera_id','frame_id','timestamp_ms','image_profile','is_simulated')):
        result['reason']='SAME_FRAME_SOURCE_REQUIRED';return result
    if material['camera_id']!=OBJECT_CAMERA_ID:
        result['reason']='OBJECT_CAMERA_REQUIRED';return result
    if any(not 0<=elapsed_ms(now_ms,v['timestamp_ms'])<=max_age_ms for v in (material,target)):
        result['reason']='STALE_OBSERVATION';return result
    if any(not finite(v.get(k)) for v in (material,target) for k in ('u_px','v_px')):
        result['reason']='PIXEL_CENTER_REQUIRED';return result
    width,height=material['image_profile'].get('width'),material['image_profile'].get('height')
    if (type(width) is not int or type(height) is not int or width<=0 or height<=0
            or any(not (0<=v['u_px']<width and 0<=v['v_px']<height) for v in (material,target))):
        result['reason']='PIXEL_CENTER_OUTSIDE_IMAGE';return result
    dx,dy=material['u_px']-target['u_px'],material['v_px']-target['v_px']
    error=sqrt(dx*dx+dy*dy)
    result.update(dx_px=dx,dy_px=dy,error_px=error,frame_id=material['frame_id'],
        camera_id=material['camera_id'],is_simulated=material['is_simulated'])
    if any(v.get('stable') is not True for v in (material,target)):
        result['reason']='STABILITY_REQUIRED';return result
    quality=alignment_quality(error,tolerance_px,stable=True)
    result.update(status='MEASURED',reason='PIXEL_ALIGNMENT_ONLY',quality=quality['quality'])
    return result


def evaluate_placement(material, target, target_slot, receipt, *, now_ms,
                       tolerance_px=10, calibrated=False):
    """Compose independent pre-placement pixel quality and post-placement score."""
    return {'alignment':measure_alignment(material,target,now_ms=now_ms,tolerance_px=tolerance_px),
            'placement_score':score_placement_receipt(target_slot,receipt or {},calibrated=calibrated),
            'motion_authorized':False}
