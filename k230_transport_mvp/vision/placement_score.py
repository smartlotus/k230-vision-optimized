"""Separate commanded placement slots (1..3) from scoring zones (1..6).

Rule points evaluate supplied evidence; they do not authorize motion.
Pixel alignment never determines a competition scoring zone.
"""
from .common import finite

RING_POINTS = {1: 15, 2: 10, 3: 7, 4: 5, 5: 3, 6: 1}


def _slot(value):
    return type(value) is int and 1 <= value <= 3


def score_ring_placement(target_slot, observed_slot, *, scoring_zone=None,
                         placement_state, stable, tilted=False, calibrated=False):
    """Score a completed observation with independently supplied slot and zone."""
    if not _slot(target_slot):raise ValueError('target_slot must be an integer in 1..3')
    if observed_slot is not None and not _slot(observed_slot):
        raise ValueError('observed_slot must be None or an integer in 1..3')
    if scoring_zone is not None and (type(scoring_zone) is not int or scoring_zone not in RING_POINTS):
        raise ValueError('scoring_zone must be None or an integer in 1..6')
    if any(type(v) is not bool for v in (stable,tilted,calibrated)):
        raise ValueError('stable, tilted and calibrated must be booleans')
    if placement_state not in ('IN_RING','OUTSIDE','FELL','UNKNOWN'):
        raise ValueError('invalid placement_state')
    result={'target_slot':target_slot,'observed_slot':observed_slot,'scoring_zone':scoring_zone,
        'placement_state':placement_state,'stable':stable,'tilted':tilted,'calibrated':calibrated,
        'official_points':None,'status':'UNKNOWN','reason':'PLACEMENT_OBSERVATION_REQUIRED'}
    if placement_state=='UNKNOWN':return result
    if tilted or placement_state in ('OUTSIDE','FELL'):
        result.update(official_points=0,status='SCORED',reason='OUTSIDE_FELL_OR_TILTED')
        return result
    if not stable:
        result['reason']='PLACEMENT_NOT_STABLE';return result
    if not calibrated:
        result['reason']='PLACEMENT_CALIBRATION_REQUIRED';return result
    if observed_slot is None:
        result['reason']='OBSERVED_SLOT_REQUIRED';return result
    if observed_slot!=target_slot:
        result.update(official_points=0,status='SCORED',reason='WRONG_SLOT');return result
    if scoring_zone is None:
        result['reason']='SCORING_ZONE_REQUIRED';return result
    result.update(official_points=RING_POINTS[scoring_zone],status='SCORED',reason='MATCHING_SLOT_AND_ZONE')
    return result


def alignment_quality(center_error_px, tolerance_px, *, stable, tilted=False):
    """Pixel-only diagnostic quality, independent of competition rule points."""
    if type(stable) is not bool or type(tilted) is not bool:
        raise ValueError('stable and tilted must be booleans')
    if center_error_px is None or tolerance_px is None:
        return {'status':'UNKNOWN','quality':None,'reason':'PIXEL_ERROR_REQUIRED'}
    if not finite(center_error_px) or not finite(tolerance_px) or center_error_px<0 or tolerance_px<=0:
        raise ValueError('finite non-negative error and positive tolerance required')
    if not stable or tilted:
        return {'status':'REJECTED','quality':0,'reason':'UNSTABLE_OR_TILTED'}
    return {'status':'MEASURED','quality':max(0.0,min(100.0,100*(1-center_error_px/tolerance_px))),
            'reason':'PIXEL_ALIGNMENT_ONLY'}


def score_placement_receipt(target_slot, receipt, *, calibrated=False):
    """Require post-placement score evidence, never an echoed command slot.

    DONE and placement_verified establish completion, not the measured location
    or scoring zone. Those need separate observed_slot, scoring_zone and
    score_observation_verified fields.
    """
    if not isinstance(receipt,dict):raise ValueError('receipt must be a mapping')
    unknown=lambda reason:dict(score_ring_placement(target_slot,None,placement_state='UNKNOWN',
        stable=False,calibrated=calibrated),reason=reason,is_simulated=receipt.get('is_simulated'))
    if (receipt.get('status')!='DONE' or receipt.get('action')!='PLACE'
            or receipt.get('placement_verified') is not True):
        return unknown('VERIFIED_PLACE_DONE_REQUIRED')
    if (receipt.get('score_observation_verified') is not True
            or type(receipt.get('is_simulated')) is not bool):
        return unknown('SCORING_OBSERVATION_REQUIRED')
    if type(receipt.get('stable')) is not bool or type(receipt.get('tilted')) is not bool:
        return unknown('EXPLICIT_PLACEMENT_STATE_REQUIRED')
    result=score_ring_placement(target_slot,receipt.get('observed_slot'),
        scoring_zone=receipt.get('scoring_zone'),placement_state=receipt.get('placement_state','UNKNOWN'),
        stable=receipt['stable'],tilted=receipt['tilted'],calibrated=calibrated)
    result['is_simulated']=receipt['is_simulated']
    return result
