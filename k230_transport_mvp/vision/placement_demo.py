"""Opt-in visual pick/place demonstration; it never issues physical actions."""
from math import sqrt
from .common import finite,elapsed_ms
from .tracker import StableTracker
from k230.camera_roles import OBJECT_CAMERA_ID


def image_footprint_contained(material,target,margin_px=2):
    """Conservative containment in image pixels, not a calibrated plane claim."""
    try:
        if target.get('proposal_only') or any(f.get('proposal_only') or f.get('boundary_complete_verified') is False for f in target['boundaries']):
            return False
        boundary=target['boundaries'][-1]
        center,axes=material['center'],material['semiaxes']
        paper_center,paper_axes=boundary['center'],boundary['semiaxes']
        if any(len(v)!=2 or not all(finite(x) for x in v) for v in (center,axes,paper_center,paper_axes)):
            return False
        if min(axes)<=0 or min(paper_axes)<=0 or not finite(margin_px) or margin_px<0:return False
        extent=sqrt((center[0]-paper_center[0])**2+(center[1]-paper_center[1])**2)+max(axes)+margin_px
        return finite(extent) and extent<min(paper_axes)
    except (KeyError,IndexError,TypeError):
        return False


class VisualPlacementDemo:
    """Real detections drive simulated events only, latched once per reset.

    Three fresh observations confirm a simulated grab. Three fresh, stationary
    material observations wholly within one independent paper target confirm a
    simulated placement. Disappearance alone never confirms a physical grab.
    """
    def __init__(self,confirmations=3,max_motion_px=8,max_gap_ms=1000):
        if type(confirmations) is not int or not 2<=confirmations<=20:
            raise ValueError('bounded confirmation count required')
        self.confirmations=confirmations;self.max_motion_px=max_motion_px;self.max_gap_ms=max_gap_ms
        self.reset()

    def reset(self):
        self.state='WAIT_MATERIAL';self.reason='WAIT_FOR_VISIBLE_MATERIAL'
        self.simulated_picks=0;self.simulated_places=0
        self.last_frame=None;self.last_timestamp=None
        self.pick_track=StableTracker(self.confirmations,self.max_motion_px,self.max_gap_ms)
        self.place_track=StableTracker(self.confirmations,self.max_motion_px,self.max_gap_ms)

    def snapshot(self):
        return {'mode':'VISUAL_SIMULATION','is_simulated':True,'state':self.state,
            'reason':self.reason,'simulated_picks':self.simulated_picks,
            'simulated_places':self.simulated_places,'confirmation_frames':self.confirmations,
            'physical_grab_verified':False,'physical_placement_verified':False,
            'motion_authorized':False,'official_points':None}

    def update(self,packet,materials,paper_targets,now_ms):
        events=[];frame=packet.get('frame_id');timestamp=packet.get('timestamp_ms')
        if (type(frame) is not int or frame<1 or type(timestamp) is not int
                or type(now_ms) is not int or type(packet.get('is_simulated')) is not bool
                or packet.get('camera_id')!=OBJECT_CAMERA_ID
                or not 0<=elapsed_ms(now_ms,timestamp)<=self.max_gap_ms):
            self.pick_track.reset();self.place_track.reset();self.reason='SOURCE_NOT_FRESH'
            return dict(self.snapshot(),events=events)
        if self.last_frame is not None and (frame<=self.last_frame or elapsed_ms(timestamp,self.last_timestamp)<=0):
            self.reason='REPLAYED_OR_OLD_FRAME'
            return dict(self.snapshot(),events=events)
        self.last_frame,self.last_timestamp=frame,timestamp
        if self.state=='SIM_PLACED':return dict(self.snapshot(),events=events)
        if len(materials)!=1:
            self.pick_track.reset();self.place_track.reset();self.reason='UNIQUE_MATERIAL_REQUIRED'
            return dict(self.snapshot(),events=events)
        material=materials[0];center=material.get('center')
        if self.state=='WAIT_MATERIAL':
            self.reason='CONFIRMING_VISIBLE_MATERIAL'
            if self.pick_track.update('SIM_PICK',frame,timestamp,center):
                self.state='SIM_HOLDING';self.reason='SIMULATED_GRAB_ONLY';self.simulated_picks=1
                events.append({'event':'SIM_PICK','is_simulated':True,'source_frame_id':frame,
                    'observation_is_simulated':packet['is_simulated'],'physical_action':False})
            return dict(self.snapshot(),events=events)
        targets=[t for t in paper_targets if image_footprint_contained(material,t)]
        if len(targets)!=1:
            self.place_track.reset();self.reason='UNIQUE_FULLY_CONTAINING_PAPER_TARGET_REQUIRED'
            return dict(self.snapshot(),events=events)
        self.reason='CONFIRMING_IMAGE_CONTAINMENT'
        if self.place_track.update('SIM_PLACE',frame,timestamp,center):
            self.state='SIM_PLACED';self.reason='SIMULATED_PLACEMENT_ONLY';self.simulated_places=1
            events.append({'event':'SIM_PLACE','is_simulated':True,'source_frame_id':frame,
                'observation_is_simulated':packet['is_simulated'],'physical_action':False,
                'containment_basis':'CONSERVATIVE_IMAGE_FOOTPRINT','target_identity_verified':False})
        return dict(self.snapshot(),events=events)
