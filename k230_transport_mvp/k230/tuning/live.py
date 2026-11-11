"""Single-owner native preview with checksummed, session-bound host commands."""
import os,json,time,gc,sys
from math import cos, sin, pi
from .schema import validate,clone,context_key,CONTEXTS,image_profile
from .store import ProfileStore,sha
from .model import TuningSession
from .runtime import TuningApp
from .material import scan_state,SourceClock,MaterialSnapshotProducer
from k230.camera_roles import MODE_CAMERA
from k230.block_profile import DRAW_COLORS
from .resources import snapshot as resource_snapshot
from vision.ellipse_native import extract_mask_ellipses, ellipse_outline
from vision.ring import cluster_circles
from vision.common import finite,observation
from vision.alignment import measure_alignment
from vision.tracker import StableTracker
from vision.placement_demo import VisualPlacementDemo

ROOT='/sdcard/transport_mvp/lv'
POLL_INTERVAL_MS=250
PREVIEW_EVERY_N_FRAMES=2
MAX_MATERIALS=3


def diagnostic_colors(single=None,multiple=None):
    if multiple is not None and single is not None:raise ValueError('Choose single or multiple diagnostic colors')
    colors=([single] if single is not None else [] if multiple is None else multiple)
    if (not isinstance(colors,(list,tuple)) or len(colors)>6
            or multiple is not None and not colors
            or any(type(v) is not int or not 1<=v<=6 for v in colors)
            or len(set(colors))!=len(colors)):
        raise ValueError('Distinct explicit diagnostic colors 1..6 required')
    return list(colors)


class LiveCommands:
    def __init__(self,session,token,mailbox=ROOT,diagnostic_material_color=None,visual_demo=False,diagnostic_material_colors=None):
        if not isinstance(token,str) or len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise ValueError('Session token required')
        self.session=session;self.token=token;self.mailbox=mailbox;self.last_seq=0;self.preview_sha=None
        if diagnostic_material_color is not None and (type(diagnostic_material_color) is not int or not 1<=diagnostic_material_color<=6):
            raise ValueError('Explicit material color 1..6 required')
        self.diagnostic_material_color=diagnostic_material_color
        self.diagnostic_material_colors=diagnostic_colors(diagnostic_material_color,diagnostic_material_colors)
        if type(visual_demo) is not bool:raise ValueError('visual_demo must be boolean')
        if visual_demo and diagnostic_material_color is None:raise ValueError('Demo requires explicit material color')
        self.visual_demo=VisualPlacementDemo() if visual_demo else None
        self.last_ack=None;self.render_mode='THRESHOLD';self.stopped=False;self.sample_requested=False;self.mailbox_polls=0
        ProfileStore(mailbox)._prepare()

    def state(self,include_profile=False):
        s=self.session;record=s.last_commit
        result={'session':self.token,'seq':self.last_seq,'profile_root':s.store.root,'service_version':1,'mode':s.mode,'station':s.station,'color':s.target_id,
            'slot':s.slot,'draft_revision':s.draft['revision'],'active_revision':s.active['revision'],
            'apply_serial':s.apply_serial,'dirty':s.dirty,'locked':s.draft['locked'],
            'preview_sha':self.preview_sha,'binding':image_profile(s.active),
            'diagnostic_material_color':self.diagnostic_material_color,
            'diagnostic_material_colors':self.diagnostic_material_colors,
            'visual_demo':self.visual_demo.snapshot() if self.visual_demo else None,
            'generation':record['generation'] if record else None,'saved_sha256':record['sha256'] if record else None,
            'saved_profile_revision':record['profile']['revision'] if record else None,
            'saved_equals_draft':record is not None and record['profile']==s.draft,
            'active_equals_draft':s.active==s.draft,'render_mode':self.render_mode,'storage_warnings':s.store.warnings}
        if include_profile:
            result['profile']=clone(s.draft)
            result['active_profile']=clone(s.active)
        return result

    def handle(self,envelope):
        if not isinstance(envelope,dict) or set(envelope)!={'session','seq','payload','sha256'}:raise ValueError('Command envelope mismatch')
        if envelope['session']!=self.token:raise ValueError('Stale session')
        seq=envelope['seq']
        if type(seq) is not int or seq<=self.last_seq or seq>2147483647:raise ValueError('Repeated/invalid command sequence')
        raw=envelope['payload']
        if not isinstance(raw,str) or len(raw)>131072 or sha(raw.encode())!=envelope['sha256']:raise ValueError('Command checksum mismatch')
        command=json.loads(raw)
        if not isinstance(command,dict) or command.get('action') not in ('status','preview','apply','save','stop','lock','view','sample','demo_reset'):raise ValueError('Unknown action')
        self.last_seq=seq;action=command['action'];s=self.session
        try:
            if action in ('preview','apply','save'):
                if set(command)!={'action','profile','context','color','slot'}:raise ValueError('Command fields mismatch')
                profile=validate(clone(command['profile']));ctx=command['context']
                if (profile['binding']['width'],profile['binding']['height']) not in ((320,180),(640,360)):raise ValueError('Live three-view output supports320x180/640x360')
                if not isinstance(ctx,str) or tuple(ctx.split('_',1)) not in CONTEXTS:raise ValueError('Unknown context')
                mode,station=ctx.split('_',1);color=command['color'];slot=command['slot']
                if type(slot) is not int or not 1<=slot<=6:raise ValueError('Slot1..6 required')
                if mode in ('COLOR','TOP') and (type(color) is not int or not 1<=color<=6):raise ValueError('Color1..6 required')
                if mode=='RING' and (type(color) is not int or not 1<=color<=3):raise ValueError('Ring1..3 required')
                if mode in ('QR','LANE') and color is not None:raise ValueError('This mode has no target ID')
                if slot!=s.slot or profile!=s.draft:
                    s.editable()
                    saved=s.store.load(slot)
                    if saved and saved['profile']['locked']:raise ValueError('Target slot locked; unlock on board first')
                # Preview must correspond to the current camera geometry/exposure.
                if action=='preview' and (profile['binding']!=s.active['binding'] or profile['camera']!=s.active['camera']):raise ValueError('Camera/binding change requires explicit Apply')
                if slot!=s.slot:
                    s.last_commit=s.saved=s.store.load(slot)
                s.draft=profile;s.slot=slot;s.dirty=not (s.last_commit and s.last_commit['profile']==profile)
                s.select(mode,station);s.target_id=color
                self.preview_sha=sha(json.dumps(profile).encode())
                if action=='apply':s.apply()
                if action=='save':
                    result=s.save();check=s.store.load(slot)
                    if check is None or check['sha256']!=result['sha256'] or check['profile']!=profile:raise OSError('Saved profile readback mismatch')
                    result['slot']=slot
            elif action=='view':
                if set(command)!={'action','selected'} or type(command['selected']) is not bool:raise ValueError('View fields mismatch')
                self.render_mode='SELECTED' if command['selected'] else 'THRESHOLD'
            elif action=='lock':
                if set(command)!={'action','locked'}:raise ValueError('Lock fields mismatch')
                s.set_locked(command['locked'])
            elif action=='stop':self.stopped=True
            elif action=='demo_reset':
                if set(command)!={'action'} or self.visual_demo is None:raise ValueError('Enabled visual demo required')
                self.visual_demo.reset()
            elif action=='sample':
                if set(command)!={'action'}:raise ValueError('Sample fields mismatch')
                self.sample_requested=True
            ack={'ok':True,'action':action,'request_sha256':envelope['sha256'],'state':self.state(action=='status')}
        except (ValueError,OSError,KeyError,TypeError) as error:
            ack={'ok':False,'action':action,'error':str(error),'request_sha256':envelope['sha256'],'state':self.state()}
        self.last_ack=ack
        return ack

    def poll(self):
        try:
            prefix=self.token[:8]+'-'
            self.mailbox_polls+=1
            name=prefix+('%08d' % (self.last_seq+1))+'.json'
            try:
                with open(self.mailbox+'/'+name) as stream:raw=stream.read(140001)
            except OSError:
                # The normal next sequence needs one lookup, not an SD directory
                # scan. Occasionally reconcile gaps left by unknown transfers.
                if self.mailbox_polls%8:return None
                names=sorted(n for n in os.listdir(self.mailbox) if n.startswith(prefix) and n.endswith('.json') and n[len(prefix):-5].isdigit() and int(n[len(prefix):-5])>self.last_seq)
                if not names:return None
                name=names[0]
                with open(self.mailbox+'/'+name) as stream:raw=stream.read(140001)
            if len(raw)>140000:return None
            envelope=json.loads(raw)
            if envelope.get('session')!=self.token or type(envelope.get('seq')) is not int or envelope['seq']<=self.last_seq:return None
            result=self.handle(envelope)
            # Requests are transient mailbox items owned by this exact session.
            try:os.remove(self.mailbox+'/'+name)
            except OSError:pass
            return result
        except (OSError,ValueError,KeyError,TypeError):return None


def draw_color_for_target(color_id):
    """Return the project color for an overlay; black targets use white ink."""
    return DRAW_COLORS.get(color_id, (255, 255, 255))


def draw_geometry(target, fits, color_id=None, x_offset=0, y_offset=0,fast_outline=False):
    for fit in fits:
        ink=draw_color_for_target(fit.get('color_id',color_id))
        cx,cy=[int(round(v)) for v in fit['center']]
        # Use source-space points for rotated geometry. Some native rotated
        # ellipse renderings distort near-circular fits. The coarse preview
        # needs fewer segments, but must preserve the fitted axes and center.
        outline=ellipse_outline(fit,steps=24 if fast_outline else 48)
        for i,point in enumerate(outline):
            end=outline[(i+1)%len(outline)]
            target.draw_line(point[0]+x_offset,point[1]+y_offset,end[0]+x_offset,end[1]+y_offset,thickness=2,color=ink)
        target.draw_cross(cx+x_offset,cy+y_offset,size=7,thickness=2,color=ink)


def _circle_outline(circle, segments=32):
    """Return a line-loop for a CanMV circle without relying on draw_circle."""
    cx,cy,r=[float(circle[k]) for k in ('x','y','r')]
    return [(int(round(cx+r*cos(2*pi*i/segments))),
             int(round(cy+r*sin(2*pi*i/segments)))) for i in range(segments)]


def draw_ring_geometry(target, circles, groups, color_id=None, x_offset=0, y_offset=0):
    """Draw measured circle proposals and clustered centers in the target ink."""
    ink=draw_color_for_target(color_id)
    for circle in circles:
        outline=_circle_outline(circle)
        for i,point in enumerate(outline):
            end=outline[(i+1)%len(outline)]
            target.draw_line(point[0]+x_offset,point[1]+y_offset,
                             end[0]+x_offset,end[1]+y_offset,thickness=1,color=ink)
    for group in groups:
        cx,cy=[int(round(v)) for v in group['center']]
        target.draw_cross(cx+x_offset,cy+y_offset,size=9,thickness=2,color=ink)


def native_ring_candidates(frame, context):
    """Run only the configured native circle proposal search for RING preview."""
    roi=tuple(context['roi'])
    try:
        found=frame.find_circles(threshold=context['circle_threshold'],
            r_min=context['r_min'],r_max=context['r_max'],r_step=context['r_step'],roi=roi)
    except Exception as error:
        return [],[],('RING_DETECTOR_ERROR: '+str(error))
    circles=[]
    for circle in list(found or ())[:128]:
        try:
            row={'x':float(circle.x()),'y':float(circle.y()),'r':float(circle.r())}
        except Exception:
            continue
        if all(v==v and abs(v)<100000 for v in row.values()) and row['r']>0:
            circles.append(row)
    groups=cluster_circles(circles,context['center_tolerance'])
    return circles,groups,'RING_CIRCLE_PROPOSALS' if circles else 'RING_NO_CIRCLE'


def ring_identity_observed(observation, target_id):
    """Count reviewed pixel identity separately from unnumbered circle proposals."""
    support=observation.get('ring_support')
    return (observation.get('kind')=='RING'
        and observation.get('status') in ('OK','UNCALIBRATED')
        and observation.get('reason')!='ID_UNKNOWN'
        and type(target_id) is int and target_id in (1,2,3)
        and type(observation.get('target_id')) is int and observation['target_id']==target_id
        and observation.get('camera_id')==MODE_CAMERA['RING']
        and observation.get('is_simulated') is False
        and type(support) is int and support>0
        and finite(observation.get('u_px')) and finite(observation.get('v_px')))


def alignment_preview_label(alignment):
    """Only label the current measured pixels; no held result or score wording."""
    if alignment.get('error_px') is None:return 'ALIGN UNKNOWN'
    state='STABLE' if alignment.get('status')=='MEASURED' else 'RAW'
    return 'PX dx=%.1f dy=%.1f %s' % (alignment['dx_px'],alignment['dy_px'],state)


def draw_material_candidates(target,candidates,color_id,x_offset=0,y_offset=0):
    """Color-only rectangles have no center cross or alignment authority."""
    for candidate in candidates:
        ink=draw_color_for_target(candidate.get('color_id',color_id))
        x,y,w,h=candidate['bbox'];x+=x_offset;y+=y_offset
        corners=((x,y),(x+w-1,y),(x+w-1,y+h-1),(x,y+h-1))
        for i,start in enumerate(corners):
            end=corners[(i+1)%4]
            target.draw_line(start[0],start[1],end[0],end[1],thickness=1,color=ink)


def _blob_candidate_roi(blobs,base_roi,margin=12):
    """Bound native contour work to quick-blob candidates with a safety margin."""
    x,y,w,h=[int(v) for v in base_roi];left=x+w;top=y+h;right=x;bottom=y
    for blob in blobs:
        try:bx,by,bw,bh=[int(v) for v in blob.rect()]
        except Exception:return (x,y,w,h)
        left=min(left,max(x,bx-margin));top=min(top,max(y,by-margin))
        right=max(right,min(x+w,bx+bw+margin));bottom=max(bottom,min(y+h,by+bh+margin))
    if right<=left or bottom<=top:return (x,y,w,h)
    return (left,top,right-left,bottom-top)


def bounded_material_results(fits,rows,maximum=MAX_MATERIALS):
    """Reject scene overflow explicitly instead of choosing arbitrary objects."""
    if len(fits)<=maximum:return fits,False
    for row in rows:
        row['raw_top_count']=row['top_count'];row['top_count']=0
    return [],True


def native_views(frame,profile,mode,station,color,selected=False,diagnostic_material_color=None,diagnostic_material_colors=None,annotated_preview=False):
    """Detect on current full-resolution pixels; never reuse a previous fit."""
    c=profile['contexts'][context_key(mode,station)]
    mask=None if annotated_preview and mode=='RING' else frame.copy()
    if mask is not None:mask.clear()
    fits=[];components=[];reason='NOT_APPLICABLE';quick_blob_count=None
    region=gray=chosen=None;ring_candidates=[];ring_groups=[];alignment_pair=None;material_fits=[]
    material_candidates=[];material_components=[];material_by_color=[];material_limit_exceeded=False;raw_material_count=0
    colors=diagnostic_colors(diagnostic_material_color,diagnostic_material_colors)
    if mode in ('COLOR','TOP'):
        threshold=c['thresholds'][str(color)]
        if threshold is None:
            reason='COLOR_PARAMETERS_REQUIRED'
        elif selected:
            quick=frame.find_blobs([tuple(threshold)],roi=tuple(c['roi']),pixels_threshold=c['min_pixels'],area_threshold=c['min_area'],merge=False)
            quick_blob_count=len(quick)
            if not quick:
                reason='NO_BLOB_GATE'
            else:
                candidate_roi=_blob_candidate_roi(quick,c['roi'])
                region=frame.copy(roi=candidate_roi);region.binary([tuple(threshold)])
                mask.draw_image(region,candidate_roi[0],candidate_roi[1])
                gray=region.to_grayscale()
                chosen=gray.copy();chosen.clear()
                result=extract_mask_ellipses(gray.to_numpy_ref(),candidate_roi,c['min_area'],c['max_area'],render_mask=chosen.to_numpy_ref(),close_kernel=3)
                components=result['components'];fits=result['accepted']
                if not result['complete']:chosen.clear()
                if mode=='TOP' and color==5 and result['complete']:
                    # The LAB body contour remains rejected. An independent
                    # luminance boundary may identify a complete dark top face.
                    import cv2
                    from vision.ellipse_edges import extract_dark_edge_ellipses
                    luma=frame.copy(roi=candidate_roi).to_grayscale()
                    edges=extract_dark_edge_ellipses(luma.to_numpy_ref(),gray.to_numpy_ref(),candidate_roi,
                        components,c['min_area'],c['max_area'],existing=fits,cv=cv2)
                    components+=edges['components']
                    if edges['complete']:
                        for fit in edges['accepted']:
                            a,b=fit['semiaxes'];cx,cy=fit['center']
                            cv2.ellipse(chosen.to_numpy_ref(),(int(round(cx-candidate_roi[0])),int(round(cy-candidate_roi[1]))),
                                (int(round(a)),int(round(b))),fit['angle_rad']*180/pi,0,360,(255,),-1)
                        fits+=edges['accepted']
                    luma=None
                reason='GEOMETRY_ONLY' if fits else (components[0]['fit']['reason'] if components else result['reason'])
                # Preserve holes in the threshold image when forming the cutout.
                chosen.b_and(gray)
                mask.clear();mask.draw_image(chosen.to_rgb565(),candidate_roi[0],candidate_roi[1])
        else:
            region=frame.copy(roi=tuple(c['roi']));region.binary([tuple(threshold)])
            mask.draw_image(region,c['roi'][0],c['roi'][1]);reason='THRESHOLD_ONLY'
    elif mode=='LANE':
        region=frame.copy(roi=tuple(c['roi']));region.to_grayscale();region.binary([(c['gray_min'],c['gray_max'])])
        mask.draw_image(region,c['roi'][0],c['roi'][1])
    elif mode=='RING':
        if selected:
            from vision.ring_native import detect_paper_targets,associate_material_target
            # Paper detection always runs from the paper ROI, even when no
            # material exists. Color and material geometry never feed this path.
            paper_roi=tuple(c['roi'])
            region=frame.copy(roi=paper_roi).to_grayscale()
            result=detect_paper_targets(region.to_numpy_ref(),paper_roi,pixel_scale=2)
            if mask is not None:
                region.to_numpy_ref()[:,:]=result['edges']
                mask.draw_image(region.to_rgb565(),paper_roi[0],paper_roi[1])
            ring_groups=result['groups'];reason=result['reason']
            fits=[fit for group in ring_groups for fit in group['boundaries']]
            mc=profile['contexts']['TOP_BUFFER']
            available=[v for v in colors if mc['thresholds'].get(str(v)) is not None]
            batch=None
            if diagnostic_material_colors is not None and available:
                batch=frame.find_blobs([tuple(mc['thresholds'][str(v)]) for v in available],roi=tuple(mc['roi']),
                    pixels_threshold=mc['min_pixels'],area_threshold=mc['min_area'],merge=False)
            for material_color in colors:
                # Material detection uses its own configured ROI and color.
                # Its output is displayed even if the paper path found nothing.
                mc=profile['contexts']['TOP_BUFFER']
                threshold=mc['thresholds'].get(str(material_color))
                summary={'color_id':material_color,'threshold_available':threshold is not None,'candidate_count':0,'top_count':0,
                    'scan_complete':False,'scan_reason':'MATERIAL_THRESHOLD_REQUIRED'}
                material_by_color.append(summary)
                if threshold is None:
                    alignment_pair={'reason':'MATERIAL_THRESHOLD_REQUIRED','material':None,'target':None}
                else:
                    quick=([blob for blob in batch if blob.code() & (1<<available.index(material_color))] if batch is not None else
                        frame.find_blobs([tuple(threshold)],roi=tuple(mc['roi']),
                            pixels_threshold=mc['min_pixels'],area_threshold=mc['min_area'],merge=False))
                    summary['candidate_count']=len(quick)
                    summary.update(scan_complete=True,scan_reason='NO_QUALIFIED_BLOB')
                    material_candidates.extend({'bbox':list(blob.rect()),'color_id':material_color,
                        'status':'COLOR_CANDIDATE','center_valid':False} for blob in quick)
                    # Semimajor <=120 plus the residual bound limits the full
                    # span; 256 includes boundary and pixel-rounding allowance.
                    # Keep oversized color candidates as diagnostics only.
                    eligible=[blob for blob in quick if max(blob.rect()[2:])<=256]
                    summary['oversize_rejected']=len(quick)-len(eligible)
                    if eligible:
                        mroi=_blob_candidate_roi(eligible,mc['roi'])
                        mmask=frame.copy(roi=mroi);mmask.binary([tuple(threshold)])
                        mgray=mmask.to_grayscale()
                        measured=extract_mask_ellipses(mgray.to_numpy_ref(),mroi,mc['min_area'],mc['max_area'],close_kernel=3)
                        summary.update(scan_complete=measured['complete'] is True,
                            scan_reason=measured.get('reason','CONTOUR_SCAN_INCOMPLETE'))
                        if material_color==5 and measured['complete'] and not measured['accepted']:
                            # Reuse TOP's independent dark-face boundary path.
                            # The color/body contour stays rejected. Circle
                            # proposals require measured contrast in all48
                            # directions and the original ellipse/interior gates.
                            from vision.ellipse_edges import extract_dark_edge_ellipses
                            luma=frame.copy(roi=mroi).to_grayscale()
                            edge_result=extract_dark_edge_ellipses(luma.to_numpy_ref(),mgray.to_numpy_ref(),
                                mroi,measured['components'],mc['min_area'],mc['max_area'])
                            measured['components'].extend(edge_result['components'])
                            if edge_result['complete']:measured['accepted'].extend(edge_result['accepted'])
                            else:
                                measured['complete']=False
                                summary.update(scan_complete=False,scan_reason=edge_result.get('reason','EDGE_SCAN_INCOMPLETE'))
                            luma=None
                        material_components.extend(dict(row,color_id=material_color) for row in measured.get('components',[]))
                        if measured['complete']:
                            summary['top_count']=len(measured['accepted'])
                            material_fits.extend(dict(row,color_id=material_color) for row in measured['accepted'])
            raw_material_count=len(material_fits)
            material_fits,material_limit_exceeded=bounded_material_results(material_fits,material_by_color)
            if diagnostic_material_color is not None:
                if profile['contexts']['TOP_BUFFER']['thresholds'].get(str(diagnostic_material_color)) is not None:
                    alignment_pair=associate_material_target(material_fits,ring_groups)
                    alignment_pair['refinement']='INDEPENDENT_DETECTIONS'
        else:
            reason='RING_DETECTOR_IDLE'
    material_complete,material_scan_reason=scan_state(colors,material_by_color,
        mode=='RING' and selected,material_limit_exceeded)
    retained=frame.copy()
    if mode!='RING':retained.b_and(mask)
    ink_id=None if mode=='RING' else color
    draw_geometry(retained,fits,ink_id,fast_outline=mode=='RING')
    draw_geometry(retained,material_fits,diagnostic_material_color)
    valid_colors={f.get('color_id') for f in material_fits}
    if diagnostic_material_colors is None:
        draw_material_candidates(retained,[c for c in material_candidates if c['color_id'] not in valid_colors],diagnostic_material_color)
    if mask is not None:draw_ring_geometry(mask,ring_candidates,ring_groups,ink_id)
    draw_ring_geometry(retained,ring_candidates,ring_groups,ink_id)
    return mask,retained,{'accepted':fits,'components':components,'reason':reason,'fresh':True,
        'quick_blob_count':quick_blob_count,'ring_candidates':ring_candidates,'ring_groups':ring_groups,
        'alignment_pair':alignment_pair,'material_fits':material_fits,
        'material_color_candidates':material_candidates,'material_components':material_components,
        'material_by_color':material_by_color,'material_limit_exceeded':material_limit_exceeded,
        'raw_material_fit_count':raw_material_count,'material_complete':material_complete,
        'material_scan_reason':material_scan_reason,'material_scan_colors':colors,
        'material_algorithm_revision':profile['revision']}


def restore_initial_state(session,state):
    """Restore protected active and draft separately, without SD profile writes."""
    if not state:return
    if state.get('profile_root')!=session.store.root:raise ValueError('Restore root mismatch')
    active=validate(clone(state['active_profile']))
    draft=validate(clone(state['profile']))
    session.draft=active;session.apply()
    session.draft=draft;session.dirty=bool(state['dirty']);session.slot=state['slot']
    session.select(state['mode'],state['station']);session.target_id=state['color']


def main(token,root='/sdcard/transport_mvp/tuning_profiles',mailbox=ROOT,max_frames=None,initial_state=None,diagnostic_material_color=None,visual_demo=False,diagnostic_material_colors=None,vector_boundary=False,annotated_preview=False):
    import image
    from vision.ellipse_native import configure_boundary_backend
    boundary_backend=configure_boundary_backend(vector_boundary)
    if type(annotated_preview) is not bool:raise ValueError('Explicit annotated preview flag required')
    app=TuningApp(root);s=app.session;restore_initial_state(s,initial_state)
    commands=LiveCommands(s,token,mailbox,diagnostic_material_color,visual_demo,diagnostic_material_colors)
    material_tracker=StableTracker();paper_tracker=StableTracker()
    material_producer=None;material_binding_key=None;material_snapshot=None
    material_clock=SourceClock(time.ticks_add(0,-1)+1 if hasattr(time,'ticks_add') else 4294967296)
    if initial_state:commands.render_mode=initial_state.get('render_mode','THRESHOLD')
    last_poll=last_report=None;errors=0;native_count=0;recognition_frames=0;accepted_frames=0;started=time.ticks_ms()
    win_start=started;win_frames=win_recognition=win_accepted=win_preview=0;win_sums=[0]*5;win_max=[0]*5;win_gc=win_gc_max=win_poll=0
    sample_writer=None;resource_cache=None;resource_at=None;win_instances={};win_ring_proposals=0
    win_alignment_raw=win_alignment_measured=win_material_frames=win_material_color_frames=0
    win_material_by_color={}
    try:
        os.exitpoint(os.EXITPOINT_ENABLE)
        app.open_display(True,to_ide=False)
        composite=None;layout=None;preview_scale=.5
        print('LIVE_TUNING_READY:'+json.dumps(commands.state(True)))
        while not commands.stopped and (max_frames is None or native_count<max_frames):
            os.exitpoint();now=time.ticks_ms()
            if last_poll is None or time.ticks_diff(now,last_poll)>=POLL_INTERVAL_MS:
                last_poll=now;ack=commands.poll();win_poll+=time.ticks_diff(time.ticks_ms(),now)
                if ack:
                    print('LIVE_TUNING_ACK:'+json.dumps(ack))
                    # Never label pre-command frames with the newly selected color/profile.
                    win_start=time.ticks_ms();last_report=win_start
                    win_frames=win_recognition=win_accepted=win_preview=0;win_sums=[0]*5;win_max=[0]*5
                    win_gc=win_gc_max=win_poll=0;win_instances={};win_ring_proposals=0
                    material_tracker.reset();paper_tracker.reset()
                    win_alignment_raw=win_alignment_measured=win_material_frames=win_material_color_frames=0
                    win_material_by_color={}
            if commands.stopped:break
            desired=(str(s.active['binding']),str(s.active['camera']),True)
            if desired!=app.applied_geometry:
                app.open_display(True,to_ide=False);composite=None
                if material_producer:material_producer.invalidate()
            b=s.active['binding'];cid=MODE_CAMERA[s.mode]
            binding_key=(id(s.active),id(s.draft),s.active['revision'],s.draft['revision'],
                s.mode,s.station,tuple(commands.diagnostic_material_colors),boundary_backend)
            if binding_key!=material_binding_key:
                material_binding_key=binding_key
                # Public observation session must not expose the mailbox token.
                source_session=sha(('material-source-v1:'+token).encode())[:32]
                material_producer=(MaterialSnapshotProducer(s.active,s.draft,source_session,
                    commands.diagnostic_material_colors,boundary_backend,s.mode,s.station)
                    if s.mode=='RING' and commands.diagnostic_material_colors else None)
                if material_producer:
                    print('MATERIAL_SNAPSHOT_CONTRACT:'+json.dumps(material_producer.contract()))
            if composite is None or layout!=(b['width'],b['height']):
                layout=(b['width'],b['height']);pw=int(b['width']*preview_scale);ph=int(b['height']*preview_scale)
                composite=image.Image(pw if annotated_preview else pw*3,ph,image.RGB565)
            try:
                if cid not in app.sensors:raise OSError(app.errors.get(cid,'Camera unavailable'))
                t0=time.ticks_ms();frame=app.sensors[cid].snapshot(timeout=100,auto_reboot=False)
                if frame is None or (frame.width(),frame.height())!=layout:raise ValueError('Fresh frame binding mismatch')
                app.frame_id+=1;t1=time.ticks_ms();material_timestamp=material_clock.update(t1)
                selected=commands.render_mode=='SELECTED'
                extra=({'diagnostic_material_colors':commands.diagnostic_material_colors} if diagnostic_material_colors is not None else {})
                if annotated_preview and s.mode=='RING':extra['annotated_preview']=True
                mask,retained,geometry=native_views(frame,material_producer.draft if material_producer else s.draft,
                    s.mode,s.station,s.target_id,selected,commands.diagnostic_material_color,**extra)
                material_snapshot=None
                if material_producer:
                    material_snapshot=material_producer.build({'camera_id':cid,'frame_id':app.frame_id,
                        'timestamp_ms':material_timestamp,'is_simulated':False,
                        'image_profile':image_profile(s.active),'active_revision':s.active['revision'],
                        'width':frame.width(),'height':frame.height()},geometry)
                t2=time.ticks_ms()
                detection=frame.copy() if s.mode in ('QR','LANE') else frame
                packet={'image':detection,'camera_id':cid,'frame_id':app.frame_id,'timestamp_ms':t1,'is_simulated':False,'image_profile':image_profile(s.active)}
                if s.mode=='RING' and not s.context(active=True)['ring_id_reviewed']:
                    # No numbered reference exists. Repeating a second Hough
                    # search cannot establish identity; show draft geometry only.
                    obs=observation('RING',packet,None,s.station,status='UNCALIBRATED',reason='ID_UNKNOWN')
                    obs.update(ring_candidates=geometry['ring_groups'],preview_only=True,
                        algorithm_profile_revision=s.draft['revision'])
                    s.observation=obs
                else:obs=s.process(packet)
                alignment={'status':'UNKNOWN','reason':'DIAGNOSTIC_MATERIAL_NOT_SELECTED','quality':None}
                if s.mode=='RING' and commands.diagnostic_material_color is not None:
                    pair=geometry.get('alignment_pair')
                    if pair and pair.get('material') and pair.get('target'):
                        material_center,paper_center=pair['material']['center'],pair['target']['center']
                        key=(s.draft['revision'],commands.diagnostic_material_color)
                        astable=material_tracker.update(key,app.frame_id,t1,material_center)
                        bstable=paper_tracker.update(key,app.frame_id,t1,paper_center)
                        meta={'status':'GEOMETRY_ONLY','frame_id':app.frame_id,'timestamp_ms':t1,
                              'camera_id':cid,'is_simulated':False,'image_profile':image_profile(s.active)}
                        alignment=measure_alignment(dict(meta,u_px=material_center[0],v_px=material_center[1],stable=astable),
                            dict(meta,u_px=paper_center[0],v_px=paper_center[1],stable=bstable),now_ms=time.ticks_ms())
                        alignment.update(material_color=commands.diagnostic_material_color,
                            paper_center_px=paper_center,material_center_px=material_center,plane_projection_verified=False,
                            limitation='TOP_FACE_HEIGHT_PARALLAX_POSSIBLE',
                            refinement=pair.get('refinement','COARSE_BOUNDARIES'))
                    else:
                        material_tracker.reset();paper_tracker.reset()
                        alignment={'status':'UNKNOWN','reason':pair['reason'] if pair else 'MATERIAL_OBSERVATION_REQUIRED','quality':None}
                    if alignment.get('error_px') is not None:win_alignment_raw+=1
                    if alignment.get('status')=='MEASURED':win_alignment_measured+=1
                demo_state=None
                if s.mode=='RING' and commands.visual_demo is not None:
                    demo_state=commands.visual_demo.update(packet,geometry.get('material_fits',[]),
                        geometry.get('ring_groups',[]),time.ticks_ms())
                    for event in demo_state['events']:
                        print('VISUAL_DEMO_EVENT:'+json.dumps(event))
                t3=time.ticks_ms()
                native_count+=1;win_frames+=1
                preview_interval=(1 if annotated_preview else PREVIEW_EVERY_N_FRAMES if s.mode=='RING' else
                    (3 if selected and geometry['reason']=='NO_BLOB_GATE' else PREVIEW_EVERY_N_FRAMES))
                preview_due=native_count%preview_interval==0
                if preview_due:
                    for source,offset in (((retained,0),) if annotated_preview else ((frame,0),(mask,pw),(retained,pw*2))):
                        composite.draw_image(source,offset,0,x_scale=preview_scale,y_scale=preview_scale)
                    if s.mode=='RING':
                        if not annotated_preview:composite.draw_string_advanced(pw+4,4,14,'PAPER %d' % len(geometry.get('ring_groups',())),color=(255,220,80))
                        label=('TOO MANY / UNKNOWN' if geometry.get('material_limit_exceeded') else 'TOP %d / COARSE' % len(geometry.get('material_fits',())) if diagnostic_material_colors is not None else
                            'MAT %d TOP %d' % (len(geometry.get('material_color_candidates',())),len(geometry.get('material_fits',()))))
                        if annotated_preview:label+=' PAPER %d' % len(geometry.get('ring_groups',()))
                        right_x=0 if annotated_preview else pw*2
                        composite.draw_string_advanced(right_x+4,4,14,label,color=(255,220,80))
                        if diagnostic_material_colors is not None:
                            names={1:'R',2:'Y',3:'B',4:'G',5:'K',6:'C'}
                            label=' '.join('%s%d' % (names[r['color_id']],r['top_count']) for r in geometry.get('material_by_color',()))
                            composite.draw_string_advanced(right_x+4,ph-19,12,'TOP '+label,color=(255,220,80))
                    if s.mode=='RING' and commands.diagnostic_material_color is not None:
                        composite.draw_string_advanced((0 if annotated_preview else pw*2)+4,ph-19,14,alignment_preview_label(alignment),color=(255,220,80))
                    if demo_state is not None:
                        composite.draw_string_advanced((0 if annotated_preview else pw*2)+4,ph-37,14,'SIM %s %d/%d' %
                            (demo_state['state'].replace('SIM_',''),demo_state['simulated_picks'],demo_state['simulated_places']),color=(0,255,255))
                    app.canvas.clear();app.canvas.draw_image(frame,0,48)
                    ink_id=None if s.mode=='RING' else s.target_id
                    draw_geometry(app.canvas,geometry['accepted'],ink_id,y_offset=48,fast_outline=s.mode=='RING')
                    draw_geometry(app.canvas,geometry.get('material_fits',()),commands.diagnostic_material_color,y_offset=48)
                    valid_colors={f.get('color_id') for f in geometry.get('material_fits',())}
                    if diagnostic_material_colors is None:
                        draw_material_candidates(app.canvas,[c for c in geometry.get('material_color_candidates',()) if c.get('color_id') not in valid_colors],commands.diagnostic_material_color,y_offset=48)
                    draw_ring_geometry(app.canvas,geometry.get('ring_candidates',()),geometry.get('ring_groups',()),ink_id,y_offset=48)
                    title=('RING candidates / ID UNKNOWN' if s.mode=='RING' and not ring_identity_observed(obs,s.target_id)
                        else s.mode+' / IMAGE PX')
                    app.canvas.draw_string_advanced(8,5,20,title+' / CSI%d' % cid,color=(255,220,80))
                    if alignment.get('error_px') is not None:
                        app.canvas.draw_string_advanced(8,29,16,'PIXEL ONLY dx=%.1f dy=%.1f / %s' %
                            (alignment['dx_px'],alignment['dy_px'],alignment['status']),color=(255,220,80))
                    app.Display.show_image(app.canvas);t4=time.ticks_ms()
                    composite.compress_for_ide(quality=60);t5=time.ticks_ms();win_preview+=1
                else:
                    t4=t5=time.ticks_ms()
                is_recognition=selected and s.mode in ('TOP','COLOR','RING')
                if is_recognition:
                    recognition_frames+=1;win_recognition+=1
                    count=(len(geometry['ring_groups']) if s.mode=='RING' else len(geometry['accepted']))
                    win_instances[count]=win_instances.get(count,0)+1
                    if s.mode=='RING' and count:win_ring_proposals+=1
                    if s.mode=='RING' and geometry.get('material_fits'):win_material_frames+=1
                    if s.mode=='RING' and geometry.get('material_color_candidates'):win_material_color_frames+=1
                    for row in geometry.get('material_by_color',()):
                        key=str(row['color_id'])
                        counts=win_material_by_color.setdefault(key,{'candidate_frames':0,'top_frames':0,'candidate_instances':0,'top_instances':0})
                        counts['candidate_frames']+=int(row['candidate_count']>0)
                        counts['top_frames']+=int(row['top_count']>0)
                        counts['candidate_instances']+=row['candidate_count'];counts['top_instances']+=row['top_count']
                    accepted=(ring_identity_observed(obs,s.target_id) if s.mode=='RING' else bool(count))
                    if accepted:accepted_frames+=1;win_accepted+=1
                for i,(end,begin) in enumerate(((t1,t0),(t2,t1),(t3,t2),(t4,t3),(t5,t4))):
                    dt=time.ticks_diff(end,begin);win_sums[i]+=dt;win_max[i]=max(win_max[i],dt)
                if commands.sample_requested:
                    commands.sample_requested=False
                    from .sampling import SampleWriter
                    if sample_writer is None:sample_writer=SampleWriter('/sdcard/transport_mvp/ellipse_perf_samples')
                    sample=sample_writer.save(frame,{'camera_id':cid,'frame_id':app.frame_id,'timestamp_ms':t1,'is_simulated':False,'image_profile':image_profile(s.active),'mode':s.mode,'station':s.station,'profile_revision':s.draft['revision']},image.rgb_to_lab,os.exitpoint)
                    # Lossless same-frame input permits threshold/morphology comparison
                    # without resampling JPEG colors or repeating a moving scene.
                    if sample.get('complete'):
                        native_path=sample['jpeg'].replace('-live.jpg','-native.rgb565')
                        native=frame.bytearray()
                        if len(native)!=b['width']*b['height']*2:raise ValueError('Native RGB565 byte count mismatch')
                        try:
                            with open(native_path,'wb') as stream:
                                written=stream.write(native)
                            sample.update(native=native_path,native_bytes=written,native_saved=written==len(native))
                        except OSError as error:sample.update(native_saved=False,native_error=str(error))
                        native=None
                    print('ELLIPSE_NATIVE_SAMPLE:'+json.dumps(sample))
                if last_report is None or time.ticks_diff(t5,last_report)>=1000:
                    last_report=t5
                    if resource_at is None or time.ticks_diff(t5,resource_at)>=5000:
                        resource_cache=resource_snapshot();resource_at=t5
                    duration=time.ticks_diff(t5,win_start)
                    stages=dict((key,{'mean_ms':win_sums[i]/win_frames,'max_ms':win_max[i]}) for i,key in enumerate(('capture','view','process','draw','compress')))
                    stages['gc']={'mean_ms':win_gc/win_frames,'max_ms':win_gc_max}
                    stages['mailbox']={'mean_ms':win_poll/win_frames}
                    print('LIVE_TUNING_FRAME:'+json.dumps({'session':token,'frame_id':app.frame_id,'native_count':native_count,'camera_id':cid,'timestamp_ms':t1,
                        'width':b['width'],'height':b['height'],'context':context_key(s.mode,s.station),'color':s.target_id,'draft_revision':s.draft['revision'],'active_revision':s.active['revision'],
                        'command_seq':commands.last_seq,'profile_root':s.store.root,'service_version':1,'errors':errors,'observation':obs,
                        'layout':'annotated' if annotated_preview else 'original-mask-retained','native_threshold':s.mode!='RING',
                        'area_semantics':'COARSE_CONCENTRIC_BOUNDARIES' if s.mode=='RING' else 'NATIVE_CONTOUR_AREA',
                        'close_kernel':3 if selected and s.mode in ('TOP','COLOR') else 0,
                        'mask_semantics':'COARSE_EDGE_MAP' if s.mode=='RING' else ('FRESH_ELLIPSE_CONTOUR' if selected else 'PIXEL_THRESHOLD'),
                        'geometry_reason':geometry['reason'],
                        'ellipse_fits':geometry['accepted'],'contour_components':geometry['components'],'ring_candidates':geometry.get('ring_candidates',[]),'ring_groups':geometry.get('ring_groups',[]),'quick_blob_count':geometry.get('quick_blob_count'),'geometry_frame_id':app.frame_id,'geometry_age_ms':0,
                        'rough_alignment':alignment,
                        'visual_demo':demo_state,
                        'paper_targets':geometry.get('ring_groups',[]),
                        'material_detections':geometry.get('material_fits',[]),
                        'material_snapshot':material_snapshot,
                        'material_complete':geometry.get('material_complete',False),
                        'material_scan_reason':geometry.get('material_scan_reason','SCAN_EVIDENCE_MISSING'),
                        'material_color_candidates':geometry.get('material_color_candidates',[]),
                        'material_components':geometry.get('material_components',[]),
                        'diagnostic_material_colors':commands.diagnostic_material_colors,
                        'material_by_color':geometry.get('material_by_color',[]),
                        'material_limit_exceeded':geometry.get('material_limit_exceeded',False),
                        'raw_material_fit_count':geometry.get('raw_material_fit_count',0),
                        'boundary_backend':boundary_backend,
                        'recognition_frames':recognition_frames,'accepted_frames':accepted_frames,'recognition_interval_ms':0,'mailbox_poll_interval_ms':POLL_INTERVAL_MS,
                        'preview_width':pw if annotated_preview else pw*3,'preview_height':ph,'preview_scale':preview_scale,'preview_interval_frames':preview_interval,'preview_frames':win_preview,
                        'window':{'frames':win_frames,'recognition_frames':win_recognition,'instance_frames':win_instances,
                            'instance_frames_semantics':'INDEPENDENT_PAPER_TARGET_COUNT' if s.mode=='RING' else 'ACCEPTED_ELLIPSE_COUNT',
                            'ring_proposal_frames':win_ring_proposals,'accepted_frames_semantics':'REVIEWED_RING_IDENTITY_PX_ONLY' if s.mode=='RING' else 'ACCEPTED_ELLIPSE',
                            'alignment_raw_frames':win_alignment_raw,'alignment_measured_frames':win_alignment_measured,
                            'material_detected_frames':win_material_frames,
                            'material_color_candidate_frames':win_material_color_frames,
                            'material_by_color':win_material_by_color,
                            'preview_frames':win_preview,'accepted_frames':win_accepted,'elapsed_ms':duration,'capture_fps':1000*win_frames/max(1,duration),'recognition_fps':1000*win_recognition/max(1,duration),'preview_fps':1000*win_preview/max(1,duration),'stages':stages},
                        'is_simulated':False,'resources':resource_cache,'elapsed_ms':time.ticks_diff(t5,started),'processing_ms':time.ticks_diff(t5,t0)}))
                    win_start=time.ticks_ms();win_frames=win_recognition=win_accepted=win_preview=0;win_sums=[0]*5;win_max=[0]*5;win_gc=win_gc_max=win_poll=0;win_instances={};win_ring_proposals=0
                    win_alignment_raw=win_alignment_measured=win_material_frames=win_material_color_frames=0
                    win_material_by_color={}
            except Exception as error:
                if 'IDE interrupt' in str(error):raise
                errors+=1;s.observation=None;s.adapter.reset_session();composite.clear()
                material_snapshot=None
                if material_producer:material_producer.invalidate()
                print('LIVE_TUNING_ERROR:'+json.dumps({'session':token,'error':str(error),'errors':errors}));time.sleep_ms(300)
                if hasattr(sys,'print_exception'):sys.print_exception(error)
            frame=mask=retained=detection=source=geometry=None
            pair=meta=alignment=packet=demo_state=None
            before_gc=time.ticks_ms()
            if native_count%8==0 or gc.mem_free()<1000000:gc.collect()
            gc_time=time.ticks_diff(time.ticks_ms(),before_gc);win_gc+=gc_time;win_gc_max=max(win_gc_max,gc_time)
            time.sleep_ms(1)
    finally:
        app.close();print('LIVE_TUNING_CLOSED:'+json.dumps({'session':token,'native_frames':native_count,'errors':errors,'cleanup_errors':app.cleanup_errors}))
