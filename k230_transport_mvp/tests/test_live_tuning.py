import json,tempfile,unittest
import types
from unittest.mock import patch
from pathlib import Path
from k230.tuning.live import (LiveCommands,restore_initial_state,_blob_candidate_roi,
                              draw_color_for_target,native_ring_candidates,ring_identity_observed,alignment_preview_label,draw_geometry)
from k230.tuning.model import TuningSession
from k230.tuning.store import ProfileStore,sha
from k230.tuning.schema import clone

class LiveCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.session=TuningSession(ProfileStore(self.temp.name+'/profiles'))
        self.commands=LiveCommands(self.session,'a'*32,self.temp.name+'/mailbox');self.seq=0

    def packet(self,action,profile=None,slot=1):
        self.seq+=1;cmd={'action':action}
        if action in ('preview','apply','save'):cmd.update(profile=clone(profile or self.session.draft),context='TOP_BUFFER',color=5,slot=slot)
        raw=json.dumps(cmd)
        return {'session':'a'*32,'seq':self.seq,'payload':raw,'sha256':sha(raw.encode())}

    def test_preview_apply_save_are_distinct_and_native_store_roundtrip(self):
        p=clone(self.session.draft);p['revision']=1;p['contexts']['TOP_BUFFER']['thresholds']['5']=[10,40,-10,10,-20,20]
        result=self.commands.handle(self.packet('preview',p));self.assertTrue(result['ok']);self.assertFalse(result['state']['active_equals_draft']);self.assertIsNone(self.session.last_commit)
        self.assertTrue(self.commands.handle(self.packet('apply',p))['state']['active_equals_draft'])
        saved=self.commands.handle(self.packet('save',p));self.assertTrue(saved['ok']);self.assertTrue(saved['state']['saved_equals_draft'])
        self.assertEqual(ProfileStore(self.temp.name+'/profiles').load()['sha256'],saved['state']['saved_sha256'])
        self.assertFalse(self.session.adapter.config['UART_ENABLED']);self.assertEqual(self.session.adapter.config['CALIBRATIONS'],{})

    def test_checksum_session_replay_invalid_profile_do_not_mutate(self):
        before=clone(self.session.draft);packet=self.packet('preview');packet['sha256']='0'*64
        with self.assertRaises(ValueError):self.commands.handle(packet)
        packet=self.packet('status');packet['session']='b'*32
        with self.assertRaises(ValueError):self.commands.handle(packet)
        packet=self.packet('status');self.commands.handle(packet)
        with self.assertRaises(ValueError):self.commands.handle(packet)
        bad=clone(before);bad['contexts']['TOP_BUFFER']['min_area']=0
        self.assertFalse(self.commands.handle(self.packet('preview',bad))['ok']);self.assertEqual(self.session.draft,before)

    def test_slot_lock_and_geometry_require_apply(self):
        p=clone(self.session.draft);p['camera']['fps']=60
        self.assertFalse(self.commands.handle(self.packet('preview',p))['ok']);self.assertEqual(self.session.active['camera']['fps'],30)
        self.session.set_locked(True);self.session.save();p=clone(self.session.draft);p['locked']=False
        self.assertFalse(self.commands.handle(self.packet('preview',p))['ok']);self.assertTrue(self.session.draft['locked'])

    def test_mailbox_ignores_partial_and_deletes_only_consumed_session_request(self):
        packet=self.packet('status');file=Path(self.temp.name+'/mailbox/aaaaaaaa-00000001.json');file.write_text('{partial')
        self.assertIsNone(self.commands.poll());self.assertTrue(file.exists())
        file.write_text(json.dumps(packet));result=self.commands.poll();self.assertTrue(result['ok']);self.assertFalse(file.exists())
        other=Path(self.temp.name+'/mailbox/other-session.json');other.write_text('keep');self.assertIsNone(self.commands.poll());self.assertTrue(other.exists())

    def test_switch_slot_does_not_claim_old_saved_generation(self):
        self.commands.handle(self.packet('save'));p=clone(self.session.draft)
        result=self.commands.handle(self.packet('preview',p,slot=2));self.assertTrue(result['ok']);self.assertIsNone(result['state']['generation']);self.assertTrue(result['state']['dirty'])

    def test_restart_restores_active_and_dirty_draft_separately_without_save(self):
        active=clone(self.session.active)
        draft=clone(active);draft['revision']=8
        draft['contexts']['TOP_BUFFER']['thresholds']['4']=[43,54,-52,-33,17,34]
        state={'profile_root':self.session.store.root,'active_profile':active,'profile':draft,
               'dirty':True,'slot':1,'mode':'TOP','station':'BUFFER','color':4}
        restore_initial_state(self.session,state)
        self.assertEqual(self.session.active,active);self.assertEqual(self.session.draft,draft)
        self.assertTrue(self.session.dirty);self.assertEqual(self.session.target_id,4)
        self.assertIsNone(self.session.store.load(1));self.assertIsNone(self.session.last_commit)
        status=self.commands.handle(self.packet('status'))['state']
        self.assertEqual(status['active_profile'],active);self.assertEqual(status['profile'],draft)

    def test_mailbox_reconciles_a_sequence_gap_without_replaying_consumed_commands(self):
        self.seq=3;packet=self.packet('status')
        file=Path(self.temp.name+'/mailbox/aaaaaaaa-00000004.json');file.write_text(json.dumps(packet))
        received=[self.commands.poll() for _ in range(8)]
        self.assertEqual(sum(r is not None for r in received),1)
        self.assertEqual(self.commands.last_seq,4);self.assertFalse(file.exists())
        self.assertTrue(all(self.commands.poll() is None for _ in range(8)))

    def test_blob_candidate_roi_is_bounded_and_keeps_margin(self):
        class Blob:
            def __init__(self, rect): self._rect=rect
            def rect(self): return self._rect
        roi=_blob_candidate_roi([Blob((20,30,40,50))],(8,10,100,90),margin=12)
        self.assertEqual(roi,(8,18,64,74))
        self.assertEqual(_blob_candidate_roi([], (8,10,100,90)), (8,10,100,90))

    def test_overlay_ink_follows_project_color_and_black_is_white(self):
        self.assertEqual(draw_color_for_target(1), (255,60,60))
        self.assertEqual(draw_color_for_target(2), (255,255,0))
        self.assertEqual(draw_color_for_target(3), (70,110,255))
        self.assertEqual(draw_color_for_target(4), (0,255,80))
        self.assertEqual(draw_color_for_target(5), (255,255,255))
        self.assertEqual(draw_color_for_target(6), (0,255,255))

    def test_native_ring_candidates_uses_configured_circle_search_and_clusters(self):
        class Circle:
            def __init__(self,x,y,r): self.values=(x,y,r)
            def x(self): return self.values[0]
            def y(self): return self.values[1]
            def r(self): return self.values[2]
        class Frame:
            def find_circles(self, **kwargs):
                self.kwargs=kwargs
                return [Circle(120,160,24),Circle(121,161,25),Circle(320,160,24)]
        frame=Frame(); context=self.session.draft['contexts']['RING_PROCESS']
        circles,groups,reason=native_ring_candidates(frame,context)
        self.assertEqual(len(circles),3)
        self.assertEqual(len(groups),2)
        self.assertEqual(reason,'RING_CIRCLE_PROPOSALS')
        self.assertEqual(frame.kwargs['roi'],tuple(context['roi']))
        self.assertEqual(frame.kwargs['r_min'],context['r_min'])

    def test_native_ring_candidates_keeps_detector_failure_explicit(self):
        class Frame:
            def find_circles(self, **kwargs): raise RuntimeError('fixture failure')
        circles,groups,reason=native_ring_candidates(Frame(),self.session.draft['contexts']['RING_PROCESS'])
        self.assertEqual((circles,groups),([],[]))
        self.assertTrue(reason.startswith('RING_DETECTOR_ERROR:'))

    def test_circle_proposals_with_unknown_id_never_count_as_accepted_ring(self):
        observation={'kind':'RING','status':'UNCALIBRATED','reason':'ID_UNKNOWN',
                     'target_id':None,'camera_id':2,'is_simulated':False,
                     'ring_candidates':[{'center':(120,160),'support':2}],
                     'u_px':120,'v_px':160}
        self.assertFalse(ring_identity_observed(observation,1))

    def test_diagnostic_material_color_is_explicit_and_not_a_ring_identity(self):
        diagnostic=LiveCommands(self.session,'b'*32,self.temp.name+'/diagnostic',diagnostic_material_color=5)
        self.assertEqual(diagnostic.state()['diagnostic_material_color'],5)
        self.assertIsNone(self.commands.state()['diagnostic_material_color'])
        for value in (True,0,7):
            with self.assertRaises(ValueError):
                LiveCommands(self.session,'b'*32,self.temp.name+'/diagnostic',diagnostic_material_color=value)

    def test_visual_demo_is_opt_in_and_reset_never_changes_active_or_draft(self):
        self.assertIsNone(self.commands.state()['visual_demo'])
        with self.assertRaises(ValueError):
            LiveCommands(self.session,'b'*32,self.temp.name+'/demo',visual_demo=True)
        commands=LiveCommands(self.session,'b'*32,self.temp.name+'/demo',diagnostic_material_color=1,visual_demo=True)
        before=clone(self.session.draft);active=clone(self.session.active)
        commands.visual_demo.state='SIM_PLACED';commands.visual_demo.simulated_places=1
        raw=json.dumps({'action':'demo_reset'})
        result=commands.handle({'session':'b'*32,'seq':1,'payload':raw,'sha256':sha(raw.encode())})
        self.assertTrue(result['ok'])
        self.assertEqual(result['state']['visual_demo']['state'],'WAIT_MATERIAL')
        self.assertFalse(result['state']['visual_demo']['motion_authorized'])
        self.assertEqual(self.session.draft,before);self.assertEqual(self.session.active,active)

    def test_preview_alignment_label_clears_old_offsets_on_lost_pair(self):
        raw={'error_px':5,'dx_px':3,'dy_px':-4,'status':'UNKNOWN'}
        self.assertEqual(alignment_preview_label(raw),'PX dx=3.0 dy=-4.0 RAW')
        self.assertIn('STABLE',alignment_preview_label(dict(raw,status='MEASURED')))
        self.assertEqual(alignment_preview_label({'status':'UNKNOWN','quality':None}),'ALIGN UNKNOWN')

    def test_rotated_near_circle_overlay_preserves_fitted_axes_without_native_distortion(self):
        lines=[];crosses=[]
        class Canvas:
            def draw_ellipse(self,*a,**k):raise AssertionError('Native rotation is not used')
            def draw_line(self,*a,**k):lines.append(a)
            def draw_cross(self,*a,**k):crosses.append(a)
        draw_geometry(Canvas(),[{'center':[100,110],'semiaxes':[71,68],'angle_rad':.537}],
                      y_offset=48,fast_outline=True)
        self.assertEqual(len(lines),24)
        points=[p for line in lines for p in ((line[0],line[1]),(line[2],line[3]))]
        w=max(p[0] for p in points)-min(p[0] for p in points)
        h=max(p[1] for p in points)-min(p[1] for p in points)
        self.assertLess(max(w,h)/min(w,h),1.1)
        self.assertEqual(crosses,[(100,158)])

    def test_native_main_draws_successful_alignment_without_overwriting_image_binding(self):
        from k230.tuning import live
        self.session.select('RING','PROCESS')
        class Canvas:
            def clear(self):pass
            def draw_image(self,*a,**k):pass
            def draw_string_advanced(self,*a,**k):pass
            def draw_line(self,*a,**k):pass
            def draw_cross(self,*a,**k):pass
            def compress_for_ide(self,*a,**k):pass
        class Frame:
            def width(self):return 640
            def height(self):return 360
        class Sensor:
            def snapshot(self,**kwargs):return Frame()
        shown=[]
        class Display:
            @staticmethod
            def show_image(canvas):shown.append(canvas)
        session=self.session
        class App:
            def __init__(self,root):
                self.session=session;self.frame_id=0;self.canvas=Canvas()
                self.sensors={2:Sensor()};self.errors={};self.cleanup_errors=[];self.Display=Display
            def open_display(self,*a,**k):
                self.applied_geometry=(str(session.active['binding']),str(session.active['camera']),True)
            def close(self):pass
        geometry={'accepted':[],'components':[],'reason':'CONCENTRIC_EDGE_GEOMETRY',
            'ring_candidates':[],'ring_groups':[],
            'alignment_pair':{'material':{'center':[103,104]},'target':{'center':[100,100]}},
            'material_by_color':[{'color_id':2,'candidate_count':2,'top_count':1},
                                 {'color_id':6,'candidate_count':1,'top_count':0}],
            'material_fits':[]}
        clock=[0]
        def ticks():clock[0]+=10;return clock[0]
        with patch.dict('sys.modules',{'image':types.SimpleNamespace(Image=lambda *a:Canvas(),RGB565=0)}), \
             patch.object(live,'TuningApp',App), \
             patch.object(live,'native_views',return_value=(Canvas(),Canvas(),geometry)), \
             patch.object(live,'resource_snapshot',return_value={}), \
             patch.object(live.os,'exitpoint',lambda *a:None,create=True), \
             patch.object(live.os,'EXITPOINT_ENABLE',1,create=True), \
             patch.object(live.time,'ticks_ms',ticks,create=True), \
             patch.object(live.time,'ticks_diff',lambda a,b:a-b,create=True), \
             patch.object(live.time,'sleep_ms',lambda a:None,create=True), \
             patch.object(live.gc,'mem_free',lambda:2000000,create=True), \
             patch('builtins.print') as output:
            live.main('c'*32,root=self.session.store.root,mailbox=self.temp.name+'/main-mailbox',
                      max_frames=3,diagnostic_material_color=1,initial_state={
                          'profile_root':self.session.store.root,'active_profile':clone(self.session.active),
                          'profile':clone(self.session.draft),'dirty':self.session.dirty,'slot':self.session.slot,
                          'mode':'RING','station':'PROCESS','color':1,'render_mode':'SELECTED'})
            live.main('d'*32,root=self.session.store.root,mailbox=self.temp.name+'/annotated-mailbox',
                      max_frames=2,diagnostic_material_color=1,annotated_preview=True,initial_state={
                          'profile_root':self.session.store.root,'active_profile':clone(self.session.active),
                          'profile':clone(self.session.draft),'dirty':self.session.dirty,'slot':self.session.slot,
                          'mode':'RING','station':'PROCESS','color':1,'render_mode':'SELECTED'})
            # Exercise the real loop's full scan -> producer -> periodic report.
            session.draft=clone(session.draft)
            for color_id in range(1,7):
                session.draft['contexts']['TOP_BUFFER']['thresholds'][str(color_id)]=[0,30,-5,5,-5,5]
            geometry.update(material_complete=True,material_limit_exceeded=False,
                raw_material_fit_count=1,material_algorithm_revision=session.draft['revision'],
                material_scan_colors=[1,2,3,4,5,6],
                material_by_color=[dict(color_id=i,candidate_count=1 if i==1 else 0,
                    top_count=1 if i==1 else 0,scan_complete=True,threshold_available=True) for i in range(1,7)],
                material_fits=[dict(accepted=True,reason='GEOMETRY_ONLY',color_id=1,
                    center=[150,150],semiaxes=[20,16],angle_rad=.2)])
            live.main('e'*32,root=self.session.store.root,mailbox=self.temp.name+'/material-mailbox',
                      max_frames=12,diagnostic_material_colors=[1,2,3,4,5,6],annotated_preview=True,
                      initial_state={'profile_root':self.session.store.root,
                          'active_profile':clone(self.session.active),'profile':clone(self.session.draft),
                          'dirty':self.session.dirty,'slot':self.session.slot,
                          'mode':'RING','station':'PROCESS','color':1,'render_mode':'SELECTED'})
        lines=[c.args[0] for c in output.call_args_list]
        self.assertFalse(any(l.startswith('LIVE_TUNING_ERROR:') for l in lines),
                         [l for l in lines if l.startswith('LIVE_TUNING_ERROR:')])
        self.assertEqual(len(shown),15)
        closed=json.loads(next(l.split(':',1)[1] for l in lines if l.startswith('LIVE_TUNING_CLOSED:')))
        self.assertEqual(closed['errors'],0)
        self.assertEqual(closed['native_frames'],3)
        reports=[json.loads(l.split(':',1)[1]) for l in lines if l.startswith('LIVE_TUNING_FRAME:')]
        self.assertTrue(reports)
        annotated=[r for r in reports if r['layout']=='annotated']
        self.assertTrue(annotated)
        self.assertTrue(all(r['preview_width']==320 and r['preview_interval_frames']==1 for r in annotated))
        for report in reports:
            if report['session']=='e'*32:continue
            count=report['window']['recognition_frames'];colors=report['window']['material_by_color']
            self.assertEqual(colors['2']['top_frames'],count)
            self.assertEqual(colors['2']['candidate_instances'],2*count)
            self.assertEqual(colors['6']['candidate_frames'],count)
            self.assertEqual(colors['6']['top_frames'],0)
        full=[r for r in reports if r['session']=='e'*32]
        self.assertTrue(full)
        self.assertEqual(full[-1]['material_snapshot']['status'],'OK')
        self.assertTrue(full[-1]['material_snapshot']['objects'][0]['stable'])
        contracts=[json.loads(line.split(':',1)[1]) for line in lines if line.startswith('MATERIAL_SNAPSHOT_CONTRACT:')]
        self.assertEqual(full[-1]['material_snapshot']['profile_id'],contracts[-1]['profile_id'])
        self.assertEqual(full[-1]['material_snapshot']['source_session'],sha(('material-source-v1:'+'e'*32).encode())[:32])
        self.assertNotEqual(full[-1]['material_snapshot']['source_session'],'e'*32)
        self.assertEqual(contracts[-1]['source_session'],full[-1]['material_snapshot']['source_session'])

    def test_paper_and_material_paths_use_separate_rois_and_keep_unpaired_detections(self):
        import numpy as np
        from k230.tuning import live
        from vision import ring_native
        profile=clone(self.session.draft)
        profile['contexts']['RING_PROCESS']['roi']=[80,40,400,240]
        profile['contexts']['TOP_BUFFER']['thresholds']['1']=[20,60,20,60,10,50]
        paper_fit={'center':[150,130],'semiaxes':[50,45],'angle_rad':.2,'rms_px':.2}
        material_fit={'center':[152,130],'semiaxes':[20,18],'angle_rad':.1,'rms_px':.2}
        group={'center':[150,130],'support':2,'boundaries':[paper_fit],'identity_verified':False}
        class Blob:
            def rect(self):return (130,110,40,40)
            def code(self):return 3
        class Image:
            def __init__(self,w=640,h=360,blobs=()):self.array=np.zeros((h,w),np.uint8);self.blobs=blobs;self.blob_rois=[]
            def copy(self,roi=None):return Image(roi[2],roi[3]) if roi else Image()
            def clear(self):pass
            def draw_image(self,*a,**k):pass
            def to_grayscale(self):return self
            def to_rgb565(self):return self
            def to_numpy_ref(self):return self.array
            def binary(self,*a):return self
            def find_blobs(self,*a,**k):self.blob_rois.append(k['roi']);return list(self.blobs)
            def draw_line(self,*a,**k):pass
            def draw_cross(self,*a,**k):pass
        for paper_present,material_present,top_valid in ((True,False,False),(False,True,True),(True,True,True),(True,True,False)):
            frame=Image(blobs=[Blob()] if material_present else [])
            paper_result={'complete':True,'groups':[group] if paper_present else [],
                'boundaries':[paper_fit] if paper_present else [],'edges':np.zeros((240,400),np.uint8),
                'reason':'INDEPENDENT_PAPER_GEOMETRY' if paper_present else 'NO_INDEPENDENT_PAPER_TARGET'}
            with self.subTest(paper=paper_present,material=material_present,top=top_valid), \
                 patch.object(ring_native,'detect_paper_targets',return_value=paper_result) as paper_detector, \
                 patch.object(live,'extract_mask_ellipses',return_value={'complete':True,
                     'accepted':[material_fit] if top_valid else [],'components':[]}):
                _,_,g=live.native_views(frame,profile,'RING','PROCESS',1,selected=True,diagnostic_material_color=1)
            self.assertEqual(paper_detector.call_count,1)
            self.assertEqual(paper_detector.call_args.args[1],(80,40,400,240))
            self.assertEqual(frame.blob_rois,[(0,0,640,360)])
            self.assertEqual(bool(g['ring_groups']),paper_present)
            self.assertEqual(bool(g['material_color_candidates']),material_present)
            self.assertEqual(bool(g['material_fits']),material_present and top_valid)
            self.assertEqual(g['alignment_pair']['target'] is not None,paper_present and material_present and top_valid)

        # Mixed diagnostics share one paper pass, retain per-color results and
        # never choose one colored material as a simulated placement target.
        profile['contexts']['TOP_BUFFER']['thresholds']['2']=[70,97,-23,-8,67,87]
        frame=Image(blobs=[Blob()])
        with patch.object(ring_native,'detect_paper_targets',return_value=paper_result) as paper_detector, \
             patch.object(live,'draw_material_candidates') as candidate_draw, \
             patch.object(live,'extract_mask_ellipses',side_effect=[
                 {'complete':True,'accepted':[material_fit],'components':[]},
                 {'complete':True,'accepted':[],'components':[]} ]):
            _,_,g=live.native_views(frame,profile,'RING','PROCESS',1,selected=True,diagnostic_material_colors=[1,2])
        self.assertEqual(paper_detector.call_count,1)
        self.assertEqual(frame.blob_rois,[(0,0,640,360)])
        self.assertEqual([r['color_id'] for r in g['material_color_candidates']],[1,2])
        self.assertEqual([(r['color_id'],r['top_count']) for r in g['material_by_color']],[(1,1),(2,0)])
        self.assertEqual(g['material_fits'][0]['color_id'],1)
        self.assertNotIn('color_id',material_fit)
        self.assertIsNone(g['alignment_pair'])
        candidate_draw.assert_not_called()

        class HugeBlob(Blob):
            def rect(self):return (0,0,640,360)
            def code(self):return 1
        profile['contexts']['TOP_BUFFER']['thresholds']['5']=[14,23,-7,2,-6,7]
        with patch.object(ring_native,'detect_paper_targets',return_value=paper_result), \
             patch.object(live,'extract_mask_ellipses') as contours:
            _,_,g=live.native_views(Image(blobs=[HugeBlob()]),profile,'RING','PROCESS',1,
                                  selected=True,diagnostic_material_colors=[5],annotated_preview=True)
        contours.assert_not_called()
        self.assertEqual(g['material_by_color'][0]['candidate_count'],1)
        self.assertEqual(g['material_by_color'][0]['oversize_rejected'],1)
        self.assertEqual(g['material_fits'],[])

        # A rejected black body must stay rejected while an independent measured
        # luminance edge may supply its top; it cannot generate a paper target.
        from vision import ellipse_edges
        body={'bbox':[130,110,40,40],'fit':{'accepted':False,'reason':'FULL_BOUNDARY_RESIDUAL'}}
        with patch.object(ring_native,'detect_paper_targets',return_value=dict(paper_result,groups=[],boundaries=[])), \
             patch.object(live,'extract_mask_ellipses',return_value={'complete':True,'accepted':[],'components':[body]}), \
             patch.object(ellipse_edges,'extract_dark_edge_ellipses',return_value={'complete':True,
                 'accepted':[dict(material_fit,backend='LUMA_EDGE_ELLIPSE')],'components':[]}) as edge_detector:
            _,_,g=live.native_views(Image(blobs=[Blob()]),profile,'RING','PROCESS',1,
                                   selected=True,diagnostic_material_colors=[5])
        edge_detector.assert_called_once()
        self.assertEqual(g['ring_groups'],[])
        self.assertEqual(g['material_fits'][0]['backend'],'LUMA_EDGE_ELLIPSE')
        self.assertEqual(g['material_fits'][0]['color_id'],5)
        self.assertEqual(g['material_by_color'][0]['top_count'],1)
        self.assertFalse(body['fit']['accepted'])

    def test_mixed_diagnostics_validate_colors_and_cannot_enable_simulated_placement(self):
        from k230.tuning.live import diagnostic_colors
        for bad in ([],[1,1],[True],[0],[7],'123'):
            with self.subTest(value=bad),self.assertRaises(ValueError):diagnostic_colors(multiple=bad)
        with self.assertRaises(ValueError):diagnostic_colors(1,[2])
        with self.assertRaises(ValueError):
            LiveCommands(self.session,'b'*32,self.temp.name+'/mixed',visual_demo=True,diagnostic_material_colors=[1,2])
        commands=LiveCommands(self.session,'b'*32,self.temp.name+'/mixed',diagnostic_material_colors=[1,2,3,4,5,6])
        self.assertIsNone(commands.state()['visual_demo'])
        self.assertEqual(commands.state()['diagnostic_material_colors'],[1,2,3,4,5,6])

    def test_more_than_three_materials_are_unknown_instead_of_silently_choosing_three(self):
        from k230.tuning.live import bounded_material_results
        fits=[{'color_id':i} for i in (1,2,3)]
        rows=[{'color_id':i,'top_count':1} for i in (1,2,3)]
        self.assertEqual(bounded_material_results(fits,rows),(fits,False))
        result,overflow=bounded_material_results(fits+[{'color_id':4}],rows)
        self.assertTrue(overflow);self.assertEqual(result,[])
        self.assertTrue(all(r['top_count']==0 and r['raw_top_count']==1 for r in rows))

    def test_mixed_overlays_use_each_fit_and_candidate_color_including_white_black(self):
        from k230.tuning.live import draw_material_candidates
        inks=[]
        class Canvas:
            def draw_line(self,*a,**k):inks.append(k['color'])
            def draw_cross(self,*a,**k):inks.append(k['color'])
        fit={'center':[100,110],'semiaxes':[20,18],'angle_rad':0}
        draw_geometry(Canvas(),[dict(fit,color_id=2),dict(fit,color_id=5)])
        self.assertEqual(set(inks),{(255,255,0),(255,255,255)})
        inks.clear()
        draw_material_candidates(Canvas(),[{'bbox':[10,10,20,20],'color_id':3},{'bbox':[40,10,20,20],'color_id':6}],None)
        self.assertEqual(set(inks),{(70,110,255),(0,255,255)})

    def test_reviewed_pixel_identity_is_distinct_from_mm_pose_or_placement_score(self):
        observation={'kind':'RING','status':'UNCALIBRATED','reason':'CALIBRATION_REQUIRED',
                     'target_id':1,'camera_id':2,'is_simulated':False,
                     'ring_support':2,'u_px':120,'v_px':160,
                     'x_mm':None,'target_pose_valid':False}
        self.assertTrue(ring_identity_observed(observation,1))
        for key,value in [('target_id',2),('camera_id',1),('is_simulated',True),
                          ('u_px',float('nan')),('ring_support',True),('reason','ID_UNKNOWN')]:
            changed=dict(observation);changed[key]=value
            with self.subTest(field=key):self.assertFalse(ring_identity_observed(changed,1))
