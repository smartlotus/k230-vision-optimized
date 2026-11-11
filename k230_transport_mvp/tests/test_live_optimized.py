"""Differential full-boundary geometry and render decoupling, host only."""
import math
import json
import tempfile
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from k230.tuning import live, live_optimized
from vision import ellipse_native as old, ellipse_native_optimized as new
from tools.benchmark_live_optimized import fixture
from k230.tuning.model import TuningSession
from k230.tuning.store import ProfileStore
from k230.tuning.schema import clone


class LiveOptimizedTests(unittest.TestCase):
    def test_real_loop_recognizes_every_frame_but_renders_only_when_due(self):
        with tempfile.TemporaryDirectory() as directory:
            session=TuningSession(ProfileStore(directory+'/profiles'))
            session.select('TOP','BUFFER')
            before_active,before_draft=clone(session.active),clone(session.draft)
            shown=[];calls=[];snapshots=[]
            class Canvas:
                def clear(self):pass
                def draw_image(self,*args,**kwargs):pass
                def draw_string_advanced(self,*args,**kwargs):pass
                def compress_for_ide(self,*args,**kwargs):pass
            class Frame:
                def width(self):return 640
                def height(self):return 360
            class Sensor:
                def snapshot(self,**kwargs):
                    frame=Frame();snapshots.append(frame);return frame
            class Display:
                @staticmethod
                def show_image(canvas):shown.append(canvas)
            class App:
                def __init__(self,root):
                    self.session=session;self.frame_id=0;self.canvas=Canvas()
                    self.sensors={2:Sensor()};self.errors={};self.cleanup_errors=[];self.Display=Display
                def open_display(self,*args,**kwargs):
                    self.applied_geometry=(str(session.active['binding']),str(session.active['camera']),True)
                def close(self):pass
            def views(frame,*args,**kwargs):
                calls.append((frame,kwargs['render']))
                geometry=dict(accepted=[],components=[],reason='NO_BLOB_GATE',ring_candidates=[],ring_groups=[])
                return (Canvas(),Canvas(),geometry) if kwargs['render'] else (None,None,geometry)
            clock=[0]
            def ticks():clock[0]+=5;return clock[0]
            with patch.dict('sys.modules',{'image':types.SimpleNamespace(Image=lambda *args:Canvas(),RGB565=0)}), \
                 patch.object(live_optimized,'TuningApp',App), \
                 patch.object(live_optimized,'native_views',side_effect=views), \
                 patch.object(live_optimized,'resource_snapshot',return_value={}), \
                 patch.object(session,'process',return_value={'status':'NOT_FOUND'}), \
                 patch.object(live_optimized.os,'exitpoint',lambda *args:None,create=True), \
                 patch.object(live_optimized.os,'EXITPOINT_ENABLE',1,create=True), \
                 patch.object(live_optimized.time,'ticks_ms',ticks,create=True), \
                 patch.object(live_optimized.time,'ticks_diff',lambda a,b:a-b,create=True), \
                 patch.object(live_optimized.time,'sleep_ms',lambda *args:None,create=True), \
                 patch.object(live_optimized.gc,'mem_free',lambda:2000000,create=True), \
                 patch('builtins.print') as output:
                live_optimized.main('f'*32,root=session.store.root,mailbox=directory+'/mailbox',max_frames=6,
                    initial_state={'profile_root':session.store.root,'active_profile':clone(session.active),
                        'profile':clone(session.draft),'dirty':session.dirty,'slot':session.slot,
                        'mode':'TOP','station':'BUFFER','color':1,'render_mode':'SELECTED'},preview_every=3)
            lines=[call.args[0] for call in output.call_args_list]
            self.assertFalse([line for line in lines if line.startswith('LIVE_TUNING_ERROR:')])
            self.assertEqual([render for _,render in calls],[False,False,True,False,False,True])
            self.assertEqual([frame for frame,_ in calls],snapshots)
            self.assertEqual(len(shown),2)
            closed=json.loads(next(line.split(':',1)[1] for line in lines if line.startswith('LIVE_TUNING_CLOSED:')))
            self.assertEqual(closed['native_frames'],6);self.assertEqual(closed['errors'],0)
            self.assertEqual(session.active,before_active);self.assertEqual(session.draft,before_draft)

    def test_ring_material_geometry_unchanged_without_render(self):
        from vision import ring_native
        results=[];copies=[]
        for module,render in ((live,True),(live_optimized,True),(live_optimized,False)):
            frame,profile=fixture()
            paper=dict(edges=np.zeros((360,640),np.uint8),groups=[],reason='NO_INDEPENDENT_PAPER_TARGET')
            profile['contexts']['RING_PROCESS']['roi']=[0,0,640,360]
            extra={} if module is live else {'render':render}
            with patch.object(ring_native,'detect_paper_targets',return_value=paper):
                result=module.native_views(frame,profile,'RING','PROCESS',1,selected=True,
                    diagnostic_material_colors=[1,2],**extra)
            results.append(result[2]);copies.append(frame.counters['copied_pixels'])
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
        self.assertLess(copies[2],copies[0])

    def test_full_boundary_matches_scalar_and_vector(self):
        rng=np.random.default_rng(20261111)
        for backend in (None,np):
            with patch.object(old,'_BOUNDARY_NP',backend),patch.object(new,'_BOUNDARY_NP',backend):
                for n in (16,48,128,512,4096):
                    for noise in (0,.3,2,5):
                        angles=np.linspace(0,2*math.pi,n,endpoint=False)
                        points=np.stack((90+35*np.cos(angles),70+24*np.sin(angles)),axis=1)
                        points+=rng.normal(0,noise,points.shape)
                        for origin in ((0,0),(10,20)):
                            kwargs=dict(origin=origin,roi=[0,0,220,180],pixel_area=math.pi*35*24,
                                        native_fit=((90,70),(70,48),0))
                            self.assertEqual(old.contour_fit(points.tolist(),**kwargs),new.contour_fit(points.tolist(),**kwargs))

    def test_native_masks_holes_clipping_and_noise_match(self):
        for seed in range(16):
            rng=np.random.default_rng(seed)
            mask=np.zeros((180,220),np.uint8)
            cv2.ellipse(mask,(90,70),(35,24),seed*10,0,360,255,-1)
            if seed%3==0:cv2.circle(mask,(90,70),20,0,-1)
            if seed%4==0:mask[70:145,75:105]=255
            mask[rng.integers(0,180,20),rng.integers(0,220,20)]=255
            for kernel in (0,3,5,7):
                self.assertEqual(old.extract_mask_ellipses(mask,[0,0,220,180],close_kernel=kernel),
                                 new.extract_mask_ellipses(mask,[0,0,220,180],close_kernel=kernel))
        points=[[1,2]]*4097
        self.assertEqual(old.contour_fit(points),new.contour_fit(points))

    def test_scalar_native_path_has_bounded_scratch_and_fallback_matches(self):
        import tracemalloc
        points=[[90+35*math.cos(i*2*math.pi/4096),70+24*math.sin(i*2*math.pi/4096)] for i in range(4096)]
        tracemalloc.start()
        try:
            actual=new.contour_fit(points,native_fit=((90,70),(70,48),0),roi=[0,0,220,180])
            peak=tracemalloc.get_traced_memory()[1]
        finally:tracemalloc.stop()
        self.assertEqual(actual,old.contour_fit(points,native_fit=((90,70),(70,48),0),roi=[0,0,220,180]))
        self.assertLess(peak,32000)
        for origin in ((0,0),(10,20)):
            self.assertEqual(old.contour_fit(points,origin=origin),new.contour_fit(points,origin=origin))

    def test_render_free_keeps_geometry_and_does_not_draw(self):
        for present in (True,False):
            legacy,profile=fixture(present)
            expected=live.native_views(legacy,profile,'TOP','BUFFER',1,selected=True)[2]
            for render in (True,False):
                frame,profile=fixture(present)
                mask,retained,actual=live_optimized.native_views(frame,profile,'TOP','BUFFER',1,selected=True,render=render)
                self.assertEqual(expected,actual)
                if not render:
                    self.assertIsNone(mask);self.assertIsNone(retained)
                    self.assertEqual(frame.counters['draw_calls'],0)
                    self.assertEqual(frame.counters['copies'],1 if present else 0)

    def test_optional_stage_timing_does_not_change_geometry(self):
        frame,profile=fixture()
        expected=live_optimized.native_views(frame,profile,'TOP','BUFFER',1,selected=True,render=False)[2]
        frame,profile=fixture()
        actual=live_optimized.native_views(frame,profile,'TOP','BUFFER',1,selected=True,render=False,profile_timing=True)[2]
        timing=actual.pop('performance_ms')
        self.assertEqual(actual,expected)
        self.assertGreaterEqual(timing['total_ms'],timing['preview_images_ms'])
        self.assertTrue(all(value>=0 for value in timing.values()))

    def test_kernel_cache_is_bounded_and_reuses_identical_backend(self):
        frame,_=fixture()
        with patch.object(new,'_KERNEL_CV',None),patch.object(new,'_CLOSE_KERNELS',{}), \
             patch.object(cv2,'getStructuringElement',wraps=cv2.getStructuringElement) as build:
            for _ in range(3):
                for kernel in (3,5,7):
                    new.extract_mask_ellipses(frame.data,[0,0,640,360],close_kernel=kernel)
            self.assertEqual(build.call_count,3)
            self.assertEqual(len(new._CLOSE_KERNELS),3)
            with self.assertRaises(ValueError):
                new.extract_mask_ellipses(frame.data,[0,0,640,360],close_kernel=9)
