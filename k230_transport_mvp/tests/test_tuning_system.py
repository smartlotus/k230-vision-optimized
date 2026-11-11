import json,tempfile,unittest
import os,time
from pathlib import Path
from unittest.mock import patch
from k230.tuning.schema import defaults,validate,compile_config,clone,image_profile
from k230.tuning.geometry import Viewport
from k230.tuning.store import ProfileStore
from k230.tuning.model import TuningSession
from k230.tuning.panel import Panel,fields
from test_block_detector import Blob,Image


class TuningGeometryTests(unittest.TestCase):
    def test_fit_preserves_both_axes_and_rejects_letterbox(self):
        view=Viewport(1920,1080,rect=(0,0,800,480),mode='FIT')
        self.assertEqual((view.w,view.h),(800,450))
        self.assertIsNone(view.image_point(100,5))
        for point in ((0,0),(960,540),(1800,1000)):
            actual=view.image_point(*view.screen_point(*point))
            self.assertLessEqual(abs(actual[0]-point[0]),2.5)
            self.assertLessEqual(abs(actual[1]-point[1]),2.5)
        self.assertEqual(view.screen_roi([960,540,240,240])[2:],(100,100))

    def test_native_pan_and_roi_keep_source_coordinates(self):
        view=Viewport(1920,1080,pan=(1000,500))
        self.assertEqual(view.image_point(0,48),(1000,500))
        roi=view.drag_roi((10,58),(109,157))
        self.assertEqual(roi,[1010,510,100,100])
        self.assertEqual(view.screen_roi(roi),(10,58,100,100))
        self.assertIsNone(view.drag_roi((10,58),(700,157)))
        small=Viewport(320,180)
        self.assertEqual((small.x,small.y),(160,138))
        self.assertIsNone(small.image_point(0,48))


class TuningStoreTests(unittest.TestCase):
    def test_selected_slot_and_locked_service_use_the_same_saved_bytes(self):
        from k230.tuning.integration import load_locked_config
        with tempfile.TemporaryDirectory() as root:
            store=ProfileStore(root);s=TuningSession(store,slot=2);s.enable_threshold();s.save()
            self.assertEqual(TuningSession(ProfileStore(root)).slot,2)
            with self.assertRaises(ValueError):load_locked_config(root)
            s.set_locked(True);record=s.save()
            config=load_locked_config(root)
            self.assertEqual(config['TUNING_SHA256'],record['sha256'])
            self.assertEqual(config['TUNING_SLOT'],2);self.assertEqual(config['TUNING_GENERATION'],2)
            self.assertEqual(config['WIDTH'],640);self.assertEqual(config['HEIGHT'],360)
            self.assertEqual(config['TOP_CONFIG']['BUFFER']['thresholds'][5],[0,100,-128,127,-128,127])
            self.assertFalse(config['UART_ENABLED']);self.assertEqual(config['CALIBRATIONS'],{})

    def test_generation_checksum_and_corrupt_or_partial_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            store=ProfileStore(root);p=defaults();first=store.save(p)
            p['revision']=1;second=store.save(p)
            Path(root,second['file']).write_text('{broken',encoding='utf8')
            Path(root,'slot1-00000003.json.tmp').write_text('partial')
            self.assertEqual(store.load()['generation'],first['generation'])
            self.assertEqual(len(store.warnings),1)
            third=store.save(p)
            self.assertEqual(third['generation'],3)
            self.assertTrue(Path(root,first['file']).exists())

    def test_save_failure_does_not_erase_previous_or_clear_dirty(self):
        with tempfile.TemporaryDirectory() as root:
            store=ProfileStore(root);session=TuningSession(store);session.save();session.enable_threshold()
            with patch('k230.tuning.store.os.rename',side_effect=OSError('SD write failure')):
                with self.assertRaises(OSError):session.save()
            self.assertTrue(session.dirty)
            restored=TuningSession(ProfileStore(root))
            self.assertIsNone(restored.context()['thresholds']['5'])

    def test_slot_isolation_and_payload_type_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            store=ProfileStore(root);store.save(defaults(),1)
            bad={'schema_version':1,'generation':2,'sha256':'x','payload':123}
            Path(root,'slot1-00000002.json').write_text(json.dumps(bad))
            self.assertEqual(store.load(1)['generation'],1)
            self.assertIsNone(store.load(2))
            with self.assertRaises(ValueError):store.save(defaults(),True)


class TuningSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.session=TuningSession(ProfileStore(self.tmp.name));self.panel=Panel(self.session)

    def test_draft_station_isolation_apply_clears_adapter_and_no_mm(self):
        s=self.session;s.select('COLOR','RAW');s.target_id=1;s.enable_threshold()
        self.assertIsNone(s.active['contexts']['COLOR_RAW']['thresholds']['1'])
        self.assertIsNone(s.draft['contexts']['COLOR_PROCESS']['thresholds']['1'])
        self.panel.set_field(['contexts','COLOR_RAW','thresholds','1',0],20)
        old=s.adapter;s.apply()
        self.assertIsNot(s.adapter,old)
        p=s.adapter.config['COLOR_CONFIG']['RAW']['block_parameters']
        self.assertEqual(p['thresholds'][1][0],20)
        self.assertIsNone(s.adapter.config['COLOR_CONFIG']['PROCESS']['block_parameters']['thresholds'][1])
        class TunedImage(Image):
            def width(self):return 640
            def height(self):return 360
        im=TunedImage([Blob(1,(100,70,40,40),1200)])
        packet={'image':im,'camera_id':2,'frame_id':1,'timestamp_ms':40,'is_simulated':True,'image_profile':image_profile(s.active)}
        result=s.process(packet)
        self.assertEqual(result['u_px'],120);self.assertIsNone(result['x_mm']);self.assertFalse(result['target_pose_valid'])
        s.apply();self.assertIsNone(s.observation);self.assertEqual(s.adapter.blocks,{})

    def test_invalid_value_lock_and_run_reject_transactionally(self):
        s=self.session;s.enable_threshold();before=clone(s.draft)
        with self.assertRaises(ValueError):self.panel.set_field(['contexts','TOP_BUFFER','thresholds','5',0],101)
        self.assertEqual(s.draft,before)
        s.set_locked(True)
        with self.assertRaises(ValueError):s.enable_threshold()
        s.running=True
        for action in (lambda:s.set_locked(False),s.apply,s.save,lambda:s.select('QR','FRONT')):
            with self.assertRaises(ValueError):action()

    def test_resolution_invalidation_and_display_only_change(self):
        s=self.session;s.draft['binding']['pose_id']='POSE_1'
        old_roi=list(s.context()['roi']);self.panel.dispatch('fit')
        self.assertEqual(s.draft['binding']['pose_id'],'POSE_1');self.assertEqual(s.context()['roi'],old_roi)
        s.set_resolution(320,180);s.apply()
        self.assertIsNone(s.active['binding']['pose_id'])
        self.assertEqual(s.context()['roi'],[0,0,320,180])
        self.assertEqual(s.adapter.config['CALIBRATIONS'],{})

    def test_schema_rejects_safety_fields_and_wrong_color_ids(self):
        for key,value in (('H',[[1,0,0]]),('UART_ENABLED',True),('MODE_CAMERA',{'TOP':0})):
            p=defaults();p[key]=value
            with self.assertRaises(ValueError):validate(p)
        p=defaults();p['contexts']['COLOR_RAW']['thresholds']['7']=None
        with self.assertRaises(ValueError):validate(p)
        p=defaults();p['qr']['confirmations']=1
        with self.assertRaises(ValueError):validate(p)

    def test_context_parameter_meanings_are_not_conflated(self):
        s=self.session;s.select('QR','FRONT')
        self.assertTrue(all(f[0][0]=='qr' for f in fields(s,'PARAM')))
        s.select('LANE','FRONT');self.assertIn('gray_min',[f[0][-1] for f in fields(s,'PARAM')])
        s.select('RING','PROCESS');self.assertIn('circle_threshold',[f[0][-1] for f in fields(s,'PARAM')])

    def test_gesture_uses_same_transform_and_rejects_toolbar(self):
        panel=self.panel;panel.page='ROI';panel.viewport=Viewport(640,360)
        panel.render_signature=panel.image_signature()
        panel.handle_touch('down',50,98);panel.handle_touch('up',149,197)
        self.assertEqual(self.session.context()['roi'],[50,50,100,100])
        before=clone(self.session.draft)
        panel.handle_touch('down',50,98);panel.handle_touch('up',750,197)
        self.assertEqual(self.session.draft,before)

    def test_display_only_changes_do_not_apply_other_dirty_drafts(self):
        s=self.session;s.enable_threshold();s.set_roi([80,40,120,120])
        self.panel.set_field(['camera','exposure_us'],12000)
        before=clone(s.active);adapter=s.adapter;serial=s.apply_serial
        self.panel.dispatch('fit');self.panel.dispatch('pan',(0,0))
        for key in ('contexts','camera','binding'):
            self.assertEqual(s.active[key],before[key])
        self.assertIs(s.adapter,adapter);self.assertEqual(s.apply_serial,serial)
        self.assertEqual(s.draft['contexts']['TOP_BUFFER']['thresholds']['5'],[0,100,-128,127,-128,127])
        self.assertTrue(s.dirty)

    def test_saved_lock_cannot_be_bypassed_by_switching_slot(self):
        s=self.session;s.set_locked(True);s.save()
        self.assertFalse(s.dirty)
        self.panel.dispatch('slot',2)
        self.assertEqual(s.slot,1);self.assertTrue(s.active['locked'])

    def test_running_controller_cannot_be_interrupted_by_preview_controls(self):
        s=self.session;s.running=True
        self.panel.dispatch('live');self.panel.dispatch('freeze')
        self.assertFalse(self.panel.live_requested);self.assertFalse(self.panel.frozen)

    def test_roi_rejects_unapplied_geometry_and_unpainted_transform(self):
        s=self.session;p=self.panel;p.page='ROI';p.viewport=Viewport(640,360)
        p.render_signature=p.image_signature();s.set_resolution(1280,720)
        before=clone(s.draft)
        p.handle_touch('down',100,148);p.handle_touch('up',199,247)
        self.assertEqual(s.draft,before)
        s.apply();p.viewport=Viewport(1280,720);p.render_signature=p.image_signature()
        p.handle_touch('down',100,148);p.dispatch('fit');p.handle_touch('up',199,247)
        self.assertEqual(s.context()['roi'],[0,0,1280,720])

    def test_separate_input_sources_cannot_complete_each_others_gesture(self):
        p=self.panel;p.page='ROI';p.viewport=Viewport(640,360);p.render_signature=p.image_signature()
        before=clone(self.session.draft)
        p.handle_touch('down',100,148,('physical',1));p.handle_touch('up',199,247,('ide',1))
        self.assertEqual(self.session.draft,before)
        p.handle_touch('up',199,247,('physical',1))
        self.assertEqual(self.session.context()['roi'],[100,100,100,100])


class TuningRuntimeTests(unittest.TestCase):
    def test_demo_event_loop_does_not_capture_or_detect(self):
        from types import SimpleNamespace
        from k230.tuning.runtime import TuningApp
        class Canvas:
            def __getattr__(self,name):return lambda *a,**k:None
        with tempfile.TemporaryDirectory() as root:
            app=TuningApp(root,demo=True);app.canvas=Canvas();app.TOUCH=SimpleNamespace(EVENT_DOWN=1,EVENT_UP=2,EVENT_MOVE=3)
            app.panel.live_requested=True
            app.Display=SimpleNamespace(show_image=lambda im:None)
            app.applied_geometry=(str(app.session.active['binding']),str(app.session.active['camera']),False)
            app.session.process=lambda frame:self.fail('DEMO must not detect')
            app.sensors={1:SimpleNamespace(snapshot=lambda **kw:self.fail('DEMO must not capture'))}
            with patch.object(time,'ticks_ms',return_value=1000,create=True),patch.object(time,'ticks_diff',side_effect=lambda a,b:a-b,create=True),patch.object(time,'sleep_ms',create=True),patch.object(os,'exitpoint',create=True):
                app.tick()
            self.assertIsNone(app.frame);self.assertEqual(app.frame_id,0);self.assertIsNone(app.session.observation)

    def test_grayscale_detection_does_not_mutate_displayed_raw_frame(self):
        from types import SimpleNamespace
        from k230.tuning.runtime import TuningApp
        class Raw:
            def __init__(self):self.gray=False;self.copies=[]
            def width(self):return 640
            def height(self):return 360
            def copy(self):other=Raw();self.copies.append(other);return other
            def to_grayscale(self):self.gray=True;return self
            def find_qrcodes(self,**kwargs):return []
        class Canvas:
            def __getattr__(self,name):return lambda *a,**k:None
        with tempfile.TemporaryDirectory() as root:
            app=TuningApp(root);app.session.select('QR','FRONT');app.canvas=Canvas();app.TOUCH=SimpleNamespace(EVENT_DOWN=1,EVENT_UP=2,EVENT_MOVE=3)
            app.panel.live_requested=True;app.Display=SimpleNamespace(show_image=lambda im:None)
            app.applied_geometry=(str(app.session.active['binding']),str(app.session.active['camera']),True)
            raw=Raw();app.sensors={1:SimpleNamespace(snapshot=lambda **kw:raw)}
            with patch.object(time,'ticks_ms',return_value=1000,create=True),patch.object(time,'ticks_diff',side_effect=lambda a,b:a-b,create=True),patch.object(time,'sleep_ms',create=True),patch.object(os,'exitpoint',create=True):
                app.tick()
            self.assertFalse(raw.gray);self.assertIs(app.frame,raw)
            self.assertEqual(len(raw.copies),1);self.assertTrue(raw.copies[0].gray)

    def test_display_cleanup_failure_does_not_skip_media_release(self):
        from k230.tuning.runtime import TuningApp
        calls=[]
        class Display:
            @staticmethod
            def deinit():calls.append('display');raise OSError('display fault')
        class Media:
            @staticmethod
            def deinit():calls.append('media')
        app=TuningApp.__new__(TuningApp);app.sensors={};app.display_ready=app.media_ready=True
        app.Display=Display;app.MediaManager=Media;app.cleanup_errors=[]
        app.close_display()
        self.assertEqual(calls,['display','media'])
        self.assertFalse(app.media_ready);self.assertFalse(app.display_ready)
        self.assertEqual(app.cleanup_errors,['display: display fault'])
