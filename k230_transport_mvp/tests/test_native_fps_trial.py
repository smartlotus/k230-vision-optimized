"""Native-trial protection and deployment stop conditions; host doubles only."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from k230.tuning.schema import defaults
from tools.native_fps_trial import trial_state,run
from tools.deploy_native_trial import deploy
from k230.tuning import live_optimized


class NativeTrialTests(unittest.TestCase):
    def test_single_preview_empty_frame_reuses_pixels_without_mask_copies(self):
        from tools.benchmark_live_optimized import fixture
        frame,profile=fixture(False)
        before=frame.data.copy()
        mask,retained,geometry=live_optimized.native_views(frame,profile,'TOP','BUFFER',2,
            selected=True,annotated_preview=True,render=True)
        self.assertIsNone(mask);self.assertIs(retained,frame)
        self.assertEqual(frame.counters['copies'],0)
        self.assertTrue((frame.data==before).all())
        self.assertFalse(geometry['accepted'])

    def test_native_62bit_ticks_extend_uint32_wrap_without_replay(self):
        with patch.object(live_optimized.time,'ticks_add',return_value=(1<<62)-1,create=True):
            clock=live_optimized.material_source_clock()
        self.assertEqual(clock.period,1<<32)
        raw=[(1<<40)+(1<<32)-2,(1<<40)+(1<<32)-1,(1<<40)+(1<<32)+1]
        self.assertEqual([clock.update(t & (clock.period-1)) for t in raw],[(1<<32)-2,(1<<32)-1,1])
        with self.assertRaises(ValueError):clock.update(1)
        with patch.object(live_optimized.time,'ticks_add',return_value=1023,create=True):
            self.assertEqual(live_optimized.material_source_clock().period,1024)
    def state(self):
        active=defaults();draft=copy.deepcopy(active);draft['revision']=25
        return dict(active_profile=active,profile=draft,profile_root='/sdcard/original',
                    dirty=True,mode='RING',station='PROCESS',color=1)

    def test_protected_active_and_draft_are_cloned_separately(self):
        before=self.state();snapshot=copy.deepcopy(before)
        result=trial_state(before,'/sdcard/transport_mvp_optimized_test','TOP',2)
        self.assertEqual(before,snapshot)
        self.assertEqual(result['active_profile'],before['active_profile'])
        self.assertEqual(result['profile'],before['profile'])
        result['profile']['revision']=99
        self.assertEqual(before['profile']['revision'],25)
        self.assertEqual(result['mode'],'TOP');self.assertEqual(result['station'],'BUFFER')
        self.assertEqual(result['profile_root'],'/sdcard/transport_mvp_optimized_test/profiles')

    def test_invalid_native_budget_rejects_before_file_or_hardware_access(self):
        for args in ({'frames':0},{'frames':301},{'duration_ms':999},{'variant':'other'}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                run('/does/not/exist','a'*32,**args)

    def test_invalid_deploy_root_never_creates_client(self):
        with patch('tools.deploy_native_trial.CanMvClient') as client:
            with self.assertRaises(ValueError):deploy('/sdcard/transport_mvp',Path('unused'))
            client.assert_not_called()

    def test_busy_or_existing_install_never_gets_written(self):
        for replies in ([{'running':True}],[{'running':False},{'exists':True}]):
            with patch('tools.deploy_native_trial.CanMvClient') as factory:
                client=factory.return_value;client.tool.side_effect=replies
                with self.assertRaises(RuntimeError):deploy('/sdcard/transport_mvp_optimized_test',Path('unused'))
                self.assertFalse(any(call.args[0]=='canmv_write_file' for call in client.tool.call_args_list))
