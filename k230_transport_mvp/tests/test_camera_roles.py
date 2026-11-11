import unittest
from k230.block_profile import make_config,PARAMETERS,parameters_for_camera
from k230.visual_inspector import VisualInspector
from vision.task_preview import TaskPreview


class RoleTests(unittest.TestCase):
    def test_current_roles_are_consistent_across_visual_modes(self):
        config=make_config()
        for mode in ('COLOR','RING','TOP'):self.assertEqual(config['MODE_CAMERA'][mode],2)
        for mode in ('QR','LANE'):self.assertEqual(config['MODE_CAMERA'][mode],1)
        self.assertEqual(config['CAMERA_PRIORITY'],(2,1,0))
        self.assertEqual(config['CAMERA_INIT_ORDER'],(2,1,0))
        self.assertEqual(config['CAMERA_QUARANTINE'],(0,))
        with self.assertRaises(ValueError):make_config(1,1)
        with self.assertRaises(ValueError):make_config(True,2)
        inspector=VisualInspector()
        for seq,mode in enumerate(('COLOR','TOP','RING','QR','LANE'),1):
            target=1 if mode in ('COLOR','TOP','RING') else None
            station='BUFFER' if mode in ('TOP','RING') else 'RAW' if mode=='COLOR' else None
            inspector.apply({'seq':seq,'mode':mode,'target_id':target,'station':station})
            self.assertEqual(inspector.camera_id(),config['MODE_CAMERA'][mode])

    def test_old_measured_profile_is_not_relabelled_as_new_camera_measurement(self):
        active=parameters_for_camera(2)
        self.assertEqual(PARAMETERS['camera_id'],2)
        self.assertEqual(active['measurement_camera_id'],2)
        self.assertTrue(active['sampling_verified'])
        self.assertNotIn('provisional',active['profile_id'])
        self.assertFalse(parameters_for_camera(1)['sampling_verified'])

    def test_new_scanner_rejects_old_camera_but_historical_restore_keeps_real_source(self):
        old=TaskPreview(qr_camera_id=1)
        for i in range(1,4):old.scan(['156+123+516+231'],i,i*40)
        current=TaskPreview(qr_camera_id=2)
        current.restore_confirmed(old.summary())
        self.assertEqual(current.summary()['accepted_camera_id'],1)
        self.assertEqual(current.summary()['scan_camera_id'],2)
        self.assertEqual(current.summary()['source'],'REAL_CSI1_QR')
        self.assertTrue(current.summary()['restored_confirmed_task'])
        fresh=TaskPreview(qr_camera_id=2)
        with self.assertRaises(ValueError):fresh.scan(['156+123+516+231'],1,40,camera_id=1)
        for i in range(1,4):fresh.scan(['156+123+516+231'],i,i*40,camera_id=2)
        self.assertEqual(fresh.summary()['source'],'REAL_CSI2_QR')
        self.assertEqual(fresh.summary()['confirmed_physical_steps'],0)


if __name__=='__main__':unittest.main()
