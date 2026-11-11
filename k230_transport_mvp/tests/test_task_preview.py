import unittest
from vision.task_preview import TaskPreview


class TaskPreviewTests(unittest.TestCase):
    def test_real_three_frames_plan_and_selected_color_never_advance_on_loss(self):
        preview=TaskPreview()
        for i in range(1,4):preview.scan(['156+123+516+231'],i,i*250)
        self.assertEqual(preview.summary()['batch2_buffer_slots'],[2,1,3])
        self.assertEqual(preview.summary()['planned_steps'],31)
        candidates=[{'target_id':1,'stable':True,'eligible':True,'center':(80,80),'track_id':7},
                    {'target_id':5,'stable':True,'eligible':True,'center':(150,80),'track_id':8}]
        self.assertEqual(preview.selected_target(candidates)['target_id'],1)
        self.assertIsNone(preview.selected_target([]))
        self.assertEqual(preview.summary()['current_preview_target'],1)
        self.assertEqual(preview.summary()['confirmed_physical_steps'],0)

    def test_invalid_ambiguous_replayed_and_simulated_qr_do_not_submit(self):
        preview=TaskPreview()
        for i in range(1,4):preview.scan(['bad'],i,i*250)
        self.assertIsNone(preview.task)
        for i in range(4,7):preview.scan(['156+123+516+231','123+123+312+231'],i,i*250)
        self.assertIsNone(preview.task)
        for i in range(3):preview.scan(['156+123+516+231'],7,2000+i*200)
        self.assertIsNone(preview.task)
        with self.assertRaises(ValueError):preview.scan(['156+123+516+231'],8,3000,is_simulated=True)
        with self.assertRaises(ValueError):preview.scan(['156+123+516+231'],8,3000,camera_id=2)

    def test_timeout_restarts_scan_but_confirmed_task_is_never_replaced(self):
        preview=TaskPreview()
        preview.scan([],1,0)
        self.assertEqual(preview.scan([],2,5001)['reason'],'QR_TIMEOUT')
        for i in range(3,6):preview.scan(['156+123+516+231'],i,6000+i*250)
        self.assertIsNotNone(preview.task)
        for i in range(6,9):
            self.assertFalse(preview.scan(['123+123+312+231'],i,7000+i*250)['submitted'])
        self.assertEqual(preview.task.payload,'156+123+516+231')

    def test_unstable_or_clipped_candidate_is_never_selected(self):
        preview=TaskPreview()
        for i in range(1,4):preview.scan(['156+123+516+231'],i,i*250)
        self.assertIsNone(preview.selected_target([{'target_id':1,'stable':False,'eligible':True}]))
        self.assertIsNone(preview.selected_target([{'target_id':1,'stable':True,'eligible':False}]))

    def test_explicit_task_restore_keeps_provenance_without_physical_progress(self):
        preview=TaskPreview()
        for i in range(1,4):preview.scan(['156+123+516+231'],i,i*250)
        restored=TaskPreview()
        restored.restore_confirmed(preview.summary())
        self.assertTrue(restored.summary()['restored_confirmed_task'])
        self.assertEqual(restored.summary()['accepted_frame_id'],3)
        self.assertEqual(restored.summary()['confirmed_physical_steps'],0)
        with self.assertRaises(ValueError):TaskPreview().restore_confirmed({'source':'SYNTHETIC'})


if __name__=='__main__':unittest.main()
