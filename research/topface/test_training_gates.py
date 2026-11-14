import unittest
from train_experiment import sanity_gate,pilot_gate,selection_score,prefer_validation


def metrics(center=2.9,map5095=.89,finite=100):
    return {'class_agnostic_recall50':1.,'matched_color_accuracy':1.,'task_mAP50':1.,'task_mAP50_95':map5095,
        'errors':{'physical_center_visible_px':{'median':center},'ellipse_iou':{'median':1.}},
        'denominators':{'matched_geometry_valid_gt':100,'finite_ellipse_predictions':finite,
                        'empty_false_positive_detections':0,'empty_observation_images':4},
        'count_groups':{'world_count_5':{'images':3,'all_observable_targets_recalled':3}}}


class TrainingGateTests(unittest.TestCase):
    def test_ninety_nine_missing_ellipses_cannot_pass_pilot(self):
        value=metrics(finite=1)
        self.assertFalse(pilot_gate(value)['pass'])
        self.assertFalse(sanity_gate(value)['pass'])

    def test_lower_scalar_score_sanity_success_is_the_saved_overfit_model(self):
        previous=metrics(center=3.1,map5095=.90);current=metrics(center=2.9,map5095=.89)
        self.assertFalse(sanity_gate(previous)['pass']);self.assertTrue(sanity_gate(current)['pass'])
        self.assertLess(selection_score(current),selection_score(previous))
        self.assertTrue(prefer_validation('overfit',selection_score(current),selection_score(previous),current))
        self.assertFalse(prefer_validation('pilot_pretrained',selection_score(current),selection_score(previous),current))

    def test_valid_angles_and_fifth_target_recall_cannot_be_omitted(self):
        value=metrics();value['denominators'].update(angle_valid_gt=20,angle_prediction_valid=0)
        self.assertFalse(pilot_gate(value)['pass'])
        value=metrics();value['count_groups']['world_count_5']['all_observable_targets_recalled']=1
        self.assertFalse(pilot_gate(value)['pass'])


if __name__=='__main__':unittest.main()
