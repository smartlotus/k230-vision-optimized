import math
import unittest
import numpy as np
import torch
from evaluate_task import evaluate
from task_geometry import letterbox_parameters


class FixedRawModel(torch.nn.Module):
    def __init__(self,raw):super().__init__();self.raw=raw
    def forward(self,images):return self.raw


class TaskEvaluationTests(unittest.TestCase):
    def fixture(self,color_channel=0):
        raw=torch.zeros(1,1275,50);raw[:,:,:6]=-20
        raw[:,:,6:38]=-20;raw[:,:,6:38].reshape(1,1275,4,8)[:,:,:,1]=20
        index=10*40+10;raw[0,index,color_channel]=20
        raw[0,index,38:40]=torch.tensor([1/8,2/8]);raw[0,index,48:50]=torch.tensor([3/8,4/8])
        raw[0,index,40:42]=20;raw[0,index,42:44]=torch.tensor([0.,math.log(5/8)])
        raw[0,index,44:46]=torch.tensor([0.,1.]);raw[0,index,46:48]=20
        target={'instance_id':1,'color_channel':0,'bbox':[72,72,88,88],'visible_area_px':256,
            'physical':[81,82],'conic':[83,84],'axes':[8,5],'orientation':[0,1],
            'center_valid':True,'center_visible':True,'geometry_valid':True,'orientation_valid':True,
            'visible_fraction':1.,'truncated':False}
        meta={'scene':'fixture','expected_world_count':1,'letterbox':letterbox_parameters(320,192)}
        loader=[(torch.zeros(1,3,192,320),[[target]],[meta])]
        return FixedRawModel(raw),loader

    def test_perfect_bbox_color_physical_and_conic_geometry_score_one(self):
        model,loader=self.fixture();result=evaluate(model,loader,torch.device('cpu'))
        self.assertAlmostEqual(result['task_mAP50_95'],1.)
        self.assertEqual(result['class_agnostic_recall50'],1.)
        self.assertEqual(result['matched_color_accuracy'],1.)
        self.assertAlmostEqual(result['errors']['physical_center_visible_px']['median'],0.)
        self.assertAlmostEqual(result['errors']['conic_center_px']['median'],0.)
        self.assertAlmostEqual(result['errors']['ellipse_iou']['median'],1.)

    def test_wrong_color_is_not_hidden_by_class_agnostic_recall(self):
        model,loader=self.fixture(1);result=evaluate(model,loader,torch.device('cpu'))
        self.assertEqual(result['class_agnostic_recall50'],1.)
        self.assertEqual(result['matched_color_accuracy'],0.)
        self.assertEqual(result['task_mAP50'],0.)

    def test_missing_ellipse_counts_as_abstention_for_valid_angle(self):
        model,loader=self.fixture()
        model.raw[:,:,44:46]=0
        result=evaluate(model,loader,torch.device('cpu'))
        self.assertEqual(result['denominators']['angle_valid_gt'],1)
        self.assertEqual(result['denominators']['angle_abstentions'],1)

    def test_empty_scene_false_positive_has_explicit_denominator(self):
        model,loader=self.fixture();images,_,meta=loader[0];meta[0]['expected_world_count']=0
        result=evaluate(model,[(images,[[]],meta)],torch.device('cpu'))
        self.assertIsNone(result['task_mAP50_95'])
        self.assertEqual(result['denominators']['empty_observation_images'],1)
        self.assertEqual(result['denominators']['empty_false_positive_detections'],1)


if __name__=='__main__':unittest.main()
