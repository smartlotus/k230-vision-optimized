import unittest
import torch
import tempfile,json,hashlib
from pathlib import Path
from task_training import TopFaceLoss,TopFaceDataset


def target(index=1,x=80,y=80,visible=True,orientation=False):
    return {'instance_id':index,'color_channel':5,'bbox':[x-12,y-12,x+12,y+12],
        'visible_area_px':576,'physical':[x+1,y+2],'conic':[x+3,y+4],
        'axes':[12,10],'orientation':[.6,.8],'center_valid':True,'center_visible':visible,
        'geometry_valid':True,'orientation_valid':orientation,'visible_fraction':.6,'truncated':False}


class TaskLossTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);self.loss=TopFaceLoss()

    def test_occluded_center_keeps_flags_but_masks_coordinate_and_round_angle_loss(self):
        raw=torch.zeros(1,1275,50,requires_grad=True)
        value,parts,counts=self.loss(raw,[[target(visible=False)]])
        value.backward();grad=raw.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertEqual(parts['physical'],0)
        self.assertEqual(float(grad[:,:,38:40].abs().sum()),0)
        self.assertGreater(float(grad[:,:,40:42].abs().sum()),0)
        self.assertGreater(float(grad[:,:,48:50].abs().sum()),0)
        self.assertEqual(float(grad[:,:,44:46].abs().sum()),0)
        self.assertEqual(counts['assigned_gt'],1)
        # color channel5 (approved color ID6) is positive; other colors remain negative.
        self.assertLess(float(grad[:,:,5].min()),0)

    def test_all_five_annotated_targets_can_receive_training_priors(self):
        items=[target(i+1,x,y) for i,(x,y) in enumerate(((40,40),(100,40),(160,40),(220,40),(280,100)))]
        raw=torch.zeros(1,1275,50,requires_grad=True)
        value,_,counts=self.loss(raw,[items]);value.backward()
        self.assertEqual(counts['world_instances'],5);self.assertEqual(counts['assigned_gt'],5)
        self.assertEqual(counts['assigned_fifth_instance'],1)
        self.assertEqual(counts['positive_priors'],15)

    def test_empty_scene_has_finite_background_loss_and_no_geometry_gradient(self):
        raw=torch.zeros(1,1275,50,requires_grad=True)
        value,_,counts=self.loss(raw,[[]]);value.backward()
        self.assertTrue(torch.isfinite(value));self.assertEqual(counts['world_instances'],0)
        self.assertEqual(float(raw.grad[:,:,6:].abs().sum()),0)

    def test_large_gt_cannot_take_fifth_instances_only_representable_prior(self):
        large=target(1,72,72);small=target(5,72,72);small['bbox']=[71,71,73,73]
        pairs,counts=self.loss.assign([large,small])
        self.assertEqual(counts['assigned_gt'],2);self.assertEqual(counts['assigned_fifth_instance'],1)
        self.assertEqual(len([gt for _,gt in pairs if gt==1]),1)

    def test_regression_is_equal_per_instance_with_different_positive_counts(self):
        large=target(1,72,72);small=target(5,160,80);small['bbox']=[159,79,161,81]
        pairs,_=self.loss.assign([large,small]);raw=torch.zeros(1,1275,50)
        for index,gt in pairs:
            item=(large,small)[gt];point=self.loss.points[index];stride=self.loss.scale[index]
            raw[0,index,38:40]=(torch.tensor(item['physical'])-point)/stride+(1. if gt==0 else 0.)
        _,parts,_=self.loss(raw,[[large,small]])
        self.assertAlmostEqual(parts['physical'],.25,places=6)

    def test_qa_member_drift_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'scene-00000';scene.mkdir()
            label={'split':'train','scene_group_id':'s','camera_group_id':'c'}
            label_path=scene/'labels.json';label_path.write_text(json.dumps(label),encoding='utf8')
            image=scene/'rgb.png';image.write_bytes(b'qa-approved-image-bytes')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            audit={'status':'PASS','scenes':[{'scene':scene.name,'split':'train','group':'s','camera_group':'c',
                'rgb_sha256':digest(image),'label_sha256':digest(label_path)}]}
            (root/'AUDIT_V11.json').write_text(json.dumps(audit),encoding='utf8')
            TopFaceDataset(root)
            image.write_bytes(b'changed-after-QA')
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                TopFaceDataset(root)


if __name__=='__main__':unittest.main()
