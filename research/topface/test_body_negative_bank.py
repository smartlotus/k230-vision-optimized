"""Synthetic evidence-gate fixtures, separate from actual Blender qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import torch

from body_negative_bank import BodyOnlyBank,POLICY,ROLE,unpack
from cloud_store import atomic_json,sha
from compile_trained_topface import load_samples
from geometry import camera_pose,intrinsics,pixel_to_top
from task_training import TopFaceDataset,TopFaceLoss
from audit_v11 import audit


def fixture(root):
    root.mkdir()
    K=intrinsics(640,360,62);pose=camera_pose(.25,90)
    center=pixel_to_top([-40,180],K,pose);transform=np.eye(4);transform[:2,3]=center[:2]
    item={'instance_id':1,'color_id':1,'object_pose':transform.tolist(),
        'bbox_xyxy_visible':None,'visible_area_px':0,'geometry_valid':False,
        'center_xy_projected_3d':[-40,180],'ellipse_conic_center':[-40,180],
        'ellipse_full':{'orientation_deg':0,'semi_major':32,'semi_minor':32},
        'center_valid':False,'center_visible':False,'orientation_valid':False,'visible_fraction':0.,'truncated':True}
    label={'split':'train','scene_group_id':'body-source','camera_group_id':'H250-P90-F62',
        'random_seed':20261123,'camera_height_above_top_mm':250.,'plane_angle_deg':90.,'hfov_deg':62.,
        'camera_intrinsics':K.tolist(),'camera_pose_world_from_cv':pose.tolist(),'negative_role':ROLE,
        'negative_edge':'left','expected_instance_count':1,'occluder':False,'hard_negative':False,'instances':[item]}
    top=np.zeros((360,640),np.uint8);body=top.copy();body[140:220,:40]=255
    image=np.full((360,640,3),70,np.uint8);image[body>0]=[30,30,160]
    cv2.imwrite(str(root/'negative-000-rgb.png'),image)
    cv2.imwrite(str(root/'negative-000-top.png'),top);cv2.imwrite(str(root/'negative-000-body.png'),body)
    atomic_json(root/'negative-000-labels.json',label)
    member={'scene':'body-source','camera_group':'H250-P90-F62','files':{}}
    for kind,suffix in [('rgb','rgb.png'),('top','top.png'),('body','body.png'),('labels','labels.json')]:
        p=root/('negative-000-'+suffix);member['files'][kind]={'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
    manifest={'source_scope':'AUDITED_SYNTHETIC_TRAIN_ONLY','policy':POLICY,
        'source_audit_sha256':'a'*64,'cad':{'original_topology_preserved':True},'members':[member]}
    atomic_json(root/'MANIFEST.json',manifest)
    return label,manifest


def refresh(root,manifest,kind):
    row=manifest['members'][0]['files'][kind];p=root/row['name']
    row.update(bytes=p.stat().st_size,sha256=sha(p));atomic_json(root/'MANIFEST.json',manifest)


def dataset_fixture(root,label,count=2):
    root.mkdir();rows=[]
    for i in range(count):
        directory=root/f'scene-{i:05d}';directory.mkdir()
        value=copy.deepcopy(label);value.update(scene_group_id=f'base-{i}',instances=[],expected_instance_count=0)
        if i==1:
            item=copy.deepcopy(label['instances'][0]);item.update(bbox_xyxy_visible=[200,140,260,200],visible_area_px=3600)
            value.update(instances=[item],expected_instance_count=1)
        atomic_json(directory/'labels.json',value)
        cv2.imwrite(str(directory/'rgb.png'),np.full((360,640,3),100,np.uint8))
        rows.append({'scene':directory.name,'group':value['scene_group_id'],'camera_group':value['camera_group_id'],
                     'split':'train','rgb_sha256':sha(directory/'rgb.png'),'label_sha256':sha(directory/'labels.json')})
    atomic_json(root/'AUDIT_V11.json',{'status':'PASS','scenes':rows})
    return rows


class Draw:
    def random(self):return 0.
    def integers(self,*args):return 0


class BodyNegativeTests(unittest.TestCase):
    def test_replacement_preserves_one_unobservable_world_instance_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bank';label,_=fixture(root);bank=BodyOnlyBank(root)
            base=copy.deepcopy(label);base.update(instances=[],scene_group_id='base')
            row={'scene':'base-slot','group':'base','camera_group':label['camera_group_id'],'split':'train'}
            image=np.zeros((360,640,3),np.uint8)
            actual,effective,event=bank.maybe_replace(image,base,row,Draw())
            self.assertEqual(len(effective['instances']),1);self.assertEqual(event['base_world_count'],0)
            self.assertEqual(event['effective_world_count'],1);self.assertEqual(event['observable_gt'],0)
            self.assertEqual(event['source_scene'],'body-source');self.assertFalse(np.array_equal(image,actual))

    def test_positive_world_scene_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bank';label,_=fixture(root);bank=BodyOnlyBank(root)
            image=np.zeros((360,640,3),np.uint8);row={'scene':'base','group':'base','camera_group':label['camera_group_id'],'split':'train'}
            actual,effective,event=bank.maybe_replace(image,label,row,Draw())
            self.assertIsNone(event);self.assertIs(actual,image);self.assertIs(effective,label)

    def test_heldout_source_or_visible_top_is_rejected_even_with_updated_hashes(self):
        for failure in ('heldout','top'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'bank';label,manifest=fixture(root)
                if failure=='heldout':
                    label['split']='synthetic_test';atomic_json(root/'negative-000-labels.json',label);refresh(root,manifest,'labels')
                else:
                    mask=np.zeros((360,640),np.uint8);mask[1,1]=255
                    cv2.imwrite(str(root/'negative-000-top.png'),mask);refresh(root,manifest,'top')
                with self.assertRaises(ValueError):BodyOnlyBank(root)

    def test_body_must_actually_be_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bank';_,manifest=fixture(root)
            cv2.imwrite(str(root/'negative-000-body.png'),np.zeros((360,640),np.uint8));refresh(root,manifest,'body')
            with self.assertRaisesRegex(ValueError,'object-ID'):BodyOnlyBank(root)

    def test_dataset_replacement_is_repeatable_and_has_no_positive_geometry_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);label,_=fixture(root/'bank');dataset_fixture(root/'data',label)
            augmented=TopFaceDataset(root/'data','train',augment=True,body_negative_bank=root/'bank')
            baseline=TopFaceDataset(root/'data','train',augment=True)
            self.assertEqual(len(augmented),len(baseline))
            positive=augmented[1];self.assertTrue(torch.equal(positive[0],baseline[1][0]))
            for epoch in range(32):
                augmented.epoch=epoch;value=augmented[0]
                if 'body_only_augmentation' in value[2]:break
            else:self.fail('Frozen draws did not exercise replacement')
            self.assertTrue(torch.equal(value[0],augmented[0][0]));self.assertEqual(value[2]['expected_world_count'],1)
            raw=torch.zeros(1,1275,50,requires_grad=True);loss,_,counts=TopFaceLoss()(raw,[value[1]])
            loss.backward();self.assertEqual(counts['world_instances'],1)
            self.assertEqual(counts['unobservable_or_under16px'],1);self.assertEqual(counts['positive_priors'],0)
            self.assertEqual(float(raw.grad[:,:,6:].abs().sum()),0.)

    def test_validation_cannot_enable_the_training_bank(self):
        with self.assertRaisesRegex(ValueError,'restricted to augmented training'):
            TopFaceDataset(Path('unused'),'val',augment=True,body_negative_bank=Path('unused'))

    def test_ptq_pairs_remain_original_while_calibration_records_actual_replacements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);label,_=fixture(root/'bank');dataset_fixture(root/'data',label,30)
            original,original_pairs=load_samples(root/'data',320,192,16,2)
            actual,pairs=load_samples(root/'data',320,192,16,2,body_negatives=BodyOnlyBank(root/'bank'))
            self.assertEqual([row[0]['scene'] for row in pairs],[row[0]['scene'] for row in original_pairs])
            self.assertTrue(all(np.array_equal(a[1],b[1]) for a,b in zip(pairs,original_pairs)))
            self.assertTrue(any(row[0]['training_body_only_augmentation']['replacement'] for row in actual))
            self.assertEqual(len(actual),len(original))

    def test_archive_traversal_is_rejected_before_writing_outside(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'unsafe.bin'
            with zipfile.ZipFile(archive,'w') as bundle:bundle.writestr('../outside','no')
            with self.assertRaisesRegex(ValueError,'Unsafe'):unpack(archive,root/'out',sha(archive),'a'*64)
            self.assertFalse((root/'outside').exists())

    def test_static_top_outside_requires_explicit_mask_qualification_route(self):
        from geometry import project,top_boundary
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);label,_=fixture(root/'bank')
            source=root/'data'/'scene-00000';source.mkdir(parents=True)
            raw=copy.deepcopy(label);raw['base_seed']=20261123
            item=raw['instances'][0];transform=np.asarray(item['object_pose'])
            local=top_boundary([0.,0.]);world=(np.column_stack([local,np.ones(len(local))])@transform.T)[:,:3]
            item.update(physical_top_center_xy=[-40.,180.],
                top_boundary_xy=project(world,np.asarray(raw['camera_intrinsics']),np.asarray(raw['camera_pose_world_from_cv'])).tolist(),
                top_face_visible_mask='visible-1.png')
            atomic_json(source/'labels_raw.json',raw)
            cv2.imwrite(str(source/'rgb.png'),np.full((360,640,3),100,np.uint8))
            for index in range(1,6):cv2.imwrite(str(source/f'visible-{index}.png'),np.zeros((360,640),np.uint8))
            body=np.zeros((360,640),np.uint8);body[140:220,:40]=255;cv2.imwrite(str(source/'body-1.png'),body)
            with self.assertRaises(RuntimeError):audit(root/'data')
            self.assertEqual(audit(root/'data',allow_body_only=True)['status'],'PASS')
            cv2.imwrite(str(source/'body-1.png'),np.zeros((360,640),np.uint8))
            with self.assertRaises(RuntimeError):audit(root/'data',allow_body_only=True)


if __name__=='__main__':unittest.main()
