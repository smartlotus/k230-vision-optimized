import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from audit_v11 import audit
from geometry import camera_pose,intrinsics,pixel_to_top,project,top_boundary
from scene_config import make_config


class V11Tests(unittest.TestCase):
    def fixture(self,root,count=1):
        directory=Path(root)/'scene-00000';directory.mkdir()
        K=intrinsics(640,360,65);pose=camera_pose(.1,80)
        center=pixel_to_top([480,100],K,pose);transform=np.eye(4);transform[:2,3]=center[:2]
        boundary=project(top_boundary(center[:2]),K,pose)
        instances=[]
        for slot in range(1,6):
            mask=np.zeros((360,640),np.uint8)
            if slot<=count:cv2.fillPoly(mask,[np.round(boundary).astype(np.int32)],255)
            cv2.imwrite(str(directory/f'visible-{slot}.png'),mask)
        if count:
            instances=[{'instance_id':1,'color_id':2,'color_material':'yellow','object_pose':transform.tolist(),
                'physical_top_center_xy':project([center],K,pose)[0].tolist(),
                'top_boundary_xy':boundary.tolist(),'top_face_visible_mask':'visible-1.png'}]
        cv2.imwrite(str(directory/'rgb.png'),np.full((360,640,3),100,np.uint8))
        raw={'schema':'topface-scene-v1.1','instances':instances,'expected_instance_count':count,
            'scene_group_id':'fixture','random_seed':123,'split':'train',
            'camera_intrinsics':K.tolist(),'camera_pose_world_from_cv':pose.tolist(),
            'camera_height_above_top_mm':100.,'plane_angle_deg':80.,'hfov_deg':65.}
        (directory/'labels_raw.json').write_text(json.dumps(raw))
        return directory

    def test_physical_center_is_distinct_from_perspective_conic_center(self):
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root);report=audit(root)
            item=json.loads((directory/'labels.json').read_text())['instances'][0]
            np.testing.assert_allclose(item['center_xy'],[480,100],atol=1e-8)
            self.assertGreater(np.linalg.norm(np.array(item['ellipse_conic_center'])-item['center_xy']),.1)
            self.assertTrue(item['center_valid']);self.assertTrue(item['center_visible'])
            self.assertEqual(report['status'],'PASS')

    def test_occluded_center_keeps_3d_truth(self):
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root);p=directory/'visible-1.png';mask=cv2.imread(str(p),0)
            mask[90:110,470:490]=0;cv2.imwrite(str(p),mask);audit(root)
            item=json.loads((directory/'labels.json').read_text())['instances'][0]
            self.assertFalse(item['center_visible']);self.assertTrue(item['center_valid'])
            np.testing.assert_allclose(item['center_xy'],[480,100],atol=1e-8)

    def test_empty_target_mask_and_unlabelled_fifth_mask_fail(self):
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root)
            cv2.imwrite(str(directory/'visible-1.png'),np.zeros((360,640),np.uint8))
            with self.assertRaises(RuntimeError):audit(root)
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root,0)
            hidden=np.zeros((360,640),np.uint8);hidden[10:20,10:20]=255
            cv2.imwrite(str(directory/'visible-5.png'),hidden)
            with self.assertRaises(RuntimeError):audit(root)

    def test_smoke_configs_are_independent_and_keep_all_five(self):
        configs=[make_config(20261114,index) for index in range(100)]
        self.assertEqual(len({c['scene_group_id'] for c in configs}),100)
        self.assertEqual(len({c['random_seed'] for c in configs}),100)
        self.assertEqual({c['expected_instance_count'] for c in configs},set(range(6)))
        self.assertEqual({o['color_id'] for c in configs for o in c['objects']},set(range(1,7)))
        self.assertEqual(min(c['camera_height_above_top_mm'] for c in configs),100)
        self.assertEqual(max(c['camera_height_above_top_mm'] for c in configs),400)
        for c in configs:
            self.assertEqual(c['expected_instance_count'],len(c['objects']))
            self.assertTrue(80<=c['plane_angle_deg']<=100)
            self.assertEqual(c,make_config(20261114,c['scene_index']))
        groups={}
        for c in configs:groups.setdefault(c['camera_group_id'],set()).add(c['split'])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))


if __name__=='__main__':unittest.main()
