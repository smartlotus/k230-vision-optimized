import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from audit_dataset import audit


class AuditTests(unittest.TestCase):
    def fixture(self,root,count):
        directory=Path(root)/"scene-00000"
        directory.mkdir()
        cv2.imwrite(str(directory/"rgb.png"),np.full((360,640,3),100,np.uint8))
        instances=[]
        for index in range(count):
            theta=np.linspace(0,2*np.pi,512,endpoint=False)
            boundary=np.column_stack([100+index*140+20*np.cos(theta),180+20*np.sin(theta)])
            mask=np.zeros((360,640),np.uint8)
            cv2.fillPoly(mask,[np.round(boundary).astype(np.int32)],255)
            cv2.imwrite(str(directory/f"visible-{index+1}.png"),mask)
            instances.append({"instance_id":index+1,"top_boundary_xy":boundary.tolist(),
                "top_face_visible_mask":f"visible-{index+1}.png"})
        raw={"instances":instances,"scene_group_id":"fixture","random_seed":1,
             "camera_pose_world_from_cv":np.eye(4).tolist(),"camera_intrinsics":np.eye(3).tolist()}
        (directory/"labels_raw.json").write_text(json.dumps(raw))
        return directory

    def test_all_four_labels_survive_and_round_direction_is_not_validated(self):
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root,4)
            report=audit(root)
            labels=json.loads((directory/"labels.json").read_text())
            self.assertEqual(report["instances"],4)
            self.assertEqual(len(labels["instances"]),4)
            self.assertTrue(all(not item["orientation_valid"] for item in labels["instances"]))
            self.assertTrue(all(item["geometry_valid"] for item in labels["instances"]))

    def test_empty_frame_and_contaminated_mask(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root,0)
            self.assertEqual(audit(root)["instances"],0)
        with tempfile.TemporaryDirectory() as root:
            directory=self.fixture(root,1)
            path=directory/"visible-1.png"
            mask=cv2.imread(str(path),0);mask[5:10,5:10]=255
            cv2.imwrite(str(path),mask)
            with self.assertRaises(RuntimeError):audit(root)
            report=json.loads((Path(root)/"AUDIT.json").read_text())
            self.assertEqual(report["issues"][0]["reason"],"VISIBLE_MASK_OUTSIDE_PROJECTED_TOP")


if __name__=="__main__":unittest.main()
