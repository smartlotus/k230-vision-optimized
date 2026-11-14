"""Independent geometric checks for the intended camera/label conventions."""
import math
import unittest
import numpy as np
from geometry import camera_pose, ellipse_from_points, intrinsics, pixel_to_top, project, top_boundary


class GeometryTests(unittest.TestCase):
    def test_vertical_centered_disk_has_known_image_diameter(self):
        K=intrinsics(640,360,65)
        pose=camera_pose(.25)
        ellipse=ellipse_from_points(project(top_boundary([0.,0.]),K,pose))
        np.testing.assert_allclose(ellipse["center"],[320,180],atol=1e-4)
        self.assertAlmostEqual(ellipse["semi_major"],K[0,0]*.015/.25,places=4)
        self.assertFalse(ellipse["angle_valid"])

    def test_vertical_height_is_top_relative_and_projection_round_trips(self):
        K=intrinsics(640,360,65)
        for height in (.1,.25,.4):
            for angle in (80,90,100):
                pose=camera_pose(height,angle,37)
                self.assertAlmostEqual(pose[2,3]-.06,height)
                rotation=pose[:3,:3]
                np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation),1.)
                measured=math.degrees(math.acos(-rotation[2,2]))
                self.assertAlmostEqual(measured,abs(90-angle),places=8)
                for pixel in ([20.,20.],[320.,180.],[620.,340.]):
                    point=pixel_to_top(pixel,K,pose)
                    np.testing.assert_allclose(project([point],K,pose)[0],pixel,atol=1e-8)

    def test_height_scaling_and_outside_camera_envelope(self):
        K=intrinsics(640,360,65)
        sizes=[ellipse_from_points(project(top_boundary([0.,0.]),K,camera_pose(h)))["semi_major"] for h in (.1,.4)]
        self.assertAlmostEqual(sizes[0]/sizes[1],4,places=5)
        for height,angle in ((.04,90),(.41,90),(.2,79),(.2,101)):
            with self.assertRaises(ValueError):camera_pose(height,angle)


if __name__=="__main__":unittest.main()
