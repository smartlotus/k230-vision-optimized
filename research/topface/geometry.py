"""Camera and ellipse conventions shared by synthesis and label auditing.

World +Z is up. Camera coordinates use OpenCV (+X right, +Y down, +Z forward).
An optical-axis-to-plane angle of 90 degrees means a vertical, downward view.
Heights are vertical lens-to-top-plane distances, not lens-to-table distances.
"""
import math
import numpy as np


def intrinsics(width, height, hfov_deg):
    if not 0 < hfov_deg < 170:
        raise ValueError("Horizontal field of view outside the supported range")
    focal = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])


def camera_pose(height_m, plane_angle_deg=90., azimuth_deg=0., roll_deg=0.):
    if not .1 <= height_m <= .4 or not 80 <= plane_angle_deg <= 100:
        raise ValueError("Camera outside the user-confirmed operating range")
    tilt, azimuth, roll = np.radians([90-plane_angle_deg, azimuth_deg, roll_deg])
    # The signed deviation is kept separately from the nonnegative physical tilt.
    forward = np.array([math.sin(tilt)*math.cos(azimuth),
                        math.sin(tilt)*math.sin(azimuth), -math.cos(tilt)])
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    axes = np.column_stack([right, down, forward])
    spin = np.array([[math.cos(roll), -math.sin(roll), 0.],
                     [math.sin(roll), math.cos(roll), 0.], [0., 0., 1.]])
    result = np.eye(4)
    result[:3, :3] = axes @ spin
    result[:3, 3] = [0., 0., .06 + height_m]
    return result


def project(points, K, world_from_cv):
    points = np.asarray(points, dtype=float)
    xyz = (points-world_from_cv[:3, 3]) @ world_from_cv[:3, :3]
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("Projection includes points behind the camera")
    pixels = xyz @ np.asarray(K).T
    return pixels[:, :2] / pixels[:, 2:]


def pixel_to_top(pixel, K, world_from_cv):
    ray = world_from_cv[:3, :3] @ np.linalg.solve(K, [*pixel, 1.])
    camera = world_from_cv[:3, 3]
    distance = (.06-camera[2])/ray[2]
    if distance <= 0:
        raise ValueError("Pixel ray does not intersect the top plane in front")
    return camera + ray*distance


def top_boundary(center_xy, samples=512):
    theta = np.linspace(0, 2*np.pi, samples, endpoint=False)
    return np.column_stack([center_xy[0]+.015*np.cos(theta),
                            center_xy[1]+.015*np.sin(theta), np.full(samples, .06)])


def ellipse_from_points(points):
    import cv2
    (cx, cy), (width, height), angle = cv2.fitEllipse(np.asarray(points, np.float32))
    if width >= height:
        major, minor, orientation = width/2, height/2, angle % 180
    else:
        major, minor, orientation = height/2, width/2, (angle+90) % 180
    return {"center": [float(cx), float(cy)], "semi_major": float(major),
            "semi_minor": float(minor), "orientation_deg": float(orientation),
            "angle_valid": minor/major < .98}
