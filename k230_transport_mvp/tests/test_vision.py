import copy
import unittest

from k230 import config as defaults
from k230.vision_adapter import VisionAdapter
from vision.common import observation, pixels
from vision.qr import QRTracker
from vision.tracker import StableTracker, TurntableTracker
from vision.color import select_candidate
from vision.ring import cluster_circles, identify_ring
from vision.lane import estimate_lane
from vision.mapping import apply_calibration, transform_point


PAYLOAD = "156+123+516+231"
PROFILE = {"width": 640, "height": 480, "crop": False, "hmirror": False,
           "vflip": False, "rotation": 0, "pose_id": "RAW_DOCK_A"}
H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def config():
    result={name: copy.deepcopy(getattr(defaults, name)) for name in dir(defaults) if name.isupper()}
    # These synthetic images/calibrations deliberately retain their explicit CSI2 source.
    result.update(CAMERA_IDS=(2,),MODE_CAMERA={mode:2 for mode in ('QR','COLOR','RING','LANE','TOP')},
                  DESKTOP_SINGLE_CAMERA=False)
    return result


class Code:
    def __init__(self, payload):
        self.value = payload

    def payload(self):
        return self.value


class Blob:
    def __init__(self, bbox):
        self.bbox = bbox

    def rect(self):
        return self.bbox


class Circle:
    def __init__(self, x, y, radius):
        self.values = (x, y, radius)

    def x(self): return self.values[0]
    def y(self): return self.values[1]
    def r(self): return self.values[2]


class GrayImage:
    def __init__(self, image):
        self.source = image

    def find_qrcodes(self, **kwargs):
        self.source.qr_calls += 1
        return [Code(p) for p in self.source.payloads]

    def get_pixel(self, x, y):
        return 100 if 200 <= x <= 400 else 0


class Image:
    def __init__(self, payloads=None, blobs=None, circles=None):
        self.payloads = payloads or []
        self.blobs = blobs or []
        self.circles = circles or []
        self.qr_calls = 0
        self.gray_calls = 0
        self.blob_kwargs = None

    def width(self): return 640
    def height(self): return 480

    def to_grayscale(self):
        self.gray_calls += 1
        return GrayImage(self)

    def find_qrcodes(self, **kwargs):
        raise AssertionError("RGB565 decoding is not the verified API path")

    def find_blobs(self, thresholds, **kwargs):
        self.blob_kwargs = (thresholds, kwargs)
        return [Blob(b) for b in self.blobs]

    def find_circles(self, **kwargs):
        return [Circle(*c) for c in self.circles]


def frame(image, number=1, **changes):
    value = {"image": image, "camera_id": 2, "frame_id": number, "timestamp_ms": number*50,
             "image_profile": dict(PROFILE), "is_simulated": False}
    value.update(changes)
    return value


def calibration(plane="RAW_TOP", station="RAW"):
    return {"calibration_id": "measured-1", "validated": True, "camera_id": 2,
            "plane": plane, "station": station, "pose_id": "RAW_DOCK_A",
            "image_profile": dict(PROFILE), "is_simulated": False, "H": copy.deepcopy(H),
            "target_pose_valid": False}


def color_config():
    value = config()
    value.update({"COLOR_THRESHOLDS": {1: (10, 80, 10, 50, -20, 40)},
                  "COLOR_ROI": (10, 10, 300, 300), "COLOR_REFERENCE": (150, 150),
                  "COLOR_MIN_AREA": 100, "COLOR_MAX_AREA": 10000})
    return value


class QRTests(unittest.TestCase):
    def test_three_fresh_frames(self):
        qr = QRTracker()
        self.assertEqual(qr.update([PAYLOAD], 1, 50)["reason"], "QR_CONFIRMING")
        self.assertEqual(qr.update([PAYLOAD], 2, 100)["status"], "NOT_FOUND")
        self.assertTrue(qr.update([PAYLOAD], 3, 150)["submitted"])
        self.assertEqual(qr.update([PAYLOAD], 4, 200)["reason"], "QR_DUPLICATE")
        qr.reset_scan()
        self.assertEqual(qr.update([PAYLOAD], 5, 250)["reason"], "QR_DUPLICATE")
        qr.reset_session()
        self.assertEqual(qr.update([PAYLOAD], 6, 300)["reason"], "QR_CONFIRMING")

    def test_replay_and_absence_reset(self):
        qr = QRTracker()
        qr.update([PAYLOAD], 1, 50)
        qr.update([PAYLOAD], 1, 50)
        self.assertEqual(qr.count, 1)
        qr.update([], 2, 100)
        self.assertEqual(qr.count, 0)
        qr.update([PAYLOAD], 3, 150)
        self.assertEqual(qr.update([PAYLOAD], 4, 200)["status"], "NOT_FOUND")

    def test_illegal_ambiguous_and_timeout(self):
        qr = QRTracker(timeout_ms=150)
        self.assertEqual(qr.update(["random text"], 1, 0)["status"], "NOT_FOUND")
        self.assertEqual(qr.update([PAYLOAD, "123+123+321+321"], 2, 50)["reason"], "QR_AMBIGUOUS")
        self.assertEqual(qr.update([PAYLOAD], 3, 150)["reason"], "QR_TIMEOUT")

    def test_long_gap_and_source_change(self):
        qr = QRTracker(max_gap_ms=100)
        qr.update([PAYLOAD], 1, 0, 2)
        qr.update([PAYLOAD], 2, 50, 2)
        self.assertFalse(qr.update([PAYLOAD], 3, 200, 2)["submitted"])
        self.assertFalse(qr.update([PAYLOAD], 4, 250, 0)["submitted"])


class StabilityTests(unittest.TestCase):
    def test_turntable_window_without_invented_motion_prediction(self):
        tracker = TurntableTracker(frames=3, max_motion_px=4)
        self.assertIsNone(tracker.last_seen)
        self.assertIsNone(tracker.update_window("RAW", 1, 50, (100, 100))["window_candidate"])
        self.assertFalse(tracker.update_window("RAW", 2, 100, (101, 100))["stable"])
        result = tracker.update_window("RAW", 3, 150, (100, 101))
        self.assertTrue(result["stable"])
        self.assertEqual(result["window_candidate"], (100, 101))
        self.assertEqual(result["last_seen"], 150)
        lost = tracker.update_window("RAW", 4, 200, None)
        self.assertFalse(lost["stable"])
        self.assertIsNone(lost["window_candidate"])
        self.assertEqual(lost["last_seen"], 150)
        changed = tracker.update_window("OTHER_DOCK", 5, 250, None)
        self.assertIsNone(changed["last_seen"])
        self.assertTrue(all(changed[key] is None for key in ("angle", "omega", "phase")))

    def test_drift_replay_gap_loss_and_context(self):
        tracker = StableTracker(frames=3, max_motion_px=4, max_gap_ms=100)
        self.assertFalse(tracker.update("a", 1, 10, (0, 0)))
        self.assertFalse(tracker.update("a", 2, 20, (3, 0)))
        self.assertFalse(tracker.update("a", 3, 30, (6, 0)))
        self.assertFalse(tracker.update("a", 4, 40, (6, 0)))
        self.assertTrue(tracker.update("a", 5, 50, (6, 0)))
        self.assertFalse(tracker.update("a", 5, 50, (6, 0)))
        self.assertFalse(tracker.update("a", 6, 200, (6, 0)))
        self.assertFalse(tracker.update("b", 7, 210, (6, 0)))
        self.assertFalse(tracker.update("b", 8, 220, None))
        self.assertFalse(tracker.update("b", 9, 230, (6, 0)))


class CandidateTests(unittest.TestCase):
    def test_color_filters_and_nearest(self):
        candidates = [{"target_id": 1, "bbox": b} for b in [(0, 5, 20, 20), (50, 50, 20, 20), (80, 80, 20, 20)]]
        candidates.append({"target_id": 2, "bbox": (40, 40, 20, 20)})
        result = select_candidate(candidates, 1, (0, 0, 200, 200), (70, 70), 100, 1000)
        self.assertEqual(result["center"], (60, 60))
        self.assertEqual(select_candidate(candidates, 1, (0, 0, 200, 200), (70, 70), 100, 1000, [(60, 60, 10)])["center"], (90, 90))

    def test_concentric_circles_not_three_rings(self):
        groups = cluster_circles([{"x": 10, "y": 20, "r": r} for r in (10, 20, 30)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["support"], 3)
        self.assertEqual(identify_ring(groups, 1, {"verified": True, "ring_id_order": [1, 2, 3]})[1], "ID_UNKNOWN")

    def test_ring_explicit_reversal_and_ambiguous_rois(self):
        groups = [{"center": (x, 20), "support": 1} for x in (10, 50, 90)]
        cfg = {"verified": True, "ring_id_order": [3, 2, 1], "min_spacing_px": 20}
        self.assertEqual(identify_ring(groups, 1, cfg)[0]["center"], (90, 20))
        self.assertEqual(identify_ring(groups[:2], 1, cfg)[1], "ID_UNKNOWN")
        cfg = {"verified": True, "id_rois": {1: (0, 0, 100, 100), 2: (0, 0, 100, 100)}}
        self.assertEqual(identify_ring(groups, 1, cfg)[1], "ID_UNKNOWN")

    def test_lane_requires_both_edges_and_multiple_consistent_rows(self):
        rows = [{"values": [0]*10+[100]*20+[0]*10} for _ in range(3)]
        self.assertEqual(estimate_lane(rows, (90, 110), 10, 30)["center_x"], 19.5)
        self.assertIsNone(estimate_lane(rows[:2], (90, 110), 10, 30))
        self.assertIsNone(estimate_lane([{"values": [100]*40}]*3, (90, 110), 10, 50))


class MappingTests(unittest.TestCase):
    def result(self):
        value = pixels(observation("COLOR", frame(Image()), 1, "RAW", "OK"), 100, 200)
        value["stable"] = True
        return value

    def test_projective_point_and_alias_without_motion_authority(self):
        self.assertEqual(transform_point(H, 3, 4), (3, 4))
        value = apply_calibration(self.result(), calibration(), "RAW_TOP")
        self.assertEqual(value["coord_frame"], "ROBOT_MM")
        self.assertFalse(value["target_pose_valid"])
        self.assertEqual(value["calibration_id"], "measured-1")

    def test_all_bindings_are_checked(self):
        for key, bad in [("camera_id", 0), ("plane", "BUFFER_PLANE"), ("station", "PROCESS"),
                         ("is_simulated", True), ("pose_id", "OTHER"), ("validated", 1),
                         ("image_profile", dict(PROFILE, hmirror=True))]:
            with self.subTest(key=key):
                c = calibration()
                c[key] = bad
                value = apply_calibration(self.result(), c, "RAW_TOP")
                self.assertEqual(value["status"], "UNCALIBRATED")
                self.assertIsNone(value["x_mm"])

    def test_invalid_h_and_unknown_pose_never_project(self):
        for bad in [[[0]*3]*3, [[1, 0, 0], [0, 1, 0], [0, 0, float("nan")]],
                    [[1, 0, 0], [0, 1, 0], [1, 0, -100]]]:
            c = calibration()
            c["H"] = bad
            self.assertIsNone(apply_calibration(self.result(), c, "RAW_TOP")["x_mm"])
        value = self.result()
        value["image_profile"]["pose_id"] = None
        c = calibration()
        c["image_profile"]["pose_id"] = None
        c["pose_id"] = None
        self.assertEqual(apply_calibration(value, c, "RAW_TOP")["status"], "UNCALIBRATED")

    def test_invalid_observation_clears_old_mm(self):
        for status in ("NOT_FOUND", "UNCALIBRATED", "ERROR", "UNAVAILABLE"):
            value = self.result()
            value.update({"status": status, "x_mm": 1, "y_mm": 2, "coord_frame": "ROBOT_MM"})
            value = apply_calibration(value, calibration(), "RAW_TOP")
            self.assertIsNone(value["x_mm"])
            self.assertEqual(value["coord_frame"], "IMAGE_PX")


class AdapterTests(unittest.TestCase):
    def test_grayscale_qr_path_and_dedup(self):
        adapter = VisionAdapter(config())
        image = Image([PAYLOAD])
        for i in (1, 2):
            self.assertEqual(adapter.detect("QR", frame(image, i))["status"], "NOT_FOUND")
        self.assertEqual(adapter.detect("QR", frame(image, 3))["status"], "OK")
        self.assertEqual(adapter.detect("QR", frame(image, 4))["reason"], "QR_DUPLICATE")
        self.assertEqual(image.gray_calls, 4)
        self.assertEqual(image.qr_calls, 4)

    def test_default_real_parameters_unset(self):
        adapter = VisionAdapter(config())
        for mode, target, station in [("COLOR", 1, "RAW"), ("RING", 1, "PROCESS"), ("TOP", 1, "BUFFER"), ("LANE", None, None)]:
            value = adapter.detect(mode, frame(Image()), target, station)
            self.assertEqual(value["status"], "UNCALIBRATED")
            self.assertIsNone(value["x_mm"])

    def test_source_and_profile_mismatch_and_no_frame(self):
        adapter = VisionAdapter(config())
        self.assertEqual(adapter.detect("QR", None)["status"], "UNAVAILABLE")
        self.assertEqual(adapter.detect("QR", frame(Image(), camera_id=0))["status"], "ERROR")
        self.assertEqual(adapter.detect("QR", frame(Image(), image_profile={"width": 320, "height": 240}))["status"], "ERROR")
        self.assertEqual(adapter.detect("QR", frame(object()))["status"], "ERROR")
        self.assertEqual(adapter.detect("QR", frame(Image(), width=320))["status"], "ERROR")
        self.assertEqual(adapter.detect("QR", frame(Image(), timestamp_ms=10.5))["status"], "ERROR")
        self.assertEqual(adapter.detect("COLOR", frame(Image()), True, "RAW")["status"], "ERROR")

    def test_caller_profile_mutation_does_not_preserve_stability(self):
        cfg = color_config()
        cfg["CALIBRATIONS"][("RAW", "RAW_TOP")] = calibration()
        adapter = VisionAdapter(cfg)
        image = Image(blobs=[(100, 100, 40, 40)])
        shared_profile = dict(PROFILE)
        adapter.detect("COLOR", frame(image, 1, image_profile=shared_profile), 1, "RAW")
        adapter.detect("COLOR", frame(image, 2, image_profile=shared_profile), 1, "RAW")
        shared_profile["hmirror"] = True
        result = adapter.detect("COLOR", frame(image, 3, image_profile=shared_profile), 1, "RAW")
        self.assertFalse(result["stable"])
        self.assertIsNone(result["x_mm"])

    def test_missing_or_non_boolean_source_never_becomes_real_mm(self):
        cfg = color_config()
        cfg["STABLE_FRAMES"] = 1
        measured = calibration()
        measured["target_pose_valid"] = True
        cfg["CALIBRATIONS"][("RAW", "RAW_TOP")] = measured
        image = Image(blobs=[(100, 100, 40, 40)])
        # First establish that every other gate would allow the candidate.
        good = VisionAdapter(cfg).detect("COLOR", frame(image), 1, "RAW")
        self.assertEqual(good["coord_frame"], "ROBOT_MM")
        self.assertTrue(good["stable"])
        self.assertTrue(good["target_pose_valid"])
        for bad in ("MISSING", None, "false", 0, 1):
            with self.subTest(source=bad):
                value = frame(image, is_simulated=bad)
                if bad == "MISSING":
                    del value["is_simulated"]
                result = VisionAdapter(cfg).detect("COLOR", value, 1, "RAW")
                self.assertEqual(result["status"], "ERROR")
                self.assertEqual(result["reason"], "FRAME_SOURCE_MISMATCH")
                self.assertIsNone(result["x_mm"])
                self.assertIsNone(result["y_mm"])
                self.assertIsNone(result["calibration_id"])
                self.assertFalse(result["target_pose_valid"])
                self.assertFalse(result["stable"])

    def test_color_px_and_mm_stability_and_loss(self):
        cfg = color_config()
        adapter = VisionAdapter(cfg)
        image = Image(blobs=[(100, 100, 40, 40)])
        value = adapter.detect("COLOR", frame(image), 1, "RAW")
        self.assertEqual((value["status"], value["coord_frame"]), ("UNCALIBRATED", "IMAGE_PX"))
        self.assertFalse(image.blob_kwargs[1]["merge"])
        cfg["CALIBRATIONS"][("RAW", "RAW_TOP")] = calibration()
        for i in (2, 3):
            value = adapter.detect("COLOR", frame(image, i), 1, "RAW")
        self.assertTrue(value["stable"])
        self.assertEqual(value["x_mm"], 120)
        lost = adapter.detect("COLOR", frame(Image(), 4), 1, "RAW")
        self.assertEqual(lost["status"], "NOT_FOUND")
        self.assertFalse(adapter.detect("COLOR", frame(image, 5), 1, "RAW")["stable"])

    def test_consumed_target_and_top_separate_plane(self):
        cfg = color_config()
        cfg["CALIBRATIONS"][("BUFFER", "BUFFER_PLANE")] = calibration("BUFFER_PLANE", "BUFFER")
        adapter = VisionAdapter(cfg)
        image = Image(blobs=[(100, 100, 40, 40)])
        adapter.mark_consumed("RAW", 1, 120, 120, 20)
        self.assertEqual(adapter.detect("COLOR", frame(image), 1, "RAW")["status"], "NOT_FOUND")
        adapter.clear_consumed("RAW", 1)
        self.assertEqual(adapter.detect("COLOR", frame(image, 2), 1, "RAW")["reason"], "CALIBRATION_REQUIRED")
        self.assertEqual(adapter.detect("TOP", frame(image, 2), 1, "BUFFER")["status"], "UNCALIBRATED")

    def test_ring_identity_unknown_and_bound_mapping(self):
        cfg = config()
        cfg["RING_CONFIG"] = {"PROCESS": {"roi": (0, 0, 600, 400), "verified": True,
            "camera_id": 2, "image_profile": dict(PROFILE), "ring_id_order": [3, 2, 1], "min_spacing_px": 30}}
        adapter = VisionAdapter(cfg)
        value = adapter.detect("RING", frame(Image(circles=[(20, 20, 10)])), 1, "PROCESS")
        self.assertEqual(value["reason"], "ID_UNKNOWN")
        self.assertIsNone(value["target_id"])
        image = Image(circles=[(20, 20, 10), (120, 20, 10), (220, 20, 10)])
        value = adapter.detect("RING", frame(image, 2), 1, "PROCESS")
        self.assertEqual(value["u_px"], 220)
        self.assertIsNone(value["x_mm"])

    def test_lane_is_auxiliary_px_only(self):
        cfg = config()
        cfg.update({"LANE_ROI": (0, 100, 640, 200), "LANE_THRESHOLD": (80, 120),
                    "LANE_MIN_WIDTH_PX": 100, "LANE_MAX_WIDTH_PX": 300})
        value = VisionAdapter(cfg).detect("LANE", frame(Image()))
        self.assertEqual(value["status"], "OK")
        self.assertTrue(value["auxiliary_only"])
        self.assertEqual(value["coord_frame"], "IMAGE_PX")
        self.assertIsNone(value["x_mm"])


if __name__ == "__main__":
    unittest.main()
