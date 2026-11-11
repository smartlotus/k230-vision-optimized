import unittest
from k230.camera_manager import CameraManager


class Image:
    def width(self): return 640
    def height(self): return 480


class Media:
    def __init__(self): self.calls = []
    def init(self): self.calls.append("init")
    def deinit(self): self.calls.append("deinit")


class Display:
    VIRT = 1
    def __init__(self): self.calls = []
    def init(self, *args, **kwargs): self.calls.append("init")
    def deinit(self): self.calls.append("deinit")
    def show_image(self, image): self.calls.append(("show", image))


class Sensor:
    RGB565 = 2
    instances = []
    fail_reset_id = None
    fail_run = False
    snapshot_failures = 0

    def __init__(self, **kwargs):
        self.id = kwargs["id"]
        self.calls = [("construct", kwargs)]
        self.instances.append(self)

    def reset(self):
        self.calls.append(("reset",))
        if self.id == self.fail_reset_id:
            raise RuntimeError("reset failed")

    def set_framesize(self, **kwargs): self.calls.append(("size", kwargs))
    def set_pixformat(self, fmt): self.calls.append(("format", fmt))
    def set_hmirror(self, value): self.calls.append(("hmirror", value))
    def set_vflip(self, value): self.calls.append(("vflip", value))
    def run(self):
        self.calls.append(("run",))
        if self.fail_run: raise RuntimeError("run failed")
    def snapshot(self, *, timeout, auto_reboot):
        self.calls.append(("snapshot", {"timeout": timeout, "auto_reboot": auto_reboot}))
        if self.snapshot_failures:
            self.snapshot_failures -= 1
            raise RuntimeError("snapshotfailed(3)")
        return Image()
    def stop(self): self.calls.append(("stop",))


class CameraTests(unittest.TestCase):
    def test_quarantined_request_rejected_before_hardware(self):
        with self.assertRaisesRegex(ValueError, 'Quarantined'):
            self.manager(CAMERA_IDS=(2,0), CAMERA_QUARANTINE=(0,)).start()
        self.assertEqual(Sensor.instances, [])

    def test_priority_starts_healthy_cameras_first_without_rebinding_source(self):
        manager=self.manager(CAMERA_IDS=(0,1,2), DESKTOP_SINGLE_CAMERA=False,
                             CAMERA_PRIORITY=(2,1,0), CAMERA_QUARANTINE=())
        manager.start()
        self.assertEqual([s.id for s in Sensor.instances], [2,1,0])
        self.assertEqual(manager.get_frame(0)['camera_id'], 0)
        manager.close()

    def setUp(self):
        Sensor.instances = []
        Sensor.fail_reset_id = None
        Sensor.fail_run = False
        Sensor.snapshot_failures = 0
        self.media = Media()
        self.display = Display()

    def manager(self, **changes):
        cfg = {"CAMERA_IDS": (2,), "MODE_CAMERA": {"QR": 2, "COLOR": 0},
               "DESKTOP_SINGLE_CAMERA": True, "PREVIEW": False}
        cfg.update(changes)
        return CameraManager(cfg, sensor_factory=Sensor, media=self.media, display=self.display, clock_ms=lambda: 500)

    def test_single_camera_actual_identity_and_lifecycle(self):
        manager = self.manager()
        self.assertIsNone(manager.get_frame(2))
        manager.start()
        manager.start()
        self.assertEqual(manager.camera_for_mode("COLOR"), 2)
        first = manager.get_frame(0)
        second = manager.get_frame(2)
        self.assertEqual((first["camera_id"], first["frame_id"], second["frame_id"]), (2, 1, 2))
        self.assertEqual(first["timestamp_ms"], 500)
        self.assertFalse(first["is_simulated"])
        self.assertEqual(first["image_profile"]["width"], 640)
        manager.show_frame(first)
        self.assertIsNone(manager.get_frame(99))
        self.assertIsNone(manager.camera_for_mode("INVALID"))
        manager.close()
        manager.close()
        calls = [call[0] for call in Sensor.instances[0].calls]
        self.assertEqual(calls.count("run"), 1)
        self.assertEqual(calls.count("stop"), 1)
        self.assertEqual(self.media.calls, ["init", "deinit"])
        self.assertEqual(self.display.calls, [])

    def test_dual_shared_start_all_stop(self):
        manager = self.manager(CAMERA_IDS=(2, 0), DESKTOP_SINGLE_CAMERA=False)
        manager.start()
        self.assertEqual(manager.get_frame(0)["camera_id"], 0)
        self.assertIsNone(manager.get_frame(1))
        manager.close()
        self.assertEqual(sum(sum(call[0] == "run" for call in s.calls) for s in Sensor.instances), 1)
        self.assertTrue(all(("stop",) in s.calls for s in Sensor.instances))

    def test_second_sensor_reset_failure_releases_both(self):
        Sensor.fail_reset_id = 0
        manager = self.manager(CAMERA_IDS=(2, 0), DESKTOP_SINGLE_CAMERA=False)
        with self.assertRaises(RuntimeError): manager.start()
        self.assertTrue(all(("stop",) in s.calls for s in Sensor.instances))
        self.assertFalse(manager.started)
        self.assertEqual(self.media.calls, [])

    def test_run_failure_releases_initialized_media_and_preview(self):
        Sensor.fail_run = True
        manager = self.manager(PREVIEW=True)
        with self.assertRaises(RuntimeError): manager.start()
        self.assertEqual(self.media.calls, ["init", "deinit"])
        self.assertEqual(self.display.calls, ["init", "deinit"])
        self.assertIn(("stop",), Sensor.instances[0].calls)

    def test_bad_ids_or_unimplemented_rotation_rejected_before_hardware(self):
        for changes in ({"CAMERA_IDS": (2, 2)}, {"ROTATION": 90}, {"CAMERA_IDS": ()}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): self.manager(**changes).start()
        self.assertEqual(Sensor.instances, [])

    def test_actual_dimensions_must_match_profile(self):
        manager = self.manager(WIDTH=320)
        manager.start()
        with self.assertRaises(RuntimeError): manager.get_frame(2)
        manager.close()

    def test_preview_shows_original_image_without_changing_source(self):
        manager = self.manager(PREVIEW=True)
        manager.start()
        frame = manager.get_frame(2)
        old_identity = (frame["image"], frame["camera_id"], frame["frame_id"], dict(frame["image_profile"]))
        manager.show_frame(frame)
        self.assertEqual(self.display.calls[-1], ("show", frame["image"]))
        self.assertEqual(old_identity, (frame["image"], frame["camera_id"], frame["frame_id"], frame["image_profile"]))
        manager.close()
        manager.show_frame(frame)
        self.assertEqual(self.display.calls[-1], "deinit")

    def test_startup_timeout_is_per_camera_then_short_and_never_reboots(self):
        manager = self.manager(CAMERA_IDS=(2, 0), DESKTOP_SINGLE_CAMERA=False)
        manager.start()
        manager.get_frame(2)
        manager.get_frame(2)
        manager.get_frame(0)
        manager.get_frame(0)
        for sensor in Sensor.instances:
            requests = [call[1] for call in sensor.calls if call[0] == "snapshot"]
            self.assertEqual(requests, [{"timeout": 2000, "auto_reboot": False},
                                        {"timeout": 100, "auto_reboot": False}])
        manager.close()

    def test_first_failure_keeps_startup_timeout_until_success(self):
        manager = self.manager(FIRST_FRAME_TIMEOUT_MS=1500, SNAPSHOT_TIMEOUT_MS=75)
        manager.start()
        Sensor.instances[0].snapshot_failures = 1
        with self.assertRaisesRegex(RuntimeError, "snapshotfailed"):
            manager.get_frame(2)
        self.assertEqual(manager.frame_id, 0)
        self.assertEqual(manager.successful_cameras, set())
        manager.get_frame(2)
        manager.get_frame(2)
        requests = [call[1] for call in Sensor.instances[0].calls if call[0] == "snapshot"]
        self.assertEqual([r["timeout"] for r in requests], [1500, 1500, 75])
        self.assertTrue(all(r["auto_reboot"] is False for r in requests))
        manager.close()
        manager.start()
        manager.get_frame(2)
        request = [call[1] for call in Sensor.instances[-1].calls if call[0] == "snapshot"][0]
        self.assertEqual(request["timeout"], 1500)
        manager.close()

    def test_timeout_above_two_seconds_or_non_integer_rejected(self):
        for changes in ({"FIRST_FRAME_TIMEOUT_MS": 2001}, {"FIRST_FRAME_TIMEOUT_MS": 0},
                        {"FIRST_FRAME_TIMEOUT_MS": True}, {"SNAPSHOT_TIMEOUT_MS": 1.5}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.manager(**changes).start()
        self.assertEqual(Sensor.instances, [])


if __name__ == "__main__":
    unittest.main()
