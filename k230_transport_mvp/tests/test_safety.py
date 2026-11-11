import copy
import unittest
from ports.motion_port import LiveMotionPort, MotionReceipt, SafetyGate


def fixture():
    binding = dict(calibration_id="H_REAL_1", camera_id="CSI2", station="RAW",
                   plane="OBJECT_TOP", image_profile={"width": 640, "height": 480, "pose_id": "p1"})
    obs = dict(binding, status="OK", kind="COLOR", target_id=1, frame_id=7,
               timestamp_ms=100, is_simulated=False, stable=True,
               coord_frame="ROBOT_MM", target_pose_valid=True, x_mm=20, y_mm=30)
    calibration = dict(binding, status="VERIFIED", is_simulated=False, pose_id="p1",
                       H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    context = dict(motion_enabled=True, motion_authorization="EXPLICIT_ENABLED",
                   active_binding=binding, calibration=calibration, received_at_ms=100,
                   current_observation_identity=("CSI2", 7, 1), max_observation_age_ms=250,
                   required_target_id=1, lane_safe=True, obstacle_clear=True,
                   workspace_mm={"x": [-100, 100], "y": [-100, 100]})
    return obs, context


class Transport:
    def __init__(self, status="DONE"):
        self.calls = []
        self.status = status

    def submit(self, command_id, action, arguments):
        self.calls.append((command_id, action, arguments))
        if self.status == "THROW":
            raise OSError("timeout after send")
        return dict(status=self.status, command_id=command_id)


class SafetyTests(unittest.TestCase):
    def test_valid_independent_gate(self):
        observation, context = fixture()
        self.assertEqual(SafetyGate().check(observation, 200, context), (True, "AUTHORIZED"))

    def test_reject_sim_stale_pixels_pose_invalid_binding(self):
        obs, context = fixture()
        changes = [dict(is_simulated=True), dict(is_simulated=None), dict(timestamp_ms=-1000),
                   dict(timestamp_ms=300), dict(coord_frame="IMAGE_PX"), dict(stable=False),
                   dict(target_pose_valid=False), dict(kind="QR"), dict(target_id=2),
                   dict(camera_id="CSI0"), dict(plane="FLOOR"), dict(calibration_id=None),
                   dict(image_profile={"width": 320}), dict(frame_id=8), dict(x_mm=9999),
                   dict(x_mm=float("nan")), dict(y_mm=None)]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(SafetyGate().check(dict(obs, **change), 200, context)[0])

    def test_reject_unknown_clearance_authorization_and_calibration(self):
        obs, context = fixture()
        changes = [dict(motion_enabled=False), dict(motion_authorization=None),
                   dict(lane_safe=None), dict(obstacle_clear=False), dict(workspace_mm=None),
                   dict(received_at_ms=None), dict(received_at_ms=-1000),
                   dict(active_binding=None), dict(calibration=None),
                   dict(calibration=dict(context["calibration"], H=None)),
                   dict(calibration=dict(context["calibration"], H=[[0]*3]*3)),
                   dict(calibration=dict(context["calibration"], status="DRAFT")),
                   dict(calibration=dict(context["calibration"], pose_id=None)),
                   dict(calibration=dict(context["calibration"], is_simulated=True))]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(SafetyGate().check(obs, 200, dict(context, **change))[0])

    def test_default_live_denied_and_four_methods(self):
        transport = Transport()
        port = LiveMotionPort(transport=transport)
        receipts = [port.drive_to("RAW"), port.pick_to_carrier(1, "RAW", 1),
                    port.place_from_carrier(1, "BUFFER", 1, 1), port.return_to_start("START_1")]
        self.assertTrue(all(r.status == "ERROR" for r in receipts))
        self.assertEqual(transport.calls, [])
        obs, context = fixture()
        port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
        port.update_observation(obs, 100)
        self.assertEqual(port.pick_to_carrier(1, "RAW", 1).status, "DONE")
        self.assertEqual(len(transport.calls), 1)
        port.disable()
        self.assertEqual(port.drive_to("PROCESS").status, "ERROR")

    def test_ack_busy_unknown_never_complete_or_auto_retry(self):
        for status in ("ACK", "BUSY", "UNKNOWN", "THROW"):
            with self.subTest(status=status):
                obs, context = fixture()
                transport = Transport(status)
                port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
                port.update_observation(obs, 100)
                receipt = port.pick_to_carrier(1, "RAW", 1)
                self.assertNotEqual(receipt.status, "DONE")
                self.assertEqual(port.pick_to_carrier(1, "RAW", 1).status, "UNKNOWN")
                self.assertEqual(len(transport.calls), 1)
                with self.assertRaises(ValueError):
                    port.resolve(dict(command_id="different", status="DONE"))
                with self.assertRaises(ValueError):
                    port.resolve(dict(command_id=receipt.command_id, status="ACK"))
                port.resolve(dict(command_id=receipt.command_id, status="DONE"))
                self.assertIsNone(port.pending)

    def test_place_ring_slot_and_top_color_are_distinct(self):
        obs, context = fixture()
        transport = Transport()
        obs["station"] = "BUFFER"
        context["active_binding"]["station"] = "BUFFER"
        context["calibration"]["station"] = "BUFFER"
        obs.update(kind="RING", target_id=2)
        context["current_observation_identity"] = ("CSI2", 7, 2)
        port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
        port.update_observation(obs, 100)
        self.assertEqual(port.place_from_carrier(6, "BUFFER", 2, 1).status, "DONE")
        obs.update(kind="TOP", target_id=6)
        obs["frame_id"] = 8
        context["current_observation_identity"] = ("CSI2", 8, 6)
        port.update_observation(obs, 100)
        self.assertEqual(port.place_from_carrier(6, "BUFFER", 2, 2).status, "DONE")
        obs["target_id"] = 2
        obs["frame_id"] = 9
        context["current_observation_identity"] = ("CSI2", 9, 2)
        port.update_observation(obs, 100)
        self.assertEqual(port.place_from_carrier(6, "BUFFER", 2, 2).status, "ERROR")
        self.assertEqual(len(transport.calls), 2)

    def test_done_consumes_frame_and_old_update_cannot_reactivate(self):
        obs, context = fixture()
        transport = Transport()
        port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
        port.update_observation(obs, 100)
        self.assertEqual(port.pick_to_carrier(1, "RAW", 1).status, "DONE")
        self.assertEqual(port.pick_to_carrier(1, "RAW", 1).reason, "OBSERVATION_ALREADY_CONSUMED")
        port.update_observation(dict(obs), 200)
        self.assertEqual(port.pick_to_carrier(1, "RAW", 1).status, "ERROR")
        obs["frame_id"] = 8
        context["current_observation_identity"] = ("CSI2", 8, 1)
        port.update_observation(obs, 200)
        self.assertEqual(port.pick_to_carrier(1, "RAW", 1).status, "DONE")
        self.assertEqual(len(transport.calls), 2)

    def test_illegal_process_slot_buffer_pick_and_process_layer_rejected(self):
        for station, action in (("PROCESS", "pick"), ("BUFFER", "pick"), ("PROCESS", "place")):
            obs, context = fixture()
            obs["station"] = station
            context["active_binding"]["station"] = station
            context["calibration"]["station"] = station
            obs["kind"] = "TOP" if action == "place" else "COLOR"
            transport = Transport()
            port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
            port.update_observation(obs, 100)
            receipt = port.place_from_carrier(1, station, 1, 2) if action == "place" else port.pick_to_carrier(1, station, None)
            self.assertEqual(receipt.status, "ERROR")
            self.assertEqual(transport.calls, [])

    def test_qr_navigation_requires_independent_fresh_route_authorization(self):
        obs, context = fixture()
        obs.update(kind="QR", coord_frame="NONE", target_pose_valid=False,
                   x_mm=None, y_mm=None, raw_payload="156+123+516+231")
        nav = dict(context, navigation_authorization="EXPLICIT_ENABLED", current_pose_verified=True,
                   current_station="RAW", route_verified=True, route_destination="PROCESS",
                   navigation_timestamp_ms=100)
        transport = Transport()
        port = LiveMotionPort(nav, transport, clock_ms=lambda: 200)
        port.update_observation(obs, 100)
        self.assertEqual(port.drive_to("PROCESS").status, "DONE")
        self.assertEqual(len(transport.calls), 1)
        for key in ("motion_enabled", "motion_authorization", "navigation_authorization", "current_pose_verified",
                    "current_station", "route_verified", "route_destination", "navigation_timestamp_ms",
                    "lane_safe", "obstacle_clear", "active_binding", "current_observation_identity"):
            denied = dict(nav)
            denied[key] = None
            transport = Transport()
            port = LiveMotionPort(denied, transport, clock_ms=lambda: 200)
            port.update_observation(obs, 100)
            self.assertEqual(port.drive_to("PROCESS").status, "ERROR", key)
            self.assertEqual(transport.calls, [], key)
        for change in (dict(is_simulated=True), dict(timestamp_ms=-1000)):
            transport = Transport()
            port = LiveMotionPort(nav, transport, clock_ms=lambda: 200)
            port.update_observation(dict(obs, **change), 100)
            self.assertEqual(port.drive_to("PROCESS").status, "ERROR")
            self.assertEqual(transport.calls, [])

    def test_live_drive_to_qr_submits_exact_destination(self):
        obs, context = fixture()
        obs.update(kind="QR", coord_frame="NONE", target_pose_valid=False,
                   x_mm=None, y_mm=None, raw_payload="156+123+516+231")
        context.update(navigation_authorization="EXPLICIT_ENABLED", current_pose_verified=True,
                       current_station="RAW", route_verified=True, route_destination="QR",
                       navigation_timestamp_ms=100)
        transport = Transport()
        port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
        port.update_observation(obs, 100)
        self.assertEqual(port.drive_to("QR").status, "DONE")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][1:], ("drive_to", {"station": "QR"}))

    def test_invalid_identity_returns_error_even_with_matching_bindings(self):
        for field, values in (("frame_id", (None, "7", True, 7.0, -1, 4294967296, [])),
                              ("camera_id", (None, True, 2.0, -1, "", [], {}))):
            for value in values:
                for navigate in (False, True):
                    with self.subTest(field=field, value=value, navigate=navigate):
                        obs, context = fixture()
                        obs[field] = value
                        if field == "camera_id":
                            context["active_binding"][field] = value
                            context["calibration"][field] = value
                        context["current_observation_identity"] = (obs["camera_id"], obs["frame_id"], obs["target_id"])
                        context.update(navigation_authorization="EXPLICIT_ENABLED", current_pose_verified=True,
                                       current_station="RAW", route_verified=True, route_destination="QR",
                                       navigation_timestamp_ms=100)
                        transport = Transport()
                        port = LiveMotionPort(context, transport, clock_ms=lambda: 200)
                        port.update_observation(obs, 100)
                        receipt = port.drive_to("QR") if navigate else port.pick_to_carrier(1, "RAW", 1)
                        self.assertEqual(receipt.status, "ERROR")
                        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
