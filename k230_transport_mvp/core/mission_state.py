"""One-command-at-a-time engine. Only confirmed DONE advances state."""
from core.stats import Stats
from core.task_parser import parse_task
from core.task_planner import build_execution_steps


class MissionEngine:
    def __init__(self, motion, camera, vision=None, display=None, logger=None):
        self.motion, self.camera = motion, camera
        self.vision, self.display, self.logger = vision or camera, display, logger
        self.status, self.reason = "READY", None
        self.events, self.observations = [], []
        self.stats = Stats()
        self.completed = {"1_PROCESS": 0, "1_BUFFER": 0, "2_PROCESS": 0, "2_BUFFER": 0}
        self.carrier = []
        self.station, self.start_id, self.task = None, None, None
        self.last_command_id, self.step_index = 0, 0
        self.confirmed_steps = 0
        self.pending_command = None
        self.elapsed_ms = 0
        self.last_frames = {}

    def _event(self, kind, **data):
        event = {"event": kind, "index": len(self.events), "is_simulated": True}
        event.update(data)
        self.events.append(event)
        if self.logger is not None:
            self.logger(event)

    def _fail(self, reason):
        self.status, self.reason = "ERROR", reason
        self._event("MISSION_ERROR", reason=reason, step_index=self.step_index,
                    pending_command=self.pending_command)
        return self.summary()

    def _observe(self, kind, target=None, station=None):
        expected_camera = 2 if kind == "QR" else 0
        frame = self.camera.get_frame(expected_camera)
        if frame is None:
            return None
        obs = self.vision.detect(kind, frame, target_id=target, station=station)
        if obs is None:
            return None
        obs = dict(obs)
        previous = self.last_frames.get(expected_camera)
        frame_valid = (frame.get("camera_id") == expected_camera
                       and frame.get("synthetic") is True
                       and type(frame.get("frame_id")) is int
                       and type(frame.get("timestamp_ms")) is int
                       and (previous is None or (frame["frame_id"] > previous[0]
                            and frame["timestamp_ms"] >= previous[1])))
        if frame_valid:
            self.last_frames[expected_camera] = (frame["frame_id"], frame["timestamp_ms"])
        stale_after = obs.get("stale_after_ms", 200)
        # Simulation uses one monotonic clock; LIVE UART clocks are deliberately
        # handled by the separate receive-time safety boundary.
        if (not frame_valid or obs.get("kind") != kind
                or obs.get("camera_id") != expected_camera
                or obs.get("frame_id") != frame["frame_id"]
                or type(obs.get("timestamp_ms")) is not int
                or type(stale_after) is not int or stale_after <= 0
                or frame["timestamp_ms"] - obs["timestamp_ms"] < 0
                or frame["timestamp_ms"] - obs["timestamp_ms"] > stale_after):
            obs.update(status="ERROR", reason="STALE_OR_MISMATCHED_OBSERVATION", x_mm=None, y_mm=None)
        self.observations.append(dict(obs))
        self._event("OBSERVATION", observation=dict(obs))
        return obs

    def _precondition(self, step):
        if step.batch == 2 and self.completed["1_BUFFER"] != 3:
            return "FIRST_BATCH_INCOMPLETE"
        if step.action == "PICK" and step.source_station == "PROCESS":
            if self.completed["%d_PROCESS" % step.batch] != 3:
                return "PROCESS_PHASE_INCOMPLETE"
        if step.action == "PICK" and len(self.carrier) >= self.capacity:
            return "CARRIER_FULL"
        if step.action == "PLACE" and (step.batch, step.target_color) not in self.carrier:
            return "CARGO_MISSING"
        if step.action == "MOVE" and "CARRIER_EMPTY" in step.preconditions and self.carrier:
            return "CARRIER_NOT_EMPTY"
        if step.action == "MOVE" and "CARRIER_LOADED" in step.preconditions and not self.carrier:
            return "CARRIER_NOT_LOADED"
        if step.action == "RETURN" and (self.carrier or self.completed["2_BUFFER"] != 3):
            return "RETURN_PRECONDITION"
        return None

    def run(self, payload, capacity=3, start_id="START_1", steps=None):
        if self.status != "READY":
            raise ValueError("MissionEngine instances cannot be replayed")
        if start_id not in ("START_1", "START_2"):
            return self._fail("INVALID_START")
        self.start_id, self.station, self.capacity = start_id, start_id, capacity
        self.task = parse_task(payload)
        planned = build_execution_steps(self.task, capacity) if steps is None else list(steps)
        self.status = "RUNNING"
        receipt = self.motion.drive_to("QR")
        error = self._confirm(receipt, "MOVE")
        if error:
            return self._fail(error)
        self.station = "QR"
        last_payload, confirmations = None, 0
        for _ in range(6):
            observation = self._observe("QR", station="QR")
            observed = observation.get("raw_payload") if observation else None
            if (not observation or observation.get("status") != "OK"
                    or observation.get("is_simulated") is not True):
                confirmations, last_payload = 0, None
                continue
            try:
                parse_task(observed)
            except ValueError:
                confirmations, last_payload = 0, None
                continue
            confirmations = confirmations + 1 if observed == last_payload else 1
            last_payload = observed
            if confirmations == 3:
                break
        if confirmations != 3 or last_payload != payload:
            return self._fail("QR_NOT_CONFIRMED")
        self._event("TASK_CONFIRMED", payload=payload, confirmations=3)
        if self.display:
            self.display.show_task(payload)
        for index, step in enumerate(planned):
            self.step_index = index
            error = self._precondition(step)
            if error:
                return self._fail(error)
            observation = None
            if step.action in ("PICK", "PLACE"):
                mode = "COLOR" if step.action == "PICK" else ("TOP" if step.layer == 2 else "RING")
                target = step.target_color if mode in ("COLOR", "TOP") else step.target_slot
                station = step.source_station if step.action == "PICK" else step.dest_station
                observation = self._observe(mode, target, station)
                if not observation or observation.get("status") != "OK":
                    return self._fail((observation or {}).get("reason", "OBSERVATION_UNAVAILABLE"))
                if (observation.get("target_id") != target or observation.get("kind") != mode
                        or observation.get("station") != station or not observation.get("stable")
                        or observation.get("camera_id") != 0
                        or observation.get("is_simulated") is not True):
                    return self._fail("OBSERVATION_TARGET_MISMATCH")
            if hasattr(self.motion, "prepare"):
                self.motion.prepare(step, observation)
            self.pending_command = step.as_dict()
            self._event("COMMAND", step=step.as_dict())
            if step.action == "MOVE":
                receipt = self.motion.drive_to(step.dest_station)
            elif step.action == "PICK":
                receipt = self.motion.pick_to_carrier(step.target_color, step.source_station, step.target_slot)
            elif step.action == "PLACE":
                receipt = self.motion.place_from_carrier(step.target_color, step.dest_station, step.target_slot, step.layer)
            elif step.action == "RETURN":
                receipt = self.motion.return_to_start(start_id)
            else:
                return self._fail("UNKNOWN_ACTION")
            error = self._confirm(receipt, step.action)
            if error:
                return self._fail(error)
            if step.action == "PICK":
                self.carrier.append((step.batch, step.target_color))
            elif step.action == "PLACE":
                self.carrier.remove((step.batch, step.target_color))
                self.completed["%d_%s" % (step.batch, step.dest_station)] += 1
            elif step.action in ("MOVE", "RETURN"):
                self.station = start_id if step.action == "RETURN" else step.dest_station
            self.pending_command = None
            self.confirmed_steps += 1
        if (self.stats.picks != 12 or self.stats.placements != 12 or self.carrier
                or any(value != 3 for value in self.completed.values()) or self.station != start_id):
            return self._fail("INCOMPLETE_FINAL_STATE")
        self.status = "DONE"
        self._event("MISSION_DONE", counters=self.stats.as_dict(), station=self.station)
        if self.display:
            self.display.show_stats(self.stats.as_dict())
        return self.summary()

    def _confirm(self, receipt, action):
        if receipt is None or not hasattr(receipt, "get"):
            self._event("RECEIPT", action=action, receipt=None)
            return "INVALID_RECEIPT"
        self._event("RECEIPT", action=action, receipt=dict(receipt))
        if receipt.get("is_simulated", True) is not True:
            return "RECEIPT_SOURCE_MISMATCH"
        command_id = receipt.get("command_id")
        if type(command_id) is not int or command_id <= self.last_command_id:
            return "STALE_OR_INVALID_RECEIPT"
        expected_command = "DRIVE" if action == "MOVE" else action
        if receipt.get("command", expected_command) != expected_command:
            return "RECEIPT_COMMAND_MISMATCH"
        self.last_command_id = command_id
        status = receipt.get("status")
        if status != "DONE":
            return receipt.get("reason") or "MOTION_%s" % status
        self.stats.confirm(action)
        self.elapsed_ms += 250 if action in ("PICK", "PLACE") else 500
        return None

    def summary(self):
        return {"status": self.status, "reason": self.reason, "is_simulated": True,
                "task": self.task.payload if self.task else None, "start_id": self.start_id,
                "station": self.station, "counters": self.stats.as_dict(),
                "completed": dict(self.completed), "confirmed_steps": self.confirmed_steps,
                "simulated_elapsed_ms": self.elapsed_ms,
                "time_claim": "Synthetic action durations; not real competition timing",
                "pending_command": self.pending_command}
