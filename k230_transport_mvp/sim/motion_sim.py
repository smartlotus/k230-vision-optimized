from sim.field_sim import FieldError


class MotionSim:
    """DONE follows mutation; ACK loss deliberately leaves execution uncertain."""
    def __init__(self, field, inject=None):
        self.field, self.inject = field, inject
        self.command_id = 0
        self.batch, self.step, self.observation = None, None, None
        self.injected = False
        self.receipts = []
        self.max_carrier_occupancy = 0

    def prepare(self, step, observation=None):
        self.batch, self.step, self.observation = step.batch, step, observation

    def _execute(self, operation, callback, mechanical=False):
        self.command_id += 1
        receipt = {"status": "DONE", "command_id": self.command_id,
                   "command": operation, "is_simulated": True}
        try:
            if mechanical:
                observation = self.observation
                if (not observation or observation.get("status") != "OK"
                        or observation.get("is_simulated") is not True or not observation.get("stable")
                        or not observation.get("target_pose_valid")
                        or observation.get("calibration_id") != "SIM_ONLY_V1"):
                    raise FieldError("SIM_OBSERVATION_REQUIRED")
                if not self.injected and self.inject in ("timeout", "carrier-full"):
                    self.injected = True
                    receipt.update(status="UNKNOWN" if self.inject == "timeout" else "ERROR",
                                   reason="ACK_TIMEOUT" if self.inject == "timeout" else "CARRIER_FULL")
                    self.receipts.append(receipt)
                    return receipt
            if not self.injected and self.inject == "obstacle" and operation == "DRIVE":
                self.injected = True
                raise FieldError("SIM_OBSTACLE_ROUTE_BLOCKED")
            callback()
            self.max_carrier_occupancy = max(self.max_carrier_occupancy, len(self.field.carrier))
            if mechanical and not self.injected and self.inject == "ack-loss":
                self.injected = True
                receipt.update(status="UNKNOWN", reason="ACK_LOST_AFTER_EXECUTION")
        except FieldError as exc:
            receipt.update(status="ERROR", reason=exc.reason)
        self.receipts.append(receipt)
        return receipt

    def drive_to(self, station):
        return self._execute("DRIVE", lambda: self.field.drive(station))

    def pick_to_carrier(self, color_id, source, slot):
        return self._execute("PICK", lambda: self.field.pick(self.batch, color_id, source, slot), True)

    def place_from_carrier(self, color_id, dest, slot, layer):
        return self._execute("PLACE", lambda: self.field.place(self.batch, color_id, dest, slot, layer), True)

    def return_to_start(self, start_id):
        return self._execute("RETURN", lambda: self.field.return_to_start(start_id))
