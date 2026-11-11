"""Independent six-object ledger. Every mutation is validated before commit."""


class FieldError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class FieldSim:
    def __init__(self, task, capacity=3, start_id="START_1"):
        if type(capacity) is not int or capacity not in (1, 2, 3):
            raise FieldError("INVALID_CAPACITY")
        if start_id not in ("START_1", "START_2"):
            raise FieldError("INVALID_START")
        self.task, self.capacity, self.start_id = task, capacity, start_id
        self.station, self.gripper = start_id, None
        self.raw = dict((batch, dict((color, "B%d:C%d" % (batch, color)) for color in colors))
                        for batch, colors in ((1, task.batch1_colors), (2, task.batch2_colors)))
        self.process, self.buffer, self.carrier = {}, {}, {}
        self.completed = {"1_PROCESS": [], "1_BUFFER": [], "2_PROCESS": [], "2_BUFFER": []}
        self.transitions = []
        self.routes = []

    def _barrier(self, batch, source):
        if batch == 2 and len(self.completed["1_BUFFER"]) != 3:
            raise FieldError("FIRST_BATCH_INCOMPLETE")
        if source == "PROCESS" and len(self.completed["%d_PROCESS" % batch]) != 3:
            raise FieldError("PROCESS_PHASE_INCOMPLETE")

    def drive(self, station):
        if station not in ("QR", "RAW", "PROCESS", "BUFFER", "START_1", "START_2"):
            raise FieldError("UNKNOWN_STATION")
        if self.gripper is not None:
            raise FieldError("GRIPPER_HELD_DURING_TRAVEL")
        # A synthetic topological lane network, not measured field geometry.
        # Every station connects through one guarded lane hub; no free-space shortcut.
        path = [self.station] if station == self.station else [self.station, "SIM_LANE_HUB", station]
        self.routes.append({"path": path, "is_simulated": True, "lane_policy": "PRESET_SAFE_NODES"})
        self.station = station

    def pick(self, batch, color, source, slot):
        self._barrier(batch, source)
        if self.station != source or source not in ("RAW", "PROCESS"):
            raise FieldError("WRONG_SOURCE_STATION")
        if self.gripper is not None:
            raise FieldError("GRIPPER_OCCUPIED")
        if len(self.carrier) >= self.capacity:
            raise FieldError("CARRIER_FULL")
        colors = self.task.batch1_colors if batch == 1 else self.task.batch2_colors
        phase = "PROCESS" if source == "RAW" else "BUFFER"
        picked = [event["color"] for event in self.transitions
                  if event["batch"] == batch and event["source"] == source]
        if len(picked) >= 3 or colors[len(picked)] != color:
            raise FieldError("PICK_ORDER_MISMATCH")
        inventory = self.raw[batch] if source == "RAW" else self.process
        key = color if source == "RAW" else slot
        obj = inventory.get(key)
        if obj != "B%d:C%d" % (batch, color):
            raise FieldError("MISSING_OR_WRONG_COLOR")
        if source == "PROCESS":
            slots = self.task.batch1_slots if batch == 1 else self.task.batch2_slots
            if slot != slots[len(picked)]:
                raise FieldError("WRONG_SOURCE_RING")
        carrier_slot = next(n for n in range(1, self.capacity + 1) if n not in self.carrier)
        del inventory[key]
        self.carrier[carrier_slot] = obj
        self.transitions.append({"batch": batch, "phase": phase, "color": color,
                                 "object": obj, "source": source, "dest": "CARRIER",
                                 "slot": carrier_slot, "layer": None})
        self.audit()

    def place(self, batch, color, dest, slot, layer):
        if self.station != dest or dest not in ("PROCESS", "BUFFER"):
            raise FieldError("WRONG_DEST_STATION")
        self._barrier(batch, "PROCESS" if dest == "BUFFER" else "RAW")
        obj = "B%d:C%d" % (batch, color)
        matches = [key for key, value in self.carrier.items() if value == obj]
        if not matches:
            raise FieldError("CARGO_MISSING")
        phase_key = "%d_%s" % (batch, dest)
        colors = self.task.batch1_colors if batch == 1 else self.task.batch2_colors
        index = len(self.completed[phase_key])
        if index >= 3 or color != colors[index]:
            raise FieldError("PLACE_ORDER_MISMATCH")
        slots = self.task.batch1_slots if batch == 1 else self.task.batch2_slots
        expected = slots[index] if dest == "PROCESS" else self.task.color_to_buffer_slot[color]
        if slot != expected or layer != (batch if dest == "BUFFER" else 1):
            raise FieldError("WRONG_RING_OR_LAYER")
        if dest == "PROCESS":
            if slot in self.process:
                raise FieldError("RING_OCCUPIED")
        else:
            layers = self.buffer.get(slot, {})
            if layer in layers:
                raise FieldError("LAYER_OCCUPIED")
            if layer == 2 and layers.get(1) != "B1:C%d" % color:
                raise FieldError("MATCHING_FIRST_LAYER_REQUIRED")
        del self.carrier[matches[0]]
        if dest == "PROCESS":
            self.process[slot] = obj
        else:
            self.buffer.setdefault(slot, {})[layer] = obj
        self.completed[phase_key].append(color)
        self.transitions.append({"batch": batch, "phase": dest, "color": color,
                                 "object": obj, "source": "CARRIER", "dest": dest,
                                 "slot": slot, "layer": layer})
        self.audit()

    def return_to_start(self, start_id):
        if start_id != self.start_id or len(self.completed["2_BUFFER"]) != 3:
            raise FieldError("RETURN_PRECONDITION")
        if self.carrier or self.gripper:
            raise FieldError("CARRIER_NOT_EMPTY")
        self.drive(start_id)

    def audit(self):
        objects = list(self.raw[1].values()) + list(self.raw[2].values())
        objects += list(self.process.values()) + list(self.carrier.values())
        objects += [obj for layers in self.buffer.values() for obj in layers.values()]
        if self.gripper is not None:
            objects.append(self.gripper)
        expected = {"B%d:C%d" % (batch, color) for batch, colors in
                    ((1, self.task.batch1_colors), (2, self.task.batch2_colors)) for color in colors}
        if len(objects) != 6 or set(objects) != expected or len(set(objects)) != 6:
            raise FieldError("OBJECT_CONSERVATION_FAILED")
        if len(self.carrier) > self.capacity:
            raise FieldError("CAPACITY_EXCEEDED")
        return True

    def snapshot(self):
        return {"station": self.station, "gripper": self.gripper,
                "raw": {batch: dict(items) for batch, items in self.raw.items()},
                "process": dict(self.process), "carrier": dict(self.carrier),
                "buffer": {slot: dict(layers) for slot, layers in self.buffer.items()},
                "routes": [{"path": list(route["path"]), "is_simulated": True,
                            "lane_policy": route["lane_policy"]} for route in self.routes],
                "completed": {key: list(items) for key, items in self.completed.items()}}

    def is_complete(self):
        self.audit()
        return (self.station == self.start_id and not self.carrier and not self.gripper
                and not self.process and not self.raw[1] and not self.raw[2]
                and all(self.buffer.get(slot) == {1: "B1:C%d" % color, 2: "B2:C%d" % color}
                        for color, slot in self.task.color_to_buffer_slot.items()))
