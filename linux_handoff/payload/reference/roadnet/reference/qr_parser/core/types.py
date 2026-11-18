"""Lightweight structures shared by CPython and MicroPython."""


class ValidationError(ValueError):
    def __init__(self, message, code="INVALID_TASK"):
        super().__init__(message)
        self.code = code


class TaskPlan:
    def __init__(self, payload, batch1_colors, batch1_slots, batch2_colors, batch2_slots):
        self.payload = payload
        self.batch1_colors = tuple(batch1_colors)
        self.batch1_slots = tuple(batch1_slots)
        self.batch2_colors = tuple(batch2_colors)
        self.batch2_slots = tuple(batch2_slots)
        self.color_to_buffer_slot = dict(zip(self.batch1_colors, self.batch1_slots))


class Step:
    def __init__(self, action, batch, phase, source_station, dest_station,
                 target_color=None, target_slot=None, pick_index=None, layer=1,
                 requires_carrier_unload=False, preconditions=()):
        self.action = action
        self.batch = batch
        self.phase = phase
        self.source_station = source_station
        self.dest_station = dest_station
        self.target_color = target_color
        self.target_slot = target_slot
        self.pick_index = pick_index
        self.layer = layer
        self.requires_carrier_unload = requires_carrier_unload
        self.preconditions = tuple(preconditions)

    def as_dict(self):
        return dict((key, getattr(self, key)) for key in (
            "action", "batch", "phase", "source_station", "dest_station",
            "target_color", "target_slot", "pick_index", "layer",
            "requires_carrier_unload", "preconditions"))
