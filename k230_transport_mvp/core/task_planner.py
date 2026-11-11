from core.types import Step, ValidationError


def build_execution_steps(task, capacity=3):
    if type(capacity) is not int or capacity not in (1, 2, 3):
        raise ValidationError("Carrier capacity must be integer 1, 2 or 3")
    steps = []
    station = "QR"
    for batch, colors, slots in ((1, task.batch1_colors, task.batch1_slots),
                                  (2, task.batch2_colors, task.batch2_slots)):
        for phase, source, dest in (("PROCESS", "RAW", "PROCESS"),
                                    ("BUFFER", "PROCESS", "BUFFER")):
            for offset in range(0, 3, capacity):
                indices = range(offset, min(offset + capacity, 3))
                if station != source:
                    steps.append(Step("MOVE", batch, phase, station, source,
                                      requires_carrier_unload=True,
                                      preconditions=("GRIPPER_EMPTY", "CARRIER_EMPTY")))
                    station = source
                for index in indices:
                    steps.append(Step("PICK", batch, phase, source, "CARRIER",
                                      colors[index], slots[index] if source == "PROCESS" else None,
                                      index, preconditions=("PHASE_BARRIER", "CAPACITY_AVAILABLE")))
                steps.append(Step("MOVE", batch, phase, source, dest,
                                  preconditions=("GRIPPER_EMPTY", "CARRIER_LOADED")))
                station = dest
                for index in indices:
                    target = (slots[index] if dest == "PROCESS"
                              else task.color_to_buffer_slot[colors[index]])
                    layer = batch if dest == "BUFFER" else 1
                    steps.append(Step("PLACE", batch, phase, "CARRIER", dest,
                                      colors[index], target, index, layer,
                                      requires_carrier_unload=True,
                                      preconditions=("MATCHING_CARGO", "TARGET_VALID")))
    steps.append(Step("RETURN", 2, "RETURN", station, "ORIGINAL_START",
                      requires_carrier_unload=True,
                      preconditions=("ALL_BUFFER_COMPLETE", "CARRIER_EMPTY", "GRIPPER_EMPTY")))
    return steps
