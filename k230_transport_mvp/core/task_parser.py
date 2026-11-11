from core.types import TaskPlan, ValidationError


def build_color_to_buffer_slot(batch1_colors, batch1_slots):
    colors, slots = tuple(batch1_colors), tuple(batch1_slots)
    if (len(colors) != 3 or len(set(colors)) != 3
            or any(type(c) is not int or c not in range(1, 7) for c in colors)
            or any(type(s) is not int for s in slots) or set(slots) != {1, 2, 3}
            or len(slots) != 3):
        raise ValidationError("Expected three distinct colors and ring permutation")
    return dict(zip(colors, slots))


def parse_task(payload):
    if not isinstance(payload, str):
        raise ValidationError("Task must be text")
    groups = payload.split("+")
    if len(groups) != 4 or any(len(g) != 3 for g in groups):
        raise ValidationError("Expected AAA+BBB+CCC+DDD")
    if any(ch not in "123456" for i in (0, 2) for ch in groups[i]):
        raise ValidationError("Color IDs must be ASCII digits 1..6")
    if any(sorted(groups[i]) != ["1", "2", "3"] for i in (1, 3)):
        raise ValidationError("Ring IDs must be a permutation of 123")
    values = [tuple(int(ch) for ch in group) for group in groups]
    if len(set(values[0])) != 3 or len(set(values[2])) != 3:
        raise ValidationError("Each batch requires three different colors")
    if set(values[0]) != set(values[2]):
        raise ValidationError("Both batches must have the same colors")
    return TaskPlan(payload, *values)
