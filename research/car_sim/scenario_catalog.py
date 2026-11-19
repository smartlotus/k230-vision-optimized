"""The six R3 development smoke cases and two negative QR observations."""

import random


ORIGINAL_TASK = "156+123+516+231"
SMOKE_PAYLOADS = (
    ORIGINAL_TASK,
    "651+312+165+213",
    "234+321+432+132",
    "245+231+542+213",
    "136+312+361+123",
    "124+132+421+321",
)


def qr_cases():
    cases = []
    for index, payload in enumerate((*SMOKE_PAYLOADS, "156+113+516+231", "")):
        seed = 3000 + index
        rng = random.Random(seed)
        legal = index < len(SMOKE_PAYLOADS)
        layout_task = payload if legal else ORIGINAL_TASK
        groups = layout_task.split("+")
        orders = [rng.sample(list(map(int, groups[batch])), 3) for batch in (0, 2)]
        cases.append({
            "key": f"qr-{seed}", "seed": seed, "split": "development" if legal else "negative",
            "payload": payload, "layout_task": layout_task,
            "start_id": f"START_{index % 2 + 1}", "orders": orders,
            "exposure_ev": (0.0, -.20, .20, -.10, .10, .05, 0.0, 0.0)[index],
            "expected": "COMPLETE" if legal else "HOLD_INVALID_TASK" if payload else "HOLD_NO_QR",
            "layout_family": "r2-gray-v1",
        })
    return cases


def drive_cases():
    perturbations = [
        {"offset_mm": -20, "yaw_deg": 0},
        {"offset_mm": 20, "yaw_deg": 0},
        {"offset_mm": -40, "yaw_deg": 0},
        {"offset_mm": 40, "yaw_deg": 0},
        {"offset_mm": 0, "yaw_deg": -5},
        {"offset_mm": 0, "yaw_deg": 5},
        {"offset_mm": 20, "yaw_deg": 0, "lateral_slip_mm_per_s": 6, "lateral_response": .68},
        {"offset_mm": -40, "yaw_deg": 0, "delay_frames": 1, "drop_frames": [5]},
    ]
    return [{"key": f"drive-{seed}", "seed": seed, "split": "development",
             "layout_family": "r2-gray-v1", **case}
            for seed, case in enumerate(perturbations, 3201)]
