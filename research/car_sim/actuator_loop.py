"""Versioned PC loopback channel for continuous virtual STM32 commands."""

import json
import math
import zlib


VERSION = "K230SIM1"
MAX_BYTES = 2048
MOTION_AXES = {
    "BASE_VELOCITY": ("BASE", {"vx_mps": .30, "vy_mps": .50, "yaw_rate_rad_s": 2.}),
    "TURRET_MOVE": ("TURRET", {"theta_deg": 180.}),
    "SLIDE_MOVE": ("SLIDE", {"slide_mm": 100.}),
    "LIFT_MOVE": ("LIFT", {"optical_height_mm": 400.}),
    "JAW_MOVE": ("JAW", {"gap_mm": 60.}),
}


def encode(frame):
    encoded = json.dumps({"version": VERSION, **frame}, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    packet = encoded + b"*" + f"{zlib.crc32(encoded):08X}".encode("ascii") + b"\n"
    if len(packet) > MAX_BYTES:
        raise ValueError("Simulator frame length")
    return packet


def decode(packet, kind):
    if len(packet) > MAX_BYTES or not packet.endswith(b"\n") or b"\n" in packet[:-1]:
        raise ValueError("Simulator framing")
    body, separator, checksum = packet[:-1].rpartition(b"*")
    if not separator or checksum != f"{zlib.crc32(body):08X}".encode("ascii"):
        raise ValueError("Simulator checksum")
    frame = json.loads(body)
    if frame.get("version") != VERSION or frame.get("type") != kind:
        raise ValueError("Simulator frame version or type")
    return frame


def check_action(action, now_ms):
    spec = MOTION_AXES.get(action.get("command"))
    if not spec or action.get("axis") != spec[0]:
        raise ValueError("AXIS_COMMAND_MISMATCH")
    values = action.get("values")
    if not isinstance(values, dict) or values.keys() != spec[1].keys():
        raise ValueError("COMMAND_VALUES_INVALID")
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > spec[1][k]
           for k, v in values.items()):
        raise ValueError("COMMAND_VALUE_LIMIT")
    key, sequence = action.get("action_id"), action.get("sequence")
    if not isinstance(key, str) or not key or len(key) > 64 or type(sequence) is not int or sequence < 0:
        raise ValueError("ACTION_ID_INVALID")
    issued, expires = action.get("issued_ms"), action.get("valid_until_ms")
    if (type(issued) is not int or type(expires) is not int or not issued <= now_ms <= expires
            or not 0 < expires - issued <= 1000):
        raise ValueError("ACTION_EXPIRED_OR_CLOCK_INVALID")
    observation = action.get("observation_id")
    if observation is not None and (type(observation) is not int or observation < 0):
        raise ValueError("OBSERVATION_ID_INVALID")


class ActuatorLoop:
    """One virtual controller per episode. Only terminal receipts can mutate control state."""

    def __init__(self):
        self.last_sequence = -1
        self.completed = {}
        self.executed = 0

    def receive(self, packet, now_ms, apply):
        action = decode(packet, "COMMAND")
        key = action.get("action_id")
        cached = self.completed.get(key) if isinstance(key, str) else None
        if cached is not None:
            if packet != cached[0]:
                raise ValueError("ACTION_ID_REBOUND")
            return [cached[1]]
        check_action(action, now_ms)
        sequence = action["sequence"]
        if sequence <= self.last_sequence:
            raise ValueError("COMMAND_SEQUENCE_REPLAY")
        self.last_sequence = sequence
        base = {"type": "RECEIPT", "action_id": key, "sequence": sequence,
                "actuator": action["axis"], "observation_id": action["observation_id"],
                "simulated": True, "received_ms": now_ms}
        ack = encode({**base, "status": "ACK"})
        feedback = apply(action["values"])
        self.executed += 1
        done = encode({**base, "status": "DONE", "feedback": feedback})
        self.completed[key] = packet, done
        return [ack, done]
