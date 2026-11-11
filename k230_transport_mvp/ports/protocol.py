"""Bounded ASCII UART protocol; compatible with CPython and MicroPython."""
import math
import time
try:
    import json
except ImportError:
    import ujson as json

MAX_LINE = 768
MODES = ("IDLE", "QR", "COLOR", "RING", "LANE", "TOP", "SELFTEST")
STATUSES = ("OK", "NOT_FOUND", "UNCALIBRATED", "UNAVAILABLE", "ERROR")
KINDS = ("QR", "COLOR", "RING", "LANE", "TOP")
ACK_STATUSES = ("ACK", "BUSY", "DONE", "ERROR", "UNKNOWN", "STOPPED", "PONG")
MATERIAL_STATUSES = ("OK", "NOT_FOUND", "UNKNOWN", "UNAVAILABLE", "ERROR")
MATERIAL_HEADER = ("version", "status", "camera_id", "frame_id", "timestamp_ms",
                   "profile_id", "width", "height", "coord_frame", "is_simulated",
                   "complete", "count", "reason", "source_session")
MATERIAL_FIELDS = ("color_id", "u_px", "v_px", "semimajor_px", "semiminor_px",
                   "angle_rad", "score_hint", "stable", "geometry_valid")


def _decode_material(parts):
    """Complete, source-pixel material snapshot; no target IDs or motion units."""
    if len(parts) < 16:
        raise ValueError("material header count")
    result = dict(zip(MATERIAL_HEADER, parts[2:16]))
    if result["version"] != "1" or result["status"] not in MATERIAL_STATUSES:
        raise ValueError("material version/status")
    result["version"] = 1
    if result["camera_id"] not in ("CSI1", "CSI2") or result["coord_frame"] != "IMAGE_PX":
        raise ValueError("material source")
    profile = result["profile_id"]
    if len(profile) != 64 or any(c not in "0123456789abcdef" for c in profile):
        raise ValueError("profile must be lowercase SHA256")
    session = result["source_session"]
    if len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
        raise ValueError("source session must be 128-bit lowercase hex")
    for name in ("frame_id", "timestamp_ms"):
        result[name] = _uint(result[name], 4294967295)
    for name in ("width", "height"):
        result[name] = _uint(result[name], 4096)
        if not result[name]:
            raise ValueError("material dimensions")
    for name in ("is_simulated", "complete"):
        result[name] = _bool(result[name])
    count = _uint(result["count"], 3)
    result["count"] = count
    if len(parts) != 16 + count * len(MATERIAL_FIELDS):
        raise ValueError("material object count")
    positive = result["status"] == "OK"
    if positive != (count > 0) or result["complete"] != (result["status"] in ("OK", "NOT_FOUND")):
        raise ValueError("material completeness/status")
    result["reason"] = unescape_field(result["reason"])
    if not positive and result["status"] != "NOT_FOUND" and not result["reason"]:
        raise ValueError("material failure needs reason")
    result["objects"] = []
    for index in range(count):
        offset = 16 + index * len(MATERIAL_FIELDS)
        row = dict(zip(MATERIAL_FIELDS, parts[offset:offset + len(MATERIAL_FIELDS)]))
        row["color_id"] = _uint(row["color_id"], 6)  # 0 is explicitly unknown.
        for name in MATERIAL_FIELDS[1:7]:
            row[name] = _number(row[name])
            if row[name] is None and name != "score_hint":
                raise ValueError("missing material geometry")
        for name in ("stable", "geometry_valid"):
            row[name] = _bool(row[name])
        u, v, a, b, angle = (row[k] for k in MATERIAL_FIELDS[1:6])
        if not row["geometry_valid"] or not 0 < b <= a <= max(result["width"], result["height"]):
            raise ValueError("material semiaxes/geometry")
        if not 0 <= angle < math.pi or not 0 <= u < result["width"] or not 0 <= v < result["height"]:
            raise ValueError("material center/angle")
        dx = math.sqrt((a * math.cos(angle)) ** 2 + (b * math.sin(angle)) ** 2)
        dy = math.sqrt((a * math.sin(angle)) ** 2 + (b * math.cos(angle)) ** 2)
        if u-dx < -0.0001 or v-dy < -0.0001 or u+dx > result["width"]-1+0.0001 or v+dy > result["height"]-1+0.0001:
            raise ValueError("material clipped boundary")
        if row["score_hint"] is not None and not 0 <= row["score_hint"] <= 100:
            raise ValueError("material score")
        result["objects"].append(row)
    return result
OBS_FIELDS = ("status", "kind", "camera_id", "frame_id", "timestamp_ms",
              "target_id", "coord_frame", "u_px", "v_px", "x_mm", "y_mm",
              "angle_deg", "score_hint", "stable", "is_simulated",
              "calibration_id", "station", "plane", "image_profile",
              "target_pose_valid", "reason", "raw_payload")


def escape_field(value):
    """Percent-encode reserved ASCII and UTF-8; no lossy profile sanitizing."""
    if value is None or value == "":
        return "-"
    output = []
    for byte in str(value).encode("utf-8"):
        if 32 <= byte <= 126 and byte not in (37, 44, 42, 36):
            output.append(chr(byte))
        else:
            output.append("%%%02X" % byte)
    return "".join(output)


def unescape_field(value):
    if value == "-":
        return None
    output = bytearray()
    index = 0
    while index < len(value):
        if value[index] == "%":
            token = value[index+1:index+3]
            if len(token) != 2 or any(c not in "0123456789ABCDEF" for c in token):
                raise ValueError("bad percent escape")
            output.append(int(token, 16))
            index += 3
        else:
            output.append(ord(value[index]))
            index += 1
    return bytes(output).decode("utf-8")


def _canonical_json(value):
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + _canonical_json(value[k]) for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json(v) for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite profile")
    return json.dumps(value)


def _reason(value):
    # Diagnostic text is bounded and may be truncated; binding fields are never truncated.
    encoded = escape_field(value)
    if len(encoded) <= 96:
        return encoded
    encoded = encoded[:93]
    while encoded.endswith("%") or (len(encoded) > 1 and encoded[-2] == "%"):
        encoded = encoded[:-1]
    # ASCII replacement prevents cutting a multibyte UTF-8 sequence.
    while True:
        try:
            unescape_field(encoded)
            return encoded + "..."
        except UnicodeError:
            encoded = encoded[:-3]


def _text(value, limit=96):
    if value is None or value == "":
        return "-"
    value = str(value)
    if len(value) > limit or any(ord(c) < 32 or ord(c) > 126 or c in ",*$" for c in value):
        raise ValueError("invalid ASCII field")
    return value


def _uint(value, upper=65535):
    if isinstance(value, bool):
        raise ValueError("boolean integer")
    token = str(value)
    if not token or any(c not in "0123456789" for c in token):
        raise ValueError("invalid integer")
    result = int(token)
    if result > upper:
        raise ValueError("integer range")
    return result


def _number(value):
    if value == "-" or value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean number")
    token = str(value)
    if not token or any(c not in "0123456789+-.eE" for c in token):
        raise ValueError("invalid number encoding")
    result = float(value)
    if not math.isfinite(result) or abs(result) > 10000000:
        raise ValueError("nonfinite/out-of-range number")
    return result


def _bool(value):
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError("boolean must be 0/1")


def _optional(value):
    return None if value == "-" else value


def _decode_body(body):
    parts = body.split(",")
    if len(parts) < 2:
        raise ValueError("field count")
    kind, seq = parts[0], _uint(parts[1])
    result = {"type": kind, "seq": seq}
    if kind == "C":
        if len(parts) != 6:
            raise ValueError("control field count")
        command, mode, target, station = parts[2:]
        if command not in ("MODE", "STOP", "PING"):
            raise ValueError("unknown control")
        if command == "MODE":
            if mode not in MODES:
                raise ValueError("unknown mode")
        elif (mode, target, station) != ("-", "-", "-"):
            raise ValueError("unused control fields")
        if target != "-":
            target = _uint(target, 6)
            if target < 1:
                raise ValueError("target range")
        result.update(command=command, mode=_optional(mode),
                      target_id=None if target == "-" else target,
                      station=_optional(station))
    elif kind == "A":
        if len(parts) != 5 or parts[3] not in ACK_STATUSES:
            raise ValueError("ack fields")
        result.update(command_seq=_uint(parts[2]), status=parts[3], reason=unescape_field(parts[4]))
    elif kind == "V":
        if len(parts) != 2 + len(OBS_FIELDS):
            raise ValueError("observation field count")
        result.update(zip(OBS_FIELDS, parts[2:]))
        if result["status"] not in STATUSES or result["kind"] not in KINDS:
            raise ValueError("observation enums")
        if result["coord_frame"] not in ("IMAGE_PX", "ROBOT_MM", "WORLD_MM", "NONE"):
            raise ValueError("coordinate frame")
        for name in ("frame_id", "timestamp_ms"):
            if name == "frame_id" and result[name] == "-" and result["status"] != "OK":
                result[name] = None
            else:
                result[name] = _uint(result[name], 4294967295)
        for name in ("u_px", "v_px", "x_mm", "y_mm", "angle_deg", "score_hint"):
            result[name] = _number(result[name])
        for name in ("stable", "is_simulated", "target_pose_valid"):
            result[name] = _bool(result[name])
        for name in ("target_id", "calibration_id", "station", "plane", "image_profile", "reason", "raw_payload"):
            result[name] = _optional(result[name])
        target = result["target_id"]
        if target is not None and all(c in "0123456789" for c in target):
            result["target_id"] = _uint(target)
        camera = result["camera_id"]
        if camera == "-":
            result["camera_id"] = None
        elif all(c in "0123456789" for c in camera):
            result["camera_id"] = _uint(camera)
        if result["score_hint"] is not None and not 0 <= result["score_hint"] <= 100:
            raise ValueError("score range")
        result["reason"] = unescape_field(parts[22])
        result["raw_payload"] = unescape_field(parts[23])
        if parts[20] != "-":
            profile = unescape_field(parts[20])
            if profile.startswith("{"):
                profile = json.loads(profile)
                if not isinstance(profile, dict):
                    raise ValueError("profile must be a dict")
            result["image_profile"] = profile
        if (result["status"] != "OK" or result["coord_frame"] in ("IMAGE_PX", "NONE")) and (result["x_mm"] is not None or result["y_mm"] is not None):
            raise ValueError("invalid millimeter coordinates")
        if result["coord_frame"] in ("ROBOT_MM", "WORLD_MM") and result["status"] == "OK" and (result["x_mm"] is None or result["y_mm"] is None):
            raise ValueError("missing millimeter coordinates")
    elif kind == "S":
        result.update(_decode_material(parts))
    else:
        raise ValueError("unknown frame type")
    return result


def encode_fields(fields):
    fields = list(fields)
    body = ",".join(_text(v, 384 if fields[0] == "V" and i == 20 else 96) for i, v in enumerate(fields))
    _decode_body(body)
    checksum = 0
    for value in body.encode("ascii"):
        checksum ^= value
    line = ("$%s*%02X\n" % (body, checksum)).encode("ascii")
    if len(line) > MAX_LINE:
        raise ValueError("frame too long")
    return line


def encode_control(seq, command, mode=None, target_id=None, station=None):
    return encode_fields(("C", seq, command, mode, target_id, station))


def encode_ack(seq, status, reason=None, command_seq=None):
    return encode_fields(("A", seq, seq if command_seq is None else command_seq, status, _reason(reason)))


def encode_observation(seq, observation):
    fields = ["V", seq]
    for key in OBS_FIELDS:
        value = observation.get(key)
        if key in ("stable", "is_simulated", "target_pose_valid"):
            if key == "target_pose_valid" and value is None:
                value = False
            if not isinstance(value, bool):
                raise ValueError("explicit boolean required: " + key)
            value = "1" if value else "0"
        if key == "image_profile" and value is not None:
            value = escape_field(_canonical_json(value) if isinstance(value, dict) else value)
        elif key == "reason":
            value = _reason(value)
        elif key == "raw_payload":
            value = escape_field(value)
        fields.append(value)
    return encode_fields(fields)


def encode_material_snapshot(seq, snapshot):
    """Caller must supply completion and per-object stability evidence explicitly.

    A fourth object is an error; the caller sends an empty UNKNOWN snapshot with
    complete=False and a SCENE_LIMIT reason instead of truncating the scene.
    """
    objects = snapshot["objects"]
    if not isinstance(objects, (list, tuple)) or len(objects) > 3:
        raise ValueError("material scene limit")
    fields = ["S", seq]
    for key in MATERIAL_HEADER:
        value = len(objects) if key == "count" else snapshot.get(key)
        if key in ("complete", "is_simulated"):
            if type(value) is not bool:
                raise ValueError("explicit boolean required: " + key)
            value = "1" if value else "0"
        elif key == "reason":
            value = _reason(value)
        fields.append(value)
    for row in objects:
        for key in MATERIAL_FIELDS:
            value = row.get(key)
            if key in ("stable", "geometry_valid"):
                if type(value) is not bool:
                    raise ValueError("explicit boolean required: " + key)
                value = "1" if value else "0"
            fields.append(value)
    return encode_fields(fields)


class StreamParser:
    """One sender session, no wrap or implicit replay reset. Feed is bounded."""
    def __init__(self, max_line=MAX_LINE, timeout_ms=250):
        self.max_line = min(max_line, MAX_LINE)
        self.timeout_ms = timeout_ms
        self.buffer = bytearray()
        self.last_seq = -1
        self.started_ms = None
        self.rejected = 0

    def reset_session(self, motors_disabled=False):
        if not motors_disabled:
            raise ValueError("session reset requires disabled motors")
        self.last_seq = -1
        self.buffer = bytearray()
        self.started_ms = None

    def expire(self, now_ms):
        elapsed = (time.ticks_diff(now_ms, self.started_ms) if hasattr(time, "ticks_diff") else now_ms - self.started_ms) if self.started_ms is not None else 0
        if self.started_ms is not None and elapsed > self.timeout_ms:
            self.buffer = bytearray()
            self.started_ms = None
            self.rejected += 1

    def feed(self, data, now_ms=0):
        self.expire(now_ms)
        output = []
        for value in data:
            if value == 36:
                self.buffer = bytearray(b"$")
                self.started_ms = now_ms
                continue
            if not self.buffer:
                continue
            self.buffer.append(value)
            if len(self.buffer) > self.max_line:
                self.buffer = bytearray()
                self.started_ms = None
                self.rejected += 1
                continue
            if value != 10:
                continue
            line = bytes(self.buffer)
            self.buffer = bytearray()
            self.started_ms = None
            try:
                if len(line) < 7 or line[-4] != 42:
                    raise ValueError("framing")
                supplied = line[-3:-1].decode("ascii")
                if len(supplied) != 2 or any(c not in "0123456789ABCDEF" for c in supplied):
                    raise ValueError("checksum encoding")
                body = line[1:-4]
                checksum = 0
                for byte in body:
                    checksum ^= byte
                if checksum != int(supplied, 16):
                    raise ValueError("checksum")
                text = body.decode("ascii")
                for i, token in enumerate(text.split(",")):
                    if not token:
                        raise ValueError("empty field")
                    _text(token, 384 if text.startswith("V,") and i == 20 else 96)
                decoded = _decode_body(text)
                if decoded["seq"] <= self.last_seq:
                    raise ValueError("replayed sequence")
                self.last_seq = decoded["seq"]
                output.append(decoded)
            except (ValueError, UnicodeError, OverflowError):
                self.rejected += 1
        return output
