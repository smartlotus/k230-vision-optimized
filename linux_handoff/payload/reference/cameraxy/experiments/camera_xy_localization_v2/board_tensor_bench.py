"""Camera-free CameraXY input/decoder bench; no camera, KModel or startup writes."""
import json


def fixture(name):
    """Deterministic planar RGB bytes, identical on CPython and MicroPython."""
    if name == "constant":
        return b"".join(bytes([c]) * (640 * 360) for c in (90, 140, 190))
    if name == "rgb_ramp":
        return b"".join(bytes([(x // 4 + c * 31) % 256 for x in range(640)]) * 360
                        for c in range(3))
    if name == "thin_lines":
        return bytes([255 if x % 4 == 0 else 0 for x in range(640)]) * (360 * 3)
    if name == "checkerboard":
        return ((b"\x00\xff" * 320 + b"\xff\x00" * 320) * 180) * 3
    raise ValueError("Unknown deterministic fixture")


def sha_file(path):
    import hashlib
    import binascii
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return binascii.hexlify(h.digest()).decode()


def sha_bytes(value):
    import hashlib
    import binascii
    return binascii.hexlify(hashlib.sha256(value).digest()).decode()


def isolated_root(root):
    parts = root.split("/")
    return (len(parts) == 3 and parts[:2] == ["", "sdcard"] and parts[2].startswith("xy")
            and len(parts[2]) > 2 and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                        for c in parts[2]))


def run(root, repeats=5):
    import gc
    import os
    import time
    import nncase_runtime as nn
    from ulab import numpy as np
    import board_probe
    decode_map = board_probe.decode_map

    if not isolated_root(root) or type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("Use an independent xy bench directory and 1..20 repeats")
    if board_probe.__file__ != root + "/board_probe.py" or __file__ != root + "/board_tensor_bench.py":
        raise ValueError("Imported native modules differ from the owned bench directory")
    report_path = root + "/native.json"
    try:
        os.stat(report_path)
    except OSError:
        pass
    else:
        raise ValueError("Immutable native report already exists")
    with open(root + "/spec.json") as stream:
        spec = json.load(stream)
    if spec["input_shape"] != [1, 3, 360, 640] or spec["output_shape"] != [1, 3, 96, 160]:
        raise ValueError("Input contract differs from current CameraXY bench")
    if [item["name"] for item in spec["fixtures"]] != ["constant", "rgb_ramp", "thin_lines", "checkerboard", "heldout"]:
        raise ValueError("Specify the five fixed native fixtures in order")
    for name, expected in spec["board_sources"].items():
        if sha_file(root + "/" + name) != expected:
            raise ValueError("Native source SHA differs: " + name)
    startup_before = {p: sha_file(p) for p in ("/sdcard/main.py", "/sdcard/boot.py")
                      if _exists(p)}
    result = dict(status="RUNNING", scope="Camera-free AI2D and learned-map decoder only",
                  firmware=str(os.uname()), runtime=nn.version(),
                  spec_sha256=sha_file(root + "/spec.json"),
                  source_sha256=spec["board_sources"], camera="NOT_RUN", kpu="NOT_RUN",
                  startup_before=startup_before, heap_initial=gc.mem_free(),
                  preprocessing=[], decode=[], invalid_decode=[], cleanup_errors=[])
    engine = builder = target = backing = source = array = raw = output = None
    try:
        engine = nn.ai2d()
        engine.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        engine.set_pad_param(True, [0, 0, 0, 0, 3, 3, 0, 0], 0, [114, 114, 114])
        engine.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        builder = engine.build(spec["input_shape"], spec["output_shape"])
        backing = np.zeros(tuple(spec["output_shape"]), dtype=np.uint8)
        target = nn.from_numpy(backing)
        for item in spec["fixtures"]:
            os.exitpoint()
            if item["kind"] == "procedural":
                raw = fixture(item["name"])
            else:
                compressed_path = item.get("compressed_input_path")
                input_path = compressed_path or root + "/heldout.input.rgb"
                parent, name = input_path.rsplit("/", 1)
                if not isolated_root(parent) or name not in ("heldout.rgb.z", "heldout.input.rgb"):
                    raise ValueError("Use an isolated bench fixture")
                with open(input_path, "rb") as stream:
                    if compressed_path:
                        import deflate
                        reader = deflate.DeflateIO(stream, deflate.ZLIB)
                        raw = reader.read()
                        reader.close()
                    else:
                        raw = stream.read()
            if len(raw) != 3 * 360 * 640 or sha_bytes(raw) != item["input_sha256"]:
                raise ValueError("Fixture input bytes differ: " + item["name"])
            array = np.frombuffer(raw, dtype=np.uint8).reshape((1, 3, 360, 640))
            source = nn.from_numpy(array)
            elapsed = []
            for _ in range(repeats):
                begin = time.ticks_us()
                builder.run(source, target)
                elapsed.append(time.ticks_diff(time.ticks_us(), begin))
            output = target.to_numpy().tobytes()
            output_name = item["name"] + ".ai2d.rgb"
            with open(root + "/" + output_name, "wb") as stream:
                stream.write(output)
            result["preprocessing"].append(dict(
                name=item["name"], input_sha256=item["input_sha256"],
                output_sha256=sha_bytes(output), output_bytes=len(output), output=output_name,
                ai2d_us=elapsed))
            source = array = raw = output = None
            gc.collect()
        centers = spec["centers_mm"]
        for item in spec["score_cases"]:
            begin = time.ticks_us()
            answer = decode_map(item["scores"], centers, spec["primary_cal"], spec["set_cal"])
            result["decode"].append(dict(sample_id=item["sample_id"], output=answer,
                                         decode_us=time.ticks_diff(time.ticks_us(), begin)))
        for name in ("truncated", "nonfinite"):
            values = list(spec["score_cases"][0]["scores"])
            if name == "truncated":
                values = values[:-1]
            else:
                values[0] = float("nan")
            try:
                decode_map(values, centers, spec["primary_cal"], spec["set_cal"])
            except ValueError:
                result["invalid_decode"].append(dict(case=name, rejected=True))
            else:
                result["invalid_decode"].append(dict(case=name, rejected=False))
        result["status"] = "AI2D_AND_DECODE_COMPLETE"
    except BaseException as exc:
        result["status"] = "RUN_FAILED"
        result["error"] = str(exc)
        raise
    finally:
        source = array = raw = output = target = backing = builder = engine = None
        gc.collect()
        try:
            nn.shrink_memory_pool()
        except Exception as exc:
            result["cleanup_errors"].append(str(exc))
        result["heap_final"] = gc.mem_free()
        result["startup_after"] = {p: sha_file(p) for p in startup_before}
        result["startup_unchanged"] = startup_before == result["startup_after"]
        if result["cleanup_errors"]:
            result["status"] = "CLEANUP_UNCONFIRMED"
        with open(report_path + ".partial", "w") as stream:
            stream.write(json.dumps(result))
        os.rename(report_path + ".partial", report_path)
        print("CAMERA_XY_TENSOR_BENCH", result["status"], report_path)
    return result


def _exists(path):
    import os
    try:
        os.stat(path)
        return True
    except OSError:
        return False
