"""CanMV bench for an isolated CameraXY deployment. Invoke run explicitly on an idle board."""
import json
import math
from neural_window import FrameWindow
from neural_decode import decode_logits


def decode(values, cell_mm, calibration, flipped_values=None):
    targets = ("camera_x_mm", "camera_y_mm")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite head tensor")
    if not cell_mm:
        if len(values) != 2 or flipped_values is None or len(flipped_values) != 2:
            raise ValueError("Direct head requires matching normal and rotated RGB inference")
        if not all(math.isfinite(x) for x in flipped_values):
            raise ValueError("Non-finite rotated head tensor")
        a = [1200 * min(1., max(-1., x)) for x in values]
        b = [1200 * min(1., max(-1., x)) for x in flipped_values]
        raw_score = -math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
        xy = [[(x+y)/2 for x, y in zip(a, b)]]
    else:
        n = 2400 // cell_mm
        count = n * n
        if len(values) != 3 * count:
            raise ValueError("Grid head tensor does not match compiler identity")
        logits = values[:count]
        chosen = sorted(range(count), key=lambda i: -logits[i])[:5]
        high = logits[chosen[0]]
        raw_score = 1. / sum(math.exp(logit - high) for logit in logits)
        xy = []
        for i in chosen:
            delta = [min(1., max(-1., x)) for x in values[count+2*i:count+2*i+2]]
            xy.append([(j + .5 + d/2) * cell_mm - 1200 for j, d in zip((i % n, i // n), delta)])
    index = sum(raw_score > edge for edge in calibration["bin_edges"])
    threshold = calibration["threshold"]
    accepted = threshold is not None and raw_score >= threshold
    result = dict(status="localized" if accepted else "unlocalized",
                  confidence=calibration["probabilities"][index],
                  top_k_camera_xy=[dict(zip(targets, position)) for position in xy])
    result.update(dict(zip(targets, xy[0] if accepted else (None, None))))
    return result


def decode_map(values, centers, primary_cal, set_cal):
    return decode_logits(values, centers, primary_cal["temperature"], primary_cal["threshold"])


def run(model_root, report_path, expected_usb_device, frames=120):
    """Use existing RoadNet bench counters and TopFace resource readers."""
    import gc
    import os
    import time
    from media.media import MediaManager
    from media.uvc import UVC
    import nncase_runtime as nn
    from ulab import numpy as np
    from board_support import Latency, sample_memory
    from board_support import digest
    from board_support import resources

    if type(frames) is not int or not 1 <= frames <= 3000 or not expected_usb_device:
        raise ValueError("Specify the operator-verified USB ID and a bounded frame count")
    if report_path in ("/sdcard/main.py", "/sdcard/boot.py"):
        raise ValueError("Use an independent CameraXY report path")
    try:
        os.stat(report_path)
    except OSError:
        pass
    else:
        raise ValueError("CameraXY run report already exists")
    for suffix in (".frames.jsonl", ".frames.jsonl.partial"):
        try:
            os.stat(report_path + suffix)
        except OSError:
            pass
        else:
            raise ValueError("Neural frame trace already exists")
    with open(model_root + "/deployment.json") as f:
        deployment = json.load(f)
    model_path = model_root + "/camera_xy.kmodel"
    identity = deployment["identity"]
    map_centers = identity.get("map_centers_mm")
    needs_flip = False
    profile = deployment["neural_window"]
    window = profile["length"]
    if len(map_centers) != 2304:
        raise ValueError("Expected 2304 camera XY anchors")
    if digest(model_path) != deployment["kmodel"]["kmodel_sha256"] or (
            identity["input_rgb_uint8_nchw"] != [1, 3 * window, 96, 160]):
        raise ValueError("Board inputs differ from frozen CameraXY KModel")
    result = dict(
        status="RUNNING", kmodel_sha256=digest(model_path), model_bytes=os.stat(model_path)[6],
        device=str(os.uname()), requested_frames=frames, frames=0, unlocalized=0,
        dropped=0, active_memory_samples=0, active_pool_peaks={}, memory_sampling_failures=0,
        cleanup_errors=[],
    )
    latencies = {name: Latency() for name in ("capture", "rgb", "ai2d", "kpu", "decode", "cycle")}
    ai2d = builder = kpu = backing = dst = history = window_input = None
    media_started = usb_started = False
    started = None
    trace = None
    try:
        trace = open(report_path + ".frames.jsonl.partial", "w")
        sample_memory(result, resources, "initial")
        found, usb_id = UVC.probe()
        if not found or usb_id != expected_usb_device:
            raise ValueError("UVC source differs from the operator-verified OV2735")
        ok, mode = UVC.select_video_mode(UVC.video_mode(1280, 720, UVC.FOURCC_MJPEG, 30))
        if not ok or (mode.width, mode.height, mode.fourcc) != (1280, 720, UVC.FOURCC_MJPEG):
            raise ValueError("OV2735 UVC mode does not match this CameraXY bench")
        result["negotiated_uvc"] = {"device": usb_id, "size": [mode.width, mode.height],
                                    "fourcc": mode.fourcc, "fps": mode.fps}
        kpu = nn.kpu()
        kpu.load_kmodel(model_path)
        if kpu.inputs_size() != 1 or kpu.outputs_size() != 1:
            raise ValueError("Unexpected KPU graph arity")
        backing = np.zeros((1, 3, 96, 160), dtype=np.uint8)
        dst = nn.from_numpy(backing)
        history = FrameWindow(np, window, profile["sample_us"], profile["max_gap_us"], time.ticks_diff)
        window_input = nn.from_numpy(history.tensor)
        kpu.set_input_tensor(0, window_input)
        result.update(qualification="EXPERIMENTAL_ACCURACY_HOLD", window=window, sample_us=profile["sample_us"], capture_decimated=0, resets=0)
        latencies["neural_window"] = Latency()
        ai2d = nn.ai2d()
        ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        ai2d.set_pad_param(True, [0, 0, 0, 0, 3, 3, 0, 0], 0, [114] * 3)
        ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        builder = ai2d.build([1, 3, 360, 640], [1, 3, 96, 160])
        media_started = True
        MediaManager.init()
        usb_started = True
        if UVC.start(cvt=False) is False:
            raise RuntimeError("OV2735 failed to start")
        started = time.ticks_us()
        for index in range(frames):
            os.exitpoint()
            cycle = time.ticks_us()
            image = UVC.snapshot(300)
            captured = time.ticks_us()
            latencies["capture"].add(time.ticks_diff(captured, cycle))
            if image is None:
                result["dropped"] += 1
                latencies["cycle"].add(time.ticks_diff(time.ticks_us(), cycle))
                if result["frames"] == 0 and result["dropped"] >= 10:
                    raise RuntimeError("No valid OV2735 frames")
                continue
            if not history.due(captured):
                result["capture_decimated"] += 1
                latencies["cycle"].add(time.ticks_diff(time.ticks_us(), cycle))
                continue
            packed = image.to_rgb888(x_size=640, y_size=360)
            hwc = packed.to_numpy_ref()
            if tuple(hwc.shape) != (360, 640, 3):
                raise ValueError("Actual RGB shape differs from model preprocessing")
            planar = hwc.reshape((360*640, 3)).transpose().copy().reshape((1, 3, 360, 640))
            latencies["rgb"].add(time.ticks_diff(time.ticks_us(), captured))
            outputs = []
            for flip in (False, True) if needs_flip else (False,):
                t = time.ticks_us()
                source = planar[:, :, ::-1, ::-1].copy() if flip else planar
                src = nn.from_numpy(source)
                builder.run(src, dst)
                before_kpu = time.ticks_us()
                latencies["ai2d"].add(time.ticks_diff(before_kpu, t))
                history.append(backing, captured)
                result["resets"] = history.resets
                result["last_window_valid_frames"] = history.count
                after_pack = time.ticks_us()
                latencies["neural_window"].add(time.ticks_diff(after_pack, before_kpu))
                before_kpu = after_pack
                kpu.run()
                latencies["kpu"].add(time.ticks_diff(time.ticks_us(), before_kpu))
                outputs.append([float(value) for value in kpu.get_output_tensor(0).to_numpy()[0]])
            t = time.ticks_us()
            if map_centers is None:
                answer = decode(outputs[0], identity["cell_mm"], identity["confidence"],
                                outputs[1] if needs_flip else None)
            else:
                answer = decode_map(outputs[0], map_centers, identity["confidence"],
                                    identity["candidate_set_confidence"])
            result["last_prediction"] = answer
            trace.write(json.dumps(dict(capture_index=index, capture_ticks_us=captured,
                captured_since_start_us=time.ticks_diff(captured, started),
                window_timestamps_us=history.tensor_timestamps_us,
                window_valid_frames=history.count, prediction=answer)) + "\n")
            if index % 10 == 0:
                trace.flush()
            result["unlocalized"] += int(answer["status"] == "unlocalized")
            result["frames"] += 1
            latencies["decode"].add(time.ticks_diff(time.ticks_us(), t))
            if index % 10 == 0:
                sample_memory(result, resources, "active")
            latencies["cycle"].add(time.ticks_diff(time.ticks_us(), cycle))
        result["status"] = "MEASURED" if result["frames"] else "NO_FRAMES"
    except BaseException as exc:
        result["status"] = "RUN_FAILED"
        result["error"] = str(exc)
        raise
    finally:
        if started is not None:
            elapsed = time.ticks_diff(time.ticks_us(), started) / 1000000
            result["wall_seconds"] = elapsed
            result["camera_to_result_fps"] = result["frames"] / elapsed if elapsed > 0 else None
        for name, enabled, close in (("uvc", usb_started, UVC.stop), ("media", media_started, MediaManager.deinit)):
            if enabled:
                try:
                    close()
                except Exception as exc:
                    result["cleanup_errors"].append([name, str(exc)])
        if result["cleanup_errors"]:
            result["status"] = "CLEANUP_UNCONFIRMED"
        ai2d = builder = kpu = backing = dst = history = window_input = None
        if trace is not None:
            try:
                trace.close()
                os.rename(report_path + ".frames.jsonl.partial", report_path + ".frames.jsonl")
                result["frames_path"] = report_path + ".frames.jsonl"
            except Exception as exc:
                result["cleanup_errors"].append(["frame_trace", str(exc)])
                result["status"] = "CLEANUP_UNCONFIRMED"
        gc.collect()
        sample_memory(result, resources, "final")
        result["latency"] = {name: histogram.report() for name, histogram in latencies.items()}
        with open(report_path + ".partial", "w") as f:
            f.write(json.dumps(result))
        os.rename(report_path + ".partial", report_path)
        print("CAMERA_XY_BOARD_RESULT", report_path, result["status"])
    return result
