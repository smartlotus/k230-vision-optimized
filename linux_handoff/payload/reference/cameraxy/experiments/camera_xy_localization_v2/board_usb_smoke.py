"""Bounded USB->AI2D->CameraXY KPU->candidate->preview engineering test."""
import json
import math


def digest(path):
    import hashlib
    import binascii
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            data = stream.read(8192)
            if not data:
                break
            h.update(data)
    return binascii.hexlify(h.digest()).decode()


def resources():
    import gc
    result = dict(python_heap_used=gc.mem_alloc(), python_heap_free=gc.mem_free())
    for name in ("sys_heap", "sys_page", "sys_mmz"):
        total, used, free = getattr(gc, name)()
        result[name] = dict(total=total, used=used, free=free)
    return result


def valid_root(root):
    parts = root.split("/")
    return (len(parts) == 3 and parts[0] == "" and parts[1] in ("tmp", "sdcard")
            and parts[2].startswith("xy") and len(parts[2]) > 2
            and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in parts[2]))


def require_data_volume():
    import os
    try:
        volume = os.statvfs('/data')
    except OSError:
        raise RuntimeError('DATA_VOLUME_UNAVAILABLE')
    if len(volume) < 4 or volume[0] <= 0 or volume[2] <= 0 or volume[3] <= 0:
        raise RuntimeError('DATA_VOLUME_UNAVAILABLE')


def engineering_contract(document):
    if (document.get("schema") != "CAMERA_XY_ENGINEERING_ONLY_V1"
            or document.get("purpose") != "USER_AUTHORIZED_HARDWARE_SMOKE"
            or document.get("production_eligible") is not False
            or document.get("require_unlocalized") is not True
            or document.get("method") != "learned_map"
            or document.get("accuracy_test") != "NOT_RUN_BY_USER_SCOPE"
            or document["identity"]["input_rgb_uint8_nchw"] != [1, 3, 96, 160]
            or document["kmodel"].get("filename") != "engineering.kmodel"):
        raise ValueError("An isolated engineering-only CameraXY artifact is required")
    centers = document["identity"]["map_centers_mm"]
    if not centers or any(len(p) != 2 or not all(math.isfinite(x) and abs(x) <= 1200 for x in p) for p in centers):
        raise ValueError("Invalid XY anchors")
    return centers


def run(root, report_name, expected_device, frames=30, preview=False):
    import os
    import gc
    import time
    import nncase_runtime as nn
    import image
    from ulab import numpy as np
    from media.media import MediaManager
    from media.uvc import UVC
    from media.display import Display
    import board_probe

    if not valid_root(root) or not report_name.endswith(".json") or not all(
            c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in report_name) or "/" in report_name:
        raise ValueError("Use an independent bench root and report basename")
    if type(frames) is not int or not 1 <= frames <= 3000 or type(preview) is not bool or not expected_device:
        raise ValueError("A verified USB source and bounded frame count are required")
    if board_probe.__file__ != root + "/board_probe.py" or __file__ != root + "/board_usb_smoke.py":
        raise ValueError("Imported module root mismatch")
    report_dir = root
    if root.startswith("/sdcard/"):
        require_data_volume()
        log_root = "/data/logs"
        report_dir = log_root + "/" + root.rsplit("/", 1)[1]
        for directory in (log_root, report_dir):
            if not _exists(directory):
                os.mkdir(directory)
    report_path = report_dir + "/" + report_name
    if _exists(report_path) or _exists(report_path + ".partial"):
        raise ValueError("Preserve existing native reports")
    with open(root + "/engineering.json") as stream:
        deployment = json.load(stream)
    centers = engineering_contract(deployment)
    model_path = root + "/engineering.kmodel"
    if digest(model_path) != deployment["kmodel"]["kmodel_sha256"]:
        raise ValueError("Engineering model byte identity differs")
    result = dict(status="RUNNING", scope="USB/AI2D/learned-map KPU/structural decode/preview/cleanup only",
        firmware=str(os.uname()), runtime=nn.version(), requested_frames=frames, attempts=0,
        frames=0, capture_none=0, ai2d_calls=0, kpu_calls=0, decoded_frames=0, displayed_frames=0,
        preview_requested=preview,
        preview_pixel_format="RGB565" if preview else None,
        model_sha256=digest(model_path), model_bytes=os.stat(model_path)[6],
        deployment_sha256=digest(root + "/engineering.json"),
        sources={name:digest(root + "/" + name) for name in ("board_usb_smoke.py", "board_probe.py")},
        accuracy_test="NOT_RUN_BY_USER_SCOPE", road_quality_test="NOT_RUN_BY_USER_SCOPE",
        confidence_domain="NOT_EVALUATED", primary_coordinates="FORCED_NULL",
        frozen_selected_method=deployment["frozen_selected_method"], tested_method="learned_map",
        production_eligible=False, cleanup_errors=[], memory_samples=[],
        capture_loss_scope="Snapshot requests returning None; sensor frame sequence unavailable")
    result["startup_before"] = {p:digest(p) for p in ("/sdcard/main.py", "/sdcard/boot.py") if _exists(p)}
    result["initial_memory"] = resources()
    times = {name:[] for name in ("capture", "rgb", "ai2d", "kpu", "readback", "decode", "preview", "cycle")}
    kpu = engine = builder = target = backing = source = raw = rgb = shown = hwc = planar = output_tensor = output_array = None
    usb_attempted = media_attempted = display_attempted = False
    started = None
    try:
        found, device = UVC.probe()
        if not found or device != expected_device:
            raise ValueError("USB source differs from preflight")
        ok, mode = UVC.select_video_mode(UVC.video_mode(1280, 720, UVC.FOURCC_MJPEG, 30))
        if not ok or (mode.width, mode.height, mode.fourcc) != (1280, 720, UVC.FOURCC_MJPEG):
            raise ValueError("UVC negotiation differs from input contract")
        result["device"] = device
        result["negotiated"] = dict(width=mode.width, height=mode.height, fourcc=mode.fourcc, fps=mode.fps)
        kpu = nn.kpu()
        kpu.load_kmodel(model_path)
        if kpu.inputs_size() != 1 or kpu.outputs_size() != 1:
            raise ValueError("Unexpected CameraXY graph arity")
        result["inputs_desc"] = str(kpu.inputs_desc(0))
        result["outputs_desc"] = str(kpu.outputs_desc(0))
        backing = np.zeros((1, 3, 96, 160), dtype=np.uint8)
        target = nn.from_numpy(backing)
        kpu.set_input_tensor(0, target)
        engine = nn.ai2d()
        engine.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        engine.set_pad_param(True, [0, 0, 0, 0, 3, 3, 0, 0], 0, [114, 114, 114])
        engine.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        builder = engine.build([1, 3, 360, 640], [1, 3, 96, 160])
        if preview:
            display_attempted = True
            Display.init(Display.VIRT, width=640, height=360, to_ide=True)
        media_attempted = True
        MediaManager.init()
        usb_attempted = True
        if UVC.start(cvt=False) is False:
            raise RuntimeError("UVC start failed")
        started = time.ticks_ms()
        for index in range(frames):
            os.exitpoint()
            if time.ticks_diff(time.ticks_ms(), started) > 300000:
                raise RuntimeError("Five-minute test bound exceeded")
            result["attempts"] += 1
            cycle = time.ticks_us()
            try:
                raw = UVC.snapshot(300)
                captured = time.ticks_us()
                times["capture"].append(time.ticks_diff(captured, cycle))
                if raw is None:
                    result["capture_none"] += 1
                    continue
                if (raw.width(), raw.height()) != (1280, 720):
                    raise ValueError("Snapshot size differs from negotiation")
                rgb = raw.to_rgb888(x_size=640, y_size=360)
                hwc = rgb.to_numpy_ref()
                if tuple(hwc.shape) != (360, 640, 3):
                    raise ValueError("Decoded RGB ABI mismatch")
                planar = hwc.reshape((360*640, 3)).transpose().copy().reshape((1, 3, 360, 640))
                source = nn.from_numpy(planar)
                prepared = time.ticks_us()
                times["rgb"].append(time.ticks_diff(prepared, captured))
                builder.run(source, target)
                result["ai2d_calls"] += 1
                before_kpu = time.ticks_us()
                times["ai2d"].append(time.ticks_diff(before_kpu, prepared))
                kpu.run()
                result["kpu_calls"] += 1
                inferred = time.ticks_us()
                times["kpu"].append(time.ticks_diff(inferred, before_kpu))
                output_tensor = kpu.get_output_tensor(0)
                output_array = output_tensor.to_numpy()
                if tuple(output_array.shape) != (1, len(centers)):
                    raise ValueError("CameraXY output score ABI mismatch")
                values = [float(v) for v in output_array[0]]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite native scores")
                result["output_shape"] = list(output_array.shape)
                result["last_score_sample"] = values[:8]
                readback = time.ticks_us()
                times["readback"].append(time.ticks_diff(readback, inferred))
                cal = dict(threshold=None, bin_edges=[], probabilities=[0.0])
                answer = board_probe.decode_map(values, centers, cal, cal)
                answer.update(confidence=None, candidate_set_confidence=None,
                              confidence_domain="NOT_EVALUATED", engineering_only=True)
                if answer["status"] != "unlocalized" or answer["camera_x_mm"] is not None or answer["camera_y_mm"] is not None:
                    raise ValueError("Engineering output must remain unlocalized")
                result["last_output"] = answer
                result["decoded_frames"] += 1
                decoded = time.ticks_us()
                times["decode"].append(time.ticks_diff(decoded, readback))
                if preview:
                    shown = rgb.to_rgb565()
                    if shown.format() != image.RGB565:
                        raise ValueError("Preview requires an actual RGB565 frame")
                    Display.show_image(shown)
                    result["displayed_frames"] += 1
                times["preview"].append(time.ticks_diff(time.ticks_us(), decoded))
                result["frames"] += 1
            finally:
                source = hwc = planar = output_tensor = output_array = None
                if shown is not None and shown is not rgb and shown is not raw:
                    try:
                        shown.__del__()
                    except BaseException as error:
                        result["cleanup_errors"].append(["frame_preview", str(error)])
                if rgb is not None and rgb is not raw:
                    try:
                        rgb.__del__()
                    except BaseException as error:
                        result["cleanup_errors"].append(["frame_rgb", str(error)])
                if raw is not None:
                    try:
                        raw.__del__()
                    except BaseException as error:
                        result["cleanup_errors"].append(["frame_raw", str(error)])
                shown = rgb = raw = None
                gc.collect()
                times["cycle"].append(time.ticks_diff(time.ticks_us(), cycle))
            if index % 30 == 0:
                result["memory_samples"].append(dict(frame=result["frames"], memory=resources()))
        complete = result["frames"] == frames and result["capture_none"] == 0
        if preview:
            complete = complete and result["displayed_frames"] == frames
        result["status"] = "ENGINEERING_CHAIN_COMPLETE" if complete else "INCOMPLETE"
        result["preview_status"] = "COMPLETE" if preview and result["displayed_frames"] == frames else "NOT_RUN" if not preview else "INCOMPLETE"
    except BaseException as error:
        result["status"] = "RUN_FAILED"
        result["error"] = str(error)
    finally:
        if started is not None:
            elapsed = time.ticks_diff(time.ticks_ms(), started) / 1000
            result["elapsed_seconds"] = elapsed
            result["processed_fps"] = result["frames"] / elapsed if elapsed else None
        source = hwc = planar = output_tensor = output_array = raw = rgb = shown = None
        kpu = builder = engine = target = backing = None
        gc.collect()
        actions = [("runtime_pool", nn.shrink_memory_pool)]
        if usb_attempted:
            actions.append(("uvc", UVC.stop))
        if display_attempted:
            actions.append(("display", Display.deinit))
        if media_attempted:
            actions.append(("media", MediaManager.deinit))
        for name, action in actions:
            try:
                action()
            except BaseException as error:
                result["cleanup_errors"].append([name, str(error)])
        gc.collect()
        result["final_memory"] = resources()
        result["startup_after"] = {p:digest(p) for p in result["startup_before"]}
        result["startup_unchanged"] = result["startup_before"] == result["startup_after"]
        result["timing"] = {name:statistics(samples) for name, samples in times.items()}
        if result["cleanup_errors"] or not result["startup_unchanged"]:
            result["status"] = "CLEANUP_UNCONFIRMED"
        with open(report_path + ".partial", "w") as stream:
            stream.write(json.dumps(result))
        os.rename(report_path + ".partial", report_path)
        print("XY_USB_ENGINEERING_RESULT", report_path, result["status"], result["frames"])
    return result


def statistics(samples):
    values = sorted(samples)
    n = len(values)
    return dict(samples=n, mean_ms=sum(values)/n/1000 if n else None,
                p50_ms=values[(n-1)//2]/1000 if n else None,
                p95_ms=values[min(n-1, (n*95+99)//100-1)]/1000 if n else None)


def _exists(path):
    import os
    try:
        os.stat(path)
        return True
    except OSError:
        return False
