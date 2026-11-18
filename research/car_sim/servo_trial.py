"""Blender camera-in-the-loop bench at an interrupted turntable and unloading dock.

The desktop Python worker only receives the camera PNG and expected feature.
All Blender truth comparisons happen after each trial stops.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def detect_worker(hub):
    from mission_vision import acquisition, read_png
    from perception import FrozenPerception
    models=FrozenPerception(hub)
    for line in sys.stdin:
        request=json.loads(line)
        img=read_png(request["image"])
        if request.get("kind")=="qr":
            import cv2
            print(json.dumps({"qr":cv2.QRCodeDetector().detectAndDecode(img)[0]}),flush=True)
            continue
        _,decoded=models.observe(img)
        result=acquisition(request["description"],img,decoded,slot=request.get("slot"),window=request.get("window"))
        if request.get("profile") and result["reason"]!="OBSERVED_COLOR_MISMATCH":
            from silhouette import black_profile
            profile=black_profile(img,result["capture_window_xyxy"],*request["profile"])
            result.update(status="ACQUIRED" if profile else "HOLD",
                          center_xy=profile["center"] if profile else None,
                          reason="BLACK_CAD_OUTLINE" if profile else "REQUIRE_BLACK_CAD_OUTLINE",
                          profile=profile)
        print(json.dumps(result),flush=True)


def run(out,hub,python):
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    from build_scene import shoot, write_json
    s=bpy.context.scene
    root=s.objects["K230 | Vehicle ground origin"]
    cam=s.objects["ROAD CAMERA | optical centre"]
    exposure=s.view_settings.exposure
    plan=json.loads((out/"round.json").read_text(encoding="utf8"))
    initial=json.loads((out/"mission_vision.json").read_text(encoding="utf8"))
    truth=json.loads((out/"mission_truth.json").read_text(encoding="utf8"))["frames"]
    observations={v["index"]:v for v in initial["observations"]}
    # One additional exposure for each rejected primary image, at unchanged pose.
    retries={}
    for row in observations.values():
        if row["status"]=="HOLD":
            s.frame_set(plan["timeline"][row["index"]]["observation_frame"])
            s.view_settings.exposure=exposure+.9
            shoot(s,cam,out/"mission_retry"/row["file"])
            retries[row["file"]]={**truth[row["file"]],"primary_sha256":row["sha256"],"exposure_delta_ev":.9}
    write_json(out/"mission_retry_truth.json",{"frames":retries})
    s.view_settings.exposure=exposure
    trials=[]
    offsets=((.016,-.009),(-.009,.013),(.011,-.014))
    with subprocess.Popen([str(python),"-B","-u",str(Path(__file__).resolve()),"--worker","--hub",str(hub)],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as worker:
        try:
            for case,index in enumerate((2,3,4,28,29,30)):
                s.frame_set(plan["timeline"][index]["observation_frame"])
                obs=observations[index]
                station=obs["station"]
                # Nominal work point taught in body coordinates, from the fixture layout.
                # The stop distance and working plane can vary independently of the lens.
                turntable=s.objects["Turntable"]
                z=(turntable.location.z+turntable.dimensions.z/2+.06) if station=="RAW" else .063
                workpoint=Vector((.215 if station=="RAW" else .235,0,z))
                def project(world_point):
                    p=world_to_camera_view(s,cam,world_point)
                    return (640*p.x,360*(1-p.y))
                x,y=offsets[case%3]
                heading=root.rotation_euler.z
                root.delta_location=Vector((x*math.cos(heading)-y*math.sin(heading),
                                            x*math.sin(heading)+y*math.cos(heading),0))
                start=root.delta_location.copy()
                stable=0
                probe=0
                trace=[]
                status="ITERATION_LIMIT"
                # Previously collected items on the deck travel with the body at this frozen keyframe.
                cargo=[]
                for obj in s.objects:
                    if "batch" in obj and abs(obj.location.z-.081)<.001:
                        cargo.append(obj)
                for iteration in range(13):
                    for obj in cargo:
                        obj.delta_location=root.delta_location
                    bpy.context.view_layer.update()
                    ref=project(root.matrix_world @ workpoint)
                    s.view_settings.exposure=exposure+(0,.9,-.65)[probe]
                    s.cycles.seed=11+iteration
                    frame=out/"servo"/f"{index:02}-{iteration:02}.png"
                    shoot(s,cam,frame)
                    request={"image":str(frame),"description":[station,obs["feature"],obs["expected_observed_color"]]}
                    worker.stdin.write(json.dumps(request)+"\n")
                    worker.stdin.flush()
                    line=worker.stdout.readline()
                    if not line:
                        raise RuntimeError("ONNX inference worker stopped")
                    detection=json.loads(line)
                    xy=detection["center_xy"]
                    event=dict(image=frame.name,reference_px=ref,estimated_px=xy,
                               station=station,status=detection["status"],reason=detection["reason"],
                               exposure_delta=(0,.9,-.65)[probe],body_offset_world_mm=[1000*v for v in root.delta_location[:2]])
                    trace.append(event)
                    if xy is None:
                        stable=0
                        if probe==2:
                            status="HOLD_NO_TARGET"
                            break
                        probe+=1
                        continue
                    du,dv=xy[0]-ref[0],xy[1]-ref[1]
                    event["error_px"]=math.hypot(du,dv)
                    if math.hypot(du,dv)<=2.5:
                        stable+=1
                        if stable==3:
                            status="PIXEL_ALIGNED"
                            break
                        continue
                    stable=0
                    # Positive image-right and image-down, transformed by the current
                    # turret angle. A small fixed step gain does not assume a pixel is a mm.
                    command=cam.matrix_world.to_quaternion() @ Vector((du*.00067,-dv*.00067,0))
                    if command.length>.008:
                        command*=.008/command.length
                    if (root.delta_location+command).length>.035:
                        status="HOLD_TRAVEL_LIMIT"
                        break
                    root.delta_location+=command
                    event["command_world_mm"]=[1000*v for v in command[:2]]
                # Ground truth is isolated from the observation and motor update above.
                bpy.context.view_layer.update()
                batch=1
                color=obs["expected_observed_color"]
                top=s.objects[f"Batch {batch} / color {color}"].matrix_world @ Vector((0,0,.060))
                error=top-root.matrix_world @ workpoint
                trials.append(dict(index=index,feature=obs["feature"],color=color,outcome=status,
                                   initial_offset_mm=[1000*v for v in start[:2]],trace=trace,
                                   independent_final_xy_error_mm=1000*math.hypot(error.x,error.y)))
                for obj in cargo:
                    obj.delta_location=(0,0,0)
                root.delta_location=(0,0,0)
        finally:
            worker.stdin.close()
            if worker.wait(timeout=15):
                raise RuntimeError("ONNX inference failure")
    write_json(out/"servo_trials.json",dict(scope="Rendered image-to-motion loop on six individual dock approaches. "
               "Nominal gripper XY and stationary block are simulation assumptions. No clamp or lift command.",
               unit="pixel errors in 640x360, vector commands in world mm",trials=trials))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--worker",action="store_true")
    ap.add_argument("--hub",required=True,type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--python",type=Path)
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.worker:
        detect_worker(args.hub)
    else:
        run(args.out.resolve(),args.hub.resolve(),args.python)
