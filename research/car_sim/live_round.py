"""Execute a fixed-field mission with image-gated transfers in a mutable Blender scene.

Vehicle waypoints and a simple gripper proxy are prescribed. Acquisitions are
rendered anew. A failed observe or contact ends the trial at the current step.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import bpy
from mathutils import Vector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build_scene import CARGO, raw_pos, shoot, target_slot, write_json
from round_plan import build_round


def plane_point(cam,uv,z):
    focal=cam.data.lens*640/cam.data.sensor_width
    ray=cam.matrix_world.to_quaternion() @ Vector(((uv[0]-320)/focal,(180-uv[1])/focal,-1))
    origin=cam.matrix_world.translation
    return origin+ray*((z-origin.z)/ray.z)


def simulate(out,hub,python,profile_review,challenge):
    s=bpy.context.scene
    root=s.objects["K230 | Vehicle ground origin"]
    cam=s.objects["ROAD CAMERA | optical centre"]
    movie=json.loads((out/"round.json").read_text(encoding="utf8"))
    table=s.objects["Turntable"]
    tray_z=table.location.z+table.dimensions.z/2
    ambient=s.view_settings.exposure-(.40 if challenge else 0)
    s.view_settings.exposure=ambient
    images="live_challenge" if challenge else "live_profile" if profile_review else "live"
    receipt=out/(images+"_round.json")
    blocks={}
    location={}
    for obj in s.objects:
        if "batch" in obj:
            ident=(obj["batch"],obj["color_id"])
            obj.animation_data_clear()
            blocks[ident]=obj
            obj.location=(0,0,-1)
            location[ident]="RESERVE"
    phase=0
    loaded=0
    cargo={}
    bases={}
    events=[]
    status="RUNNING"
    picks=placements=0
    with subprocess.Popen([str(python),"-B","-u",str(HERE/"servo_trial.py"),"--worker","--hub",str(hub)],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as engine:
        def observe(path,**request):
            engine.stdin.write(json.dumps({"image":str(path),**request})+"\n")
            engine.stdin.flush()
            line=engine.stdout.readline()
            if not line:
                raise RuntimeError("Inference process stopped")
            return json.loads(line)
        try:
            s.frame_set(movie["timeline"][0]["end"])
            qr_image=out/images/"qr.png"
            shoot(s,s.objects["QR camera / assumed mount"],qr_image)
            qr=observe(qr_image,kind="qr")["qr"]
            if qr!=movie["plan"]["task"]:
                write_json(receipt,{"status":"HOLD_QR","decoded":qr})
                return
            steps=build_round(qr,movie["plan"]["start_id"])["steps"]
            for i,step in enumerate(steps[1:],1):
                act=step["action"]
                timing=movie["timeline"][i]
                s.frame_set(timing["observation_frame"] or timing["end"])
                for ident,carrier in cargo.items():
                    blocks[ident].location=root.matrix_world @ Vector(CARGO[carrier])
                if act not in ("PICK","PLACE"):
                    events.append({"step":i,"action":act,"state":"PRESCRIBED_MOTION"})
                    continue
                station=step["source_station"] if act=="PICK" else step["dest_station"]
                wanted=step["target_color"]
                slot=step["target_slot"]
                batch=step["batch"]
                layer=step["layer"]
                if station=="RAW" and batch!=loaded:
                    assert not any(site=="RAW" for site in location.values())
                    for ident in blocks:
                        if ident[0]==batch:
                            location[ident]="RAW"
                    loaded=batch
                    phase=1 if challenge else 0
                support_top=act=="PLACE" and layer==2
                feature="pick_top" if act=="PICK" else "support_top" if support_top else "empty_rings"
                expected=bases.get(slot) if support_top else wanted if act=="PICK" else None
                plane_z=(tray_z+.060 if station=="RAW" else .063) if feature!="empty_rings" else .003
                event={"step":i,"action":act,"station":station,"slot":slot,"color":wanted,
                       "feature":feature,"attempts":[]}
                events.append(event)
                estimate=None
                # A freely indexed three-stop turntable presents all three phases
                # in order. The controller cannot request a particular material.
                phase_count=3 if station=="RAW" else 1
                for stop in range(phase_count):
                    if station=="RAW":
                        for ident in blocks:
                            if location[ident]=="RAW":
                                blocks[ident].location=raw_pos(*ident,-phase*math.tau/3,tray_z)
                    use_outline=profile_review and station!="RAW" and expected==5
                    probes=((0,0),(-.65,0),(.9,0),(-.35,.016),(.20,-.016))
                    if use_outline:
                        probes=((0,0),(-.15,.012),(.25,-.012),(-.55,.018),(-.30,-.018))
                    candidates=[]
                    for probe,(ev,along_lane) in enumerate(probes):
                        if challenge:
                            along_lane+=.005
                        yaw=root.rotation_euler.z
                        root.delta_location=(-math.sin(yaw)*along_lane,math.cos(yaw)*along_lane,0)
                        bpy.context.view_layer.update()
                        for ident,carrier in cargo.items():
                            blocks[ident].location=root.matrix_world @ Vector(CARGO[carrier])
                        s.view_settings.exposure=ambient+ev
                        img=out/images/f"{i:02}-{stop}-{probe}.png"
                        shoot(s,cam,img)
                        request={"description":[station,feature,expected]}
                        if feature=="empty_rings":
                            request["slot"]=slot
                        if use_outline:
                            request["profile"]=[cam.matrix_world.translation.z,plane_z]
                        detection=observe(img,**request)
                        event["attempts"].append(dict(file=img.name,state=detection["status"],
                                   reason=detection["reason"],exposure_ev=ev,lane_offset_m=along_lane))
                        if detection["status"]=="ACQUIRED":
                            measured=plane_point(cam,detection["center_xy"],plane_z)
                            if use_outline:
                                event["attempts"][-1]["outline_iou"]=detection["profile"]["silhouette_iou"]
                                match=next((point for point in candidates if (point-measured).length<.004),None)
                                if match is None:
                                    candidates.append(measured)
                                    continue
                                measured=(measured+match)/2
                            estimate=measured
                            break
                        if detection["reason"]=="OBSERVED_COLOR_MISMATCH" and station=="RAW":
                            break
                    if estimate is not None:
                        break
                    phase=(phase+1)%3
                if estimate is None:
                    status="HOLD_VISION"
                    break
                event["estimated_world_xy_mm"]=[1000*v for v in estimate[:2]]
                # The following operations are the gripper/contact simulator. Only
                # the simulated success receipt is fed back to task progression.
                if act=="PICK":
                    candidates=[k for k,pos in location.items() if pos==station]
                    distances={k:math.dist(blocks[k].location[:2],estimate[:2]) for k in candidates}
                    touched=[k for k,d in distances.items() if d<.008]
                    if len(touched)!=1 or touched[0]!=(batch,wanted):
                        status="HOLD_GRASP_RECEIPT"
                        break
                    ident=touched[0]
                    event["contact_offset_mm"]=1000*distances[ident]
                    cargo[ident]=next(j for j in range(3) if j not in cargo.values())
                    location[ident]="CARGO"
                    picks+=1
                else:
                    ident=(batch,wanted)
                    assert ident in cargo
                    transfer_z=.003
                    center=target_slot(station,slot)
                    if support_top:
                        support=blocks[1,expected]
                        if location[1,expected]!="BUFFER" or expected!=wanted:
                            status="HOLD_WRONG_BASE"
                            break
                        center=support.location
                        transfer_z=center.z+.060
                    offset=math.dist(estimate[:2],center[:2])
                    event["release_offset_mm"]=1000*offset
                    # Conservative simulation receipt inside base support / first ring.
                    if offset>.008:
                        status="HOLD_RELEASE_RECEIPT"
                        break
                    del cargo[ident]
                    location[ident]=station
                    blocks[ident].location=(estimate.x,estimate.y,transfer_z)
                    if station=="BUFFER" and not support_top:
                        bases[slot]=wanted
                    placements+=1
                event["state"]="SIMULATED_TRANSFER"
                root.delta_location=(0,0,0)
                s.view_settings.exposure=ambient
            else:
                status="DONE" if picks==placements==12 and len(bases)==3 and not cargo else "INCOMPLETE"
        finally:
            engine.stdin.close()
            if engine.wait(timeout=15):
                raise RuntimeError("Inference worker failure")
    referee={}
    if status=="DONE":
        start=steps[0]["poses"][0]
        start_error=math.dist(root.matrix_world.translation[:2],start[:2])
        stacks=[]
        for slot,color in sorted(bases.items()):
            bottom,upper=blocks[1,color].location,blocks[2,color].location
            support_error=1000*math.dist(bottom[:2],upper[:2])
            ring_error=1000*math.dist(bottom[:2],target_slot("BUFFER",slot)[:2])
            dz=1000*(upper.z-bottom.z)
            stacks.append(dict(slot=slot,color=color,first_ring_error_mm=ring_error,
                               second_xy_offset_mm=support_error,layer_height_mm=dz))
        referee=dict(start_center_error_mm=1000*start_error,stacks=stacks)
        assert start_error<.001 and all(v=="BUFFER" for v in location.values())
        assert all(s["first_ring_error_mm"]<=8 and s["second_xy_offset_mm"]<=8
                   and abs(s["layer_height_mm"]-60)<.1 for s in stacks)
        shoot(s,s.objects["Camera"],out/(images+"_finish.png"))
    write_json(receipt,dict(status=status,qr=qr,picks=picks,placements=placements,black_paper_profile=profile_review,
              challenge=challenge,referee=referee,base_exposure=ambient,
              last_step=events[-1]["step"],events=events,
              scope="One fixed layout. QR and per-action RGB drive a mutable object world. Chassis follows recorded "
              "waypoints, fixture heights and pose known, contact receipts approximate. No hardware I/O."))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("out","hub","python"):
        ap.add_argument("--"+name,required=True,type=Path)
    ap.add_argument("--profile-review",action="store_true")
    ap.add_argument("--challenge",action="store_true")
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:])
    if args.challenge and not args.profile_review:
        ap.error("--challenge uses the profile-review configuration")
    simulate(args.out.resolve(),args.hub.resolve(),args.python,args.profile_review,args.challenge)
