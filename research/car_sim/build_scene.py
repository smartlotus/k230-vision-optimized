"""Build the photo-based rig and a playable task replay in Blender.

Run with Blender --background --python-exit-code 1 --python build_scene.py --
--hub K230_PROJECT --out NEW_DIRECTORY. Scene units are metres.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "topface"), str(HERE.parents[1] / "k230_transport_mvp")]
from native_cad import import_cad, material
from round_plan import build_round
from vehicle import build_vehicle, set_vehicle_pose

TASK = "156+123+516+231"
COLORS = {1: (.73, .014, .015), 5: (.016, .017, .019), 6: (.06, .50, .82)}
OTHER_COLORS = {2: (.72, .58, .008), 3: (.01, .04, .6), 4: (.018, .47, .056)}
RAW_ORDERS = ((1, 5, 6), (5, 1, 6))
CARGO = [(-.068, -.053, .081), (0, -.053, .081), (.068, -.053, .081)]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def configure(scene, samples=4):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 640, 360
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = 20
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8


def camera(name, pos, aim, ortho=None, parent=None):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = pos
    obj.rotation_euler = (Vector(aim) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    data.clip_start = .001
    if ortho:
        data.type, data.ortho_scale = "ORTHO", ortho
    return obj


def shoot(scene, cam, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = cam
    scene.render.filepath = str(path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def studio_light(name, pos, power, size, target=(0, 0, .1)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size = power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = pos
    obj.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()


def screen_view(cam):
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                region = area.spaces.active.region_3d
                region.view_perspective = "CAMERA"
                area.spaces.active.overlay.show_overlays = False
                area.spaces.active.shading.color_type = "MATERIAL"


def vehicle_asset(out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    configure(s, 12)
    s.world = bpy.data.worlds.new("Studio")
    s.world.color = (.40, .40, .40)
    rig = build_vehicle()
    s.render.resolution_x, s.render.resolution_y = 960, 540
    presentation = camera("Vehicle overview", (.68, .91, .66), (.055, 0, .208), .96)
    plan_view = camera("Plan / front left, right up", (.060, 0, .82), (.060,0,0), .90)
    plan_view.rotation_euler = (0, 0, -math.pi/2)
    studio_light("Key", (.40, .34, .78), 12, .48)
    studio_light("Rim", (-.36, -.4, .6), 9, .4)
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -.001))
    floor = bpy.context.object
    floor.name = "Studio ground"
    floor.data.materials.append(material("Studio warm gray", (.40, .42, .44)))
    for f, deployed, z in ((1,0,330),(15,0,330),(40,1,400),(55,1,400),(85,1,100),(110,1,400)):
        set_vehicle_pose(rig, deployed, z)
        for obj, prop in ((rig["turret"], "rotation_euler"), (rig["lift"], "location")):
            obj.keyframe_insert(prop, frame=f)
    s.frame_end = 110
    s.timeline_markers.new("STOWED", frame=1)
    s.timeline_markers.new("DEPLOYED", frame=40)
    s.timeline_markers.new("LIFT MIN 100 mm", frame=85)
    s.frame_set(1)
    s.camera = presentation
    screen_view(presentation)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "vehicle.blend"), compress=True)
    for f, name in ((1,"stowed"), (40,"deployed")):
        s.frame_set(f)
        shoot(s, presentation, out / (name + ".png"))
        shoot(s, plan_view, out / ("plan_" + name + ".png"))
    poses = []
    for f,h in ((1,330),(40,400),(85,100)):
        s.frame_set(f)
        opt = rig["camera"].matrix_world
        x,y,z = opt.translation
        axis = rig["turret"].matrix_world.translation
        radius = math.hypot(x-axis.x,y-axis.y)
        assert abs(z*1000-h)<.001 and abs(radius-.180)<.00001
        assert (opt.to_quaternion() @ Vector((0,0,-1))).z < -.99999
        holder=rig["holder"]
        endpoints=[holder.matrix_world @ Vector((0,0,z_local)) for z_local in
                   (min(v.co.z for v in holder.data.vertices),max(v.co.z for v in holder.data.vertices))]
        holder_axis=(endpoints[1]-endpoints[0]).normalized()
        assert holder_axis.z>.999999
        hubs=[wheel.matrix_world.translation for wheel in rig["right_wheels"]]
        mid=(hubs[0]+hubs[1])/2
        if f!=1:
            assert abs(y-mid.y)<1e-6 and x>mid.x
        assert (x>.140 if f!=1 else abs(x)<.140 and abs(y)<.1475)
        deps=bpy.context.evaluated_depsgraph_get()
        mesh_objects=[o.evaluated_get(deps) for o in rig["collection"].objects if o.type=="MESH"]
        points=[o.matrix_world @ v.co for o in mesh_objects for v in o.data.vertices]
        spans=[1000*(max(v[i] for v in points)-min(v[i] for v in points)) for i in range(3)]
        if f==1:
            assert all(v<=limit+.001 for v,limit in zip(spans,(300.,300.,400.)))
        poses.append(dict(frame=f,xyz_mm=[1000*v for v in opt.translation],radius_mm=1000*radius,
                          mesh_xyz_span_mm=spans,holder_axis=list(holder_axis),
                          right_wheel_midpoint_xy_mm=[1000*v for v in mid[:2]],
                          lens_minus_midpoint_xy_mm=[1000*(x-mid.x),1000*(y-mid.y)],
                          right_edge_overhang_mm=1000*x-140,
                          turret_xy_mm=[1000*v for v in axis[:2]]))
    write_json(out/"rig_check.json", poses)


def add_qr_board(texture):
    def panel(name, x, width, height):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x, .08, .155),
                                         rotation=(math.pi/2, 0, -math.pi/2))
        obj = bpy.context.object
        obj.name = name
        obj.scale = (width, height, 1)
        obj["road_label"] = 6
        return obj
    board = panel("QR / A4 / assumed pose", 1.204, .297, .210)
    board.data.materials.append(material("Paper", (.92, .92, .92)))
    patch = panel("QR / task code", 1.2038, .080, .080)
    mat = material("Printed task", (1, 1, 1))
    image = bpy.data.images.load(str(texture))
    image.pack()
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image, node.interpolation = image, "Closest"
    mat.node_tree.links.new(node.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    patch.data.materials.append(mat)


def make_blocks(asset_dir, orders=RAW_ORDERS):
    first, _, receipt = import_cad(asset_dir)
    blocks = {}
    palette = {**COLORS, **OTHER_COLORS}
    for batch, order in enumerate(orders, 1):
        for color in sorted(order):
            obj = first if not blocks else first.copy()
            if blocks:
                bpy.context.collection.objects.link(obj)
            obj.name = f"Batch {batch} / color {color}"
            if not obj.data.materials:
                obj.data.materials.append(None)
            slot = obj.material_slots[0]
            slot.link = "OBJECT"
            slot.material = bpy.data.materials.get(f"Task color {color}") or material(f"Task color {color}", palette[color])
            obj["road_label"], obj["batch"], obj["color_id"] = 5, batch, color
            blocks[batch, color] = obj
    return blocks, receipt


def local_to_world(pose, local):
    x,y,angle = pose
    u,v,z = local
    return Vector((x+math.cos(angle)*u-math.sin(angle)*v, y+math.sin(angle)*u+math.cos(angle)*v, z))


def target_slot(station, slot, layer=1):
    t = (slot-2)*.15
    if station == "PROCESS":
        return Vector((-t, -1.125, .003))
    return Vector((-1.125, t, .003+.060*(layer-1)))


def raw_pos(batch, color, spin, top_z, orders=RAW_ORDERS):
    order = orders[batch - 1]
    a = -math.pi/2 + order.index(color)*math.tau/3 + spin
    return Vector((.09*math.cos(a), 1.27+.09*math.sin(a), top_z))


def print_slot_numbers():
    ink=material("Station numerals / black ink",(.015,.015,.015))
    for station,rotation in (("PROCESS",math.pi),("BUFFER",math.pi/2)):
        for slot in (1,2,3):
            point=target_slot(station,slot)
            surface="rough_processing_zone" if station=="PROCESS" else "temporary_storage_zone"
            panel=bpy.data.objects[surface]
            point.z=max((panel.matrix_world @ v.co).z for v in panel.data.vertices)+.0001
            bpy.ops.object.text_add(location=point,rotation=(0,0,rotation))
            label=bpy.context.object
            label.name=f"{station} / printed {slot}"
            label.data.body=str(slot)
            label.data.align_x="CENTER"
            label.data.align_y="CENTER"
            label.data.size=.024
            label.data.materials.append(ink)
            bpy.ops.object.convert(target="MESH")
            bpy.context.object["road_label"]=6


def timeline(scene, rig, blocks, plan, table_z, orders=RAW_ORDERS):
    steps = plan["steps"]
    pose = list(steps[0]["poses"][0])
    frame, fraction, spin, batch = 1, 0., 0., 1
    carried, placed = {}, {}
    records, views = [], []
    def key(override=None):
        root = rig["root"]
        root.location = (*pose[:2], 0.)
        root.rotation_euler.z = pose[2]
        set_vehicle_pose(rig, fraction)
        for obj, prop in ((root, "location"),(root, "rotation_euler"),(rig["turret"], "rotation_euler"),(rig["lift"], "location")):
            obj.keyframe_insert(prop, frame=frame)
        for ident, obj in blocks.items():
            if override and override[0] == ident:
                pos = override[1]
            elif ident in carried:
                pos = local_to_world(pose,CARGO[carried[ident]])
            elif ident in placed:
                pos = placed[ident]
            elif ident[0] == batch:
                pos = raw_pos(*ident, spin, table_z, orders)
            else:
                pos = (0, 0, -1.)
            obj.location = pos
            obj.keyframe_insert("location",frame=frame)
    key()
    def shift(new_fraction):
        nonlocal frame, fraction
        if abs(fraction-new_fraction)>1e-4:
            frame += 6
            fraction = new_fraction
            key()
    def move(next_pose, driving=True):
        nonlocal frame,pose
        x,y,raw_angle = next_pose
        # Continuous yaw avoids full turns at the +-pi seam.
        da = (raw_angle-pose[2]+math.pi)%math.tau-math.pi
        distance = math.hypot(x-pose[0],y-pose[1])
        if max(distance,abs(da)) < 1e-7:
            return
        if driving:
            shift(1.)
        start=list(pose)
        segments=max(1,math.ceil(abs(da)/.15))
        ticks=max(segments, math.ceil(20*max(distance/.42,abs(da)/1.6)))
        start_frame=frame
        for k in range(1,segments+1):
            t=k/segments
            pose=[start[0]+(x-start[0])*t,start[1]+(y-start[1])*t,start[2]+da*t]
            frame=start_frame+round(t*ticks)
            key()
    for index,step in enumerate(steps):
        action=step["action"]
        begin=frame
        scene.timeline_markers.new(f"{index:02} {action} {step.get('target_color') or ''}",frame=frame)
        for waypoint in step["poses"]:
            if frame==1 and math.dist(pose[:2],waypoint[:2])>.04:
                # The event rules allow unfolding after departure. The chassis
                # travels the first 40 mm using its declared folded outline.
                ratio=.04/math.dist(pose[:2],waypoint[:2])
                move([pose[0]+ratio*(waypoint[0]-pose[0]),pose[1]+ratio*(waypoint[1]-pose[1]),pose[2]],False)
            move(waypoint)
        arrival=frame
        capture=None
        if action in ("PICK","PLACE"):
            ident=(step["batch"],step["target_color"])
            station=step["source_station"] if action=="PICK" else step["dest_station"]
            if station=="RAW" and ident[0]!=batch:
                frame+=1; batch=ident[0]; spin=0.; key()
            if action=="PICK" and station=="RAW":
                order=orders[batch - 1]
                desired=-order.index(ident[1])*math.tau/3
                delta=desired-spin
                count=max(1,math.ceil(abs(delta)/.15))
                for k in range(count):
                    spin+=delta/count; frame+=max(1,round(abs(delta)/count*20*8/math.tau)); key()
            views.append((frame,index,action,station,"arrival"))
            capture=frame
            shift(0.)
            if action=="PICK":
                src=raw_pos(*ident,spin,table_z,orders) if station=="RAW" else placed.pop(ident)
                carrier_slot=next(i for i in range(3) if i not in carried.values())
                dst=local_to_world(pose,CARGO[carrier_slot])
            else:
                src=local_to_world(pose,CARGO[carried.pop(ident)])
                dst=target_slot(station,step["target_slot"],step["layer"])
            # Known waypoints only; arm IK, contact and grip force are unmeasured.
            clearance=max(src.z,dst.z)+.075
            for point in (src,Vector((src.x,src.y,clearance)),Vector((dst.x,dst.y,clearance)),dst):
                frame+=5; key((ident,point))
            if action=="PICK":
                carried[ident]=carrier_slot
            else:
                placed[ident]=dst
            frame+=4; key()
            views.append((frame,index,action,station,"stowed_after"))
        elif action in ("QR","SCAN_QR","SCAN"):
            shift(0.)
            frame+=4; key()
            views.append((frame,index,"QR","QR","front"))
        elif action=="RETURN":
            shift(0.)
        else:
            views.append((arrival,index,action,step.get("dest_station"),"arrival"))
        records.append(dict(index=index, action=action,start=begin,arrival=arrival,end=frame,observation_frame=capture,
                            batch=step.get("batch"),color=step.get("target_color"),slot=step.get("target_slot"),
                            layer=step.get("layer")))
    for obj in [*blocks.values(),rig["root"],rig["turret"],rig["lift"]]:
        action=obj.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation="LINEAR"
    assert len(placed)==6 and not carried
    for color in orders[0]:
        assert (placed[2,color]-placed[1,color]-Vector((0,0,.06))).length<1e-6
    scene.frame_end=frame
    scene.frame_set(1)
    return records, views


def road_label(obj):
    if "road_label" in obj:
        return int(obj["road_label"])
    if obj.name == "Outside":
        return 255
    if obj.data and getattr(obj.data, "materials", None):
        mat = obj.data.materials[0]
        if mat and mat.name.startswith("Class "):
            return int(mat.name.split()[1])
    return 255


def labels(scene, cam, blocks, path):
    """Ray-cast the rendered geometry at pixel centres, including self-occlusion."""
    import struct, zlib
    width,height = 160,90
    deps = bpy.context.evaluated_depsgraph_get()
    origin=cam.matrix_world.translation
    rotation=cam.matrix_world.to_quaternion()
    fx=cam.data.lens/cam.data.sensor_width*width
    rows=[]
    for y in range(height):
        row=bytearray()
        for x in range(width):
            ray=rotation @ Vector(((x+.5-width/2)/fx,(height/2-y-.5)/fx,-1))
            hit,_,_,_,obj,_=scene.ray_cast(deps,origin,ray,distance=8)
            row.append(road_label(obj) if hit else 255)
        rows.append(b"\0"+row)
    def chunk(tag,data):
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,0,0,0,0))+chunk(b"IDAT",zlib.compress(b"".join(rows)))+chunk(b"IEND",b""))
    visible=[]
    for (batch,color),obj in blocks.items():
        center=obj.matrix_world @ Vector((0,0,.06))
        p=world_to_camera_view(scene,cam,center)
        if not (p.z>0 and 0<=p.x<1 and 0<p.y<=1):
            continue
        hit,point,_,_,other,_=scene.ray_cast(deps,origin,center-origin,distance=8)
        if hit and other==obj and (point-center).length<.0002:
            visible.append(dict(color_id=color,center_xy=[640*p.x,360*(1-p.y)]))
    return visible


def prepare_round(hub, task=TASK, start_id="START_1", orders=RAW_ORDERS, qr_texture=None, gripper=None):
    from core.task_parser import parse_task

    parsed = parse_task(task)
    if len(orders) != 2 or any(
        len(order) != 3 or set(order) != set(colors)
        for order, colors in zip(orders, (parsed.batch1_colors, parsed.batch2_colors))
    ):
        raise ValueError("Each turntable order must contain exactly the colors of its batch")
    field=hub/"worktrees/ops-temporal-camera-v1/docs/temporal_camera/map-correction/field_gray_v1.blend"
    assert hashlib.sha256(field.read_bytes()).hexdigest()=="b5bea8c6738bc7ebd8d57cdae34af99d1cf10b3fc782ea9c6c986e8802441e95"
    bpy.ops.wm.open_mainfile(filepath=str(field))
    scene=bpy.data.scenes["Scene"]
    bpy.context.window.scene=scene
    bpy.data.scenes.remove(bpy.data.scenes["Semantic object IDs"])
    configure(scene,3)
    for obj in list(scene.objects):
        if obj.name.startswith("Block "):
            bpy.data.objects.remove(obj,do_unlink=True)
        elif obj.hide_render:
            obj.hide_viewport=True
    print_slot_numbers()
    add_qr_board(qr_texture or HERE.parents[1]/"k230_transport_mvp/data/task_qr.png")
    rig=build_vehicle(gripper)
    front=camera("QR camera / assumed mount",(0,.062,.155),(0,.30,.155),parent=rig["root"])
    front.data.lens=26
    overhead=scene.objects["Camera"]
    overhead.data.type="ORTHO"
    overhead.data.ortho_scale=5.05
    overhead.location=(0,0,4)
    overhead.rotation_euler=(0,0,0)
    cad=hub/"webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/k230_material_block_model"
    blocks,receipt=make_blocks(cad, orders)
    plan=build_round(task, start_id)
    table=scene.objects["Turntable"]
    records,views=timeline(scene,rig,blocks,plan,table.location.z+table.dimensions.z/2,orders)
    scene.camera=overhead
    screen_view(overhead)
    scene["task"]=task
    scene["scope"]="Prescribed kinematic mission; movable side camera, real CAD, frozen perception evaluated on separate renders."
    scene["body_axes"]="+Y front, +X right, +Z up"
    return scene, rig, front, overhead, blocks, plan, records, views, receipt


def run(hub,out):
    out.mkdir(parents=True,exist_ok=True)
    vehicle_asset(out)
    scene,rig,front,overhead,blocks,plan,records,views,receipt=prepare_round(hub)
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"competition_round.blend"),compress=True)
    write_json(out/"round.json",dict(plan=plan,timeline=records,cad=receipt,frame_end=scene.frame_end))
    scene.frame_set(scene.frame_end)
    shoot(scene,overhead,out/"finish.png")
    scene.frame_set(records[0]["end"])
    shoot(scene,front,out/"qr.png")
    mission_truth = {}
    for frame,index,action,station,state in views:
        if action not in ("PICK","PLACE") or state!="arrival":
            continue
        scene.frame_set(frame)
        cam=rig["camera"]
        filename=f"{index:02d}.png"
        shoot(scene,cam,out/"mission_camera"/filename)
        step=plan["steps"][index]
        slot=step["target_slot"]
        support=None
        if action=="PICK":
            obj=blocks[step["batch"],step["target_color"]]
            point=obj.matrix_world @ Vector((0,0,.060))
        elif step["layer"]==2:
            base_step=next(s for s in plan["steps"][:index] if s["action"]=="PLACE" and
                           s["dest_station"]==station and s["target_slot"]==slot and s["layer"]==1)
            support=base_step["target_color"]
            obj=blocks[base_step["batch"],support]
            point=obj.matrix_world @ Vector((0,0,.060))
        else:
            obj=None
            point=target_slot(station,slot)
        projected=world_to_camera_view(scene,cam,point)
        hit,near,_,_,surface,_=scene.ray_cast(bpy.context.evaluated_depsgraph_get(),
                    cam.matrix_world.translation,point-cam.matrix_world.translation,distance=3)
        visible=hit and (near-point).length<.008 and (obj is None or surface==obj)
        in_frame=projected.z>0 and 0<=projected.x<=1 and 0<=projected.y<=1
        mission_truth[filename]=dict(station=station,support_color_id=support,
                    target_center_xy=[640*projected.x,360*(1-projected.y)],
                    visible_target=bool(in_frame and visible))
    write_json(out/"mission_truth.json",dict(frames=mission_truth))
    # Distinct sampled route windows, lighting conditions and docking perturbations.
    groups=[("train",31),("train",52),("train",83),("val",94),("test",115)]
    rows,truth=[],{}
    choices=[v for v in views if v[4]=="arrival" and v[0]>1]
    base_exposure = scene.view_settings.exposure
    for i,(split,seed) in enumerate(groups):
        rng=random.Random(seed)
        group=f"{split}-{seed}"
        for j,category in enumerate(("road","raw","process","buffer","stowed","road2")):
            if category.startswith("road"):
                candidates=[r for r in records if r["action"] in ("MOVE","RETURN") and r["end"]-r["start"]>40]
                rec=rng.choice(candidates)
                frame=rng.randint(rec["start"]+12,rec["end"]-8)
            elif category=="stowed":
                frame=rng.choice([v[0] for v in views if v[4]=="stowed_after"])
            else:
                frame=rng.choice([v[0] for v in choices if v[3]==category.upper() and v[2] in ("PICK","PLACE")])
            scene.frame_set(frame)
            root=rig["root"]
            root.delta_location=(rng.uniform(-.024,.024),rng.uniform(-.024,.024),0)
            root.delta_rotation_euler.z=rng.uniform(-.105,.105)
            scene.view_settings.exposure=base_exposure+(-.45,-.12,.15,.32,.48)[i]
            cam=rig["camera"]
            rgb=f"{group}/{j}-{category}.png"
            mask=f"masks/{rgb}"
            shoot(scene,cam,out/"rgb"/rgb)
            objects=labels(scene,cam,blocks,out/mask)
            rows.append(dict(group=group,split=split,pose=category,rgb=f"rgb/{rgb}",mask=mask,frame=frame,
                             world_from_camera=[list(line) for line in cam.matrix_world]))
            truth[rgb]=dict(road_mask=mask,topfaces=objects)
        write_json(out/"splits.json",dict(frames=rows))
        write_json(out/"truth.json",dict(frames=truth))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--hub",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:])
    run(args.hub.resolve(),args.out.resolve())
