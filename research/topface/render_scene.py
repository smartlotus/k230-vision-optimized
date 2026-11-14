import blenderproc as bproc
"""Original fused CAD, face-level visible masks, and analytic full-top geometry.

Executed by blenderproc run. This smoke/preview renderer never starts training.
One material-index pass labels top triangles without splitting the mesh.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).parent))
from geometry import camera_pose, intrinsics, pixel_to_top, project, top_boundary


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def material(name, color, roughness=.65, top_id=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.)
    shader.inputs["Roughness"].default_value = roughness
    mat.pass_index = top_id
    return mat


def device_probe():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    attempts = []
    chosen = "CPU"
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            devices = [{"name": d.name, "type": d.type} for d in preferences.devices]
            attempts.append({"backend": backend, "devices": devices})
            usable = [d for d in preferences.devices if d.type == backend]
            if usable:
                for item in preferences.devices:
                    item.use = item.type == backend
                chosen = backend
                break
        except Exception as error:
            attempts.append({"backend": backend, "error_type": type(error).__name__})
    bpy.context.scene.cycles.device = "CPU" if chosen == "CPU" else "GPU"
    return {"selected_backend": chosen, "attempts": attempts,
            "gpu_render_verified": False}


def import_cad(asset_dir):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(asset_dir/"material_block.glb"))
    meshes = [o for o in set(bpy.data.objects)-before if o.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError("Original GLB must import as one fused mesh")
    obj = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.rotation_mode = "XYZ"
    coordinates = np.array([list(v.co) for v in obj.data.vertices])
    size = np.ptp(coordinates, axis=0)
    correction = 0.
    # The supplied GLB stores the source STL's Z-up geometry without converting
    # it to glTF's Y-up convention. Native Blender import consequently puts the
    # 60mm axis along Y. Correct only that coordinate frame; preserve topology.
    if np.allclose(size, [.05, .06, .05], atol=2e-4):
        y_min, y_max = coordinates[:,1].min(), coordinates[:,1].max()
        if abs(y_max) < 1e-5 and abs(y_min+.06) < 1e-5:
            correction = -90.
        elif abs(y_min) < 1e-5 and abs(y_max-.06) < 1e-5:
            correction = 90.
        else:
            raise RuntimeError("Unexpected GLB Y-axis origin; require explicit coordinate review")
        obj.rotation_euler.x = math.radians(correction)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj.data.update()
        coordinates = np.array([list(v.co) for v in obj.data.vertices])
        size = np.ptp(coordinates, axis=0)
    if not np.allclose(size, [.05, .05, .06], atol=2e-4):
        raise RuntimeError(f"Unexpected original CAD dimensions/orientation: {size.tolist()}")
    if abs(coordinates[:, 2].min()) > 1e-5:
        raise RuntimeError("Original CAD origin must remain at its bottom center")
    top_faces = []
    top_area = 0.
    for face in obj.data.polygons:
        vertices = coordinates[list(face.vertices)]
        if np.all(np.abs(vertices[:, 2]-.06) < 2e-5) and face.normal.z > .999:
            if np.any(np.linalg.norm(vertices[:, :2], axis=1) > .01502):
                raise RuntimeError("Top material selection contains a side or oversized face")
            top_faces.append(face.index)
            top_area += face.area
    if abs(top_area-math.pi*.015**2) > math.pi*.015**2*.005:
        raise RuntimeError("Face-level top selection does not cover the top disk")
    # No vertices, triangles, or connected components are changed for labeling.
    receipt = {"mesh_count": len(meshes), "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons), "size_m": size.tolist(),
        "top_face_count": len(top_faces), "top_area_m2": top_area,
        "glb_up_axis_correction_deg": correction,
        "original_topology_preserved": True, "units": "m"}
    return obj, set(top_faces), receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if not 1 <= args.count <= 200:
        raise ValueError("This entrypoint is limited to smoke/preview, maximum 200 frames")
    args.output.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.max_bounces = 3
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.image_settings.file_format = "PNG"
    device = device_probe()
    save_json(args.output/"render_device.json", device)
    original, top_faces, cad_receipt = import_cad(args.assets)
    save_json(args.output/"cad_import.json", cad_receipt)
    original.hide_render = True
    colors = {"red": (.7,.025,.02), "yellow": (.8,.65,.01), "blue": (.01,.04,.6),
              "green": (.015,.45,.025), "black": (.007,.007,.007), "light_blue": (.02,.45,.65)}
    instances = []
    for index in range(4):
        obj = original.copy()
        obj.data = original.data.copy()
        bpy.context.collection.objects.link(obj)
        obj.data.materials.clear()
        obj.data.materials.append(material(f"body{index}", colors["red"]))
        obj.data.materials.append(material(f"top{index}", colors["red"], top_id=index+1))
        for face in obj.data.polygons:
            face.material_index = int(face.index in top_faces)
        instances.append(obj)
    bproc.object.create_primitive("PLANE", scale=[2,2,1]).set_materials([
        bproc.types.Material(material("table", (.3,.28,.23)))])
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([-.25,-.25,.7])
    light.set_energy(70)
    light.blender_obj.data.shape = "DISK"
    light.blender_obj.data.size = .4
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (.5,.5,.5,1.)
    background.inputs["Strength"].default_value = .3
    # ID masks are face material indexes in the same Cycles render as the RGB.
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layer = tree.nodes.new("CompositorNodeRLayers")
    bpy.context.view_layer.use_pass_material_index = True
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layer.outputs["Image"], composite.inputs["Image"])
    mask_outputs = []
    for index in range(4):
        mask = tree.nodes.new("CompositorNodeIDMask")
        mask.index = index+1
        mask.use_antialiasing = False
        tree.links.new(layer.outputs["IndexMA"], mask.inputs["ID value"])
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.format.color_depth = "8"
        tree.links.new(mask.outputs["Alpha"], output.inputs[0])
        mask_outputs.append(output)
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    for frame in range(args.count):
        directory = args.output/f"scene-{frame:05d}"
        directory.mkdir(exist_ok=False)
        # The first frame is a deterministic center-projection qualification image.
        height = .25 if frame == 0 else float(rng.uniform(.1, .4))
        angle = 90. if frame == 0 else float(rng.uniform(80, 100))
        azimuth = float(rng.uniform(0,360))
        hfov = 65. if frame == 0 else float(rng.uniform(62,68))
        pose = camera_pose(height, angle, azimuth)
        K = intrinsics(640,360,hfov)
        bproc.camera.set_intrinsics_from_K_matrix(K, 640,360)
        # Convert from OpenCV camera axes to Blender (+X right, +Y up, -Z forward).
        blender_pose = pose @ np.diag([1.,-1.,-1.,1.])
        scene.camera.matrix_world = Matrix(blender_pose.tolist())
        count = 1 if frame == 0 else frame % 5
        placed = []
        labels = []
        for index, obj in enumerate(instances):
            obj.hide_render = index >= count
            if index >= count:
                continue
            # Keep all four labels; never cap the annotation count to three.
            for attempt in range(300):
                pixel = [320,180] if frame == 0 else rng.uniform([15,15], [625,345])
                location = pixel_to_top(pixel, K, pose)[:2]
                if all(np.linalg.norm(location-other) >= .052 for other in placed):
                    break
            else:
                raise RuntimeError("Could not place nonintersecting original CAD blocks")
            placed.append(location)
            obj.location = [*location, 0.]
            obj.rotation_euler = [0.,0.,float(rng.uniform(0,2*math.pi))]
            color_name = list(colors)[(frame+index) % 6]
            for mat in obj.data.materials:
                shader = mat.node_tree.nodes.get("Principled BSDF")
                shader.inputs["Base Color"].default_value = (*colors[color_name],1.)
                shader.inputs["Roughness"].default_value = float(rng.uniform(.35,.85))
            bpy.context.view_layer.update()
            boundary = project(top_boundary(location), K, pose)
            labels.append({"instance_id": index+1, "color_material": color_name,
                "object_pose": [list(row) for row in obj.matrix_world],
                "top_boundary_xy": boundary.tolist(),
                "physical_top_center_xy": project([[*location,.06]], K, pose)[0].tolist(),
                "top_face_visible_mask": f"visible-{index+1}.png"})
        for index, output in enumerate(mask_outputs):
            output.base_path = str(directory)
            output.file_slots[0].path = f"mask-{index+1}-"
        scene.frame_set(frame)
        scene.render.filepath = str(directory/"rgb.png")
        bpy.ops.render.render(write_still=True)
        for index in range(4):
            outputs = list(directory.glob(f"mask-{index+1}-*.png"))
            if len(outputs) != 1:
                raise RuntimeError("Expected one material-index output per instance")
            outputs[0].rename(directory/f"visible-{index+1}.png")
        save_json(directory/"labels_raw.json", {"schema": "topface-render-v1",
            "scene_group_id": f"preview-{args.seed}-{frame}", "random_seed": args.seed,
            "camera_intrinsics": K.tolist(), "camera_pose_world_from_cv": pose.tolist(),
            "camera_pose_world_from_blender": blender_pose.tolist(),
            "camera_height_above_top_mm": height*1000, "plane_angle_deg": angle,
            "absolute_tilt_from_normal_deg": abs(90-angle), "hfov_deg": hfov,
            "intrinsics_status": "estimated", "distortion": "none_smoke_qualification",
            "object_pose_status": "upright", "width": 640, "height": 360,
            "instances": labels})
        save_json(args.output/"PROGRESS.json", {"completed_frames": frame+1,
            "requested_frames": args.count, "elapsed_seconds": time.monotonic()-start})
    device["gpu_render_verified"] = device["selected_backend"] != "CPU"
    device["successful_frames"] = args.count
    device["blender_version"] = bpy.app.version_string
    save_json(args.output/"render_device.json", device)


if __name__ == "__main__":
    main()
