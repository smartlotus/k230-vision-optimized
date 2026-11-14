"""Native Blender helpers, reused from the validated original-CAD importer."""
import math
import bpy
import numpy as np

def material(name, color, roughness=.65, top_id=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.)
    shader.inputs["Roughness"].default_value = roughness
    mat.pass_index = top_id
    return mat

def device_probe(force_cpu=False):
    if force_cpu:
        bpy.context.scene.cycles.device = "CPU"
        return {"selected_backend":"CPU","attempts":[],"gpu_render_verified":False}
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
