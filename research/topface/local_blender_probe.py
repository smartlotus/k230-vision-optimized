"""Native Blender integration probe for the same CAD and mask functions.

Run in factory-startup/background mode; CPU only. This validates native Blender
geometry/compositor behavior, not BlenderProc installation or Kaggle admission.
"""
import ast
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix

args=sys.argv[sys.argv.index("--")+1:]
source,assets,output=[Path(value).resolve() for value in args]
output.mkdir(parents=True,exist_ok=True)
namespace={"bpy":bpy,"np":np,"math":__import__("math")}
module=ast.parse(source.read_text(encoding="utf8"))
functions=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in {"import_cad","material"}]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),namespace)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
obj,faces,receipt=namespace["import_cad"](assets)
obj.data.materials.clear()
obj.data.materials.append(namespace["material"]("body",(.7,.02,.02)))
obj.data.materials.append(namespace["material"]("top",(.7,.02,.02),top_id=1))
for face in obj.data.polygons:face.material_index=int(face.index in faces)
bpy.ops.mesh.primitive_plane_add(size=2)
table=bpy.context.object
table.data.materials.append(namespace["material"]("table",(.25,.25,.25)))
bpy.ops.object.light_add(type="AREA",location=(.1,-.15,.5))
light=bpy.context.object
light.data.energy=30
light.data.size=.3
bpy.ops.object.camera_add(location=(0,0,.31))
camera=bpy.context.object
camera.matrix_world=Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,.31],[0,0,0,1]])
camera.data.type="PERSP"
camera.data.lens=36/(2*np.tan(np.radians(65)/2))
camera.data.sensor_width=36
camera.data.sensor_fit="HORIZONTAL"
scene=bpy.context.scene
scene.camera=camera
scene.render.engine="CYCLES"
scene.cycles.device="CPU"
scene.cycles.samples=12
scene.render.threads_mode="FIXED"
scene.render.threads=2
scene.render.resolution_x=640
scene.render.resolution_y=360
scene.render.resolution_percentage=100
scene.render.image_settings.file_format="PNG"
bpy.context.view_layer.use_pass_material_index=True
scene.use_nodes=True
tree=scene.node_tree
tree.nodes.clear()
layer=tree.nodes.new("CompositorNodeRLayers")
composite=tree.nodes.new("CompositorNodeComposite")
tree.links.new(layer.outputs["Image"],composite.inputs["Image"])
mask=tree.nodes.new("CompositorNodeIDMask")
mask.index=1
mask.use_antialiasing=False
tree.links.new(layer.outputs["IndexMA"],mask.inputs["ID value"])
file=tree.nodes.new("CompositorNodeOutputFile")
file.base_path=str(output)
file.file_slots[0].path="visible-"
file.format.file_format="PNG"
file.format.color_mode="BW"
file.format.color_depth="8"
tree.links.new(mask.outputs["Alpha"],file.inputs[0])
scene.frame_set(0)
scene.render.filepath=str(output/"rgb.png")
bpy.ops.render.render(write_still=True)
receipt.update(scope="NATIVE_LOCAL_BLENDER_CPU_ONLY",blender_version=bpy.app.version_string)
(output/"cad_probe.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
