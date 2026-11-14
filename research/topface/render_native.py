"""V1.1 native Blender renderer; no BlenderProc or pip installation required.

blender --background --factory-startup --python-exit-code 1 --python this.py -- ...
The original fused CAD uses top-face material indexes in the same RGB render.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import project,top_boundary
from native_cad import import_cad,material,device_probe
from scene_config import COLORS,make_config


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temporary.replace(path)


def printed_material(name,color,top_id=0):
    mat=material(name,color,top_id=top_id)
    tree=mat.node_tree;shader=tree.nodes.get('Principled BSDF')
    coords=tree.nodes.new('ShaderNodeTexCoord');noise=tree.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value=1500.;noise.inputs['Detail'].default_value=2.
    bump=tree.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00015
    tree.links.new(coords.outputs['Object'],noise.inputs['Vector'])
    tree.links.new(noise.outputs['Fac'],bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'],shader.inputs['Normal'])
    return mat


def set_camera(scene,K,world_from_cv):
    camera=scene.camera;data=camera.data
    scene.render.resolution_x=640;scene.render.resolution_y=360
    scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=1.;scene.render.pixel_aspect_y=1.
    if abs(K[0,0]-K[1,1])>1e-6:raise ValueError('This inferred camera uses square pixels')
    data.type='PERSP';data.sensor_fit='HORIZONTAL';data.sensor_width=36.
    data.lens=float(K[0,0]*36./640)
    data.shift_x=float((320-K[0,2])/640)
    data.shift_y=float((K[1,2]-180)/640)
    data.clip_start=.001;data.clip_end=10.
    camera.matrix_world=Matrix((world_from_cv@np.diag([1.,-1.,-1.,1.])).tolist())
    bpy.context.view_layer.update()
    # Independent Blender projection check, separate from our camera math.
    points=np.array([[0.,0.,.06],[.01,0.,.06],[0.,.01,.06]])
    expected=project(points,K,world_from_cv)
    actual=[]
    for point in points:
        uv=world_to_camera_view(scene,camera,Vector(point.tolist()))
        actual.append([640*uv.x,360*(1-uv.y)])
    error=float(np.max(np.abs(expected-np.array(actual))))
    if error>.01:raise RuntimeError(f'Blender/camera projection mismatch {error} px')
    return error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261114)
    parser.add_argument('--device',choices=['auto','cpu'],default='auto')
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800)
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    args=parser.parse_args(argv)
    if not 1<=args.count<=20000:raise ValueError('Count outside registered dataset range')
    args.assets=args.assets.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+args.max_seconds
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    scene.cycles.samples=args.samples;scene.cycles.max_bounces=3;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.image_settings.file_format='PNG'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    device=device_probe(force_cpu=args.device=='cpu')
    write_json(args.output/'render_device.json',device)
    original,top_faces,cad_receipt=import_cad(args.assets)
    write_json(args.output/'cad_import.json',cad_receipt)
    original.hide_render=True
    instances=[]
    for index in range(5):
        obj=original.copy();obj.data=original.data.copy();bpy.context.collection.objects.link(obj)
        obj.data.materials.clear()
        obj.data.materials.append(printed_material(f'body-{index}',(.5,.05,.02)))
        obj.data.materials.append(printed_material(f'top-{index}',(.5,.05,.02),top_id=index+1))
        for face in obj.data.polygons:face.material_index=int(face.index in top_faces)
        instances.append(obj)
    bpy.ops.mesh.primitive_plane_add(size=4)
    table=bpy.context.object;table_mat=material('background',(.3,.3,.3));table.data.materials.append(table_mat)
    bpy.ops.object.light_add(type='AREA',location=(0,0,.7))
    light=bpy.context.object;light.data.shape='DISK'
    bpy.ops.object.camera_add();scene.camera=bpy.context.object
    scene.world.use_nodes=True;background=scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value=(.7,.75,.8,1.)
    # A separate non-target occluder never receives a semantic top-face ID.
    bpy.ops.mesh.primitive_cube_add(size=1)
    occluder=bpy.context.object;occluder.dimensions=(.011,.07,.008)
    occluder.data.materials.append(material('non-target-occluder',(.12,.12,.12)))
    bpy.ops.mesh.primitive_torus_add(major_radius=.022,minor_radius=.004,major_segments=48,minor_segments=12)
    negative=bpy.context.object;negative.location=(.09,-.05,.004)
    negative.data.materials.append(material('hard-negative-ring',(.06,.18,.55)))
    scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear()
    layers=tree.nodes.new('CompositorNodeRLayers')
    bpy.context.view_layer.use_pass_material_index=True
    composite=tree.nodes.new('CompositorNodeComposite');tree.links.new(layers.outputs['Image'],composite.inputs['Image'])
    mask_outputs=[]
    for index in range(5):
        mask=tree.nodes.new('CompositorNodeIDMask');mask.index=index+1;mask.use_antialiasing=False
        tree.links.new(layers.outputs['IndexMA'],mask.inputs['ID value'])
        output=tree.nodes.new('CompositorNodeOutputFile')
        output.format.file_format='PNG';output.format.color_mode='BW';output.format.color_depth='8'
        tree.links.new(mask.outputs['Alpha'],output.inputs[0]);mask_outputs.append(output)
    for frame in range(args.count):
        if time.monotonic()+30>=deadline:
            write_json(args.output/'PROGRESS.json',{'status':'BUDGET_STOP','completed_frames':frame,'requested_frames':args.count})
            raise RuntimeError('Stopped at a scene boundary before registered rendering deadline')
        config=make_config(args.seed,frame,args.qualification)
        directory=args.output/f'scene-{frame:05d}';directory.mkdir(exist_ok=False)
        scene.frame_set(frame)
        K=np.array(config['camera_intrinsics']);pose=np.array(config['camera_pose_world_from_cv'])
        projection_error=set_camera(scene,K,pose)
        light.location=config['light_position'];light.data.energy=config['light_energy'];light.data.size=config['light_size']
        background.inputs['Strength'].default_value=config['world_strength']
        table_mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*config['background_rgb'],1.)
        scene.view_settings.exposure=config['exposure_stops']
        negative.hide_render=not config['hard_negative']
        occluder.hide_render=not config['occluder']
        labels=[]
        for index,obj in enumerate(instances):
            obj.hide_render=index>=config['expected_instance_count']
            if obj.hide_render:continue
            item=config['objects'][index]
            obj.location=item['position_m'];obj.rotation_euler=(0.,0.,item['yaw_rad'])
            color=tuple(min(1.,v*item['brightness']) for v in COLORS[item['color_id']][1])
            for mat in obj.data.materials:
                shader=mat.node_tree.nodes.get('Principled BSDF')
                shader.inputs['Base Color'].default_value=(*color,1.)
                shader.inputs['Roughness'].default_value=item['roughness']
            bpy.context.view_layer.update()
            transform=np.array([list(row) for row in obj.matrix_world])
            center_world=(transform@np.array([0.,0.,.06,1.]))[:3]
            local=top_boundary([0.,0.]);world=(np.column_stack([local,np.ones(len(local))])@transform.T)[:,:3]
            labels.append({'instance_id':index+1,'color_id':item['color_id'],'color_material':item['color_material'],
                'object_pose':transform.tolist(),'top_center_world':center_world.tolist(),
                'physical_top_center_xy':project([center_world],K,pose)[0].tolist(),
                'top_boundary_xy':project(world,K,pose).tolist(),
                'top_face_visible_mask':f'visible-{index+1}.png'})
        if config['occluder']:
            occluder.location=Vector(config['objects'][0]['position_m'])+Vector((0.,0.,.068))
            occluder.rotation_euler=(0.,0.,config['objects'][0]['yaw_rad'])
        for index,output in enumerate(mask_outputs):
            output.base_path=str(directory);output.file_slots[0].path=f'mask-{index+1}-'
        scene.render.filepath=str(directory/'rgb.png')
        bpy.ops.render.render(write_still=True)
        for index in range(5):
            files=list(directory.glob(f'mask-{index+1}-*.png'))
            if len(files)!=1:raise RuntimeError('Expected one face-index mask per slot')
            files[0].rename(directory/f'visible-{index+1}.png')
        config.update(instances=labels,blender_projection_max_error_px=projection_error,
                      material_index_slots=5,blender_version=bpy.app.version_string)
        write_json(directory/'labels_raw.json',config)
        progress={'status':'RUNNING','completed_frames':frame+1,'requested_frames':args.count,
                  'elapsed_seconds':time.monotonic()-started}
        write_json(args.output/'PROGRESS.json',progress)
        print('TOPFACE_PROGRESS',json.dumps(progress),flush=True)
    device.update(gpu_render_verified=device['selected_backend']!='CPU',successful_frames=args.count,
                  blender_version=bpy.app.version_string,numpy_version=np.__version__)
    write_json(args.output/'render_device.json',device)
    progress['status']='COMPLETE';write_json(args.output/'PROGRESS.json',progress)


if __name__=='__main__':main()
