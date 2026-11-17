"""Render RGB and pixel IDs using the same Blender camera and geometry."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import BLOCK_COLORS, LAYOUT, MASK_COLORS, PALETTE, RINGS, STAGE, capture_records


def material(name, color, texture=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = .78
    if texture:
        coords = tree.nodes.new('ShaderNodeTexCoord')
        noise = tree.nodes.new('ShaderNodeTexNoise')
        ramp = tree.nodes.new('ShaderNodeValToRGB')
        noise.inputs['Scale'].default_value = 85
        tree.links.new(coords.outputs['Object'], noise.inputs['Vector'])
        tree.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        tree.links.new(ramp.outputs['Color'],bsdf.inputs['Base Color'])
    return mat


def identify(obj, idx, mat):
    obj.color = (*MASK_COLORS[idx],1) if idx != 255 else (0,0,0,1)
    obj['class_id'] = idx
    obj.data.materials.append(mat)
    return obj


def mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces)
    obj = bpy.data.objects.new(name,mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def rect(name, center, size, z):
    x,y = center
    dx,dy = (value/2 for value in size)
    return mesh_object(name,[(x-dx,y-dy,z),(x+dx,y-dy,z),
                             (x+dx,y+dy,z),(x-dx,y+dy,z)],[(0,1,2,3)])


def cylinder(name, radius, depth, top=None):
    bpy.ops.mesh.primitive_cone_add(vertices=48,radius1=radius,
                                   radius2=radius if top is None else top,depth=depth)
    obj = bpy.context.object
    obj.name = name
    return obj


def make_rings(name, centers):
    vertices, faces = [], []
    n = 64
    for x,y in centers:
        for diameter in RINGS:
            start = len(vertices)
            for r in (diameter/2,diameter/2-.0015):
                vertices.extend((x+r*math.cos(i*math.tau/n),y+r*math.sin(i*math.tau/n),.0027) for i in range(n))
            faces.extend((start+i,start+(i+1)%n,start+n+(i+1)%n,start+n+i) for i in range(n))
    return mesh_object(name,vertices,faces)


def configure(scene, size, engine):
    scene.render.engine = engine
    scene.render.resolution_x,scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    if engine == 'BLENDER_WORKBENCH':
        shading = scene.display.shading
        shading.light = 'FLAT'
        shading.color_type = 'OBJECT'
        shading.background_type = 'WORLD'
        shading.show_shadows = False
        shading.show_cavity = False
        shading.show_object_outline = False
        shading.show_specular_highlight = False
        scene.display.render_aa = 'OFF'
    else:
        scene.eevee.taa_render_samples = 8
        scene.view_settings.exposure = -3.5


def build(size):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new('Ambient')
    scene.world.use_nodes = True
    scene.world.color = (0,0,0)
    mats = [material(f'Class {i}',color,i<4) for i,color in enumerate(PALETTE)]
    stock = [material(f'Stock {i+1}',color) for i,color in enumerate(BLOCK_COLORS)]
    def plane(name, center, dimensions, z, idx):
        return identify(rect(name,center,dimensions,z),idx,mats[idx])
    width,height = (LAYOUT['field'][key] for key in ('width','height'))
    road = LAYOUT['road']['arm_width']
    identify(rect('Outside',(0,0),(12,12),-.001),255,material('Outside',(.17,.16,.15)))
    plane('2.4 m square',(0,0),(width,height),0,2)
    plane('Longitudinal road',(0,0),(road,height),.001,0)
    plane('Transverse road',(0,0),(width,road),.0011,0)
    for key,idx in (('yellow_zones',1),('blue_start_zones',3)):
        zone = LAYOUT[key]
        for i,center in enumerate(zone['centers']):
            plane(f'{key} {i}',center,zone['size'],.002,idx)
    for key in ('temporary_storage_zone','rough_processing_zone'):
        zone = LAYOUT[key]
        x,y = zone['center']
        plane(key,(x,y),zone['size'],.0024,2)
        centers = [(x,y+d) if key.startswith('temporary') else (x+d,y) for d in (-.15,0,.15)]
        identify(make_rings(key+' rings',centers),6,mats[6])
    table = identify(cylinder('Turntable',.15,1),6,mats[6])
    obstacles = [identify(cylinder(f'Obstacle {i}',.025,.10),4,mats[4]) for i in range(4)]
    blocks = [[identify(cylinder(f'Block {i} / {j}',r,d,top),5,stock[0])
               for j,(r,d,top) in enumerate(((.025,.008,.025),(.025,.044,.015),(.015,.008,.015)))]
              for i in range(4)]
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36
    camera.data.clip_start = .02
    scene.camera = camera
    bpy.ops.object.light_add(type='AREA')
    key = bpy.context.object
    key.data.shape = 'DISK'
    bpy.ops.object.light_add(type='AREA',location=(0,0,2.5))
    fill = bpy.context.object
    fill.data.size = 4
    labels = bpy.data.scenes.new('Semantic object IDs')
    for collection in scene.collection.children:
        labels.collection.children.link(collection)
    labels.camera = camera
    labels.world = scene.world
    configure(scene,size,'BLENDER_EEVEE')
    configure(labels,size,'BLENDER_WORKBENCH')
    return scene, labels, camera, mats, stock, table, obstacles, blocks, key, fill


def apply_record(parts, rec, size):
    scene,labels,camera,mats,stock,table,obstacles,blocks,key,fill = parts
    w,h = size
    f = math.hypot(w,h)/(2*math.tan(math.radians(rec['diag_fov_deg'])/2))
    camera.data.type = 'PERSP'
    camera.data.lens = 36*f/w
    camera.location = rec['center']
    camera.rotation_euler = tuple(math.radians(rec[f'camera_{axis}_deg'])
                                  for axis in ('pitch','roll','yaw'))
    bpy.context.view_layer.update()
    rec['camera_height_m'] = rec['center'][2]
    rec['intrinsics_pinhole'] = {'fx':f,'fy':f,'cx':(w-1)/2,'cy':(h-1)/2,'width':w,'height':h}
    rec['camera_to_world_blender'] = [list(row) for row in camera.matrix_world]
    light = rec['lighting_id']
    for mat,color in zip(mats,rec['palette_linear']):
        nodes = mat.node_tree.nodes
        bsdf = nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = (*color,1)
        ramp = nodes.get('Color Ramp')
        if ramp:
            for element,gain in zip(ramp.color_ramp.elements,(.84,1.13)):
                element.color = (*(min(.99,c*gain) for c in color),1)
            bsdf.inputs['Roughness'].default_value = .27 if light == 'reflection' else .78
    power,fill_power,size,temperature = {
        'nominal': (410,165,2.2,(1,.96,.91)),
        'low_light': (110,76,1.6,(.91,.95,1)),
        'shadow': (310,48,.24,(1,.94,.87)),
        'reflection': (620,140,.65,(.94,.97,1)),
    }[light]
    ambient = scene.world.node_tree.nodes['Background']
    ambient.inputs[0].default_value = (*temperature,1)
    ambient.inputs[1].default_value = .28
    key.location = (*rec['lamp_xy'],1.85)
    key.data.energy,key.data.size,key.data.color = power,size,temperature
    fill.data.energy = fill_power

    tx,ty = LAYOUT['raw_material_turntable']['center']
    dz = rec['turntable_height_m']
    table.location = (tx,ty,dz/2)
    table.scale.z = dz
    for i,obj in enumerate(obstacles):
        instances = rec['obstacle_instances']
        obj.hide_render = i >= len(instances)
        if not obj.hide_render:
            obj.location = (*instances[i]['center_xy'],.05)
    blocks_visible = []
    for i in range(3):
        phi = rec['turntable_phase_rad'] + math.tau*i/3
        blocks_visible.append((tx+.09*math.cos(phi),ty+.09*math.sin(phi),dz,rec['turntable_color_ids'][i]))
    station = rec['station_item_color']
    if station is not None:
        name = 'temporary_storage_zone' if rec['stratum'] == 'storage' else 'rough_processing_zone'
        sx,sy = LAYOUT[name]['center']
        blocks_visible.append((sx,sy,.003,station))
    for i,parts in enumerate(blocks):
        for obj,offset in zip(parts,(.004,.030,.056)):
            obj.hide_render = i >= len(blocks_visible)
            if not obj.hide_render:
                x,y,z,c = blocks_visible[i]
                obj.location = (x,y,z+offset)
                obj.data.materials[0] = stock[c]
    rec['material_instances'] = [{'base_xyz':[x,y,z],'color_id':c+1,'class_id':5} for x,y,z,c in blocks_visible]


def png_gray(path, mask):
    h,w = mask.shape
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    signature = b'\x89PNG\r\n\x1a\n'
    hdr = struct.pack('>IIBBBBB',w,h,8,0,0,0,0)
    scanlines = np.pad(mask,((0,0),(1,0))).tobytes()
    path.write_bytes(signature+chunk(b'IHDR',hdr)+chunk(b'IDAT',zlib.compress(scanlines,3))+chunk(b'IEND',b''))


def render(parts, rec, out, size):
    scene,labels = parts[:2]
    scene.render.filepath = str(out/rec['image_path'])
    bpy.ops.render.render(write_still=True,scene=scene.name)
    code_path = out/'object_id_rgb.png'
    labels.render.filepath = str(code_path)
    bpy.ops.render.render(write_still=True,scene=labels.name)
    ids = bpy.data.images.load(str(code_path),check_existing=False)
    w,h = size
    values = np.empty(w*h*4,np.float32)
    ids.pixels.foreach_get(values)
    bpy.data.images.remove(ids)
    code_path.unlink()
    rgb = values.reshape(h,w,4)[::-1,:,:3]
    if np.any((rgb>.01)&(rgb<.99)):
        raise ValueError('Unexpected interpolated or color-managed object ID')
    codes = (rgb>.5).astype(np.uint8) @ np.array([4,2,1],np.uint8)
    lut = np.full(8,255,np.uint8)
    for i,c in enumerate(MASK_COLORS):
        lut[sum(int(v)*n for v,n in zip(c,(4,2,1)))] = i
    png_gray(out/rec['mask_path'],lut[codes])
    for key in ('image','mask'):
        rec[key+'_sha256'] = hashlib.sha256((out/rec[key+'_path']).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=STAGE,default='smoke')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--width',type=int,default=320)
    parser.add_argument('--overview',action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.width < 160 or args.width%16:
        parser.error('--width must be at least 160 and divisible by 16')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    out = args.out.resolve()
    incomplete = out/'manifest.incomplete'
    completed = out/'manifest.jsonl'
    scheduled = list(capture_records(args.stage))
    recorded = []
    if args.resume:
        manifest = completed if completed.exists() else incomplete
        recorded = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
        if len(recorded) > len(scheduled):
            raise ValueError('Recorded rows exceed the frozen schedule')
        for row, expected in zip(recorded, scheduled):
            expected = json.loads(json.dumps(expected))
            if any(row.get(k) != v for k,v in expected.items()):
                raise ValueError(f"Resume schedule drift at {row['id']}")
            k = row['intrinsics_pinhole']
            if (k['width'], k['height']) != (args.width,args.width*9//16):
                raise ValueError('Resume resolution differs from prior captures')
            for prefix in ('image','mask'):
                digest = hashlib.sha256((out/row[prefix+'_path']).read_bytes()).hexdigest()
                if digest != row[prefix+'_sha256']:
                    raise ValueError(f"Resume file drift at {row['id']}")
        if completed.exists():
            if len(recorded) != len(scheduled):
                raise ValueError('Completed manifest contains only a subset')
            print(f'ROADNET_ALREADY_COMPLETE={len(recorded)}',flush=True)
            return
    else:
        out.mkdir(parents=True)
        (out/'rgb').mkdir()
        (out/'masks').mkdir()
    start = len(recorded)
    stop = min(len(scheduled), start+args.limit) if args.limit is not None else len(scheduled)
    for rec in scheduled[start:stop]:
        if any((out/rec[prefix+'_path']).exists() for prefix in ('image','mask')):
            raise ValueError(f"Unrecorded output exists for {rec['id']}")
    size = (args.width,args.width*9//16)
    parts = build(size)
    print(f'ROADNET_BEGIN Blender {bpy.app.version_string} size={size}',flush=True)
    with incomplete.open('a' if args.resume else 'x',encoding='utf8') as stream:
        for i in range(start,stop):
            rec = scheduled[i]
            apply_record(parts,rec,size)
            render(parts,rec,out,size)
            stream.write(json.dumps(rec,separators=(',',':'))+'\n')
            stream.flush()
            if i==0 or (i+1)%25==0:
                print(f"ROADNET_COMPLETED={i+1} {rec['id']}",flush=True)
    finished = stop == len(scheduled)
    if finished:
        incomplete.rename(completed)
    if args.overview and finished:
        scene,labels,camera = parts[:3]
        rec = next(r for r in capture_records('smoke') if r['stratum']=='intersection')
        apply_record(parts,rec,size)
        camera.location,camera.rotation_euler = (0,0,4),(0,0,0)
        camera.data.type,camera.data.ortho_scale = 'ORTHO',3.0
        for item in (scene,labels):
            item.render.resolution_x = item.render.resolution_y = 800
        render(parts,{'image_path':'field.png','mask_path':'field_mask.png'},out,(800,800))
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'field.blend'))
    print(f'ROADNET_BATCH_DONE={stop}/{len(scheduled)} stage={args.stage} complete={finished}',flush=True)


if __name__ == '__main__':
    main()

