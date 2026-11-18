"""Correct only existing field material/class using official Figure1.

The source .blend is immutable. Geometry signatures must remain identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import bpy
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/roadnet'))
import blender_scene as road


def geometry():
    result={}
    for obj in bpy.data.objects:
        if obj.type!='MESH':continue
        h=hashlib.sha256()
        for v in obj.data.vertices:h.update(struct.pack('<3d',*v.co))
        for p in obj.data.polygons:h.update(struct.pack('<'+'I'*len(p.vertices),*p.vertices))
        h.update(np.asarray(obj.matrix_world,dtype=np.float64).tobytes())
        result[obj.name]={'vertices':len(obj.data.vertices),'polygons':len(obj.data.polygons),'geometry_sha256':h.hexdigest()}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.source.resolve();out=args.out.resolve()
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!='9a289ad14541b986fdc2d2cb7a213412c407b1becc5770338e76f984a3b41996':raise ValueError('Expected audited RoadNet .blend')
    out.mkdir(parents=True,exist_ok=False);bpy.ops.wm.open_mainfile(filepath=str(source))
    before=geometry();field=bpy.data.objects['2.4 m square'];old=field.data.materials[0].name
    field.data.materials[0]=bpy.data.materials['Class 0']
    field['class_id']=0;field.color=(1.,0.,0.,1.)
    after=geometry()
    if before!=after:raise ValueError('Geometry changed during material-only repair')
    target=out/'field_gray_v1.blend';bpy.ops.wm.save_as_mainfile(filepath=str(target))
    # Render a static orthographic reference from the corrected saved scene.
    scene=bpy.data.scenes['Scene'];labels=bpy.data.scenes['Semantic object IDs'];camera=scene.camera
    camera.location=(0,0,4);camera.rotation_euler=(0,0,0);camera.data.type='ORTHO';camera.data.ortho_scale=3.
    for obj in bpy.data.objects:
        if obj.name.startswith(('Obstacle ','Block ','Turntable')):obj.hide_render=True
    road.configure(scene,(800,800),'BLENDER_WORKBENCH');scene.display.shading.color_type='MATERIAL'
    road.configure(labels,(800,800),'BLENDER_WORKBENCH')
    scene.view_settings.exposure=0.
    labels.view_settings.exposure=0.
    for s in (scene,labels):
        for key in ('use_stamp','use_stamp_date','use_stamp_time','use_stamp_render_time','use_stamp_filename'):
            if hasattr(s.render,key):setattr(s.render,key,False)
    road.render((scene,labels),{'image_path':'field_preview.png','mask_path':'field_mask.png'},out,(800,800))
    report={'status':'MATERIAL_AND_SEMANTIC_CLASS_CORRECTED_GEOMETRY_IDENTICAL',
        'authority':'Official2027 PDF Figure1 and user confirmation on2026-11-17',
        'source_blend_sha256':source_sha,'corrected_blend_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_sha,
        'changed_object':'2.4 m square','old_material':old,'new_material':'Class 0','new_class_id':0,
        'geometry_before':before,'geometry_after':after,'mesh_count':len(before),
        'preserved':'field bounds, zones, stations, CAD meshes and all mesh transforms',
        'meaning':'all non-zone field base including perimeter is gray road; station strips remain white'}
    (out/'CORRECTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('geometry_before','geometry_after')}),flush=True)


if __name__=='__main__':main()
