"""Render competition paths on the corrected copy of the existing .blend."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from competition_paths import schedule
from dataset import digest, write_jsonl

sys.path.insert(0, str(HERE/'vendor/roadnet'))
import blender_scene as road
import sim


def main():
    parser = argparse.ArgumentParser()
    for name in ('out', 'blend', 'cad', 'camera-renderer'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stage', choices=('smoke','pilot','full'), required=True)
    parser.add_argument('--pilot-gate', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.stage=='full':
        gate = json.loads(args.pilot_gate.read_text()) if args.pilot_gate else {}
        if gate.get('status')!='PASS' or gate.get('map_sha256')!=digest(args.blend):
            parser.error('A passed competition Pilot gate bound to the current map is required')
    input_sha = digest(args.blend)
    if input_sha!='b5bea8c6738bc7ebd8d57cdae34af99d1cf10b3fc782ea9c6c986e8802441e95':
        raise ValueError('Expected the audited corrected field scene')
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    if bpy.data.objects['2.4 m square'].get('class_id')!=0:
        raise ValueError('Field perimeter semantic class must be gray road')
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'rgb').mkdir()
    (out/'masks').mkdir()
    spec = importlib.util.spec_from_file_location('camera_xy_v2_render',args.camera_renderer.resolve())
    camera_render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_render)
    size = (160,90)
    parts = camera_render.existing_scene(road,size)
    camera_render.use_cad(args.cad,parts.blocks)
    board = camera_render.add_qr_board(road,parts)
    for scene in (parts.scene,parts.labels):
        for key in ('use_stamp','use_stamp_date','use_stamp_time','use_stamp_render_time','use_stamp_filename'):
            if hasattr(scene.render,key):
                setattr(scene.render,key,False)
    rows = list(schedule(args.stage))
    starts = {r['trajectory_id']:r for r in rows if r['frame_index']==0}
    row_groups = {}
    for row in rows:
        row_groups.setdefault(row['trajectory_id'],[]).append(row)
    records = {}
    for trajectory, row in starts.items():
        xy = row['camera_xy_target']
        item = {'group':trajectory,'variant':0,'provenance':row['seed'],
                'x':xy['camera_x_mm'],'y':xy['camera_y_mm']}
        record, appearance_seed, _ = camera_render.render_record(item,sim)
        route = [tuple(x['camera_xy_target'][k]/1000 for k in ('camera_x_mm','camera_y_mm'))
                 for x in row_groups[trajectory]]
        record['obstacle_instances'] = [o for o in record['obstacle_instances']
            if min(math.dist(o['center_xy'],p) for p in route)>.21]
        records[trajectory]=(record,appearance_seed)
    with (out/'manifest.incomplete.jsonl').open('x',encoding='utf8') as manifest:
        for i,row in enumerate(rows,1):
            original, appearance_seed = records[row['trajectory_id']]
            rec = dict(original)
            meta,target = row['metadata'],row['camera_xy_target']
            rec.update(center=[target['camera_x_mm']/1000,target['camera_y_mm']/1000,meta['camera_z_mm']/1000],
                       camera_yaw_deg=meta['camera_yaw_deg'],camera_pitch_deg=meta['camera_pitch_deg'],
                       camera_roll_deg=meta['camera_roll_deg'],diag_fov_deg=meta['diagonal_fov_deg'],
                       lighting_id=row['lighting_profile'],image_path=row['image_path'],mask_path=row['mask_path'])
            rec['turntable_phase_rad'] += rec['turntable_direction'] * math.tau * row['timestamp_s']/rec['turntable_period_s']
            camera_render.apply(road,parts,rec,size,board)
            x,y,z=parts.camera.matrix_world.translation
            if max(abs(x*1000-target['camera_x_mm']),abs(y*1000-target['camera_y_mm']))>.001:
                raise ValueError('Optical center does not match scheduled position')
            row['camera_xy_target']={'camera_x_mm':float(x)*1000,'camera_y_mm':float(y)*1000}
            meta.update(camera_to_world_blender=rec['camera_to_world_blender'],
                        intrinsics_pinhole=rec['intrinsics_pinhole'],source_appearance_seed=appearance_seed)
            row['appearance_profile_id']=f'gray-appearance-{appearance_seed}'
            row['map_variant']='official-fig1-gray-perimeter-v1'
            road.render(parts,rec,out,size)
            row.update(image_sha256=rec['image_sha256'],mask_sha256=rec['mask_sha256'])
            manifest.write(json.dumps(row,separators=(',',':'))+'\n')
            if i%25==0:
                manifest.flush()
                print(f'COMPETITION_FRAMES={i}',flush=True)
    write_jsonl(out/'manifest.jsonl',rows)
    (out/'RENDER_IDENTITY.json').write_text(json.dumps({'geometry_sha256':input_sha,'cad_sha256':digest(args.cad),
        'renderer_sha256':digest(__file__),'camera_v2_renderer_sha256':digest(args.camera_renderer),
        'roadnet_renderer_sha256':digest(road.__file__),'source_official_pdf_sha256':'1ac7fa340193591874e3ec84f9427567864c16ab39fb500db814a4d2cfc000e4',
        'blender':bpy.app.version_string,'stage':args.stage,'resolution':size,
        'labels':['camera_x_mm','camera_y_mm'],'start_policy':'both zones, opposite goal per user'} ,indent=2)+'\n')


if __name__=='__main__':
    main()
