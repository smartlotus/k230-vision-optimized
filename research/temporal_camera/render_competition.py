"""Render competition paths on the corrected copy of the existing .blend."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from competition_paths import schedule
from dataset import digest, write_jsonl
from camera_preprocess import ai2d_downsample_4x

sys.path.insert(0, str(HERE/'vendor/roadnet'))
import blender_scene as road
import sim


def compact_sensor_frame(out, record, metadata, size):
    rgb_path = out / record['image_path']
    raw = bpy.data.images.load(str(rgb_path), check_existing=False)
    raw.colorspace_settings.name = 'Non-Color'
    values = np.empty(size[0] * size[1] * 4, np.float32)
    raw.pixels.foreach_get(values)
    bpy.data.images.remove(raw)
    rgb = np.rint(values.reshape(size[1],size[0],4)[::-1,:,:3] * 255).astype(np.uint8)
    model_input = ai2d_downsample_4x(rgb.transpose(2,0,1))
    road.png_bytes(rgb_path, model_input[:,3:93,:].transpose(1,2,0), 2)
    record['image_sha256'] = digest(rgb_path)
    intrinsics = dict(metadata['intrinsics_pinhole'])
    intrinsics.update(width=160,height=90,fx=intrinsics['fx']/4,fy=intrinsics['fy']/4,
                      cx=(intrinsics['cx']+.5)/4-.5,cy=(intrinsics['cy']+.5)/4-.5)
    metadata['rgb_intrinsics_pinhole'] = intrinsics


def main():
    parser = argparse.ArgumentParser()
    for name in ('out', 'blend', 'cad', 'camera-renderer'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stage', choices=('smoke','pilot','full'), required=True)
    parser.add_argument('--pilot-gate', type=Path)
    parser.add_argument('--sensor-resolution', action='store_true', help='Render 640×360 for the 4:1 CanMV AI2D path')
    parser.add_argument('--compact-sensor', action='store_true',
                        help='Save the exact sensor-matched model RGB at 160×90; keep full-resolution masks')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.compact_sensor and not args.sensor_resolution:
        parser.error('--compact-sensor requires --sensor-resolution')
    size = (640,360) if args.sensor_resolution else (160,90)
    rgb_size = (160,90) if args.compact_sensor else size
    if args.stage=='full':
        gate = json.loads(args.pilot_gate.read_text()) if args.pilot_gate else {}
        if (gate.get('status')!='PASS' or gate.get('map_sha256')!=digest(args.blend)
                or gate.get('resolution')!=list(size)):
            parser.error('A passed competition Pilot gate for the current map and image resolution is required')
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
            if args.compact_sensor:
                compact_sensor_frame(out,rec,meta,size)
            row.update(image_sha256=rec['image_sha256'],mask_sha256=rec['mask_sha256'])
            manifest.write(json.dumps(row,separators=(',',':'))+'\n')
            if i%25==0:
                manifest.flush()
                print(f'COMPETITION_FRAMES={i}',flush=True)
    write_jsonl(out/'manifest.jsonl',rows)
    (out/'RENDER_IDENTITY.json').write_text(json.dumps({'geometry_sha256':input_sha,'cad_sha256':digest(args.cad),
        'renderer_sha256':digest(__file__),'camera_v2_renderer_sha256':digest(args.camera_renderer),
        'roadnet_renderer_sha256':digest(road.__file__),'source_official_pdf_sha256':'1ac7fa340193591874e3ec84f9427567864c16ab39fb500db814a4d2cfc000e4',
        'blender':bpy.app.version_string,'stage':args.stage,'resolution':size,'rgb_storage_resolution':rgb_size,
        'rgb_storage':'AI2D_HALF_PIXEL_4X_UNPADDED' if args.compact_sensor else 'RAW_RENDER',
        'labels':['camera_x_mm','camera_y_mm'],'start_policy':'both zones, opposite goal per user'} ,indent=2)+'\n')


if __name__=='__main__':
    main()
