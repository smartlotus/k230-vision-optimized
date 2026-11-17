"""Reuse frozen RoadNet Blender scene; labels from actual camera.matrix_world."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

import bpy

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from dataset import schedule, write_jsonl
sys.path.insert(0,str(HERE/'vendor/roadnet'))
from blender_scene import build, apply_record
from sim import capture_records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('smoke','pilot'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'rgb').mkdir()
    size=(160,90);parts=build(size);scene=parts[0];camera=parts[2]
    scene.eevee.taa_render_samples=4
    for key in ('use_stamp','use_stamp_date','use_stamp_time','use_stamp_render_time','use_stamp_filename'):
        if hasattr(scene.render,key):setattr(scene.render,key,False)
    templates=list(capture_records('pilot'))
    rows=[]
    with (out/'manifest.incomplete.jsonl').open('x',encoding='utf8') as f:
        for row in schedule(args.stage):
            trajectory_number=int(row['trajectory_id'].rsplit('tr',1)[1])
            group,scenario_index=divmod(trajectory_number,10)
            source_strata=list(range(10))
            random.Random(20261117+group).shuffle(source_strata)
            # Every trajectory group contains one source from each original
            # RoadNet stratum. Source domain must not be an alias for split.
            template_index=source_strata[scenario_index]*100+(group%10)*10
            rec=json.loads(json.dumps(templates[template_index]))
            row['appearance_profile_id']=f"roadnet-scene-{rec['scene_seed']}-{row['lighting_profile']}"
            row['metadata']['source_appearance_scene_seed']=rec['scene_seed']
            row['metadata']['source_stratum']=rec['stratum']
            meta=row['metadata'];target=row['camera_xy_target']
            rec.update(center=[target['camera_x_mm']/1000,target['camera_y_mm']/1000,meta['camera_z_mm']/1000],
                       camera_yaw_deg=meta['camera_yaw_deg'],camera_pitch_deg=0,camera_roll_deg=0,
                       diag_fov_deg=96,image_path=row['image_path'])
            rec['lighting_id']=row['lighting_profile']
            apply_record(parts,rec,size)
            bpy.context.view_layer.update()
            xyz=camera.matrix_world.translation
            row['camera_xy_target']={'camera_x_mm':float(xyz.x)*1000,'camera_y_mm':float(xyz.y)*1000}
            meta['camera_to_world_blender']=[list(v) for v in camera.matrix_world]
            meta['intrinsics_pinhole']=rec['intrinsics_pinhole']
            # Appearance/geometry remain coherent for an entire trajectory.
            scene.render.filepath=str(out/row['image_path'])
            bpy.ops.render.render(write_still=True,scene=scene.name)
            row['image_sha256']=hashlib.sha256((out/row['image_path']).read_bytes()).hexdigest()
            rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
            if len(rows)%25==0:print('TEMPORAL_FRAMES='+str(len(rows)),flush=True)
    write_jsonl(out/'manifest.jsonl',rows)
    (out/'RENDER_IDENTITY.json').write_text(json.dumps({'blender':bpy.app.version_string,
        'scene_source_sha256':hashlib.sha256((HERE/'vendor/roadnet/blender_scene.py').read_bytes()).hexdigest(),
        'map_sha256':hashlib.sha256((HERE/'vendor/roadnet/configs/field_layout.json').read_bytes()).hexdigest(),
        'labels':'camera.matrix_world.translation x,y *1000 ONLY','resolution':size,
        'native_camera_validation':'NOT RUN'},indent=2)+'\n')


if __name__=='__main__':main()
