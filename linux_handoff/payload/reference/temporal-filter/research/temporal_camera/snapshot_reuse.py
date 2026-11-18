"""One-time verified input migration; never writes to the active source tree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    source=Path('C:/Users/28389/Desktop/k230-competition-temporal/experiments/competition_temporal_camera_localization_v1')
    data=source/'.runtime/sequences-pilot-a01'
    manifest=data/'manifest.jsonl'
    expected='ca2d839c34659fee39d676a50e27ae0c78142625d6e577642ffef13589e11840'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:raise ValueError('Source dataset changed')
    spec=importlib.util.spec_from_file_location('source_sequence_qa',source/'sequence_qa.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,qa=module.verify(data)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'rgb').mkdir()
    new=[]
    for r in rows:
        target=out/r['image'];shutil.copy2(data/r['image'],target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=r['image_sha256']:raise ValueError('Input copy differs')
        new.append({'sample_id':r['sample_id'],'trajectory_id':r['trajectory_id'],'scene_id':r['scene_id'],
            'spatial_block_id':r['spatial_block_id'],'pose_group_id':r['pose_group_id'],
            'appearance_profile_id':'original-scene-seed-'+str(r['seed']),
            'split':r['split'],'timestamp_s':r['timestamp_s'],'frame_index':r['frame_index'],
            'scenario':r['scenario'],'evaluation_partition':'trajectory_holdout',
            'camera_xy_target':r['camera_xy_target'],'image_path':r['image'],'image_sha256':r['image_sha256'],
            'metadata':r['camera_metadata'],'original_trajectory_seed':r['seed']})
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:raise ValueError('Source changed during copy')
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in new))
    (out/'SOURCE_MANIFEST.jsonl').write_bytes(manifest.read_bytes())
    (out/'SOURCE_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    (out/'RENDER_IDENTITY.json').write_text(json.dumps({'source':str(data),'source_manifest_sha256':expected,
        'source_tree_access':'READ_ONLY_OTHER_ACTIVE_OWNER','images_copied':len(rows),
        'assets_rebuilt':False,'source_assets':json.loads((data/'qa.json').read_text())['assets'],
        'target_fields':['camera_x_mm','camera_y_mm'],'source_camera_matrix_units':'mm'},indent=2)+'\n')
    print(json.dumps({'status':'PASS','frames':len(new),'original_manifest_sha256':expected,
                      'copied_rgb_bytes':sum(p.stat().st_size for p in (out/'rgb').glob('*.png'))}))


if __name__=='__main__':main()
