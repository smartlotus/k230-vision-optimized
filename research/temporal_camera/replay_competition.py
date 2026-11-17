"""Read-only paired replay of the frozen CameraXY model on corrected trajectories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from dataset import audit, digest
from localization_runtime import CameraXYTracker
from map_models import MapCameraXYNet
from train_map_temporal import pixels, stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows=[json.loads(line) for line in (args.data/'manifest.jsonl').read_text().splitlines()]
    audit(rows,args.data)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    model=MapCameraXYNet(parent).eval()
    centers=np.array(parent['centers_mm'])
    torch.set_num_threads(2)
    selected=[row for row in rows if row['split']=='val']
    evaluated=defaultdict(list)
    tracks={}
    zone_correct={'two_zone':0,'global':0}
    zone_count=0
    with torch.inference_mode():
        for i in range(0,len(selected),16):
            chunk=selected[i:i+16]
            inputs=torch.from_numpy(np.stack([pixels(args.data/row['image_path']) for row in chunk]))
            for row,logits in zip(chunk,model(inputs).numpy()):
                truth=np.array([row['camera_xy_target'][key] for key in ('camera_x_mm','camera_y_mm')])
                evaluated['single_frame'].append(float(np.linalg.norm(centers[logits.argmax()]-truth)))
                if row['frame_index']==0:
                    zone_count+=1
                for name,startup in (('two_zone','competition'),('global','global')):
                    key=(name,row['trajectory_id'])
                    if key not in tracks:
                        tracks[key]=CameraXYTracker(centers,temperature=1.,startup=startup)
                    result=tracks[key].update(logits.tolist(),row['timestamp_s'])
                    xy=np.array([result['candidate_x_mm'],result['candidate_y_mm']])
                    evaluated[name].append(float(np.linalg.norm(xy-truth)))
                    if row['frame_index']==0 and xy[1]*truth[1]>0:
                        zone_correct[name]+=1
    report={'status':'HOST_FROZEN_MODEL_REPLAY','split':'val','optimization_updates':0,
            'map_parent_sha256':digest(args.parent),'dataset_manifest_sha256':digest(args.data/'manifest.jsonl'),
            'results':{k:stats(values) for k,values in evaluated.items()},
            'initial_zone_correct':zone_correct,'initial_zone_trials':zone_count,
            'scope':'diagnostic candidate errors before Kaggle adaptation; no real confidence calibration'}
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
