"""Extract native LAB samples from frozen grid ROIs, avoiding moving-scene races."""
import argparse
import hashlib
import json
from pathlib import Path


def sample(grid,rois):
    result={}
    for cid,roi in rois.items():
        x,y,w,h=roi
        values=[]
        for row_index,row in enumerate(grid['lab_grid']):
            py=grid['y0']+row_index*grid['stride']
            if not y<=py<y+h:continue
            for col_index,pixel in enumerate(row):
                px=grid['x0']+col_index*grid['stride']
                if x<=px<x+w:values.append(pixel)
        if len(values)<8:raise ValueError('ROI has fewer than eight native samples')
        channels=[sorted(v[c] for v in values) for c in (0,1,2)]
        result[cid]={'roi':roi,'native_samples':len(values),
            'p05':[c[int((len(c)-1)*.05)] for c in channels],
            'median':[c[len(c)//2] for c in channels],
            'p95':[c[int((len(c)-1)*.95)] for c in channels]}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('grid',type=Path);parser.add_argument('rois',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    grid=json.loads(args.grid.read_text(encoding='utf-8'))
    labels=json.loads(args.rois.read_text(encoding='utf-8'))
    result={'source':'Frozen real CSI2 native RGB565 LAB grid','grid_sha256':hashlib.sha256(args.grid.read_bytes()).hexdigest(),
        'distance_cm':grid['distance_cm'],'camera_mount_fixed':grid['camera_mount_fixed'],
        'samples':sample(grid,labels['rois']),'label_scope':labels['scope']}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
