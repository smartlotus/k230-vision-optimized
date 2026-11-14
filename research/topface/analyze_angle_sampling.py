"""Measure train-only raster sensitivity of nearly circular top-face angles."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np

from cloud_store import atomic_json,sha
from dataset_diagnostics import stats
from geometry import ellipse_from_points


def analyze(dataset,output):
    if output.exists():
        raise FileExistsError('Preserve the existing angle sampling report')
    index=json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    if index['status']!='PASS':
        raise ValueError('Expected a verified simulation dataset')
    groups=defaultdict(list)
    cases=[]
    shifts=(0.,.25,.5,.75)
    for row in index['scenes']:
        if row['split']!='train':
            continue
        path=dataset/row['scene']/'labels.json'
        if sha(path)!=row['label_sha256']:
            raise ValueError('Train label bytes differ from the original audit')
        label=json.loads(path.read_text(encoding='utf8'))
        for item in label['instances']:
            if not item['geometry_valid'] or not item['orientation_valid']:
                continue
            ellipse=item['ellipse_full']
            major=ellipse['semi_major']*.5
            minor=ellipse['semi_minor']*.5
            theta=math.radians(ellipse['orientation_deg'])
            ratio=minor/major
            width=int(math.ceil(2*major+8))
            errors=[]
            yy,xx=np.mgrid[0:width,0:width]
            for dx in shifts:
                for dy in shifts:
                    x=xx-width/2+dx
                    y=yy-width/2+dy
                    u=x*math.cos(theta)+y*math.sin(theta)
                    v=-x*math.sin(theta)+y*math.cos(theta)
                    pixels=((u/major)**2+(v/minor)**2<=1).astype(np.uint8)
                    contours,_=cv2.findContours(pixels,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                    if len(contours)!=1 or len(contours[0])<5:
                        raise ValueError('Ellipse raster is insufficient for the sampling diagnostic')
                    fitted=ellipse_from_points(contours[0].reshape(-1,2))
                    errors.append(abs((fitted['orientation_deg']-ellipse['orientation_deg']+90)%180-90))
            band='lt95' if ratio<.95 else '95to97'
            value={'scene':row['scene'],'instance_id':item['instance_id'],'height_mm':label['camera_height_above_top_mm'],
                   'axis_ratio':ratio,'semimajor_model_px':major,'axis_gap_model_px':major-minor,
                   'angle_shift_errors_deg':errors,'worst_shift_angle_error_deg':max(errors)}
            cases.append(value)
            groups[band].append(value)
    report={'schema':'topface-train-angle-sampling-v1.1','scope':'TRAIN_ONLY_BINARY_RASTER_DIAGNOSTIC',
            'status':'MEASURED','source_audit_sha256':sha(dataset/'AUDIT_V11.json'),'model_input_wh':[320,192],
            'subpixel_offsets_model_px':list(shifts),'cases':cases,
            'groups':{key:{'instances':len(rows),'axis_gap_model_px':stats([r['axis_gap_model_px'] for r in rows]),
                          'worst_offset_angle_error_deg':stats([r['worst_shift_angle_error_deg'] for r in rows])}
                      for key,rows in groups.items()},
            'interpretation':'Fit the exact continuous ellipse after binary sampling at sixteen subpixel offsets. This isolates raster sensitivity. It does not estimate the learned model error or the calibrated camera uncertainty.',
            'original_labels_modified':False}
    atomic_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    result=analyze(**vars(parser.parse_args()))
    print(json.dumps(result['groups'],indent=2))
