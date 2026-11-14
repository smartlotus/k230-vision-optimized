"""Reproduce the old four-update pilot validation without consuming test."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'k230_transport_mvp'
sys.path.insert(0,str(PROJECT))
from tools.material_dataset import verified_rgb,review_gate
from tools.material_preprocess import model_uint8,float_reference
from cloud_store import atomic_json,sha

MODEL_SHA='df9c639010e342ff54d5189678c096b9f10e7d3337734b5ad4b081e3bf0e3fdb'
VAL_IDS={'48b365d9c5ff60085a0f835f974986f91321e91a752b5593bca9a2d57f662b35',
         '86d28d9912c476108454dad00cfaefb1bc4926d295188153fa58d8d034ed88dd'}


def main(output):
    output=Path(output)
    if output.exists():raise ValueError('Immutable validation report required')
    manifest_path=PROJECT/'data/ai_pilot/material_manifest_reviewed_r4_20261110.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    path=ROOT/'.ops/local/frozen-model-20261112-p1/onnx-source/material.onnx'
    if sha(path)!=MODEL_SHA:raise ValueError('Frozen old-model identity changed')
    records=sorted((r for r in manifest['images'] if r.get('split')=='val'),key=lambda r:r['image_id'])
    if {r['image_id'] for r in records}!=VAL_IDS:raise ValueError('Frozen val membership changed')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    rows=[]
    for record in records:
        reasons,objects=review_gate(record)
        if reasons:raise ValueError('Validation approval gate no longer passes')
        image=verified_rgb(PROJECT,record)
        tensor,_=model_uint8(image,224)
        raw=session.run(None,{'images':float_reference(tensor)})[0]
        if raw.shape!=(1,5,1029) or not np.isfinite(raw).all() or np.any(raw[:,4,:]<0) or np.any(raw[:,4,:]>1):
            raise ValueError('Old declared decoded-box/probability contract failed')
        score=float(raw[0,4,:].max())
        # This frozen result needs no NMS/matching: no raw score qualifies.
        # If a different result is observed, require complete decoder evaluation
        # rather than return fabricated zeros or silently tune the threshold.
        if score>=.25:raise ValueError('Nonzero proposals now qualify; use complete frozen decoder/matching')
        rows.append({'image_id':record['image_id'],'gt':len(objects),'max_score':score,
            'predictions':0,'tp':0,'fp':0,'fn':len(objects),'color_scope':'red-bench only'})
    result={'schema':'topface-legacy-pilot-val-retest-v1.1','scope':'HOST_ONNX_VAL_ONLY',
        'model':'historicalYOLOv8n visible_top,1epoch4optimizerupdates','model_sha256':MODEL_SHA,
        'manifest_sha256':sha(manifest_path),'input':'verifiedRGB565->RGB224x126 pad49/49value114 NCHW /255once',
        'confidence_threshold':.25,'nms_iou':.45,'matching_iou':.5,
        'images':2,'gt':sum(r['gt'] for r in rows),'predictions':0,'tp':0,'fp':0,'fn':sum(r['fn'] for r in rows),
        'recall':0.,'precision':None,'precision_reason':'undefined:no predictions',
        'rows':rows,'test_pixels_read':False,'optimizer_updates_this_probe':0,'board_run':False,
        'new_topfacenet_result':False,'color_or_ellipse_accuracy_verified':False,
        'limits':'Two red-bench val images do not establish overall accuracy. Existing old test was already used historically; no new test tuning.',
        'ort':ort.__version__,'numpy':np.__version__}
    atomic_json(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
