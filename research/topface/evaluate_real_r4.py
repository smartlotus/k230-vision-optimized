"""Evaluate real detection and color on the approved historical CSI2 test."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from cloud_store import atomic_json, sha
from export_nanodet_candidate import session
from task_geometry import box_iou, decode, preprocess_bgr


PROJECT = Path(__file__).resolve().parents[2]/'k230_transport_mvp'
MANIFEST_SHA = '6d21a2e9d0a251c1161ee3b63345909d64ad1f8cda6d3f9be0e0c0454843e198'


def evaluate_real(export_dir, output, phase='formal_historical_test',backend='onnx'):
    export_dir,output=export_dir.resolve(),output.resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing r4 evaluation result')
    if backend=='onnx':
        return _evaluate_real(export_dir,output,phase)
    if backend!='simulator':
        raise ValueError('Explicit frozen ONNX or KModel Simulator backend required')
    from nncase_simulator import KModelRuntime
    release=json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    identity=json.loads((export_dir/'identity.json').read_text(encoding='utf8'))
    if identity['onnx_sha256']!=release['onnx_sha256']:
        raise ValueError('Floating and quantized real evaluation refer to different selected weights')
    work=PROJECT.parent/'.ops/local'/('topface-real-simulator-'+uuid.uuid4().hex)
    with KModelRuntime(export_dir/'topface.kmodel',export_dir/'identity.json',work) as simulator:
        return _evaluate_real(export_dir,output,phase,simulator)


def _evaluate_real(export_dir, output, phase, simulator=None):
    if output.exists():
        raise FileExistsError('Preserve the existing r4 test result')
    sys.path.insert(0, str(PROJECT))
    from tools.material_dataset import review_gate, verified_rgb

    path = PROJECT/'data/ai_pilot/material_manifest_reviewed_r4_20261110.json'
    model = export_dir/'topface.onnx'
    release = json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    allowed = ('formal_full',) if phase=='formal_historical_test' else ('pilot_pretrained','pilot_random_control','pilot_geometry_refine')
    if phase not in ('formal_historical_test','pilot_validation'):
        raise ValueError('Explicit real evaluation phase required')
    if release['experiment'] not in allowed or release['status'] != 'PYTORCH_ONNX_PARITY_PASS':
        raise ValueError('Real test requires a frozen formal ONNX artifact')
    if sha(path) != MANIFEST_SHA or sha(model) != release['onnx_sha256']:
        raise ValueError('Exact model and approved r4 manifest are required')
    runtime = session(model) if simulator is None else None
    width, height = release['width'], release['height']
    records = json.loads(path.read_text(encoding='utf8'))['images']
    rows = []
    counts = Counter()
    colors = {color:Counter() for color in range(1,7)}
    for record in records:
        if record.get('split')!=('test' if phase=='formal_historical_test' else 'val'):
            continue
        reasons, objects = review_gate(record)
        if reasons:
            continue
        truth = []
        for item in objects:
            x,y,w,h = item['bbox_xywh']
            truth.append([x,y,x+w,y+h])
            if item.get('color_id') is not None:
                colors[item['color_id']]['gt'] += 1
        rgb = verified_rgb(PROJECT, record)
        bgr=np.ascontiguousarray(rgb[:,:,::-1])
        if simulator is None:
            tensor, _ = preprocess_bgr(bgr, width, height)
            raw = runtime.run(None, {runtime.get_inputs()[0].name:tensor[None]})[0][0]
        else:
            if (simulator.width,simulator.height)!=(width,height):
                raise ValueError('Simulator real input geometry differs')
            raw=simulator.raw_from_bgr(bgr)
        decoded = decode(raw, width, height, score_threshold=.25, nms_iou=.5)
        unused = set(range(len(truth)))
        matched = []
        for detection in decoded['detections']:
            color = detection['color_id']
            colors[color]['predictions'] += 1
            indices = sorted(unused)
            if not indices:
                continue
            overlaps = box_iou(np.asarray(detection['bbox_xyxy_visible']),np.asarray([truth[i] for i in indices]))
            best = int(np.argmax(overlaps))
            if overlaps[best]<.5:
                continue
            index = indices[best]
            unused.remove(index)
            expected = objects[index].get('color_id')
            if expected is not None:
                counts['color_labeled_matches'] += 1
                counts['correct_color'] += int(color==expected)
                colors[expected]['correct'] += int(color==expected)
            matched.append({'bbox_iou':float(overlaps[best]),'approved_color':expected,'predicted_color':color})
        detections = decoded['detections']
        counts.update(images=1, gt=len(truth), predictions=len(detections), tp=len(matched),
                      fp=len(detections)-len(matched), fn=len(unused))
        rows.append({'image_id':record['image_id'],'scene_group':record['scene_group_id'],
                     'gt':len(truth),'predictions':len(detections),'matched':matched,'detections':detections})
    if not rows:
        raise ValueError('No approved real test frames')
    result = {'status':'HISTORICAL_REAL_TEST_EVALUATED' if phase=='formal_historical_test' else 'PILOT_REAL_VALIDATION_EVALUATED',
              'phase':phase,'evaluation_split':'test' if phase=='formal_historical_test' else 'val',
              'backend':'ONNX_RUNTIME' if simulator is None else 'NNCASE_SIMULATOR',
              'model_sha256':release['onnx_sha256'] if simulator is None else simulator.identity['kmodel_sha256'],
              'reference_onnx_sha256':release['onnx_sha256'],
              'manifest_sha256':MANIFEST_SHA,'camera_id':2,'score_threshold':.25,'matching_iou':.5,
              'denominators':dict(counts),'recall':counts['tp']/counts['gt'] if counts['gt'] else None,
              'precision':counts['tp']/counts['predictions'] if counts['predictions'] else None,
              'matched_color_accuracy':counts['correct_color']/counts['color_labeled_matches'] if counts['color_labeled_matches'] else None,
              'colors':{str(color):dict(data) for color,data in colors.items()},'rows':rows,
              'center_ellipse_source_truth':'UNAVAILABLE','fresh_real_blind_test':'NOT_RUN'}
    if simulator is not None:
        result['simulator_timing_seconds']={'model_load':simulator.load_seconds,
            'inference':simulator.inference_seconds,'scope':'HOST_SIMULATOR; not native KPU latency'}
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('formal_historical_test','pilot_validation'),default='formal_historical_test')
    parser.add_argument('--backend',choices=('onnx','simulator'),default='onnx')
    print(json.dumps(evaluate_real(**vars(parser.parse_args())),indent=2))
