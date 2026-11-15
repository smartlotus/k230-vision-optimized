"""Measure when a frozen detector redetects the original simulated target."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from cloud_store import atomic_json, sha
from export_nanodet_candidate import session
from task_geometry import box_iou, decode, preprocess_bgr


def measure(export_dir, dataset, output, phase='formal', backend='onnx'):
    export_dir,dataset,output=export_dir.resolve(),dataset.resolve(),output.resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing sequence result')
    trained = json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    index = json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    model_path = export_dir/'topface.onnx'
    if phase not in ('formal','pilot_preliminary') or backend not in ('onnx','simulator'):
        raise ValueError('Explicit frozen-model evaluation phase and backend required')
    allowed=('formal_full',) if phase=='formal' else ('pilot_pretrained','pilot_geometry_refine')
    if (trained['experiment'] not in allowed or trained['status']!='PYTORCH_ONNX_PARITY_PASS'
            or sha(model_path)!=trained['onnx_sha256'] or index['status']!='PASS'):
        raise ValueError('Sequence model does not match its evaluation phase or audited frames')
    if backend=='onnx':
        context=nullcontext(None)
    else:
        from nncase_simulator import KModelRuntime
        identity=json.loads((export_dir/'identity.json').read_text(encoding='utf8'))
        if identity['onnx_sha256']!=trained['onnx_sha256']:
            raise ValueError('Quantized sequence model differs from the selected floating model')
        context=KModelRuntime(export_dir/'topface.kmodel',export_dir/'identity.json',
            Path(__file__).resolve().parents[2]/'.ops/local'/('topface-sequence-simulator-'+uuid.uuid4().hex))
    with context as simulator:
        return _measure(export_dir,dataset,output,trained,index,phase,simulator)


def _measure(export_dir,dataset,output,trained,index,phase,simulator):
    runtime = session(export_dir/'topface.onnx') if simulator is None else None
    width,height = trained['width'],trained['height']
    groups = defaultdict(list)
    for row in index['scenes']:
        folder = dataset/row['scene']
        if sha(folder/'rgb.png')!=row['rgb_sha256'] or sha(folder/'labels.json')!=row['label_sha256']:
            raise ValueError('Sequence frame changed after the audit')
        label = json.loads((folder/'labels.json').read_text(encoding='utf8'))
        if label['split']!='synthetic_test' or len(label['instances'])!=1:
            raise ValueError('Redetection fixtures require a single audited test target')
        target = label['instances'][0]
        visible = target['visible_area_px']>=16
        if visible!=label['expected_visible']:
            raise ValueError('Observed sequence visibility differs from the event labels')
        image=cv2.imread(str(folder/'rgb.png'))
        if simulator is None:
            tensor,_ = preprocess_bgr(image,width,height)
            raw = runtime.run(None,{runtime.get_inputs()[0].name:tensor[None]})[0][0]
        else:
            if (simulator.width,simulator.height)!=(width,height):
                raise ValueError('Simulator sequence geometry differs from the floating contract')
            raw=simulator.raw_from_bgr(image)
        detections = decode(raw,width,height)['detections']
        hit = False;localized=False;predicted_color=None
        if visible:
            for detection in detections:
                if box_iou(
                    np.asarray(detection['bbox_xyxy_visible']),
                    np.asarray([target['bbox_xyxy_visible']]))[0]>=.5:
                    localized=True;predicted_color=detection['color_id']
                    hit = predicted_color==target['color_id']
                    break
        groups[label['sequence_group_id']].append({'frame':label['sequence_frame'],'visible':visible,
            'event':label['sequence_event'],'recovery_window_start':label['recovery_window_start'],
            'matched':hit,'localized':localized,'target_color':target['color_id'],
            'matched_color':predicted_color,'detections':len(detections),
            'false_positives':len(detections)-int(localized)})
    records = []
    for name,frames in sorted(groups.items()):
        frames.sort(key=lambda row:row['frame'])
        start = frames[0]['recovery_window_start']
        eligible = [row for row in frames if row['frame']>=start and row['visible']]
        if not eligible or eligible[0]['frame']!=start:
            raise ValueError('Recovery interval has no visible target on its first frame')
        first = next((row['frame'] for row in eligible if row['matched']),None)
        records.append({'sequence':name,'event':frames[0]['event'],'first_visible_frame_detected':first==start,
                        'recovered':first is not None,'frames_to_recovery':None if first is None else first-start,
                        'false_positives_when_hidden':sum(row['detections'] for row in frames if not row['visible']),
                        'frames':frames})
    frames=[frame for group in groups.values() for frame in group]
    gt=sum(row['visible'] for row in frames);tp=sum(row['localized'] for row in frames)
    predictions=sum(row['detections'] for row in frames)
    correct_color=sum(row['matched'] for row in frames)
    result = {'status':'SYNTHETIC_REDETECTION_MEASURED' if phase=='formal' else 'PILOT_PRELIMINARY_SYNTHETIC_REDETECTION_MEASURED',
              'phase':phase,'model_experiment':trained['experiment'],'final_acceptance':False,
              'eligible_for_formal_acceptance':phase=='formal',
              'backend':'ONNX_RUNTIME' if simulator is None else 'NNCASE_SIMULATOR',
              'model_sha256':trained['onnx_sha256'] if simulator is None else simulator.identity['kmodel_sha256'],
              'reference_onnx_sha256':trained['onnx_sha256'],
              'dataset_index_sha256':sha(dataset/'AUDIT_V11.json'),'sequence_count':len(records),
              'recoveries':sum(row['recovered'] for row in records),'records':records,
              'denominators':{'frames':len(frames),'visible_gt':gt,'hidden_frames':len(frames)-gt,
                  'predictions':predictions,'tp':tp,'fp':predictions-tp,'fn':gt-tp,
                  'correct_color':correct_color,'color_labeled_matches':tp},
              'precision':tp/predictions if predictions else None,'recall':tp/gt if gt else None,
              'matched_color_accuracy':correct_color/tp if tp else None,
              'scope':'SYNTHETIC_SEQUENCE_FIXTURE; 175 mm, 90 degrees; blue and black only; not full-range or real accuracy',
              'matching':{'iou':.5,'score':.25,'nms_iou':.5,
                  'localization_color_match':False,'recovery_color_match':True},'tracking_enabled':False}
    if simulator is not None:
        result['simulator_timing_seconds']={'model_load':simulator.load_seconds,
            'inference':simulator.inference_seconds,'scope':'HOST_SIMULATOR; not native KPU latency'}
    atomic_json(output,result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('export-dir','dataset','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--phase',choices=('formal','pilot_preliminary'),default='formal')
    parser.add_argument('--backend',choices=('onnx','simulator'),default='onnx')
    result = measure(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ('status','sequence_count','recoveries')},indent=2))
