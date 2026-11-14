"""Measure when a frozen detector redetects the original simulated target."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from cloud_store import atomic_json, sha
from export_nanodet_candidate import session
from task_geometry import box_iou, decode, preprocess_bgr


def measure(export_dir, dataset, output):
    if output.exists():
        raise FileExistsError('Preserve the existing sequence result')
    trained = json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    index = json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    model_path = export_dir/'topface.onnx'
    if (trained['experiment']!='formal_full' or trained['status']!='PYTORCH_ONNX_PARITY_PASS'
            or sha(model_path)!=trained['onnx_sha256'] or index['status']!='PASS'):
        raise ValueError('Sequence evaluation requires a frozen formal model and audited frames')
    runtime = session(model_path)
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
        tensor,_ = preprocess_bgr(cv2.imread(str(folder/'rgb.png')),width,height)
        raw = runtime.run(None,{runtime.get_inputs()[0].name:tensor[None]})[0][0]
        detections = decode(raw,width,height)['detections']
        hit = False
        if visible:
            for detection in detections:
                if detection['color_id']==target['color_id'] and box_iou(
                    np.asarray(detection['bbox_xyxy_visible']),
                    np.asarray([target['bbox_xyxy_visible']]))[0]>=.5:
                    hit = True
                    break
        groups[label['sequence_group_id']].append({'frame':label['sequence_frame'],'visible':visible,
            'event':label['sequence_event'],'recovery_window_start':label['recovery_window_start'],
            'matched':hit,'detections':len(detections)})
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
    result = {'status':'SYNTHETIC_REDETECTION_MEASURED','model_sha256':trained['onnx_sha256'],
              'dataset_index_sha256':sha(dataset/'AUDIT_V11.json'),'sequence_count':len(records),
              'recoveries':sum(row['recovered'] for row in records),'records':records,
              'matching':{'iou':.5,'score':.25,'color_match':True},'tracking_enabled':False}
    atomic_json(output,result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('export-dir','dataset','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    result = measure(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ('status','sequence_count','recoveries')},indent=2))
