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


RECOVERY_EVENTS = {'occlusion', 'exit_and_reentry'}


def load_sequence_frames(dataset, index):
    frames = []
    groups = defaultdict(list)
    for row in index['scenes']:
        folder = dataset / row['scene']
        if sha(folder / 'rgb.png') != row['rgb_sha256'] or sha(folder / 'labels.json') != row['label_sha256']:
            raise ValueError('Sequence frame changed after the audit')
        label = json.loads((folder / 'labels.json').read_text(encoding='utf8'))
        if label['split'] != 'synthetic_test' or len(label['instances']) != 1:
            raise ValueError('Redetection fixtures require a single audited test target')
        if (label['instances'][0]['visible_area_px'] >= 16) != label['expected_visible']:
            raise ValueError('Observed sequence visibility differs from the event labels')
        groups[label['sequence_group_id']].append(label)
        frames.append((folder, label))
    if not groups:
        raise ValueError('Recovery fixture has no audited sequences')
    for labels in groups.values():
        labels.sort(key=lambda label: label['sequence_frame'])
        first = labels[0]
        length = first.get('sequence_length', 12)
        if [label['sequence_frame'] for label in labels] != list(range(length)):
            raise ValueError('Recovery sequence has missing, duplicate, or unexpected frames')
        for label in labels:
            if (label['sequence_event'], label['recovery_window_start'], label['instances'][0]['color_id'],
                    label.get('sequence_length', 12)) != (
                    first['sequence_event'], first['recovery_window_start'],
                    first['instances'][0]['color_id'], length):
                raise ValueError('Recovery sequence metadata changes between frames')
            hidden = label['sequence_event'] in RECOVERY_EVENTS and 4 <= label['sequence_frame'] < 8
            if label['expected_visible'] == hidden:
                raise ValueError('Recovery event visibility schedule differs from its protocol')
        start = first['recovery_window_start']
        if not 0 <= start < length or not labels[start]['expected_visible']:
            raise ValueError('Recovery interval starts without a visible target')
    return frames


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
    audited_frames = load_sequence_frames(dataset, index)
    runtime = session(export_dir/'topface.onnx') if simulator is None else None
    width,height = trained['width'],trained['height']
    groups = defaultdict(list)
    for frame_number, (folder, label) in enumerate(audited_frames, start=1):
        target = label['instances'][0]
        visible = target['visible_area_px']>=16
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
            'height_mm':label['camera_height_above_top_mm'], 'plane_angle_deg':label['plane_angle_deg'],
            'hfov_deg':label['hfov_deg'], 'camera_group_id':label['camera_group_id'],
            'matched':hit,'localized':localized,'target_color':target['color_id'],
            'matched_color':predicted_color,'detections':len(detections),
            'false_positives':len(detections)-int(localized)})
        if simulator is not None and (frame_number % 32 == 0 or frame_number == len(audited_frames)):
            print(json.dumps({'simulator_frames_completed': frame_number,
                              'total_frames': len(audited_frames)}), flush=True)
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
    recovery_records = [row for row in records if row['event'] in RECOVERY_EVENTS]
    continuity_frames = [frame for row in records if row['event'] not in RECOVERY_EVENTS for frame in row['frames']]
    coverage = {'height_mm': sorted({frame['height_mm'] for frame in frames}),
                'plane_angle_deg': sorted({frame['plane_angle_deg'] for frame in frames}),
                'hfov_deg': sorted({frame['hfov_deg'] for frame in frames}),
                'colors': sorted({frame['target_color'] for frame in frames}),
                'camera_groups': sorted({frame['camera_group_id'] for frame in frames}),
                'events': sorted({row['event'] for row in records})}
    result = {'status':'SYNTHETIC_REDETECTION_MEASURED' if phase=='formal' else 'PILOT_PRELIMINARY_SYNTHETIC_REDETECTION_MEASURED',
              'phase':phase,'model_experiment':trained['experiment'],'final_acceptance':False,
              'eligible_for_formal_acceptance':phase=='formal',
              'backend':'ONNX_RUNTIME' if simulator is None else 'NNCASE_SIMULATOR',
              'model_sha256':trained['onnx_sha256'] if simulator is None else simulator.identity['kmodel_sha256'],
              'reference_onnx_sha256':trained['onnx_sha256'],
              'dataset_index_sha256':sha(dataset/'AUDIT_V11.json'),'sequence_count':len(records),
              'recoveries':sum(row['recovered'] for row in recovery_records),'records':records,
              'recovery_events': {'sequences': len(recovery_records),
                  'recovered': sum(row['recovered'] for row in recovery_records),
                  'first_visible_frame_detected': sum(row['first_visible_frame_detected'] for row in recovery_records),
                  'frames_to_recovery': [row['frames_to_recovery'] for row in recovery_records]},
              'recovery_by_event': {event: {
                  'sequences': sum(row['event'] == event for row in recovery_records),
                  'recovered': sum(row['event'] == event and row['recovered'] for row in recovery_records),
                  'first_visible_frame_detected': sum(row['event'] == event and row['first_visible_frame_detected']
                                                      for row in recovery_records)}
                  for event in sorted(RECOVERY_EVENTS)},
              'continuous_visibility': {'sequences': len(records) - len(recovery_records),
                  'visible_frames': sum(frame['visible'] for frame in continuity_frames),
                  'localized_frames': sum(frame['localized'] for frame in continuity_frames),
                  'correct_color_frames': sum(frame['matched'] for frame in continuity_frames)},
              'coverage': coverage,
              'denominators':{'frames':len(frames),'visible_gt':gt,'hidden_frames':len(frames)-gt,
                  'predictions':predictions,'tp':tp,'fp':predictions-tp,'fn':gt-tp,
                  'correct_color':correct_color,'color_labeled_matches':tp},
              'precision':tp/predictions if predictions else None,'recall':tp/gt if gt else None,
              'matched_color_accuracy':correct_color/tp if tp else None,
              'scope':'SYNTHETIC_SEQUENCE_FIXTURE; discrete inferred conditions listed in coverage; real accuracy unmeasured',
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
