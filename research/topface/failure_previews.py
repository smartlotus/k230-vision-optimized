"""Select bounded error examples from saved formal raw tensors without inference."""
import json
from pathlib import Path

import cv2
import numpy as np

from cloud_store import atomic_json,sha
from simulate_dataset import order_sha
from task_geometry import box_iou,decode


def compare_frame(predictions,instances):
    truth=[item for item in instances if item['bbox_xyxy_visible'] is not None and item['visible_area_px']>=16]
    unused=set(range(len(truth)));matched=0;wrong_color=0;large_center=0;missing_ellipse=0
    for detection in predictions:
        indices=sorted(unused)
        if not indices:continue
        overlaps=box_iou(np.asarray(detection['bbox_xyxy_visible']),
                         np.asarray([truth[index]['bbox_xyxy_visible'] for index in indices]))
        best=int(np.argmax(overlaps))
        if overlaps[best]<.5:continue
        target=truth[indices[best]];unused.remove(indices[best]);matched+=1
        wrong_color+=int(target['color_id']!=detection['color_id'])
        if target.get('center_visible'):
            large_center+=int(np.linalg.norm(np.asarray(target['center_xy_projected_3d'])-
                                            detection['center_xy_projected_3d'])>10.)
        if target.get('geometry_valid'):missing_ellipse+=int(detection['ellipse'] is None)
    return {'observable_gt':len(truth),'predictions':len(predictions),'tp':matched,
        'fp':len(predictions)-matched,'fn':len(unused),'wrong_color':wrong_color,
        'matched_visible_center_over10_source_px':large_center,'matched_valid_gt_missing_ellipse':missing_ellipse}


def render(dataset,predictions,metrics,output,limit=12):
    dataset,predictions,output=map(Path,(dataset,predictions,output))
    if output.exists():raise FileExistsError('Preserve the original failure evidence')
    state=json.loads((predictions/'RESULT.json').read_text(encoding='utf8'))
    index=json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    rows=[row for row in index['scenes'] if row['split']=='synthetic_test']
    if (not 1<=limit<=12 or state['scope']!='HELD_OUT_SYNTHETIC_TEST' or
            state['status']!='NNCASE_AND_ORT_TEST_RAW_COMPLETE' or
            state['dataset_index_sha256']!=sha(dataset/'AUDIT_V11.json') or index['status']!='PASS' or
            state['images']!=state['completed_images'] or len(rows)!=state['images'] or
            state['scene_order_sha256']!=order_sha(rows) or metrics['identity']!=state['identity'] or
            metrics['scope']!='HELD_OUT_SYNTHETIC_TEST'):
        raise ValueError('Failure evidence requires completed actual formal test outputs and audited GT')
    files={row['name']:row for row in state['files']};arrays={}
    for backend in ('ort','simulator'):
        name=backend+'-raw.npy';path=predictions/name
        if path.stat().st_size!=files[name]['bytes'] or sha(path)!=files[name]['sha256']:
            raise ValueError('Failure evidence raw outputs changed')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.dtype!=np.float32 or list(values.shape)!=state['raw_shape']:
            raise ValueError('Failure raw tensor dimensions differ')
        arrays[backend]=values
    width,height=state['identity']['model_wh'];candidates=[]
    for number,row in enumerate(rows):
        directory=dataset/row['scene']
        if sha(directory/'labels.json')!=row['label_sha256']:
            raise ValueError('Failure example label changed after QA')
        label=json.loads((directory/'labels.json').read_text(encoding='utf8'))
        compared={backend:compare_frame(decode(raw[number],width,height,score_threshold=.25,nms_iou=.5)['detections'],
                                        label['instances']) for backend,raw in arrays.items()}
        priority=sum(10*(r['fp']+r['fn'])+5*r['wrong_color']+r['matched_visible_center_over10_source_px']+
                     r['matched_valid_gt_missing_ellipse'] for r in compared.values())
        if priority:candidates.append({'scene':row['scene'],'row':number,'priority':priority,'measurements':compared})
    candidates.sort(key=lambda row:(-row['priority'],row['scene']))
    output.mkdir(parents=True);examples=[]
    for ordinal,candidate in enumerate(candidates[:limit],1):
        row=rows[candidate['row']];directory=dataset/row['scene']
        if sha(directory/'rgb.png')!=row['rgb_sha256']:raise ValueError('Failure image differs from QA')
        label=json.loads((directory/'labels.json').read_text(encoding='utf8'));image=cv2.imread(str(directory/'rgb.png'))
        tiles=[]
        for backend in ('ort','simulator'):
            tile=image.copy()
            for target in label['instances']:
                if target['bbox_xyxy_visible'] is None or target['visible_area_px']<16:continue
                x1,y1,x2,y2=target['bbox_xyxy_visible']
                cv2.rectangle(tile,(x1,y1),(x2,y2),(0,220,0),1)
                cv2.putText(tile,'GT '+str(target['color_id']),(x1,max(12,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,0),1)
            predictions_row=decode(arrays[backend][candidate['row']],width,height,score_threshold=.25,nms_iou=.5)['detections']
            for detection in predictions_row:
                x1,y1,x2,y2=map(round,detection['bbox_xyxy_visible'])
                cv2.rectangle(tile,(x1,y1),(x2,y2),(0,0,255),1)
                text=f"P{detection['color_id']} {detection['score']:.2f}"
                cv2.putText(tile,text,(max(0,x1),min(355,y2+12)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            header=np.zeros((28,640,3),np.uint8)
            cv2.putText(header,backend+' '+row['scene']+' green=GT red=prediction',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(np.vstack([header,tile]))
        path=output/f'case-{ordinal:02d}.jpg';cv2.imwrite(str(path),np.hstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,85])
        examples.append({**candidate,'name':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
            'source_rgb_sha256':row['rgb_sha256'],'source_label_sha256':row['label_sha256']})
    evidence={'status':'BOUNDED_FORMAL_TEST_ERROR_EXAMPLES_SELECTED','scope':'HELD_OUT_SYNTHETIC_TEST',
        'identity':state['identity'],'dataset_index_sha256':state['dataset_index_sha256'],
        'images_evaluated':len(rows),'error_images_at_registered_rules':len(candidates),'examples':examples,
        'selection':'Highest10*(FP+FN)+5*wrongColor+centerError>10px+missingEllipse; deterministic scene tie break',
        'thresholds':{'score':.25,'nms_iou':.5,'matching_iou':.5},
        'note':'Diagnostic selection after frozen testing; no tuning, inference replay or claim of all geometric failures.',
        'new_inference_passes':0}
    atomic_json(output/'EVIDENCE.json',evidence);return evidence
