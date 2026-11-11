"""Evaluate manual boxes against frozen real-board detections; no fake truth."""
import argparse
import json
import math
from pathlib import Path


def iou(a,b):
    x,y,w,h=a;u,v,p,q=b
    intersection=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return intersection/(w*h+p*q-intersection) if w*h+p*q>intersection else 0


def evaluate(report,labels):
    tp=fp=fn=0
    errors=[]
    matches=[]
    samples={s['frame_id']:s for s in report['samples']}
    for label in labels['frames']:
        if label['frame_id'] not in samples:raise ValueError('Missing recorded frame')
        predictions=samples[label['frame_id']]['detections']
        remaining=set(range(len(predictions)))
        for truth in label['objects']:
            options=[(iou(truth['bbox'],p['bbox']),i,p) for i,p in enumerate(predictions)
                     if i in remaining and p['target_id']==truth['target_id']]
            options.sort(reverse=True,key=lambda t:t[0])
            if not options or options[0][0]<labels.get('iou_threshold',0.5):
                fn+=1
                continue
            overlap,index,prediction=options[0]
            remaining.remove(index)
            tp+=1
            truth_center=(truth['bbox'][0]+truth['bbox'][2]/2,truth['bbox'][1]+truth['bbox'][3]/2)
            error=math.sqrt(sum((prediction['center'][i]-truth_center[i])**2 for i in (0,1)))
            errors.append(error)
            matches.append({'frame_id':label['frame_id'],'target_id':truth['target_id'],
                            'iou':overlap,'center_error_px':error})
        fp+=len(remaining)
    return {'scope':labels['scope'],'label_source':labels['label_source'],'frames':len(labels['frames']),
            'ground_truth_objects':tp+fn,'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'mean_center_error_px':sum(errors)/len(errors) if errors else None,
            'max_center_error_px':max(errors) if errors else None,'matches':matches,
            'is_generalization_acceptance':False,'physical_motion_acceptance':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('labels',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=evaluate(json.loads(args.report.read_text(encoding='utf-8')),json.loads(args.labels.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
