"""Separate detection/color/physical-center/conic-ellipse measurements.

Task AP uses visible top boxes, up to5 detections per image and101-point
interpolation. It is not labelled pycocotools COCO AP. No test-set tuning.
"""
from collections import Counter,defaultdict
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from dataset_diagnostics import stats
from task_geometry import decode,box_iou


def ellipse_mask(center,axes,theta):
    canvas=np.zeros((360,640),np.uint8)
    if not np.isfinite([*center,*axes,theta]).all() or min(axes)<=0 or max(axes)>2000:return canvas
    cv2.ellipse(canvas,tuple(np.rint(center).astype(int)),tuple(np.maximum(1,np.rint(axes)).astype(int)),theta,0,360,1,-1)
    return canvas


def task_ap(records,threshold):
    rows=[]
    for color in range(6):
        positives=sum(sum(g['color_channel']==color for g in r['gt']) for r in records)
        detections=[]
        for image_index,r in enumerate(records):
            for d in r['pred']:
                if d['color_id']==color+1:detections.append((d['score'],image_index,d))
        detections.sort(key=lambda r:-r[0]);matched=defaultdict(set);tp=[];fp=[]
        for _,image_index,d in detections:
            gt=records[image_index]['gt'];eligible=[(i,g) for i,g in enumerate(gt) if g['color_channel']==color and i not in matched[image_index]]
            if eligible:
                ious=box_iou(np.asarray(d['bbox_xyxy_visible']),np.asarray([g['bbox_source'] for _,g in eligible]))
                best=int(np.argmax(ious));hit=ious[best]>=threshold
            else:hit=False
            tp.append(int(hit));fp.append(int(not hit))
            if hit:matched[image_index].add(eligible[best][0])
        if positives:
            t=np.cumsum(tp);f=np.cumsum(fp);recall=t/positives;precision=t/np.maximum(t+f,1)
            ap=float(np.mean([precision[recall>=r].max() if (recall>=r).any() else 0 for r in np.linspace(0,1,101)]))
        else:ap=None
        rows.append({'color_id':color+1,'gt':positives,'predictions':len(detections),'ap':ap})
    available=[r['ap'] for r in rows if r['ap'] is not None]
    return {'iou':threshold,'macro_ap':float(np.mean(available)) if available else None,'per_color':rows}


def evaluate(model,loader,device,score_threshold=.25,ap_score_floor=.001,max_batches=None):
    model.eval();records=[];counts=Counter();errors=defaultdict(list);times=[];groups=defaultdict(Counter)
    with torch.inference_mode():
        for batch_index,(images,targets,metadata) in enumerate(loader):
            if max_batches is not None and batch_index>=max_batches:break
            images=images.to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter();raw=model(images).float().cpu().numpy()
            if device.type=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-started)/len(images)*1000)
            height,width=images.shape[2:]
            for output,items,meta in zip(raw,targets,metadata):
                gain=meta['letterbox']['gain'];top,_,left,_=meta['letterbox']['padding_tblr'];offset=np.array([left,top])
                gt=[];counts['world_instances']+=len(items)
                for item in items:
                    if item['bbox'] is None or item['visible_area_px']<16:
                        counts['unobservable_or_under16px_retained']+=1;continue
                    source=(np.array(item['bbox'])-np.tile(offset,2))/gain
                    gt.append({**item,'bbox_source':source.tolist()})
                ap_pred=decode(output,width,height,score_threshold=ap_score_floor)['detections']
                operating=decode(output,width,height,score_threshold=score_threshold)
                pred=operating['detections'];records.append({'scene':meta['scene'],'gt':gt,'pred':ap_pred})
                counts['images']+=1;counts['observable_gt']+=len(gt);counts['operating_predictions']+=len(pred)
                counts['capacity_'+operating['capacity_state']]+=1
                group='world_count_'+str(meta['expected_world_count']);groups[group]['images']+=1
                groups[group]['observable_gt']+=len(gt)
                matched=set();hits=0
                for d in sorted(pred,key=lambda x:-x['score']):
                    eligible=[i for i in range(len(gt)) if i not in matched]
                    if not eligible:counts['fp']+=1;continue
                    ious=box_iou(np.asarray(d['bbox_xyxy_visible']),np.asarray([gt[i]['bbox_source'] for i in eligible]))
                    best=int(np.argmax(ious))
                    if ious[best]<.5:counts['fp']+=1;continue
                    index=eligible[best];matched.add(index);g=gt[index];hits+=1;counts['tp_class_agnostic']+=1
                    counts['matched_color_correct']+=int(d['color_id']==g['color_channel']+1)
                    counts['matched_color_total']+=1
                    physical=(np.asarray(g['physical'])-offset)/gain
                    if g['center_valid']:
                        key='physical_center_visible_px' if g['center_visible'] else 'physical_center_occluded_px'
                        errors[key].append(float(np.linalg.norm(physical-d['center_xy_projected_3d'])))
                    if g['geometry_valid']:
                        counts['matched_geometry_valid_gt']+=1;e=d['ellipse']
                        if e is None:counts['missing_finite_ellipse']+=1;continue
                        counts['finite_ellipse_predictions']+=1
                        conic=(np.asarray(g['conic'])-offset)/gain;axes=np.asarray(g['axes'])/gain
                        theta=.5*math.atan2(g['orientation'][0],g['orientation'][1])*180/math.pi
                        expected=ellipse_mask(conic,axes,theta)
                        actual=ellipse_mask(e['conic_center_xy'],[e['semi_major'],e['semi_minor']],e['orientation_deg'])
                        errors['ellipse_iou'].append(float((expected&actual).sum()/max(1,(expected|actual).sum())))
                        errors['conic_center_px'].append(float(np.linalg.norm(conic-e['conic_center_xy'])))
                        errors['semi_major_relative'].append(abs(e['semi_major']-axes[0])/axes[0])
                        errors['semi_minor_relative'].append(abs(e['semi_minor']-axes[1])/axes[1])
                        if g['orientation_valid']:
                            counts['angle_valid_gt']+=1
                            if e['orientation_valid']:
                                counts['angle_prediction_valid']+=1
                                errors['orientation_error_deg'].append(abs((e['orientation_deg']-theta+90)%180-90))
                            else:counts['angle_abstentions']+=1
                counts['fn_class_agnostic']+=len(gt)-hits;groups[group]['matched_gt']+=hits
                groups[group]['all_observable_targets_recalled']+=int(hits==len(gt))
                if not gt:
                    counts['empty_observation_images']+=1;counts['empty_false_positive_detections']+=len(pred)
    ap=[task_ap(records,float(t)) for t in np.linspace(.5,.95,10)]
    available_ap=[a['macro_ap'] for a in ap if a['macro_ap'] is not None]
    return {'schema':'topface-task-evaluation-v1.1','scope':'SYNTHETIC_ONLY' if records else 'EMPTY',
        'images':len(records),'denominators':dict(counts),'operating_score_threshold':score_threshold,
        'ap_score_floor':ap_score_floor,'max_detections_per_image':5,'ap_method':'task-specific101-point interpolation; not pycocotoolsCOCO AP',
        'task_mAP50':ap[0]['macro_ap'],'task_mAP50_95':float(np.mean(available_ap)) if available_ap else None,
        'ap_by_iou':ap,'class_agnostic_recall50':counts['tp_class_agnostic']/max(1,counts['observable_gt']),
        'matched_color_accuracy':counts['matched_color_correct']/max(1,counts['matched_color_total']),
        'errors':{key:stats(value) for key,value in errors.items()},'count_groups':{key:dict(value) for key,value in groups.items()},
        'host_forward_ms_per_image_batch_average':stats(times),'board_latency_measured':False,
        'real_accuracy_verified':False,'scene_names':[r['scene'] for r in records]}
