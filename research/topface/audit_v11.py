"""Strict V1.1 labels: physical circle centre, conic centre and visible mask."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from geometry import ellipse_from_points,project,top_boundary
from cloud_store import atomic_json,sha
from scene_config import camera_partition


def audit(root,require_smoke=False):
    root=Path(root);rows=[];issues=[];seeds=set();groups=set()
    count_hist={i:0 for i in range(6)};color_hist={i:0 for i in range(1,7)}
    image_hashes={};height_values=[];angle_values=[];max_center_error=0.;camera_splits={}
    for path in sorted(root.glob('scene-*/labels_raw.json')):
        raw=json.loads(path.read_text());directory=path.parent
        camera_group,effective_split=camera_partition(raw['camera_height_above_top_mm'],raw['plane_angle_deg'],raw['hfov_deg'])
        # Split assignment is an audited derivation, before any model training.
        # Preserve the producer's original hint in raw files and derived labels.
        raw['split_generation_hint']=raw.get('split')
        raw['split']=effective_split;raw['camera_group_id']=camera_group
        camera_splits.setdefault(camera_group,set()).add(effective_split)
        def issue(reason,**fields):issues.append(dict(scene=directory.name,reason=reason,**fields))
        if raw['scene_group_id'] in groups or raw['random_seed'] in seeds:issue('DUPLICATE_GROUP_OR_SEED')
        groups.add(raw['scene_group_id']);seeds.add(raw['random_seed'])
        expected=raw['expected_instance_count'];count_hist[expected]+=1
        if len(raw['instances'])!=expected:issue('INSTANCE_COUNT_MISMATCH')
        rgb=cv2.imread(str(directory/'rgb.png'))
        if rgb is None or rgb.shape!=(360,640,3):raise ValueError('Expected RGB 640x360')
        image_digest=sha(directory/'rgb.png')
        if image_digest in image_hashes:issue('DUPLICATE_RGB_BYTES',other=image_hashes[image_digest])
        image_hashes[image_digest]=directory.name
        K=np.array(raw['camera_intrinsics']);pose=np.array(raw['camera_pose_world_from_cv'])
        labels=[];total=np.zeros((360,640),np.uint8);active_ids=set()
        for item in raw['instances']:
            index=item['instance_id'];active_ids.add(index)
            color=item['color_id']
            if color not in color_hist:raise ValueError('Unverified color ID')
            color_hist[color]+=1
            visible=cv2.imread(str(directory/item['top_face_visible_mask']),0)
            if visible is None or visible.shape!=(360,640):raise ValueError('Missing visible top mask')
            visible=(visible>=128).astype(np.uint8)
            if np.any(visible&total):issue('VISIBLE_INSTANCE_MASK_OVERLAP',instance_id=index)
            total|=visible
            transform=np.array(item['object_pose'])
            center_world=(transform@np.array([0.,0.,.06,1.]))[:3]
            physical=project([center_world],K,pose)[0]
            error=float(np.linalg.norm(physical-np.array(item['physical_top_center_xy'])))
            max_center_error=max(max_center_error,error)
            if error>.01:issue('PHYSICAL_CENTER_REPROJECTION_MISMATCH',instance_id=index,error_px=error)
            local=top_boundary([0.,0.]);world=(np.column_stack([local,np.ones(len(local))])@transform.T)[:,:3]
            boundary=project(world,K,pose).astype(np.float32)
            if np.max(np.abs(boundary-np.array(item['top_boundary_xy'])))>.01:
                issue('TOP_BOUNDARY_REPROJECTION_MISMATCH',instance_id=index)
            full=np.zeros_like(visible);cv2.fillPoly(full,[np.round(boundary).astype(np.int32)],1)
            allowance=cv2.dilate(full,np.ones((3,3),np.uint8))
            outside=int((visible&(1-allowance)).sum())
            if outside:issue('VISIBLE_MASK_OUTSIDE_PROJECTED_TOP',instance_id=index,pixels=outside)
            visible_area=int(visible.sum());full_area=float(cv2.contourArea(boundary));oncanvas=int(full.sum())
            truncated=bool(np.any(boundary[:,0]<0) or np.any(boundary[:,0]>=640) or
                           np.any(boundary[:,1]<0) or np.any(boundary[:,1]>=360))
            if not visible_area and oncanvas>=16 and not raw.get('fully_occluded',False):
                issue('UNEXPECTED_EMPTY_TOP_MASK',instance_id=index)
            center_valid=bool(np.isfinite(physical).all() and 0<=physical[0]<640 and 0<=physical[1]<360)
            u,v=np.floor(physical).astype(int)
            center_visible=bool(center_valid and visible[v,u])
            ellipse=ellipse_from_points(boundary)
            conic_mask=np.zeros_like(visible)
            axes=tuple(np.maximum(1,np.round([ellipse['semi_major'],ellipse['semi_minor']])).astype(int))
            conic_center=tuple(np.round(ellipse['center']).astype(int))
            cv2.ellipse(conic_mask,conic_center,axes,ellipse['orientation_deg'],0,360,1,-1)
            fit_iou=float((full&conic_mask).sum()/max(1,(full|conic_mask).sum()))
            if fit_iou<.9:issue('ELLIPSE_FIT_IOU',instance_id=index,iou=fit_iou)
            yy,xx=np.where(visible)
            bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
            fraction=min(1.,visible_area/max(1.,full_area))
            occlusion=1-min(1.,visible_area/max(1,oncanvas))
            label={**item,'top_face_full_mask':f'full-{index}.png','bbox_xyxy_visible':bbox,
                'bbox_xyxy':bbox,'center_xy':physical.tolist(),'center_xy_projected_3d':physical.tolist(),
                'projected_3d_circle_center':physical.tolist(),'center_visible':center_visible,
                'center_valid':center_valid,'ellipse_conic_center':ellipse['center'],
                'ellipse_full':{'conic_center':ellipse['center'],'semi_major':ellipse['semi_major'],
                    'semi_minor':ellipse['semi_minor'],'orientation_deg':ellipse['orientation_deg']},
                'orientation_valid':ellipse['angle_valid'],'visible_fraction':fraction,
                'occlusion_level':occlusion,'occluded':occlusion>.05,'truncated':truncated,
                'geometry_valid':visible_area>=16 and fraction>=.4 and not truncated,
                'visible_area_px':visible_area,'full_projection_area_px':full_area,
                'ellipse_raster_iou':fit_iou,'center_reprojection_error_px':error,
                'visible_mask_centroid_diagnostic':([float(xx.mean()),float(yy.mean())] if len(xx) else None)}
            label.pop('top_boundary_xy',None);labels.append(label)
            cv2.imwrite(str(directory/f'full-{index}.png'),full*255)
            cv2.ellipse(rgb,conic_center,axes,ellipse['orientation_deg'],0,360,(0,255,255),1)
            if center_valid:cv2.drawMarker(rgb,tuple(np.round(physical).astype(int)),(255,0,255),cv2.MARKER_CROSS,9,1)
            if bbox:cv2.rectangle(rgb,tuple(bbox[:2]),tuple(bbox[2:]),(0,255,0),1)
        for index in range(1,6):
            if index not in active_ids:
                hidden=cv2.imread(str(directory/f'visible-{index}.png'),0)
                if hidden is None or hidden.shape!=(360,640):raise ValueError('All five mask slots required')
                if np.any(hidden>=128):issue('UNLABELLED_TOP_MASK',instance_id=index)
        raw['instances']=labels;atomic_json(directory/'labels.json',raw)
        cv2.putText(rgb,f"{directory.name}: {expected} targets",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(directory/'overlay.jpg'),rgb)
        rows.append({'scene':directory.name,'group':raw['scene_group_id'],'split':raw['split'],
                     'camera_group':camera_group,'seed':raw['random_seed'],'instances':expected,'rgb_sha256':image_digest,
                     'label_sha256':sha(directory/'labels.json')})
        height_values.append(raw['camera_height_above_top_mm']);angle_values.append(raw['plane_angle_deg'])
    if not rows:raise ValueError('No scene data found')
    if require_smoke:
        if len(rows)!=100:issues.append({'reason':'SMOKE_MUST_HAVE_100_INDEPENDENT_SCENES'})
        if any(not value for value in count_hist.values()):issues.append({'reason':'MISSING_0_TO_5_COUNT_COVERAGE'})
        if any(not value for value in color_hist.values()):issues.append({'reason':'MISSING_SIX_COLOR_COVERAGE'})
    report={'schema':'topface-v1.1-label-audit','status':'PASS' if not issues else 'FAIL',
        'images':len(rows),'instances':sum(r['instances'] for r in rows),'issues':issues,
        'count_histogram':count_hist,'color_histogram':color_hist,
        'unique_scene_groups':len(groups),'unique_scene_seeds':len(seeds),
        'camera_configuration_groups':len(camera_splits),
        'camera_group_cross_split_count':sum(len(s)>1 for s in camera_splits.values()),
        'split_policy':'SHA256 of height/pitch/HFOV configuration group; frozen before training',
        'height_mm_range':[min(height_values),max(height_values)],
        'plane_angle_deg_range':[min(angle_values),max(angle_values)],
        'max_center_reprojection_error_px':max_center_error,
        'strong_perspective_outside_user_envelope':'excluded, not claimed as covered',
        'real_geometry_accuracy_verified':False,'scenes':rows}
    atomic_json(root/'AUDIT_V11.json',report)
    if issues:raise RuntimeError(f'V1.1 audit failed: {len(issues)} issues; see AUDIT_V11.json')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--require-smoke',action='store_true');args=parser.parse_args()
    print(json.dumps(audit(args.root,args.require_smoke)))
