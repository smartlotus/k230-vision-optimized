"""Explicit 6+32+12 losses and train-only synthetic-label ingestion.

The original NanoDet auxiliary training head is not imported. This project's
new geometry channels are never passed to the original 6+32-only split.
"""
from collections import Counter
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from task_geometry import priors,preprocess_bgr
from cloud_store import sha


class TopFaceDataset(Dataset):
    def __init__(self,root,split='train',width=320,height=192,scene_names=None,augment=False,seed=20261114,
                 background_bank=None,body_negative_bank=None):
        self.root=Path(root);self.width=width;self.height=height
        self.augment=augment;self.seed=seed;self.epoch=0;self.rows=[]
        self.backgrounds=None
        self.body_only=None;self.body_counts=Counter();self.body_used=set();self.body_trace=None
        if body_negative_bank is not None:
            if split!='train' or not augment:raise ValueError('Body-only augmentation is restricted to augmented training')
            from body_negative_bank import BodyOnlyBank
            self.body_only=BodyOnlyBank(body_negative_bank)
        if background_bank is not None:
            if split!='train' or not augment:
                raise ValueError('Real background augmentation is restricted to augmented training')
            from background_augmentation import BackgroundBank
            self.backgrounds=BackgroundBank(background_bank)
        audit=json.loads((self.root/'AUDIT_V11.json').read_text(encoding='utf8'))
        if audit['status']!='PASS':raise ValueError('Dataset QA must pass before training')
        requested=None if scene_names is None else set(scene_names)
        for row in audit['scenes']:
            if row['split']==split and (requested is None or row['scene'] in requested):self.rows.append(row)
        if not self.rows:raise ValueError('Requested approved split is empty')
        if requested is not None and set(r['scene'] for r in self.rows)!=requested:
            raise ValueError('Selected scenes must all belong to the requested split')
        scene_groups={};camera_groups={}
        for row in audit['scenes']:
            for registry,key in ((scene_groups,row['group']),(camera_groups,row['camera_group'])):
                if key in registry and registry[key]!=row['split']:raise ValueError('Dataset group crosses splits')
                registry[key]=row['split']
        for row in self.rows:
            directory=(self.root/row['scene']).resolve()
            if not directory.is_relative_to(self.root.resolve()):raise ValueError('Scene outside approved dataset')
            label_path=directory/'labels.json'
            if sha(directory/'rgb.png')!=row['rgb_sha256'] or sha(label_path)!=row['label_sha256']:
                raise ValueError('Dataset bytes changed after approved QA')
            label=json.loads(label_path.read_text(encoding='utf8'))
            if (label['split'],label['scene_group_id'],label['camera_group_id'])!=(row['split'],row['group'],row['camera_group']):
                raise ValueError('Frozen scene/camera group identity changed')
        self.audit_sha256=sha(self.root/'AUDIT_V11.json')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];directory=self.root/row['scene']
        label=json.loads((directory/'labels.json').read_text(encoding='utf8'))
        if label['split']!=row['split']:raise ValueError('Split identity changed')
        image=cv2.imread(str(directory/'rgb.png'))
        replacement=None
        if self.augment:
            if self.body_only is not None:
                self.body_counts['read_slots']+=1
                if not label['instances']:self.body_counts['world_empty_slots']+=1
                body_seed=[self.seed,self.epoch,index,0xB0]
                image,label,replacement=self.body_only.maybe_replace(image,label,row,
                    np.random.default_rng(np.random.SeedSequence(body_seed)))
                if replacement is not None:
                    self.body_counts['replacements']+=1;self.body_used.add(replacement['source_scene'])
                    replacement.update(epoch=self.epoch,base_slot=index,seed_components=body_seed)
                    if self.body_trace is not None:
                        with Path(self.body_trace).open('a',encoding='utf8') as stream:
                            stream.write(json.dumps(replacement)+'\n')
            if self.backgrounds is not None and replacement is None:
                background_rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,0xB6]))
                image,_=self.backgrounds.apply(image,label['instances'],background_rng)
            # No hue shift/channel swap; only small photometric degradation.
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            image=np.clip(image.astype(np.float32)*rng.uniform(.9,1.1)+rng.normal(0,rng.uniform(0,2),image.shape),0,255).astype(np.uint8)
            if rng.random()<.2:image=cv2.GaussianBlur(image,(3,3),.5)
        image,info=preprocess_bgr(image,self.width,self.height)
        top,_,left,_=info['padding_tblr'];offset=np.array([left,top],np.float32);gain=info['gain']
        targets=[]
        for item in label['instances']:
            # Retain every world instance and its explicit observation status.
            # Loss eligibility depends on visibility/representability, never ID.
            bbox=item['bbox_xyxy_visible'];ellipse=item['ellipse_full']
            theta=math.radians(ellipse['orientation_deg'])
            targets.append({'instance_id':item['instance_id'],'color_channel':item['color_id']-1,
                'bbox':None if bbox is None else (np.asarray(bbox,np.float32)*gain+np.tile(offset,2)).tolist(),
                'visible_area_px':item['visible_area_px'],
                'physical':(np.asarray(item['center_xy_projected_3d'],np.float32)*gain+offset).tolist(),
                'conic':(np.asarray(item['ellipse_conic_center'],np.float32)*gain+offset).tolist(),
                'axes':(np.array([ellipse['semi_major'],ellipse['semi_minor']],np.float32)*gain).tolist(),
                'orientation':[math.sin(2*theta),math.cos(2*theta)],
                'center_valid':item['center_valid'],'center_visible':item['center_visible'],
                'geometry_valid':item['geometry_valid'],'orientation_valid':item['orientation_valid'],
                'visible_fraction':item['visible_fraction'],'truncated':item['truncated']})
        meta={'scene':row['scene'] if replacement is None else replacement['source_scene'],
            'group':label['scene_group_id'],'camera_group':label['camera_group_id'],
            'split':row['split'],'expected_world_count':len(targets),'letterbox':info,
            'camera_height_mm':label.get('camera_height_above_top_mm'),'camera_plane_angle_deg':label.get('plane_angle_deg')}
        if replacement is not None:meta['body_only_augmentation']=replacement
        return torch.from_numpy(image),targets,meta


def collate(items):
    return torch.stack([i[0] for i in items]),[i[1] for i in items],[i[2] for i in items]


def giou_loss(pred,target):
    inter=(torch.minimum(pred[:,2:],target[:,2:])-torch.maximum(pred[:,:2],target[:,:2])).clamp(min=0).prod(dim=1)
    pred_area=(pred[:,2:]-pred[:,:2]).clamp(min=0).prod(dim=1)
    target_area=(target[:,2:]-target[:,:2]).clamp(min=0).prod(dim=1)
    union=(pred_area+target_area-inter).clamp(min=1e-7)
    enclosing=(torch.maximum(pred[:,2:],target[:,2:])-torch.minimum(pred[:,:2],target[:,:2])).clamp(min=0).prod(dim=1).clamp(min=1e-7)
    return 1-inter/union+(enclosing-union)/enclosing


class TopFaceLoss(nn.Module):
    def __init__(self,width=320,height=192,positives_per_gt=3,ellipse_shape_loss=False):
        super().__init__();points,scale=priors(width,height)
        self.points_numpy=points;self.scales_numpy=scale
        self.register_buffer('points',torch.from_numpy(points));self.register_buffer('scale',torch.from_numpy(scale))
        self.positives_per_gt=positives_per_gt
        self.ellipse_shape_loss=ellipse_shape_loss
        self.weights={'color':1.,'dfl':.25,'bbox':2.,'physical':1.,'conic':1.,'axes':.5,
                      'orientation':.2,'center_flags':.2,'geometry_confidence':.2,'visibility_quality':.2}
        if ellipse_shape_loss:
            self.weights.update(ellipse_anisotropy=.25,ellipse_log_shape=.5)

    def assign(self,targets):
        points=self.points_numpy;scales=self.scales_numpy
        pairs=[];counts=Counter(world_instances=len(targets));candidate_lists={}
        for gt_index,item in enumerate(targets):
            if not 0<=item['color_channel']<6:raise ValueError('Approved color ID1..6 must map to channel0..5')
            if item['bbox'] is None or item['visible_area_px']<16:
                counts['unobservable_or_under16px']+=1;continue
            counts['observable_eligible_gt']+=1
            box=np.asarray(item['bbox']);center=(box[:2]+box[2:])/2
            distances=np.column_stack([points[:,0]-box[0],points[:,1]-box[1],box[2]-points[:,0],box[3]-points[:,1]])/scales[:,None]
            valid=(distances>=0).all(axis=1)&(distances<7-1e-4).all(axis=1)
            ideal=max(8.,min(64.,2.**round(math.log2(max(float(np.max(box[2:]-box[:2]))/4,1)))))
            cost=np.linalg.norm((points-center)/scales[:,None],axis=1)+np.abs(np.log2(scales/ideal))
            candidates=np.flatnonzero(valid)
            candidates=candidates[np.argsort(cost[candidates],kind='stable')]
            if not len(candidates):
                counts['no_representable_prior']+=1;continue
            candidate_lists[gt_index]=[int(i) for i in candidates]
        # Maximum bipartite matching gives every representable GT a primary
        # prior before any GT receives optional extra positives. <=5 GTs.
        owner={};primary={}
        def match(gt_index,seen):
            for index in candidate_lists[gt_index]:
                if index in seen:continue
                seen.add(index)
                if index not in owner or match(owner[index],seen):
                    owner[index]=gt_index;primary[gt_index]=index;return True
            return False
        for gt_index in sorted(candidate_lists,key=lambda i:len(candidate_lists[i])):
            if not match(gt_index,set()):counts['assignment_conflicts']+=1
        occupied=set(owner)
        for gt_index in sorted(primary):
            item=targets[gt_index]
            selected=[primary[gt_index]]
            for index in candidate_lists[gt_index]:
                if len(selected)>=self.positives_per_gt:break
                if index not in occupied:selected.append(index);occupied.add(index)
            counts['assigned_gt']+=1
            if item['instance_id']==5:counts['assigned_fifth_instance']+=1
            for index in selected:occupied.add(index);pairs.append((index,gt_index))
        counts['positive_priors']=len(pairs)
        return pairs,dict(counts)

    def forward(self,raw,batch_targets):
        if raw.shape!=(len(batch_targets),len(self.points),50) or not torch.isfinite(raw).all():
            raise ValueError('Expected finite [B,N,50] declared task logits')
        zero=raw.sum()*0
        sums={key:zero for key in self.weights};assignments=[];denom=0;counts=Counter()
        color_positive=zero;color_background=zero
        for batch_index,targets in enumerate(batch_targets):
            pairs,one_count=self.assign(targets);counts.update(one_count);denom+=len(pairs)
            if one_count.get('assignment_conflicts',0):raise ValueError('Representable GT assignment conflict; no silent target loss')
            expected=torch.zeros_like(raw[batch_index,:,:6])
            instance_counts=Counter(gt for _,gt in pairs)
            point_weights=torch.zeros(len(self.points),dtype=raw.dtype,device=raw.device)
            for index,gt_index in pairs:expected[index,targets[gt_index]['color_channel']]=1.
            for index,gt_index in pairs:point_weights[index]=1/instance_counts[gt_index]
            logits=raw[batch_index,:,:6];prob=logits.sigmoid()
            correct_prob=prob*expected+(1-prob)*(1-expected)
            alpha=.25*expected+.75*(1-expected)
            focal=(F.binary_cross_entropy_with_logits(logits,expected,reduction='none')*alpha*(1-correct_prob).square()).sum(dim=1)
            color_positive=color_positive+(focal*point_weights).sum()
            color_background=color_background+focal[point_weights==0].sum()
            assignments.extend((batch_index,index,targets[gt],1/instance_counts[gt]) for index,gt in pairs)
        normalizer=max(denom,1)
        sums['color']=color_positive/max(1,counts['assigned_gt'])+color_background/normalizer
        if assignments:
            bi=torch.tensor([a[0] for a in assignments],device=raw.device)
            pi=torch.tensor([a[1] for a in assignments],device=raw.device)
            items=[a[2] for a in assignments];pred=raw[bi,pi];point=self.points[pi];stride=self.scale[pi,None]
            weights=torch.tensor([a[3] for a in assignments],dtype=raw.dtype,device=raw.device)
            def tensor(key):return torch.as_tensor([i[key] for i in items],dtype=raw.dtype,device=raw.device)
            def weighted_mean(per_prior,mask=None):
                if per_prior.ndim>1:per_prior=per_prior.reshape(len(per_prior),-1).mean(dim=1)
                if mask is None:return (per_prior*weights).sum()/weights.sum().clamp(min=1e-7)
                return (per_prior[mask]*weights[mask]).sum()/weights[mask].sum().clamp(min=1e-7) if mask.any() else zero
            box=tensor('bbox')
            distances=torch.cat([point-box[:,:2],box[:,2:]-point],dim=1)/stride
            dfl=pred[:,6:38].reshape(-1,4,8)
            low=distances.floor().long();high=low+1
            frac=distances-low
            sums['dfl']=weighted_mean(F.cross_entropy(dfl.reshape(-1,8),low.reshape(-1),reduction='none').reshape(-1,4)*(1-frac)+
                         F.cross_entropy(dfl.reshape(-1,8),high.reshape(-1),reduction='none').reshape(-1,4)*frac)
            decoded=(dfl.softmax(dim=2)*torch.arange(8,dtype=raw.dtype,device=raw.device)).sum(dim=2)*stride
            decoded=torch.cat([point-decoded[:,:2],point+decoded[:,2:]],dim=1)
            sums['bbox']=weighted_mean(giou_loss(decoded,box))
            def masked_smooth_l1(value,target,mask):
                return weighted_mean(F.smooth_l1_loss(value,target,reduction='none'),mask)
            valid=tensor('center_valid').bool();visible=tensor('center_visible').bool()
            geometry=tensor('geometry_valid').bool();orientation=tensor('orientation_valid').bool()&geometry
            sums['physical']=masked_smooth_l1(pred[:,38:40],(tensor('physical')-point)/stride,valid&visible)
            sums['conic']=masked_smooth_l1(pred[:,48:50],(tensor('conic')-point)/stride,geometry)
            axes=tensor('axes')/stride
            if (axes<=0).any():raise ValueError('Invalid positive ellipse semiaxis truth')
            sums['axes']=masked_smooth_l1(pred[:,42:44],axes.log(),geometry)
            angle_vector=tensor('orientation')
            if self.ellipse_shape_loss:
                # The anisotropic part of a log ellipse shape tensor is invariant
                # to swapping the two axes and rotating the angle by 90 degrees.
                predicted_gap=pred[:,42]-pred[:,43]
                true_gap=axes[:,0].log()-axes[:,1].log()
                unit=F.normalize(pred[:,44:46],dim=1,eps=1e-4)
                scale=.02
                sums['ellipse_anisotropy']=weighted_mean(
                    F.smooth_l1_loss(predicted_gap/scale,true_gap/scale,reduction='none'),geometry)
                sums['ellipse_log_shape']=weighted_mean(F.smooth_l1_loss(
                    predicted_gap[:,None]*unit/scale,true_gap[:,None]*angle_vector/scale,
                    reduction='none'),geometry)
            if orientation.any():
                vector=pred[orientation,44:46];norm=vector.norm(dim=1)
                normalized=vector/norm.clamp(min=1e-4)[:,None]
                angle_loss=1-(normalized*angle_vector[orientation]).sum(dim=1)+.1*(norm-1).abs()
                sums['orientation']=(angle_loss*weights[orientation]).sum()/weights[orientation].sum()
            sums['center_flags']=weighted_mean(F.binary_cross_entropy_with_logits(pred[:,40:42],torch.stack([valid,visible],dim=1).to(raw.dtype),reduction='none'))
            sums['geometry_confidence']=weighted_mean(F.binary_cross_entropy_with_logits(pred[:,46],geometry.to(raw.dtype),reduction='none'))
            sums['visibility_quality']=weighted_mean(F.binary_cross_entropy_with_logits(pred[:,47],tensor('visible_fraction'),reduction='none'))
        total=sum(sums[key]*weight for key,weight in self.weights.items())
        if not torch.isfinite(total):raise ValueError('Nonfinite multi-task loss')
        return total,{key:float(value.detach()) for key,value in sums.items()},dict(counts)
