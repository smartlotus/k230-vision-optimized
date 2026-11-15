"""One finite logical training experiment; cloud controller owns publication.

No implicit resume, no automatic experiment replay, and no test evaluation.
Checkpoints include optimizer/scaler/shuffle RNG and exact input identities.
"""
from collections import Counter
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cloud_store import atomic_json,sha
from evaluate_task import evaluate
from nanodet_topface import NanoDetTopFace,CHECKPOINT_SHA256,output_contract
from task_training import TopFaceDataset,TopFaceLoss,collate


def selection_score(metrics):
    errors=metrics['errors']
    center=errors.get('physical_center_visible_px',{}).get('median',20.)
    ellipse=errors.get('ellipse_iou',{}).get('median',0.)
    return metrics['task_mAP50_95']+.1*metrics['matched_color_accuracy']+.1*ellipse-.01*min(center,20.)


def sanity_gate(metrics):
    errors=metrics['errors'];denom=metrics['denominators']
    checks={'recall50_at_least95pct':metrics['class_agnostic_recall50']>=.95,
        'color_at_least95pct':metrics['matched_color_accuracy']>=.95,
        'task_map50_at_least95pct':metrics['task_mAP50']>=.95,
        'visible_center_median_at_most3_source_px':errors.get('physical_center_visible_px',{}).get('median',float('inf'))<=3.,
        'ellipse_iou_median_at_least75pct':errors.get('ellipse_iou',{}).get('median',0.)>=.75,
        'finite_ellipse_coverage_at_least90pct':denom.get('finite_ellipse_predictions',0)/max(1,denom.get('matched_geometry_valid_gt',0))>=.9,
        'empty_scene_present_and_has_no_false_positives':(
            denom.get('empty_observation_images',0)>0 and denom.get('empty_false_positive_detections',0)==0)}
    if denom.get('angle_valid_gt',0):
        checks['valid_angle_prediction_coverage_at_least90pct']=denom.get('angle_prediction_valid',0)/denom['angle_valid_gt']>=.9
        checks['angle_median_at_most10deg']=errors.get('orientation_error_deg',{}).get('median',float('inf'))<=10
    for count in (4,5):
        group=metrics['count_groups'].get('world_count_'+str(count),{})
        if group.get('images',0):checks[f'all_observable_{count}_targets_recalled']=group.get('all_observable_targets_recalled',0)==group['images']
    return {'pass':all(checks.values()),'checks':checks,'scope':'overfit train sanity only, not generalization'}


def pilot_gate(metrics):
    denom=metrics['denominators']
    checks={'recall50_at_least80pct':metrics['class_agnostic_recall50']>=.8,
        'color_at_least90pct':metrics['matched_color_accuracy']>=.9,
        'task_map50_at_least75pct':metrics['task_mAP50']>=.75,
        'visible_center_median_at_most5_source_px':metrics['errors'].get('physical_center_visible_px',{}).get('median',float('inf'))<=5.,
        'ellipse_iou_median_at_least60pct':metrics['errors'].get('ellipse_iou',{}).get('median',0.)>=.6,
        'finite_ellipse_coverage_at_least90pct':denom.get('finite_ellipse_predictions',0)/max(1,denom.get('matched_geometry_valid_gt',0))>=.9,
        'empty_fp_per_image_at_most_point1':denom.get('empty_false_positive_detections',0)/max(1,denom.get('empty_observation_images',0))<=.1}
    if denom.get('angle_valid_gt',0):
        checks['valid_angle_prediction_coverage_at_least90pct']=denom.get('angle_prediction_valid',0)/denom['angle_valid_gt']>=.9
        checks['angle_median_at_most15deg']=metrics['errors'].get('orientation_error_deg',{}).get('median',float('inf'))<=15
    for count in (4,5):
        group=metrics['count_groups'].get('world_count_'+str(count),{})
        if group.get('images',0):checks[f'all_observable_{count}_recapture_at_least80pct']=group.get('all_observable_targets_recalled',0)/group['images']>=.8
    return {'pass':all(checks.values()),'checks':checks,'scope':'registered synthetic Pilot gate, not real acceptance'}


def prefer_validation(kind,score,best_score,metrics):
    # Sanity is a conjunction, not the scalar used to rank general validation.
    # Always preserve the exact model that first satisfies the overfit gate.
    return score>best_score or (kind=='overfit' and sanity_gate(metrics)['pass'])


def atomic_checkpoint(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    torch.save(value,temporary);temporary.replace(path)


def choose_overfit(root,width=320,height=192,count=16,seed=20261114):
    dataset=TopFaceDataset(root,'train',width,height)
    loss=TopFaceLoss(width,height);candidates=[]
    for index,row in enumerate(dataset.rows):
        _,items,meta=dataset[index];_,counts=loss.assign(items)
        if counts.get('assignment_conflicts',0) or counts.get('no_representable_prior',0):continue
        if counts.get('observable_eligible_gt',0)!=len(items):continue
        features={'count'+str(len(items))}|{'color'+str(i['color_channel']) for i in items}
        features|={'count'+str(len(items))+'_color'+str(i['color_channel']) for i in items}
        if any(i['geometry_valid'] for i in items):features.add('geometry')
        if any(not i['center_visible'] for i in items):features.add('hidden_center')
        if any(i['truncated'] for i in items):features.add('boundary')
        candidates.append((row['scene'],features))
    if len(candidates)<count:raise ValueError('Not enough fully representable train scenes for sanity selection')
    rng=np.random.default_rng(seed);rng.shuffle(candidates);chosen=[];covered=set()
    for world_count in (0,1,2,3,4,5):
        feature='count'+str(world_count)
        eligible=[j for j,(_,features) in enumerate(candidates) if feature in features]
        if not eligible or count<6:
            raise ValueError('Overfit subset requires at least one scene of every object count')
        best=max(eligible,key=lambda j:len(candidates[j][1]-covered))
        scene,features=candidates.pop(best);chosen.append(scene);covered|=features
    while len(chosen)<count:
        best=max(range(len(candidates)),key=lambda j:len(candidates[j][1]-covered))
        scene,features=candidates.pop(best);chosen.append(scene);covered|=features
    if any('color'+str(i) not in covered for i in range(6)) or 'geometry' not in covered:
        raise ValueError('Overfit subset requires all six colors and geometry labels')
    return {'scenes':chosen,'covered_features':sorted(covered),'seed':seed,'selection':'train-only label-stratified, before any optimizer updates'}


def train(root,output,checkpoint,kind='pilot_pretrained',width=320,height=192,
          seed=20261114,batch_size=32,max_epochs=20,max_updates=None,deadline=None,selected_scenes=None,
          initial_checkpoint=None,initial_checkpoint_sha256=None,use_amp=False,
          formal_identity=None,resume_directory=None,resume_sha256=None,session_epochs=None,
          ellipse_shape_loss=False,background_bank=None):
    if kind not in ('overfit','pilot_pretrained','pilot_random_control','pilot_geometry_refine','formal_full'):
        raise ValueError('Unknown finite experiment')
    if background_bank is not None and kind!='formal_full':
        raise ValueError('The qualified real background policy belongs to the frozen formal experiment')
    if not torch.cuda.is_available():raise RuntimeError('Kaggle GPU training admission requires CUDA; no silent CPU full training')
    if max_epochs<1 or max_epochs>200 or (max_updates is not None and not 1<=max_updates<=1000):
        raise ValueError('Training budget outside registered limits')
    if kind=='overfit' and max_updates is None:raise ValueError('Overfit requires an explicit update budget')
    if formal_identity is not None and (kind!='formal_full' or not 100<=max_epochs<=200 or
            use_amp or max_updates is not None or selected_scenes is not None):
        raise ValueError('Formal resume supports the frozen FP32 epoch schedule')
    if resume_directory is not None and (not formal_identity or not resume_sha256 or initial_checkpoint is not None):
        raise ValueError('Resume requires the recorded formal identity and previous checkpoint hash')
    if session_epochs is not None and (not formal_identity or session_epochs<1):
        raise ValueError('A session epoch cap is only valid for formal training')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0');started=time.monotonic()
    deadline=deadline or started+7200
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    if formal_identity is not None:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise ValueError('Set the deterministic CUDA workspace before starting formal training')
        # CUDA bilinear interpolation backward in the upstream GhostPAN lacks
        # a deterministic kernel. Keep its trained semantics and report this
        # limitation instead of failing the first backward pass.
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    model=NanoDetTopFace().to(device)
    initialization=model.initialize_pretrained(checkpoint,CHECKPOINT_SHA256) if kind!='pilot_random_control' and resume_directory is None and initial_checkpoint is None else {'initialization':'random same architecture/new-head seed','optimizer_updates':0}
    if initial_checkpoint is not None:
        if kind not in ('formal_full','pilot_geometry_refine') or not initial_checkpoint_sha256 or sha(initial_checkpoint)!=initial_checkpoint_sha256:
            raise ValueError('Only a frozen selected Pilot checkpoint can initialize formal training')
        parent=torch.load(initial_checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(parent['model'],strict=True)
        initialization={'initialization':'selected frozen Pilot best','parent_sha256':initial_checkpoint_sha256,
            'parent_project_updates':parent['state']['optimizer_updates'],'new_optimizer_updates':0}
    # Overfit starts fresh from original features; Pilot does not inherit its
    # fitted sixteen-image weights. Both Pilot controls start with the same RNG.
    dataset=TopFaceDataset(root,'train',width,height,scene_names=selected_scenes,augment=kind!='overfit',seed=seed,
                          background_bank=background_bank)
    validation=TopFaceDataset(root,'train' if kind=='overfit' else 'val',width,height,
                              scene_names=selected_scenes if kind=='overfit' else None)
    criterion=TopFaceLoss(width,height,ellipse_shape_loss=ellipse_shape_loss).to(device)
    preflight=Counter()
    for index in range(len(dataset)):
        _,items,_=dataset[index];_,counts=criterion.assign(items);preflight.update(counts)
    if preflight.get('assignment_conflicts',0):raise ValueError('Representable target assignment conflict before training')
    missing=preflight.get('no_representable_prior',0)/max(1,preflight.get('observable_eligible_gt',0))
    if missing>.02:raise ValueError('More than2pct visible eligible GTs have no representable prior; inspect input scale')
    shuffle=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=min(batch_size,len(dataset)),shuffle=True,generator=shuffle,
                      num_workers=0,collate_fn=collate,pin_memory=True)
    val_loader=DataLoader(validation,batch_size=min(batch_size,len(validation)),shuffle=False,
                          num_workers=0,collate_fn=collate,pin_memory=True)
    lr={'overfit':.003,'pilot_geometry_refine':.0003}.get(kind,.001)
    if kind=='pilot_geometry_refine':
        predictions=set(model.head.gfl_cls.parameters())
        parameters=[{'params':[p for p in model.parameters() if p not in predictions],'lr_scale':.25},
                    {'params':list(model.head.gfl_cls.parameters()),'lr_scale':1.}]
    else:
        parameters=model.parameters()
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.001 if kind=='overfit' else .05)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    state={'schema':'topface-logical-training-v1.1','kind':kind,'status':'ADMITTED',
        'started_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'width':width,'height':height,
        'batch_size':loader.batch_size,'max_epochs':max_epochs,'max_updates':max_updates,
        'optimizer_updates':0,'epoch':0,'initialization':initialization,'preflight_assignment':dict(preflight),
        'training_scenes':[r['scene'] for r in dataset.rows], 'validation_scenes':[r['scene'] for r in validation.rows],
        'dataset_audit_sha256':dataset.audit_sha256,'learning_rate':lr,'loss_weights':criterion.weights,
        'mixed_precision':use_amp,
        'ellipse_shape_loss':ellipse_shape_loss,
        'real_background_augmentation':dataset.backgrounds.identity if dataset.backgrounds is not None else None,
        'validation_selection':('Prefer full Pilot gate, then rank by validation score' if kind=='pilot_geometry_refine'
                                else 'mAP50_95 +.1color +.1medianEllipseIoU -.01min(medianVisibleCenterPx,20)'),
        'input_contract':output_contract(width,height),'test_evaluation':False,'real_evaluation':False,
        'automatic_replay':False,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
        'epoch_records':[]}
    end_epoch=max_epochs
    first_epoch=0
    best_score=-float('inf');best_metrics=None;steps=0;timings=[];stale_epochs=0
    target_updates=max_updates if max_updates is not None else max_epochs*len(loader)
    warmup=min(100,max(10,target_updates//10))
    schedule={'target_updates':target_updates,'warmup_updates':warmup,'epoch_batches':len(loader)}
    runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
             'cudnn':torch.backends.cudnn.version(),'python':platform.python_version(),
             'numpy':np.__version__,'opencv':cv2.__version__}
    if formal_identity is not None:
        state.update(formal_identity=formal_identity,schedule=schedule,runtime=runtime,
                     checkpoint_boundary='epoch',completed_epochs=0,
                     reproducibility={'data_order_rng_and_optimizer_restored':True,
                         'bitwise_cuda_resume_verified':False,
                         'deterministic_algorithms':'warn_only',
                         'limitation':'Upstream bilinear GhostPAN CUDA backward may be nondeterministic.'})
    if resume_directory is not None:
        resume_directory=Path(resume_directory)
        last=resume_directory/'last.pt'
        if sha(last)!=resume_sha256:raise ValueError('Formal checkpoint digest differs')
        previous=torch.load(last,map_location='cpu',weights_only=True)
        recorded=previous['state']
        for key in ('formal_identity','schedule','runtime','dataset_audit_sha256','training_scenes',
                    'validation_scenes','width','height','batch_size','max_epochs','seed','learning_rate',
                    'loss_weights','mixed_precision','real_background_augmentation'):
            if recorded[key]!=state[key]:raise ValueError(f'Formal checkpoint differs in {key}')
        first_epoch=recorded['completed_epochs']
        steps=recorded['optimizer_updates']
        if not 0<first_epoch<max_epochs or steps!=first_epoch*len(loader):
            raise ValueError('Formal checkpoint is outside a completed epoch boundary')
        best_file=resume_directory/'best.pt'
        if sha(best_file)!=recorded['best_checkpoint_sha256']:
            raise ValueError('Previous best checkpoint identity differs')
        shutil.copy2(best_file,output/'best.pt')
        best_metrics=recorded['best_metrics']
        best_score=recorded['best_validation_score']
        stale_epochs=recorded['stale_epochs']
        atomic_json(output/'BEST_METRICS.json',best_metrics)
        for key in ('initialization','epoch','completed_epochs','optimizer_updates','best_epoch',
                    'best_checkpoint_sha256','epoch_records'):
            state[key]=recorded[key]
        state['resumed_from_sha256']=resume_sha256
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        shuffle.set_state(previous['shuffle_rng'])
        torch.set_rng_state(previous['torch_rng'])
        torch.cuda.set_rng_state_all(previous['cuda_rng'])
        random.setstate(previous['python_rng'])
        numpy_rng=previous['numpy_rng']
        np.random.set_state((numpy_rng[0],np.asarray(numpy_rng[1],dtype=np.uint32),*numpy_rng[2:]))
    if session_epochs is not None:
        end_epoch=min(max_epochs,first_epoch+session_epochs)
    atomic_json(output/'CONFIG.json',state)
    writer=None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(output/'tensorboard'));state['tensorboard']='ENABLED'
    except ImportError:state['tensorboard']='UNAVAILABLE_JSON_METRICS_PRESERVED'
    def save(name):
        value={'schema':'topface-training-checkpoint-v1.1',
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'shuffle_rng':shuffle.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,
            'source_checkpoint_sha256':CHECKPOINT_SHA256}
        if formal_identity is not None:
            numpy_rng=np.random.get_state()
            value.update(python_rng=random.getstate(),numpy_rng=(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]))
        atomic_checkpoint(output/name,value)
    try:
        if first_epoch==0:
            baseline=evaluate(model,val_loader,device,include_slices=kind=='pilot_geometry_refine',
                              angle_diagnostics=kind=='pilot_geometry_refine')
            atomic_json(output/'BASELINE.json',baseline)
        stop=False
        epochs_to_run=math.ceil(max_updates/len(loader)) if kind=='overfit' else end_epoch
        state['effective_epoch_limit']=epochs_to_run
        for epoch in range(first_epoch,epochs_to_run):
            if formal_identity is not None and state['epoch_records']:
                elapsed=[record.get('epoch_wall_seconds',0.) for record in state['epoch_records'][-3:]]
                if time.monotonic()+max(120.,1.5*max(elapsed))+120>=deadline:
                    state['status']='FORMAL_SESSION_BOUNDARY' if state['completed_epochs']>first_epoch else 'FORMAL_NO_EPOCH_PROGRESS'
                    break
            epoch_started=time.monotonic()
            dataset.epoch=epoch;model.train();losses=[];assignment_counts=Counter()
            for images,targets,_ in loader:
                if time.monotonic()+120>=deadline:
                    state['status']='BUDGET_CHECKPOINT_STOP';stop=True;break
                images=images.to(device,non_blocking=True)
                progress=steps/max(1,target_updates)
                factor=(.1+.9*steps/warmup) if steps<warmup else .05+.95*.5*(1+math.cos(math.pi*min(progress,1)))
                for group in optimizer.param_groups:group['lr']=lr*factor*group.get('lr_scale',1.)
                optimizer.zero_grad(set_to_none=True);tick=time.perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=use_amp):raw=model(images)
                loss,parts,counts=criterion(raw.float(),targets)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<old_scale:
                    state['amp_skipped_steps']=state.get('amp_skipped_steps',0)+1
                    if state['amp_skipped_steps']>5:raise RuntimeError('Repeated AMP overflow; preserve failure instead of hiding it')
                    continue
                steps+=1;state['optimizer_updates']=steps;losses.append(float(loss.detach()))
                assignment_counts.update(counts);torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
                if writer:
                    writer.add_scalar('train/total_loss',losses[-1],steps)
                    for name,value in parts.items():writer.add_scalar('loss/'+name,value,steps)
                if steps%100==0:
                    state['epoch']=epoch+1
                    if formal_identity is None:save('last.pt')
                    atomic_json(output/'PROGRESS.json',state)
                    print('TOPFACE_TRAIN_PROGRESS',json.dumps({'kind':kind,'updates':steps,'epoch':epoch+1,'loss':losses[-1]}),flush=True)
                if max_updates is not None and steps>=max_updates:stop=True;break
            state['epoch']=epoch+1
            # Reserve publication budget. Checkpoint stops do not claim val pass.
            if state.get('status')=='BUDGET_CHECKPOINT_STOP':
                if formal_identity is None:save('last.pt')
                else:
                    state['uncommitted_optimizer_updates']=steps-state['completed_epochs']*len(loader)
                    state['status']='FORMAL_BUDGET_STOP_MID_EPOCH'
                break
            if kind=='overfit' and steps%25!=0 and not stop:continue
            validation_started=time.monotonic()
            metrics=evaluate(model,val_loader,device,include_slices=kind=='pilot_geometry_refine',
                             angle_diagnostics=kind=='pilot_geometry_refine')
            score=selection_score(metrics)
            if kind=='pilot_geometry_refine' and pilot_gate(metrics)['pass']:
                score+=2.
            validation_seconds=time.monotonic()-validation_started
            state['epoch_records'].append({'epoch':epoch+1,'updates':steps,'loss_mean':float(np.mean(losses)) if losses else None,
                'validation_score':score,'task_mAP50':metrics['task_mAP50'],'task_mAP50_95':metrics['task_mAP50_95'],
                'recall50':metrics['class_agnostic_recall50'],'matched_color_accuracy':metrics['matched_color_accuracy'],
                'training_wall_seconds':validation_started-epoch_started,'validation_wall_seconds':validation_seconds,
                'assignment_counts':dict(assignment_counts)})
            atomic_json(output/f'validation-epoch{epoch+1:03d}.json',metrics)
            if writer:writer.add_scalar('val/task_mAP50',metrics['task_mAP50'],steps)
            sanity_pass=kind=='overfit' and sanity_gate(metrics)['pass']
            if prefer_validation(kind,score,best_score,metrics):
                best_score=score;best_metrics=metrics;stale_epochs=0;state['best_epoch']=epoch+1
                save('best.pt');atomic_json(output/'BEST_METRICS.json',metrics)
                if formal_identity is not None:state['best_checkpoint_sha256']=sha(output/'best.pt')
            else:stale_epochs+=1
            if formal_identity is not None:
                state.update(completed_epochs=epoch+1,best_validation_score=best_score,
                             best_metrics=best_metrics,stale_epochs=stale_epochs)
            state['epoch_records'][-1]['epoch_wall_seconds']=time.monotonic()-epoch_started
            save('last.pt');atomic_json(output/'PROGRESS.json',state)
            print('TOPFACE_TRAIN_EPOCH',json.dumps(state['epoch_records'][-1]),flush=True)
            if sanity_pass:stop=True
            if kind=='formal_full' and epoch+1>=100 and stale_epochs>=20:stop=True
            if stop:break
        if state['status']=='ADMITTED':
            state['status']=('FORMAL_SESSION_BOUNDARY' if formal_identity is not None
                             and state['completed_epochs']<max_epochs and not stop
                             else 'FINITE_TRAINING_COMPLETE')
        state['best_validation_score']=best_score if math.isfinite(best_score) else None
        state['gate']=(sanity_gate(best_metrics) if kind=='overfit' else pilot_gate(best_metrics)) if best_metrics else {'pass':False,'reason':'NO_COMPLETED_VALIDATION'}
        state['step_seconds']={'n':len(timings),'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95))} if timings else {'n':0}
        state['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_CHECKPOINTS',error_type=type(error).__name__)
        if steps and formal_identity is None:save('last.pt')
        raise
    finally:
        if writer:writer.close()
        state['elapsed_seconds']=time.monotonic()-started
        state['checkpoints']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.glob('*.pt'))]
        atomic_json(output/'RESULT.json',state)
    return state
