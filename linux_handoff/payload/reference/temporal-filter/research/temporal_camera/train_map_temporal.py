"""Kaggle CameraXY reuse experiment; frozen visual/map parent, causal adapters."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import audit,digest
from embedded_filter import SoftGridFilter
from map_models import MapCameraXYNet
from competition_paths import seed_start_belief
from camera_preprocess import rgb_uint8


def put(path,value):
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def pixels(path):
    return rgb_uint8(path).astype(np.float32)/255


def stats(error):
    x=np.asarray(error)
    return {'n':len(x),'mean_mm':float(x.mean()),'median_mm':float(np.median(x)),
            'p95_mm':float(np.quantile(x,.95)),'max_mm':float(x.max()),
            'within50_rate':float((x<=50).mean()),'within200_rate':float((x<=200).mean())}


def soft_target(xy,centers):
    return torch.softmax(-((xy[:,None,:]-centers[None,:,:])**2).sum(-1)/(2*38**2),dim=1)


def grid_prob(prob,centers,grid):
    index=np.floor((centers+1200)/(2400/grid)).astype(int).clip(0,grid-1)
    return np.bincount(index[:,1]*grid+index[:,0],weights=prob,minlength=grid*grid).reshape(grid,grid)


def evaluate(model,features,blurred_features,rows,groups,centers,temperature,competition=False):
    # All states and stress branches share observation indices; drops never
    # remain in the neural window. Blackouts replace actual temporal inputs.
    result={};details=[]
    test=[ids for ids in groups.values() if rows[ids[0]]['split']=='test']
    row_xy=np.array([[r['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')] for r in rows])
    modes=('global','competition_two_zone') if competition else ('global','start_region_300mm','oracle_start_cell','wrong_history')
    with torch.inference_mode():
        blackout=model.image_embedding(torch.zeros(1,3,96,160,device=features.device))[0]
    for condition in ('clean','frame_drop','variable_dt','blur','visual_failure','relocalization'):
        states={}
        for initial in modes:
            for grid in (24,48):
                raw_error=[];raw_grid_error=[];filtered_error=[];jumps=0;recoveries=0;lost=0;latencies=[]
                by_scenario=defaultdict(list);start_checks=[];trajectory_recoveries=[]
                for tr_index,ids in enumerate(test):
                    filt=SoftGridFilter([[True]*grid for _ in range(grid)],cell_mm=2400/grid,
                                        memory_seconds=2.,global_floor=.01)
                    xy0=row_xy[ids[0]];last=None
                    if initial=='competition_two_zone':
                        seed_start_belief(filt)
                    elif initial!='global':
                        if initial=='wrong_history':xy0=-xy0
                        yy,xx=np.mgrid[:grid,:grid];cell=2400/grid
                        mx=(xx+.5)*cell-1200;my=(yy+.5)*cell-1200
                        if initial=='start_region_300mm':p=(abs(mx-xy0[0])<=150)&(abs(my-xy0[1])<=150)
                        else:
                            p=np.zeros((grid,grid),bool)
                            col,row=np.floor((xy0+1200)/cell).astype(int).clip(0,grid-1);p[row,col]=True
                        filt.belief[:]=(p/p.sum()).ravel().tolist();filt.state='TRACKING';filt.last_ts=rows[ids[0]]['timestamp_s']
                    visible=[]
                    recovery_error=[]
                    for frame,i in enumerate(ids):
                        if condition=='frame_drop' and frame in (6,7,8,9,10,11):continue
                        target=i
                        if condition=='relocalization' and frame>=15:
                            other=max(test,key=lambda clip:np.linalg.norm(row_xy[clip[15]]-row_xy[ids[15]])) if competition else test[(tr_index+len(test)//2)%len(test)]
                            target=other[frame]
                        healthy=not ((condition=='visual_failure' and 6<=frame<=11) or
                                     (competition and condition=='relocalization' and 9<=frame<=14))
                        z=features[target] if healthy else blackout
                        if condition=='blur':z=blurred_features[target]
                        visible.append(z)
                        window=visible[-model.window:];window=[window[0]]*(model.window-len(window))+window
                        with torch.inference_mode():logits=model.score_features(torch.stack(window)[None])[0].cpu().numpy()
                        logits=(logits-logits.max())/temperature;prob=np.exp(logits);prob/=prob.sum()
                        anchor=int(prob.argmax());pred=centers[anchor];truth=row_xy[target]
                        belief=grid_prob(prob,centers,grid)
                        stamp=rows[i]['timestamp_s']*(3 if condition=='variable_dt' else 1)
                        tic=time.perf_counter();r=filt.update(belief.ravel().tolist(),stamp,visual_quality=1. if healthy else 0.)
                        latencies.append((time.perf_counter()-tic)*1000)
                        filtered=np.array([r['camera_x_mm'],r['camera_y_mm']])
                        raw_index=int(belief.argmax());raw_grid=np.array([(raw_index%grid+.5)*2400/grid-1200,(raw_index//grid+.5)*2400/grid-1200])
                        raw_error.append(float(np.linalg.norm(pred-truth)));raw_grid_error.append(float(np.linalg.norm(raw_grid-truth)))
                        filtered_error.append(float(np.linalg.norm(filtered-truth)))
                        by_scenario[rows[i]['scenario']].append(filtered_error[-1])
                        if competition and frame==0:
                            start_checks.append(bool(filtered[1]*truth[1]>0))
                        recovery_frame=15 if condition=='relocalization' else 12
                        if condition in ('visual_failure','relocalization') and frame>=recovery_frame:
                            recovery_error.append(filtered_error[-1])
                        if last is not None and np.linalg.norm(filtered-last)>500:jumps+=1
                        last=filtered;recoveries+=int(r['relocalized']);lost+=int(r['state']=='LOST')
                        details.append({'sample_id':rows[target]['sample_id'],'trajectory_id':rows[i]['trajectory_id'],
                            'frame':frame,'condition':condition,'initial':initial,'grid':grid,
                            'raw_error_mm':raw_error[-1],'filtered_error_mm':filtered_error[-1],
                            'posterior_mass':r['confidence'],'state':r['state'],'sensor_healthy':healthy})
                    if recovery_error:
                        hit=next((j for j in range(len(recovery_error)-2)
                                  if max(recovery_error[j:j+3])<=200),None)
                        trajectory_recoveries.append(hit)
                states[f'{initial}/grid{grid}']={'raw_anchor':stats(raw_error),'raw_grid':stats(raw_grid_error),
                    'filtered_grid':stats(filtered_error),'jumps_over500':jumps,'lost_frames':lost,
                    'global_recovery_events':recoveries,'host_cpu_filter_p95_ms':float(np.quantile(latencies,.95)),
                    'initialization_provenance':{'global':'NO_PRIOR','competition_two_zone':'TWO_OFFICIAL_AREAS_EQUAL_WEIGHT_WITH_MARGIN'}.get(initial,'SYNTHETIC_ORACLE'),
                    'by_scenario':{key:stats(value) for key,value in by_scenario.items()},
                    'initial_zone_correct':sum(start_checks) if competition else None,
                    'initial_zone_trials':len(start_checks),
                    'recovered_with_3_consecutive_within200mm':sum(x is not None for x in trajectory_recoveries),
                    'recovery_trajectory_trials':len(trajectory_recoveries),
                    'motion':'UNKNOWN; no hard exclusion','confidence':'UNCALIBRATED_POSTERIOR_MASS'}
            result[condition]=states
    return result,details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--run-id',required=True);ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--max-seconds',type=int,default=2200);ap.add_argument('--encoder');ap.add_argument('--learning')
    ap.add_argument('--all-endpoints',action='store_true');ap.add_argument('--settings',default='cnn1,gru4,gru8,tcn4,tcn8')
    args=ap.parse_args();start=time.monotonic();deadline=start+args.max_seconds
    if not torch.cuda.is_available() or not Path('/kaggle/working').is_dir():raise RuntimeError('Kaggle GPU required')
    torch.set_num_threads(2);torch.manual_seed(117);np.random.seed(117);random.seed(117)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (args.data/'manifest.jsonl').read_text().splitlines()];qa,_=audit(rows,args.data)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['trajectory_id']].append(i)
    parent_path=args.data.parent/'parent.pt';parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    expected='c16e0ddf9f458fb7dfbcda5902f60c09b4a9e0426bb55a7e9b3ebea3d486d989'
    identity=json.loads((args.data/'RENDER_IDENTITY.json').read_text())
    competition=identity.get('start_policy')=='both zones, opposite goal per user'
    if competition:
        reference=json.loads((args.data.parent/'REFERENCE.json').read_text())
        qa_file=json.loads((args.data/'QA.json').read_text())
        if reference['parent_sha256']!=expected or reference['field_mask_sha256']!=qa_file['field_mask_sha256']:
            raise ValueError('Corrected map reference lineage differs')
        expected=reference['checkpoint_sha256']
    if digest(parent_path)!=expected:raise ValueError('Verified parent changed')
    adaptation_path=args.data.parent/'MAP_ADAPTATION.json'
    adaptation=json.loads(adaptation_path.read_text()) if adaptation_path.is_file() else {}
    adapt_rgb=adaptation.get('rgb_projection_learning_rate')
    if adapt_rgb is not None and (not competition or adapt_rgb!=.0003):
        raise ValueError('Unexpected RGB adaptation recipe')
    centers=np.asarray(parent['centers_mm'],np.float32);ct=torch.from_numpy(centers).cuda()
    baseline=MapCameraXYNet(parent).eval().cuda();images=torch.from_numpy(np.stack([pixels(args.data/r['image_path']) for r in rows])).cuda()
    with torch.inference_mode():
        encode=baseline.cache_spatial_features if adapt_rgb is not None else baseline.image_embedding
        cached=torch.cat([encode(x) for x in images.split(32)])
        cached_blur=torch.cat([encode(F.avg_pool2d(x,9,stride=1,padding=4)) for x in images.split(32)])
    cached=cached.clone();cached_blur=cached_blur.clone()
    del images,baseline,encode
    xy=torch.tensor([[r['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')] for r in rows],device='cuda')
    target=soft_target(xy,ct);split={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    lookup={}
    for ids in groups.values():
        for offset,i in enumerate(ids):
            history=ids[max(0,offset-7):offset+1];lookup[i]=[history[0]]*(8-len(history))+history
    available={'cnn1':('cnn',1),'gru4':('gru',4),'gru8':('gru',8),'tcn4':('tcn',4),'tcn8':('tcn',8)}
    summaries={};budgets={}
    for name in args.settings.split(','):
        if time.monotonic()>deadline:raise TimeoutError('Frozen map reuse budget exhausted')
        route,t=available[name];torch.manual_seed(117)
        model=MapCameraXYNet(parent,route,t,single_frame_adjustment=competition).cuda().eval();folder=out/name;folder.mkdir()
        windows=torch.tensor([lookup[i][-t:] for i in range(len(rows))],device='cuda')
        temporal_params=[p for p in model.parameters() if p.requires_grad]
        optimizer=None
        if temporal_params:
            groups_and_rates=[{'params':temporal_params,'lr':.001}]
            if adapt_rgb is not None:
                model.enable_rgb_projection_training()
                groups_and_rates.append({'params':[p for p in model.rgb.parameters() if p.requires_grad],
                                         'lr':adapt_rgb})
            optimizer=torch.optim.AdamW(groups_and_rates,weight_decay=.002)
        def logits_at(indices):
            values=cached[windows[indices]]
            if adapt_rgb is not None:
                values=model.project_spatial_features(values)
            return model.score_features(values)
        history=[];best=float('inf');best_state=None;updates=0;rng=np.random.default_rng(117)
        epochs=0 if optimizer is None else args.epochs
        for epoch in range(epochs+1):
            if time.monotonic()>deadline:raise TimeoutError('Frozen map reuse budget exhausted')
            total=0.
            if epoch:
                model.train()
                for ids in np.array_split(rng.permutation(split['train']),16):
                    logits=logits_at(ids)
                    loss=-(target[ids]*F.log_softmax(logits,1)).sum(1).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    total+=float(loss.detach())*len(ids);updates+=1
            model.eval()
            with torch.inference_mode():
                val=torch.cat([logits_at(ids) for ids in np.array_split(split['val'],8)])
                nll=float(-(target[split['val']]*F.log_softmax(val,1)).sum(1).mean())
            record={'epoch':epoch,'optimizer_updates':updates,'validation_xy_nll':nll,'train_xy_nll':total/len(split['train']) if epoch else None}
            history.append(record);put(folder/'history.json',history);print(json.dumps({'route':name,**record}),flush=True)
            if nll<best:
                best=nll;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=epoch
                torch.save({'state':best_state,'route':route,'window':t,'parent_sha256':expected,
                    'centers_mm':parent['centers_mm'],'reference_views':parent['reference_views'],
                    'dataset_manifest_sha256':digest(args.data/'manifest.jsonl'),'selected_epoch':selected,
                    'single_frame_adjustment':model.single_frame_adjustment,
                    'targets':['camera_x_mm','camera_y_mm']},folder/'best.pt')
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict() if optimizer else None,
                'epoch':epoch,'optimizer_updates':updates,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'parent_sha256':expected},folder/'last.pt')
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():
            val=torch.cat([logits_at(ids) for ids in np.array_split(split['val'],8)])
            temperatures=(.5,1.,2.,4.,8.)
            temperature=min(temperatures,key=lambda temp:float(-(target[split['val']]*F.log_softmax(val/temp,1)).sum(1).mean()))
            features=model.project_spatial_features(cached) if adapt_rgb is not None else cached
            blurred_features=model.project_spatial_features(cached_blur) if adapt_rgb is not None else cached_blur
        stress,details=evaluate(model,features,blurred_features,rows,groups,centers,temperature,competition)
        # Keep per-frame evidence in the cloud archive, bounded summaries on host.
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in details))
        cpu=model.cpu().eval();dummy=torch.zeros(1,3*t,96,160)
        torch.onnx.export(cpu,dummy,str(folder/'camera_xy.onnx'),input_names=['rgb_window'],output_names=['location_logits'],opset_version=13,dynamo=False)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        session=ort.InferenceSession(str(folder/'camera_xy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        sample=np.concatenate([pixels(args.data/rows[i]['image_path']) for i in lookup[split['test'][0]][-t:]],axis=0)[None]
        with torch.inference_mode():ref=cpu(torch.from_numpy(sample)).numpy()
        actual=session.run(None,{'rgb_window':sample})[0];np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-3)
        summary={'route':route,'window':t,'parent_sha256':expected,'encoder_frozen':adapt_rgb is None,'reference_frozen':True,
            'rgb_projection_adaptation_lr':adapt_rgb,
            'selected_epoch':selected,'epochs_completed':epochs,'optimizer_updates':updates,
            'temperature':temperature,'stress':stress,'onnx_ort_max_abs':float(abs(actual-ref).max()),
            'onnx_sha256':digest(folder/'camera_xy.onnx'),'best_sha256':digest(folder/'best.pt'),
            'parameters':sum(p.numel() for p in cpu.parameters()),'native_k230':'NOT RUN','real_confidence':'UNKNOWN'}
        put(folder/'METRICS.json',summary);summaries[name]=summary;put(out/'COMPARISON.json',summaries)
        model.cuda()  # cached feature tensors and later routes remain on GPU
    gate={'status':'REPORT_ONLY_REQUIRES_COMPETITION_QUALIFICATION','qa':qa,
        'full_generation_authorized_by_gate':False,
        'parent_training_reused':'10000 positions,18epochs4500updates','elapsed_seconds':time.monotonic()-start,
        'native_board':'NOT RUN','global_confidence_calibration':'NOT QUALIFIED'}
    if competition:
        evidence=summaries['cnn1']['stress']['clean']['competition_two_zone/grid48']
        recovery=summaries['cnn1']['stress']['relocalization']['competition_two_zone/grid48']
        errors=evidence['filtered_grid']
        full=(errors['median_mm']<=200 and errors['p95_mm']<=500 and
              evidence['initial_zone_correct']/evidence['initial_zone_trials']>=.8 and
              recovery['recovered_with_3_consecutive_within200mm']/recovery['recovery_trajectory_trials']>=.5)
        gate.update(status='PASS' if full else 'NEEDS_IMPROVEMENT',map_sha256=identity['geometry_sha256'],
                    resolution=identity['resolution'],dataset_manifest_sha256=digest(args.data/'manifest.jsonl'),
                    full_generation_authorized_by_gate=full,
                    criterion={'median_mm_max':200,'p95_mm_max':500,'correct_initial_zone_fraction_min':.8,
                               'recovered_trajectory_fraction_min':.5},
                    measured={'clean':evidence,'relocalization':recovery})
    put(out/'PILOT_GATE.json',gate)


if __name__=='__main__':main()
