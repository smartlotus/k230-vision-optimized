"""Bounded Kaggle pilot: same targets, trajectory splits, five model settings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import audit, build_sequences, digest
from models import CameraXYNet
from filter import TemporalGridFilter


def save_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def prediction(logits, offsets, temperature=1.):
    p=torch.softmax(logits.flatten(1)/temperature,1)
    idx=p.argmax(1)
    xy=torch.stack((idx%24,idx//24),1)*100+offsets*100-1200
    return xy,p


def metric(errors):
    a=np.asarray(errors)
    if not len(a):return {'n':0,'status':'NOT RUN'}
    return {'n':len(a),'mean_mm':float(a.mean()),'median_mm':float(np.median(a)),
            'p95_mm':float(np.quantile(a,.95)),'max_mm':float(a.max())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--max-seconds',type=int,default=2400)
    ap.add_argument('--encoder',choices=('baseline','spatial'),default='baseline')
    ap.add_argument('--all-endpoints',action='store_true')
    ap.add_argument('--learning',choices=('ce','xy_smooth'),default='ce')
    ap.add_argument('--settings',default='cnn1,gru4,gru8,tcn4,tcn8')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+args.max_seconds
    if not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required; no local CPU training fallback')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    rows=[json.loads(x) for x in (args.data/'manifest.jsonl').read_text().splitlines()]
    qa,groups=audit(rows,args.data)
    seq=build_sequences(rows,8)
    if args.all_endpoints:
        seq=[]
        for tr,frames in groups.items():
            for i,current in enumerate(frames):
                clip=frames[max(0,i-7):i+1];clip=[clip[0]]*(8-len(clip))+clip
                seq.append({'sequence_id':tr+f'-warm-T8-{i:03d}','sample_id':current['sample_id'],
                            'frames':[r['image_path'] for r in clip],'split':current['split'],
                            'camera_xy_target':current['camera_xy_target']})
    bypath={r['image_path']:r for r in rows}
    images={r['image_path']:torch.from_numpy(np.asarray(Image.open(args.data/r['image_path']).convert('RGB')).copy()).permute(2,0,1).float()/255 for r in rows}
    # All five settings have exactly the same training/validation/test endpoints.
    endpoint_sha=hashlib.sha256(json.dumps([s['sample_id'] for s in seq]).encode()).hexdigest()
    if args.encoder=='spatial':
        # Separate train-only learning qualification; discard this model before
        # fair model training. Never use validation/test to fit this probe.
        from models_spatial import CameraXYNet as SpatialNet
        probe_seq=[s for s in seq if s['split']=='train'][::19][:32]
        px=torch.stack([images[s['frames'][-1]] for s in probe_seq]).cuda()
        py=torch.tensor([[s['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')] for s in probe_seq],device='cuda')
        pc=((py+1200)//100).long().clamp(0,23);pl=pc[:,1]*24+pc[:,0]
        torch.manual_seed(117);probe=SpatialNet('cnn',1).cuda().train()
        optimizer=torch.optim.AdamW(probe.parameters(),lr=.003,weight_decay=.0001)
        for step in range(160):
            logits,_=probe(px);loss=nn.functional.cross_entropy(logits.flatten(1),pl)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.inference_mode():
            logits,_=probe(px);accuracy=float((logits.flatten(1).argmax(1)==pl).float().mean())
            classes=int(logits.flatten(1).argmax(1).unique().numel())
            probe.eval();eval_logits,_=probe(px)
            eval_accuracy=float((eval_logits.flatten(1).argmax(1)==pl).float().mean())
        qualification={'status':'PASS' if accuracy>=.8 and float(loss)<.7 else 'FAILED',
            'scope':'train-only learning probe, weights discarded','optimizer_updates':160,
            'samples':len(probe_seq),'sample_ids':[s['sample_id'] for s in probe_seq],
            'train_ce':float(loss),'train_grid_accuracy':accuracy,'predicted_classes':classes}
        qualification['eval_same_train_samples_grid_accuracy']=eval_accuracy
        save_json(out/'TINY_LEARNING_GATE.json',qualification)
        del probe,optimizer
        if qualification['status']!='PASS':raise ValueError('Spatial encoder tiny learning gate failed')
    requested=args.settings.split(',')
    available={'cnn1':('cnn',1),'gru4':('gru',4),'gru8':('gru',8),'tcn4':('tcn',4),'tcn8':('tcn',8)}
    if len(set(requested))!=len(requested) or any(n not in available for n in requested) or 'cnn1' not in requested:
        raise ValueError('Invalid frozen settings')
    configs=[available[n] for n in requested]
    summaries={}
    for route,t in configs:
        if time.monotonic()>deadline:raise TimeoutError('Frozen pilot wall budget exceeded')
        name=f'{route}{t}';folder=out/name;folder.mkdir()
        random.seed(117);np.random.seed(117);torch.manual_seed(117);torch.cuda.manual_seed_all(117)
        if args.encoder=='spatial':
            from models_spatial import CameraXYNet as SpatialNet
            model=SpatialNet(route,t).cuda()
        else:model=CameraXYNet(route,t).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
        splits={}
        for split in ('train','val','test'):
            clips=[s for s in seq if s['split']==split]
            x=torch.stack([torch.cat([images[p] for p in s['frames'][-t:]],0) for s in clips])
            xy=torch.tensor([[s['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')] for s in clips])
            cell=torch.floor((xy+1200)/100).long().clamp(0,23)
            label=cell[:,1]*24+cell[:,0];offset=(xy+1200)/100-cell
            splits[split]=(clips,x,xy,label,offset)
        train=splits['train'];gen=torch.Generator().manual_seed(117)
        loader=DataLoader(TensorDataset(train[1],train[3],train[4]),batch_size=32,shuffle=True,generator=gen)
        augment_rng=torch.Generator(device='cuda').manual_seed(117)
        best=float('inf');stale=0;updates=0;history=[]
        prior=args.data.parent/'prior_cnn1'
        reuse=route=='cnn' and prior.exists()
        if reuse:
            original=torch.load(prior/'best.pt',map_location='cpu',weights_only=False)
            if (original['route'],original['window'],original['endpoint_sha256'],original['dataset_manifest_sha256']) != (
                    route,t,endpoint_sha,digest(args.data/'manifest.jsonl')):
                raise ValueError('Original CNN experiment identity differs; no replay allowed')
            for filename in ('best.pt','last.pt','history.json'):
                shutil.copy2(prior/filename,folder/filename)
            history=json.loads((prior/'history.json').read_text())
            updates=history[-1]['updates']
        for epoch in ([] if reuse else range(1,args.epochs+1)):
            if time.monotonic()>deadline:raise TimeoutError('Frozen pilot wall budget exceeded')
            model.train();loss_sum=0
            for x,cls,offset in loader:
                x,cls,offset=x.cuda(),cls.cuda(),offset.cuda()
                if args.learning=='xy_smooth':
                    from learning import augment_window
                    x=augment_window(x,augment_rng)
                logits,refine=model(x)
                if args.learning=='xy_smooth':
                    from learning import xy_heatmap_loss
                    cell=torch.stack((cls%24,cls//24),1)
                    xy=(cell+offset)*100-1200
                    loss=xy_heatmap_loss(logits,refine,xy,cell)
                else:loss=nn.functional.cross_entropy(logits.flatten(1),cls)+nn.functional.smooth_l1_loss(refine,offset)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                updates+=1;loss_sum+=loss.item()*len(x)
            scheduler.step();model.eval()
            with torch.inference_mode():
                val_logits,val_offset=model(splits['val'][1].cuda())
                val_error=torch.linalg.vector_norm(prediction(val_logits,val_offset)[0].cpu()-splits['val'][2],dim=1)
                score=float(val_error.mean())
            history.append({'epoch':epoch,'updates':updates,'train_loss':loss_sum/len(train[1]),'val_mean_mm':score})
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                   'epoch':epoch,'optimizer_updates':updates,'random':random.getstate(),'numpy_rng':np.random.get_state(),
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':gen.get_state(),
                   'route':route,'window':t,'endpoint_sha256':endpoint_sha,
                   'dataset_manifest_sha256':digest(args.data/'manifest.jsonl'),
                   'source_sha256':digest(Path(__file__)),'run_id':args.run_id,
                   'encoder':args.encoder,'all_endpoints':args.all_endpoints,
                   'learning':args.learning,'augmentation_rng':augment_rng.get_state(),
                   'label_fields':['camera_x_mm','camera_y_mm']}
            torch.save(state,folder/'last.pt')
            if score<best:
                best=score;stale=0;torch.save(state,folder/'best.pt')
            else:stale+=1
            save_json(folder/'history.json',history)
            print(json.dumps({'route':name,**history[-1]}),flush=True)
            if stale>=12:break
        state=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['model']);model.eval()
        with torch.inference_mode():
            vl,vo=model(splits['val'][1].cuda())
            candidates=torch.logspace(-1,1,41,device='cuda')
            nll=torch.stack([nn.functional.cross_entropy(vl.flatten(1)/v,splits['val'][3].cuda()) for v in candidates])
            temperature=float(candidates[nll.argmin()])
            logits,offset=model(splits['test'][1].cuda())
            xy,p=prediction(logits,offset,temperature)
            errors=torch.linalg.vector_norm(xy.cpu()-splits['test'][2],dim=1).tolist()
            partitions={part:metric([e for e,s in zip(errors,splits['test'][0])
                         if bypath[s['frames'][-1]]['evaluation_partition']==part])
                         for part in ('trajectory_holdout','spatial_challenge','appearance_challenge')}
        # Export the actual validation-selected weights, not a fresh model.
        cpu=model.cpu();example=torch.zeros(1,3*t,90,160)
        torch.onnx.export(cpu,example,str(folder/'camera_xy.onnx'),input_names=['rgb_window'],
                          output_names=['location_logits','xy_refinement'],opset_version=13,dynamo=False)
        import onnxruntime as ort
        session=ort.InferenceSession(str(folder/'camera_xy.onnx'),providers=['CPUExecutionProvider'])
        paired=splits['test'][1][:1].numpy()
        with torch.inference_mode():ref=[v.numpy() for v in cpu(torch.from_numpy(paired))]
        actual=session.run(None,{'rgb_window':paired})
        for a,b in zip(actual,ref):np.testing.assert_allclose(a,b,rtol=1e-3,atol=1e-4)
        # On identical full trajectories, reset between trajectories and stress
        # cases; all routes score frames>=7 with real past-only windows.
        stress={}
        for condition in ('clean','frame_drop','variable_dt','blur','visual_failure','relocalization'):
            raw_errors=[];raw_grid_errors=[];filtered_errors=[];pred_rows=[]
            test_groups=[v for v in groups.values() if v[0]['split']=='test']
            for trajectory_index,frames in enumerate(test_groups):
                tr=frames[0]['trajectory_id']
                if frames[0]['split']!='test':continue
                filt=TemporalGridFilter(np.ones((24,24),bool))
                visible=[]
                for i in range(len(frames)):
                    if condition=='frame_drop' and i in (3,5,8):continue
                    current=frames[i]
                    if condition=='relocalization' and i>=6:
                        # Actual kidnapped-camera image+GT switch, not a reset
                        # of the filter with unchanged observation.
                        current=test_groups[(trajectory_index+len(test_groups)//2)%len(test_groups)][i]
                    current={**current,'sensor_valid':not (condition=='visual_failure' and 1<=i<=5)}
                    visible.append(current)
                    clip=visible[-t:]
                    clip=[clip[0]]*(t-len(clip))+clip
                    tensor=torch.cat([images[r['image_path']] if r['sensor_valid'] else torch.zeros_like(images[r['image_path']]) for r in clip],0)[None]
                    if condition=='blur':
                        tensor=nn.functional.avg_pool2d(tensor,9,stride=1,padding=4)
                    quality=0. if condition=='visual_failure' and 1<=i<=5 else 1.
                    tic=time.perf_counter()
                    raw=session.run(None,{'rgb_window':tensor.numpy()})
                    host_ms=(time.perf_counter()-tic)*1000
                    pred,prob=prediction(torch.from_numpy(raw[0]),torch.from_numpy(raw[1]),temperature)
                    gt=np.array([current['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')])
                    stamp=frames[i]['timestamp_s']*(3 if condition=='variable_dt' else 1)
                    tic=time.perf_counter();f=filt.update(prob.numpy().reshape(24,24),stamp,visual_quality=quality)
                    filter_ms=(time.perf_counter()-tic)*1000
                    a=float(np.linalg.norm(pred.numpy()[0]-gt))
                    grid_idx=int(prob.argmax());raw_grid=np.array([-1200+(grid_idx%24+.5)*100,-1200+(grid_idx//24+.5)*100])
                    grid_error=float(np.linalg.norm(raw_grid-gt))
                    b=float(np.linalg.norm(np.array([f['camera_x_mm'],f['camera_y_mm']])-gt))
                    if i>=7:raw_errors.append(a);raw_grid_errors.append(grid_error);filtered_errors.append(b)
                    pred_rows.append({'sample_id':current['sample_id'],'trajectory_id':tr,
                        'frame_index':i,'scored':i>=7,'visible_sample_ids':[r['sample_id'] for r in clip],
                        'sensor_valid':current['sensor_valid'],
                        'evaluation_partition':frames[0]['evaluation_partition'],'raw_grid_error_mm':grid_error,
                        'timestamp_s':stamp,'error_mm':a,'filtered_error_mm':b,'state':f['state'],
                        'confidence':f['confidence'],'host_ort_ms':host_ms,'host_filter_ms':filter_ms})
            stress[condition]={'raw_refined':metric(raw_errors),'raw_grid':metric(raw_grid_errors),
                               'filtered_grid':metric(filtered_errors),'predictions':pred_rows,
                               'comparison_contract':'raw_grid vs filtered_grid; raw_refined separate'}
        summary={'route':route,'window':t,'epochs_completed':history[-1]['epoch'],
                 'selected_epoch':state['epoch'],'optimizer_updates':updates,'parameters':sum(p.numel() for p in cpu.parameters()),
                 'training_reused_without_optimizer_replay':reuse,
                 'encoder':args.encoder,'all_endpoints':args.all_endpoints,
                 'learning':args.learning,
                 'temperature':temperature,'test':partitions['trajectory_holdout'],
                 'test_partitions':partitions,'stress':stress,
                 'endpoint_sha256':endpoint_sha,'best_sha256':digest(folder/'best.pt'),
                 'onnx_sha256':digest(folder/'camera_xy.onnx'),'onnx_ort_parity':'PASS',
                 'kmodel':'NOT RUN','k230_fps':'NOT RUN','real_camera_test':'NOT RUN'}
        save_json(folder/'METRICS.json',summary);summaries[name]=summary
        save_json(out/'COMPARISON.json',summaries)
    save_json(out/'PILOT_GATE.json',{'status':'PASS' if summaries['cnn1']['test']['median_mm']<=200 and summaries['cnn1']['test']['p95_mm']<=500 else 'NEEDS_IMPROVEMENT',
        'thresholds':{'median_mm_max':200,'p95_mm_max':500},'qa':qa,
        'full_generation_authorized_by_gate':False,'note':'Review stability, ambiguity, coverage and calibration before scale-up',
        'elapsed_seconds':time.monotonic()-start,'device':torch.cuda.get_device_name(0),
        'native_board':'NOT RUN'})


if __name__=='__main__':main()
