"""Reviewed-data-only bounded CPU pilot, with counted optimizer updates."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.material_dataset import audit,canonical,verified_rgb,source_file
from tools.material_preprocess import model_uint8
from tools.file_integrity import sha256_file as sha
from vision.yolo_candidates import input_contract


def now():return datetime.now(timezone.utc).isoformat()


def write_receipt(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def plan_training(root,manifest,initial_weights,size=224,epochs=1,batch=2,max_updates=200,max_seconds=300,seed=20261109):
    report=audit(root,manifest)
    if not report['export_ready']:
        raise ValueError('Reviewed complete train/val/test groups and positive training tops required')
    input_contract(size)
    for value,low,high,label in ((epochs,1,5,'epochs'),(batch,1,4,'batch'),
            (max_updates,1,200,'max_updates'),(max_seconds,30,600,'max_seconds'),(seed,0,2147483647,'seed')):
        if type(value) is not int or not low<=value<=high:raise ValueError('Bounded integer '+label+' required')
    if report['eligible_images']>5000:raise ValueError('CPU pilot supports at most5000 reviewed images')
    weights=Path(initial_weights)
    if not weights.is_file() or weights.suffix!='.pt' or not 0<weights.stat().st_size<=64*1024*1024:
        raise ValueError('Existing local trusted pretrained .pt required; no implicit download')
    updates=math.ceil(report['eligible_split_counts']['train']/batch)*epochs
    if updates>max_updates:raise ValueError('Epoch/batch schedule exceeds explicit optimizer update budget')
    return dict(schema_version=1,status='PLAN_ONLY_NOT_STARTED',dataset_manifest_sha256=report['manifest_sha256'],
        initial_weights_sha256=sha(weights),initial_weights_bytes=weights.stat().st_size,
        architecture='yolov8n.yaml',initialization='EXPLICIT_LOCAL_PRETRAINED',classes=['visible_top'],
        input_contract=input_contract(size),epochs=epochs,batch=batch,seed=seed,
        max_optimizer_updates=max_updates,max_seconds=max_seconds,planned_optimizer_updates=updates,
        split_counts=report['eligible_split_counts'],training_positive_instances=report['train_positive_instances'],
        held_out_test_consumed=False,device='cpu',workers=0,amp=False,
        normalization='UINT8_RGB_NCHW_TO_FLOAT_DIV255_ONCE',
        augmentation='NONE_FIRST_PILOT_PRELETTERBOXED_CENTER114',
        optimizer=dict(name='AdamW',lr0=.001,nbs=batch,warmup_epochs=0),
        real_optimizer_updates=0,task_accuracy_verified=False,board_run=False)


def stage_dataset(root,manifest,output,size):
    """Freeze only train/val pixels with the exact deployment resize/pad contract."""
    report=audit(root,manifest)
    if not report['export_ready']:raise ValueError('Reviewed grouped complete dataset required')
    output=Path(output)
    if output.exists():raise ValueError('Dataset staging directory is immutable')
    output.mkdir(parents=True)
    records={r['image_id']:r for r in manifest['images']};rows=[]
    for row in report['rows']:
        if not row['eligible'] or row['split']=='test':continue
        record=records[row['image_id']];split=row['split'];key=row['image_id']
        tensor,contract=model_uint8(verified_rgb(root,record),size)
        rgb=np.ascontiguousarray(tensor[0].transpose(1,2,0))
        image=output/'images'/split/(key+'.png');image.parent.mkdir(parents=True,exist_ok=True)
        label=output/'labels'/split/(key+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise OSError('Staged PNG write failed')
        loaded=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        if not np.array_equal(loaded,rgb):raise ValueError('Staged PNG pixels changed')
        lines=[];gain=contract['gain'];top=contract['pad_tblr'][0];left=contract['pad_tblr'][2]
        for obj in row['objects']:
            x,y,w,h=obj['bbox_xywh']
            lines.append('0 %.9f %.9f %.9f %.9f'%(((x+w/2)*gain+left)/size,
                ((y+h/2)*gain+top)/size,w*gain/size,h*gain/size))
        label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        rows.append(dict(image_id=key,split=split,image=image.relative_to(output).as_posix(),
                        image_sha256=sha(image),tensor_uint8_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
                        label=label.relative_to(output).as_posix(),label_sha256=sha(label)))
    # No test path/pixels available to the training library. Validation is explicit.
    yaml=output/'dataset.yaml'
    yaml.write_text('path: '+json.dumps(str(output.resolve()))+'\ntrain: images/train\nval: images/val\nnames:\n  0: visible_top\n',encoding='utf-8')
    receipt=dict(schema_version=1,dataset_manifest_sha256=report['manifest_sha256'],
        input_contract=input_contract(size),images=rows,yaml_sha256=sha(yaml),test_pixels_staged=False,
        held_out_test_ids=sorted(r['image_id'] for r in report['rows'] if r['eligible'] and r['split']=='test'))
    write_receipt(output/'staging.json',receipt)
    return receipt


def verify_staging(output,receipt):
    output=Path(output)
    if sha(output/'dataset.yaml')!=receipt['yaml_sha256']:raise ValueError('Training YAML drift')
    expected=set()
    for row in receipt['images']:
        for kind in ('image','label'):
            path=source_file(output,row[kind]);expected.add(path.resolve())
            if sha(path)!=row[kind+'_sha256']:raise ValueError('Staged training bytes changed')
    for directory in ('images','labels'):
        for path in (output/directory).rglob('*'):
            if path.relative_to(output).as_posix() in ('labels/train.cache','labels/val.cache'):continue
            if path.is_file() and path.resolve() not in expected:
                raise ValueError('Unregistered staged data file')
    if (output/'images/test').exists():raise ValueError('Held-out test images exposed to trainer')


class PilotBudgetReached(RuntimeError):pass


class OptimizerLedger:
    """Count actual torch optimizer executions, including no AMP-skipped calls."""
    def __init__(self,receipt,path,max_updates,max_seconds,clock=time.monotonic):
        self.receipt,self.path,self.limit,self.clock=receipt,Path(path),max_updates,clock
        self.deadline=clock()+max_seconds;self.handles=[]

    def check_time(self,*unused):
        if self.clock()>=self.deadline:raise PilotBudgetReached('CPU pilot time budget reached')

    def before_step(self,*unused):
        self.check_time()
        if self.receipt['real_optimizer_updates']>=self.limit:
            raise PilotBudgetReached('CPU pilot optimizer budget reached')

    def after_step(self,*unused):
        self.receipt['real_optimizer_updates']+=1
        self.receipt['last_optimizer_step_at']=now()
        write_receipt(self.path,self.receipt)

    def attach(self,optimizer):
        if self.handles:raise ValueError('Only one optimizer writer allowed')
        self.handles=[optimizer.register_step_pre_hook(self.before_step),
                      optimizer.register_step_post_hook(self.after_step)]

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]


def trainer_class(ledger,source_guard):
    """Delay heavy imports until after real data admission."""
    from collections import defaultdict
    from copy import copy
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.data import build_yolo_dataset
    class PilotValidator(DetectionValidator):
        def run_callbacks(self,event):
            if event in ('on_val_start','on_val_batch_start'):ledger.check_time()
    class PilotTrainer(DetectionTrainer):
        def run_callbacks(self,event):
            # This runner owns only local evidence callbacks, no service integrations.
            if event=='on_train_start':
                ledger.attach(self.optimizer)
                ledger.receipt['training_loop_started']=True
                write_receipt(ledger.path,ledger.receipt)
            if event in ('on_train_epoch_start','on_train_batch_start','on_val_batch_start'):
                ledger.check_time()
            if event=='on_train_epoch_start':source_guard()
            if event=='on_fit_epoch_end':
                ledger.receipt['completed_epochs']=int(self.epoch)+1
                write_receipt(ledger.path,ledger.receipt)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.callbacks=defaultdict(list)  # Validator inherits local-only checks.
            self.callbacks['on_val_batch_start']=[ledger.check_time]
        def build_dataset(self,img_path,mode='train',batch=None):
            # Both splits use deterministic inference transforms. No accidental
            # Albumentations, identity-affine label filtering, or rectangular batches.
            return build_yolo_dataset(self.args,img_path,batch,self.data,mode='val',rect=False,stride=32)
        def get_validator(self):
            self.loss_names='box_loss','cls_loss','dfl_loss'
            return PilotValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
    return PilotTrainer


def run_training(root,manifest_path,initial_weights,output,size=224,epochs=1,batch=2,max_updates=200,max_seconds=300,seed=20261109):
    root=Path(root).resolve();manifest_path=Path(manifest_path).resolve();initial_weights=Path(initial_weights).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    plan=plan_training(root,manifest,initial_weights,size,epochs,batch,max_updates,max_seconds,seed)
    if importlib.metadata.version('ultralytics')!='8.3.235':raise ValueError('Pinned YOLO8.3.235 required')
    output=Path(output).resolve()
    if output.exists():raise ValueError('New immutable pilot run directory required; no automatic retry/resume')
    output.mkdir(parents=True)
    receipt=dict(plan,status='PREPARING',started_at=now(),training_loop_started=False,completed_epochs=0,
                 actual_backend='ULTRALYTICS_CPU',source_manifest_file='reviewed-manifest.json')
    receipt_path=output/'training.json';write_receipt(receipt_path,receipt)
    write_receipt(output/'reviewed-manifest.json',manifest)
    old_env={key:os.environ.get(key) for key in ('YOLO_CONFIG_DIR','YOLO_OFFLINE')}
    os.environ['YOLO_CONFIG_DIR']=str(output/'settings');os.environ['YOLO_OFFLINE']='true'
    ledger=OptimizerLedger(receipt,receipt_path,max_updates,max_seconds)
    try:
        staged=stage_dataset(root,manifest,output/'dataset',size)
        receipt['staging_sha256']=sha(output/'dataset/staging.json')
        def guard():
            if canonical(json.loads(manifest_path.read_text(encoding='utf-8')))!=canonical(manifest):
                raise ValueError('Persistent review/split/source manifest changed')
            if audit(root,manifest)['manifest_sha256']!=plan['dataset_manifest_sha256']:
                raise ValueError('Original reviewed data changed')
            if sha(initial_weights)!=plan['initial_weights_sha256']:raise ValueError('Initialization weights changed')
            verify_staging(output/'dataset',staged)
        guard()
        import torch
        torch.set_num_threads(2)
        overrides=dict(model='yolov8n.yaml',pretrained=str(initial_weights),data=str(output/'dataset/dataset.yaml'),
            project=str(output),name='fit',exist_ok=False,device='cpu',workers=0,epochs=epochs,batch=batch,
            imgsz=size,seed=seed,deterministic=True,amp=False,cache=False,rect=False,multi_scale=False,
            optimizer='AdamW',lr0=.001,nbs=batch,warmup_epochs=0,patience=epochs+1,
            save=True,save_period=-1,plots=False,val=True,resume=False,compile=False,
            hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
            flipud=0,fliplr=0,bgr=0,mosaic=0,mixup=0,cutmix=0,copy_paste=0,close_mosaic=0,
            erasing=0,augment=False)
        receipt['overrides']=overrides;receipt['versions']={key:importlib.metadata.version(key) for key in ('torch','ultralytics','numpy')}
        receipt['status']='TRAINER_INITIALIZING';write_receipt(receipt_path,receipt)
        trainer=trainer_class(ledger,guard)(overrides=overrides)
        if Path(trainer.save_dir).resolve()!=output/'fit':raise ValueError('Unexpected training directory')
        trainer.train();guard()
        if (receipt['real_optimizer_updates']!=plan['planned_optimizer_updates'] or
                not 0<receipt['real_optimizer_updates']<=max_updates or receipt['completed_epochs']!=epochs):
            raise ValueError('Missing optimizer or completed epoch evidence')
        artifacts=[]
        for name in ('best.pt','last.pt'):
            path=source_file(output,'fit/weights/'+name)
            if not path.is_file():raise ValueError('Missing expected trained checkpoint')
            artifacts.append(dict(file=path.relative_to(output).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
        receipt.update(status='TRAINING_FINISHED_NOT_TASK_ACCEPTED',weights=artifacts,finished_at=now(),
                       test_evaluation_executed=False,task_accuracy_verified=False)
    except PilotBudgetReached as error:
        receipt.update(status='BUDGET_REACHED_PARTIAL_NOT_EXPORTABLE',error=str(error),finished_at=now())
    except BaseException as error:
        receipt.update(status='FAILED_OR_UNKNOWN_NOT_EXPORTABLE',error=str(error),finished_at=now())
        raise
    finally:
        try:ledger.close()
        finally:
            try:write_receipt(receipt_path,receipt)
            finally:
                for key,value in old_env.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
    return receipt


def verify_training_run(root,run,checkpoint='best.pt'):
    run=Path(run).resolve();receipt=json.loads((run/'training.json').read_text(encoding='utf-8'))
    if (receipt.get('status')!='TRAINING_FINISHED_NOT_TASK_ACCEPTED' or
            type(receipt.get('real_optimizer_updates')) is not int or receipt['real_optimizer_updates']<=0 or
            receipt.get('real_optimizer_updates')!=receipt.get('planned_optimizer_updates') or
            type(receipt.get('max_optimizer_updates')) is not int or
            receipt['real_optimizer_updates']>receipt['max_optimizer_updates'] or
            receipt.get('completed_epochs')!=receipt.get('epochs') or receipt.get('training_loop_started') is not True):
        raise ValueError('Completed counted task training receipt required')
    manifest=json.loads((run/'reviewed-manifest.json').read_text(encoding='utf-8'))
    report=audit(root,manifest)
    if not report['export_ready'] or report['manifest_sha256']!=receipt['dataset_manifest_sha256']:
        raise ValueError('Training dataset/review identity mismatch')
    if sha(run/'dataset/staging.json')!=receipt['staging_sha256']:raise ValueError('Training staging receipt changed')
    staged=json.loads((run/'dataset/staging.json').read_text(encoding='utf-8'))
    if (staged['dataset_manifest_sha256']!=receipt['dataset_manifest_sha256'] or
            staged['input_contract']!=receipt['input_contract']):
        raise ValueError('Staging/source manifest identity mismatch')
    verify_staging(run/'dataset',staged)
    if checkpoint not in ('best.pt','last.pt'):raise ValueError('Explicit best/last checkpoint required')
    matches=[row for row in receipt['weights'] if row['file']=='fit/weights/'+checkpoint]
    if len(matches)!=1:raise ValueError('Checkpoint receipt missing or duplicated')
    row=matches[0];path=source_file(run,row['file'])
    if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Trained checkpoint bytes changed')
    return receipt,path,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','train'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--initial-weights',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--size',type=int,choices=(224,320),default=224)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--max-updates',type=int,default=200);p.add_argument('--max-seconds',type=int,default=300)
    a=p.parse_args()
    try:
        if a.action=='plan':
            result=plan_training(a.root,json.loads(a.manifest.read_text(encoding='utf-8')),a.initial_weights,
                                 a.size,a.epochs,a.batch,a.max_updates,a.max_seconds)
        else:
            if a.output is None:raise ValueError('Explicit new training output required')
            result=run_training(a.root,a.manifest,a.initial_weights,a.output,a.size,a.epochs,a.batch,a.max_updates,a.max_seconds)
        print(json.dumps(result,indent=2))
    except (ValueError,OSError) as error:
        print(json.dumps(dict(status='NOT_ADMITTED',reason=str(error),task_accuracy_verified=False)))
        raise SystemExit(2)
