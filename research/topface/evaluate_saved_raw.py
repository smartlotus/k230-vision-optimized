"""Score actual saved ORT/Simulator predictions with the same frozen GT and metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cloud_store import atomic_json,sha
from evaluate_task import evaluate
from simulate_dataset import order_sha
from task_training import TopFaceDataset,collate


class SavedRaw(torch.nn.Module):
    def __init__(self,values):
        super().__init__();self.values=values;self.cursor=0

    def forward(self,images):
        end=self.cursor+len(images)
        if end>len(self.values):
            raise ValueError('Cached raw result denominator is exhausted')
        result=torch.from_numpy(np.array(self.values[self.cursor:end],copy=True))
        self.cursor=end
        return result


def evaluate_raw(dataset,predictions,export_dir,output,qualification=False):
    dataset,predictions,export_dir,output=(Path(path).resolve() for path in (dataset,predictions,export_dir,output))
    if output.exists():
        raise FileExistsError('Preserve the frozen raw-output metrics')
    state=json.loads((predictions/'RESULT.json').read_text(encoding='utf8'))
    release=json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    scope='ENGINEERING_TRAIN_FIXTURE' if qualification else 'HELD_OUT_SYNTHETIC_TEST'
    split='train' if qualification else 'synthetic_test'
    if (state['status']!='NNCASE_AND_ORT_TEST_RAW_COMPLETE' or state['scope']!=scope or
            state['split']!=split or state['images']!=state['completed_images'] or
            release['onnx_sha256']!=state['identity']['onnx_sha256']):
        raise ValueError('Raw prediction identity or split differs from the frozen model')
    if not qualification and release['experiment']!='formal_full':
        raise ValueError('The independent test requires the selected formal model')
    width,height=state['identity']['model_wh']
    index=json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    rows=[row for row in index['scenes'] if row['split']==split]
    if qualification:
        if not 1<=state['images']<=6:
            raise ValueError('Engineering fixture exceeds its finite count')
        rows=rows[:state['images']]
    chosen=TopFaceDataset(dataset,split,width,height,scene_names=[row['scene'] for row in rows])
    if (chosen.audit_sha256!=state['dataset_index_sha256'] or len(chosen)!=state['images'] or
            order_sha(chosen.rows)!=state['scene_order_sha256']):
        raise ValueError('Raw prediction rows differ from the verified GT loader order')
    files={row['name']:row for row in state['files']}
    if set(files)!={'ort-raw.npy','simulator-raw.npy'} or len(state['files'])!=2:
        raise ValueError('Unexpected paired raw member inventory')
    result={'status':'PAIRED_RAW_TASK_METRICS_COMPLETE','scope':scope,'images':len(chosen),
        'identity':state['identity'],'dataset_index_sha256':chosen.audit_sha256,
        'raw_result_sha256':sha(predictions/'RESULT.json'),'backends':{},'optimizer_updates':0}
    loader=DataLoader(chosen,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate)
    for backend in ('ort','simulator'):
        name=backend+'-raw.npy';path=predictions/name
        if path.stat().st_size!=files[name]['bytes'] or sha(path)!=files[name]['sha256']:
            raise ValueError('Saved raw tensor bytes differ from the inference result')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(values.shape)!=state['raw_shape'] or values.dtype!=np.float32:
            raise ValueError('Saved raw tensor dimensions or type differ')
        cached=SavedRaw(values)
        metrics=evaluate(cached,loader,torch.device('cpu'),score_threshold=.25,include_slices=True,angle_diagnostics=True)
        if cached.cursor!=len(chosen):
            raise ValueError('Metrics did not consume every actual inference row')
        metrics['cached_array_read_ms_per_image_batch_average']=metrics.pop('host_forward_ms_per_image_batch_average')
        metrics.update(scope=scope,backend='ONNX_RUNTIME' if backend=='ort' else 'NNCASE_SIMULATOR',
            model_sha256=state['identity']['onnx_sha256' if backend=='ort' else 'kmodel_sha256'],
            dataset_audit_sha256=chosen.audit_sha256,physical_mm_error='UNAVAILABLE_WITHOUT_CALIBRATION',
            inference_timing_note='Metrics read saved raw outputs; actual Simulator execution timing is in the raw result.')
        result['backends'][backend]=metrics
    atomic_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('dataset','predictions','export-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--qualification',action='store_true')
    result=evaluate_raw(**vars(parser.parse_args()))
    print(json.dumps({'status':result['status'],'scope':result['scope'],'images':result['images'],
        'backends':{name:{key:row[key] for key in ('task_mAP50','class_agnostic_recall50','denominators')}
                    for name,row in result['backends'].items()}},indent=2))
