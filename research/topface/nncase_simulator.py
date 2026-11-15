"""Run the frozen RGB-uint8 KModel with the matching nncase2.11 Simulator."""
import gc
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from cloud_store import sha
from task_geometry import letterbox_bgr


class KModelRuntime:
    def __init__(self,model,identity,work_dir=None):
        started=time.perf_counter();self.inference_seconds=[]
        model=Path(model).resolve()
        self.identity=json.loads(Path(identity).read_text(encoding='utf8'))
        if sha(model)!=self.identity['kmodel_sha256'] or self.identity['input']!='NCHW_RGB_UINT8':
            raise ValueError('KModel differs from its frozen RGB contract')
        versions={name:importlib.metadata.version(name) for name in ('nncase','nncase-kpu')}
        if versions!=self.identity['compiler'] or set(versions.values())!={'2.11.0'}:
            raise ValueError('Simulator version differs from the compiled artifact')
        self.width,self.height=self.identity['model_wh']
        self.previous_cwd=Path.cwd()
        self.previous={key:os.environ.get(key) for key in ('NNCASE_PLUGIN_PATH','PATH')}
        self.dll=None;self.simulator=None
        plugin=Path(importlib.util.find_spec('_nncase').origin).parent
        try:
            os.environ['NNCASE_PLUGIN_PATH']=str(plugin)
            os.environ['PATH']=str(plugin)+os.pathsep+os.environ.get('PATH','')
            self.dll=os.add_dll_directory(str(plugin)) if os.name=='nt' else None
            if work_dir is not None:
                work_dir=Path(work_dir).resolve();work_dir.mkdir(parents=True,exist_ok=False)
                os.chdir(work_dir)
            import nncase
            self.nncase=nncase
            self.simulator=nncase.Simulator();self.simulator.load_model(model.read_bytes())
            self.load_seconds=time.perf_counter()-started
        except Exception:
            self.close()
            raise

    def raw_from_bgr(self,image):
        canvas,_=letterbox_bgr(image,self.width,self.height)
        rgb=canvas[:,:,::-1].transpose(2,0,1)[None].copy()
        return self.raw_from_rgb_nchw(rgb)

    def raw_from_rgb_nchw(self,rgb):
        if rgb.dtype!=np.uint8 or list(rgb.shape)!=[1,3,self.height,self.width]:
            raise ValueError('Prepared Simulator input must be exact uint8 NCHW RGB')
        started=time.perf_counter()
        self.simulator.set_input_tensor(0,self.nncase.RuntimeTensor.from_numpy(rgb))
        self.simulator.run()
        raw=self.simulator.get_output_tensor(0).to_numpy().copy()
        self.inference_seconds.append(time.perf_counter()-started)
        if list(raw.shape)!=self.identity['output_shape'] or not np.isfinite(raw).all():
            raise ValueError('Simulator raw output differs from its finite tensor contract')
        return raw[0]

    def close(self):
        self.simulator=None
        gc.collect()
        if self.dll is not None:
            self.dll.close();self.dll=None
        os.chdir(self.previous_cwd)
        for key,value in self.previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
