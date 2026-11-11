"""PC-only material producer -> ASCII -> compiled C trace, no board access."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from k230.tuning.material import MaterialSnapshotProducer
from k230.tuning.schema import defaults,image_profile
from ports.protocol import encode_material_snapshot,StreamParser


def geometry(fits):
    return dict(material_fits=fits,raw_material_fit_count=len(fits),material_complete=True,
                material_limit_exceeded=False,material_scan_colors=[1,2,3,4,5,6],
                material_algorithm_revision=0,
                material_by_color=[dict(color_id=i,scan_complete=True,scan_reason='COMPLETE',threshold_available=True,
                    top_count=sum(row['color_id']==i for row in fits)) for i in range(1,7)])


def export(output):
    active=defaults()
    for color in range(1,7):active['contexts']['TOP_BUFFER']['thresholds'][str(color)]=[0,30,-5,5,-5,5]
    maker=MaterialSnapshotProducer(active,active,'b'*32,[1,2,3,4,5,6],'SCALAR_FULL_BOUNDARY')
    fits=[dict(accepted=True,reason='GEOMETRY_ONLY',color_id=c,center=[u,150],semiaxes=[20,16],angle_rad=.2)
          for c,u in ((1,100),(1,250),(5,400))]
    frames=[];wire=[];commands=[]
    for n in range(1,10):
        meta=dict(camera_id=2,frame_id=n,timestamp_ms=n*100,width=640,height=360,is_simulated=False,
                  image_profile=image_profile(active),active_revision=0)
        g=geometry(fits)
        if n==4:g.update(material_limit_exceeded=True,raw_material_fit_count=4,material_fits=[])
        if n==8:
            g['material_by_color'][4].update(scan_complete=False,scan_reason='EDGE_PROPOSAL_LIMIT')
            g['material_complete']=False
            g['material_scan_reason']='EDGE_PROPOSAL_LIMIT'
        if n==9:g=geometry([])
        value=maker.build(meta,g)
        encoded=encode_material_snapshot(n,value)
        assert StreamParser().feed(encoded)[0]['objects']==value['objects']
        frames.append(dict(source=meta,measurement=g,produced_snapshot=value,synthetic=True))
        wire.append(encoded.decode('ascii'))
        commands.append('F %d %s\n'%(n*10,encoded.hex()))
    compiler=shutil.which('gcc')
    if not compiler and Path('C:/msys64/mingw64/bin/gcc.exe').exists():
        compiler='C:/msys64/mingw64/bin/gcc.exe'
    if not compiler:raise RuntimeError('No C compiler: consumer proof UNKNOWN')
    env=dict(os.environ);env['PATH']=str(Path(compiler).parent)+os.pathsep+env.get('PATH','')
    with tempfile.TemporaryDirectory(prefix='material-producer-') as temp:
        driver=Path(temp)/'driver.c'
        driver.write_text((ROOT/'stm32/host_material.c').read_text(encoding='utf-8').replace('a'*64,maker.profile_id),encoding='utf-8')
        driver_sha=hashlib.sha256(driver.read_bytes()).hexdigest()
        executable=Path(temp)/('consumer.exe' if os.name=='nt' else 'consumer')
        compiled=subprocess.run([compiler,'-std=c99','-Wall','-Wextra','-Werror','-I',str(ROOT/'stm32'),
            str(ROOT/'stm32/protocol.c'),str(ROOT/'stm32/material_receiver.c'),str(driver),'-lm','-o',str(executable)],
            capture_output=True,text=True,env=env,check=True)
        run=subprocess.run([str(executable)],input=''.join(commands),capture_output=True,text=True,env=env,check=True)
    rows=[json.loads(line) for line in run.stdout.splitlines()]
    assert [r['usable'] for r in rows]==[0,0,1,0,0,0,1,0,0]
    assert [r['count'] for r in rows]==[3,3,3,0,3,3,3,0,0]
    assert [r['status'] for r in rows]==[1,1,1,3,1,1,1,3,2]
    # Bounded host overhead measurement; no native detection/capture/preview included.
    benchmark=MaterialSnapshotProducer(active,active,'b'*32,[1,2,3,4,5,6],'SCALAR_FULL_BOUNDARY')
    durations=[]
    for n in range(1,204):
        meta=dict(camera_id=2,frame_id=n,timestamp_ms=n*100,width=640,height=360,is_simulated=False,
                  image_profile=image_profile(active),active_revision=0)
        measured=geometry(fits)
        begin=perf_counter();benchmark.build(meta,measured);ms=(perf_counter()-begin)*1000
        if n>3:durations.append(ms)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    trace=dict(scope='Synthetic PC measurements only; not camera recognition, GT or training',
               contract=maker.contract(),frames=frames,wire_ascii=wire,consumer_rows=rows)
    (output/'producer-to-c-trace.json').write_text(json.dumps(trace,indent=2)+'\n',encoding='utf-8')
    sources=['k230/tuning/material.py','k230/tuning/live.py','vision/material_tracker.py',
             'ports/protocol.py','stm32/protocol.c','stm32/material_receiver.c',
             'stm32/material_receiver.h','stm32/host_material.c','tests/test_material_producer.py',
             'tests/test_material_protocol.py','tests/test_live_tuning.py','tools/export_material_producer_evidence.py']
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='HOST_PRODUCER_TO_C_PASS',
                board_access=False,uart_access=False,training_steps=0,human_gt=0,
                compiler_flags=['-std=c99','-Wall','-Wextra','-Werror'],compile_stderr=compiled.stderr,
                runner_configuration=dict(expected_profile_id=maker.profile_id,source_session='b'*32,
                                          generated_driver_sha256=driver_sha),
                host_timing=dict(warmups=3,iterations=len(durations),mean_ms=sum(durations)/len(durations),
                                 max_ms=max(durations),scope='Producer only on PC; not K230 FPS or end-to-end latency'),
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (output/'producer-verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','board_access','uart_access','host_timing')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    export(p.parse_args().output)
