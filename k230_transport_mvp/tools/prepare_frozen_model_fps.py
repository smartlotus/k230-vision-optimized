"""Prepare the existing four-update pilot for speed testing; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.material_dataset import audit,prepare_calibration,verified_rgb,source_file
from tools.material_preprocess import model_uint8
from tools.material_model_contract import verify_training_provenance
from tools.compile_material_kmodel import compile_model

ROOT=Path(__file__).resolve().parents[1]


def upload_bundle(prepared,remote,receipt_path):
    """Transfer to a new short SD directory with per-part readback; no retries."""
    from tools.deploy_native_trial import CanMvClient
    import base64,time
    prepared=Path(prepared).resolve();receipt_path=Path(receipt_path)
    if not remote.startswith('/sdcard/kmfps') or '/' in remote[8:] or '..' in remote:raise ValueError('Short isolated model directory required')
    client=CanMvClient()
    if client.tool('canmv_script_running').get('running') is not False:raise RuntimeError('Board busy')
    if client.tool('canmv_stat_file',{'path':remote}).get('exists') is not False:raise ValueError('New immutable board directory required')
    if client.tool('canmv_mkdir',{'path':remote}).get('success') is not True:raise RuntimeError('Directory creation failed')
    identity=json.loads((prepared/'identity.json').read_text())
    assets=[(prepared/'compiled/material.kmodel','material.kmodel'),(prepared/'identity.json','identity.json'),
        (ROOT/'tools/frozen_model_fps.py','frozen_model_fps.py'),
        (ROOT.parent/'docs/validation/native-fps-20261112/protected-state.json','protected-state.json')]
    for row in identity['images']:
        assets.extend((prepared/row[key],row[key]) for key in ('native_file','rgb_file','input_file'))
    jobs=[];receipt={'root':remote,'status':'INCOMPLETE','parts':[]}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    for source,name in assets:
        raw=source.read_bytes();parts=[]
        for index,start in enumerate(range(0,len(raw),262144)):
            block=raw[start:start+262144];part=remote+'/'+name+'.p'+str(index)
            if len(part.encode())>=64:raise ValueError('SD protocol path too long')
            if client.tool('canmv_write_file',{'path':part,'encoding':'base64','content':base64.b64encode(block).decode()}).get('success') is not True:raise RuntimeError('Write refused; no retry')
            read=client.tool('canmv_read_file',{'path':part,'encoding':'base64'})
            if base64.b64decode(read['content'])!=block:raise RuntimeError('Part readback mismatch')
            parts.append(part);receipt['parts'].append({'path':part,'sha256':hashlib.sha256(block).hexdigest(),'bytes':len(block)})
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        jobs.append({'target':remote+'/'+name,'parts':parts,'sha256':hashlib.sha256(raw).hexdigest()})
        print(json.dumps({'uploaded_asset':name,'bytes':len(raw)}),flush=True)
    script="import json,hashlib,binascii\njobs="+json.dumps(jobs)+"\nverified={}\nfor job in jobs:\n h=hashlib.sha256()\n with open(job['target'],'wb') as dst:\n  for part in job['parts']:\n   with open(part,'rb') as src:\n    while True:\n     b=src.read(4096)\n     if not b:break\n     dst.write(b);h.update(b)\n value=binascii.hexlify(h.digest()).decode()\n if value!=job['sha256']:raise ValueError('Join hash mismatch')\n verified[job['target']]=value\nprint('MODEL_BUNDLE_JOINED:'+json.dumps(verified))\n"
    client.tool('canmv_terminal_output',{'clear':True})
    client.tool('canmv_run_script',{'script':script})
    for _ in range(50):
        if client.tool('canmv_script_running').get('running') is False:break
        time.sleep(.2)
    terminal=client.tool('canmv_terminal_output',{'clear':False})['text']
    lines=[line for line in terminal.splitlines() if line.startswith('MODEL_BUNDLE_JOINED:')]
    if not lines:raise RuntimeError('Join outcome unconfirmed; reconcile before retry')
    verified=json.loads(lines[-1].split(':',1)[1])
    if verified!={job['target']:job['sha256'] for job in jobs}:raise RuntimeError('Join readback identity mismatch')
    receipt.update(status='VERIFIED_FROZEN_BUNDLE',files=verified)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'files':len(verified)}),flush=True)


def sha(path):
    from tools.file_integrity import sha256_file
    return sha256_file(path)


def prepare(onnx,training_run,output,manifest_path):
    onnx=Path(onnx).resolve();training_run=Path(training_run).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Immutable benchmark directory required')
    manifest_path=Path(manifest_path).resolve();manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    report=audit(ROOT,manifest)
    if not report['export_ready']:raise ValueError('Reviewed frozen dataset required')
    provenance=verify_training_provenance(ROOT,manifest,onnx,training_run)
    if provenance['real_optimizer_updates']!=4:raise ValueError('Expected the original four-update pilot')
    output.mkdir(parents=True)
    (output/'onnx-source').mkdir()
    shutil.copyfile(onnx,output/'onnx-source/material.onnx')
    shutil.copyfile(onnx.parent/'export.json',output/'onnx-source/export.json')
    calibration=prepare_calibration(ROOT,manifest,output/'calibration',size=224,maximum=4)
    approved={row['image_id'] for row in report['rows'] if row['eligible'] and row['split']=='train'}
    images=[]
    for i,record in enumerate(sorted((r for r in manifest['images'] if r['image_id'] in approved),key=lambda r:r['image_id'])):
        rgb=verified_rgb(ROOT,record);tensor,contract=model_uint8(rgb,224)
        raw=source_file(ROOT,record['captures'][0]['files']['rgb565']['path'])
        native=output/('frame%d.rgb565'%i);model_input=output/('input%d.bin'%i)
        shutil.copyfile(raw,native);model_input.write_bytes(tensor.tobytes())
        planar=output/('frame%d.rgb888p'%i)
        planar.write_bytes(rgb.transpose(2,0,1)[None,...].tobytes())
        images.append({'image_id':record['image_id'],'split':'train','source':raw.relative_to(ROOT).as_posix(),
            'native_file':native.name,'native_sha256':sha(native),'input_file':model_input.name,
            'input_sha256':sha(model_input),'input_contract':contract,
            'rgb_file':planar.name,'rgb_sha256':sha(planar)})
    identity={'onnx_sha256':sha(onnx),'training_provenance':provenance,'images':images,
        'calibration_train_images':calibration['samples_count'],'held_out_test_consumed':False,
        'retrained':False,'task_accuracy_verified':False,'scope':'FROZEN_REAL_TRAIN_IMAGES_SPEED_ONLY'}
    (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'COMPILING_EXISTING_TRAINED_PILOT','onnx_sha256':identity['onnx_sha256'],
                      'images':len(images),'updates':4}),flush=True)
    result=compile_model(ROOT,manifest,output/'onnx-source/material.onnx',output/'calibration',224,
                         output/'compiled',manifest_path=manifest_path,training_run=training_run)
    identity.update(kmodel_file='material.kmodel',kmodel_sha256=result['kmodel_sha256'])
    (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':result['status'],'kmodel_bytes':result['kmodel_bytes'],
                      'kmodel_sha256':result['kmodel_sha256']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',type=Path,required=True);p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/ai_pilot/material_manifest_reviewed_r4_20261110.json')
    a=p.parse_args();prepare(a.onnx,a.training_run,a.output,a.manifest)
