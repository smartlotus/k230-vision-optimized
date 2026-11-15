"""Export and quantize a verified checkpoint while all training data stays in Kaggle."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from cloud_artifacts import extract_verified_archive
from cloud_store import archive_directory,atomic_json,sha
from compiler_runtime import install
from downstream_job import mount_training_source
from formal_job import unpack_session
from full_dataset import load_shards
from native_job import verify_inputs


def original_pilot(config,mount,root,deadline):
    from combine_datasets import combine
    sources=[]
    for name in ('primary','colorfix'):
        source=config['pilot_sources'][name]
        archive=source['archive']
        matches=list(mount.rglob(archive['name']))
        if len(matches)!=1:
            raise ValueError('Expected one mounted Pilot archive')
        path=matches[0].resolve()
        if (not path.is_relative_to(mount) or path.stat().st_size!=archive['bytes']
                or sha(path,deadline)!=archive['sha256']):
            raise ValueError('Mounted Pilot archive differs from the accepted bytes')
        target=root/name
        extract_verified_archive(path,target)
        sources.append((name,target/'pilot',archive['sha256']))
    destination=root/'dataset'
    combine(sources,destination)
    return destination


def run(input_root,runtime_input,run_id,training_input_root=None):
    input_root=Path(input_root).resolve()
    runtime_input=Path(runtime_input).resolve()
    verify_inputs(input_root)
    verify_inputs(runtime_input)
    config=json.loads((input_root/'MODEL_ARTIFACT_CONFIG.json').read_text(encoding='utf8'))
    if config['run_id']!=run_id:
        raise ValueError('Model conversion run identity differs')
    root=Path('/kaggle/working')
    scratch=(Path('/kaggle/temp') if Path('/kaggle/temp').exists() else Path('/tmp'))/run_id
    scratch.mkdir(exist_ok=False)
    output=root/run_id
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    deadline=started+config.get('max_seconds',3600)
    state={'run_id':run_id,'status':'STARTED','optimizer_updates':0,
           'input_manifest_sha256':sha(input_root/'INPUT_MANIFEST.json'),
           'runtime_input_manifest_sha256':sha(runtime_input/'INPUT_MANIFEST.json')}
    try:
        from native_export_runtime import install as install_native
        state['native_export_dependencies']=install_native(input_root,scratch/'native-site')
        import importlib.util
        state['environment_modules']={name:importlib.util.find_spec(name) is not None
                                      for name in ('onnx','onnxruntime','torch')}
        print('TOPFACE_MODEL_ARTIFACT_STAGE NATIVE_EXPORT_ENVIRONMENT '+json.dumps(state['environment_modules']),flush=True)
        import onnx
        import onnxruntime
        import torch
        state['native_library_versions']={'onnx':onnx.__version__,'onnxruntime':onnxruntime.__version__,'torch':str(torch.__version__)}
        mount=Path('/kaggle/input').resolve()
        if config['phase']=='pilot_conversion':
            dataset=original_pilot(config,mount,scratch,deadline-600)
        else:
            dataset=scratch/'dataset'
            load_shards(config['shards'],mount,dataset,config['total_scenes'],config['data_seed'],
                        ('train','val','synthetic_test'),deadline=deadline-600)
        if sha(dataset/'AUDIT_V11.json')!=config['dataset_index_sha256']:
            raise ValueError('Conversion dataset differs from the selected checkpoint')
        print('TOPFACE_MODEL_ARTIFACT_STAGE INPUTS_VERIFIED',flush=True)
        ptq_backgrounds=None
        if config.get('ptq_background_bank') is not None:
            training_input_root=Path(training_input_root).resolve()
            if sha(training_input_root/'INPUT_MANIFEST.json')!=config['training_input']['manifest_sha256']:
                raise ValueError('PTQ background source differs from the frozen formal input')
            verify_inputs(training_input_root)
            from background_augmentation import unpack
            bank=config['ptq_background_bank']
            ptq_backgrounds=unpack(training_input_root/'real-background-bank.bin',scratch/'ptq-backgrounds',
                                   bank['archive_sha256'],bank['manifest_sha256'])
        parent=config['parent']
        source=unpack_session(mount,scratch/'parent',parent,parent['experiment'],('best.pt',),deadline-600)
        checkpoint=source/'best.pt'
        if sha(checkpoint)!=parent['checkpoint_sha256']:
            raise ValueError('Selected checkpoint digest differs')
        code=output/'source'
        mount_training_source(input_root,code)
        for path in input_root.glob('*.py'):
            shutil.copy2(path,code/path.name)
        from export_trained_topface import export
        exported=output/'exported'
        state['export']=export(checkpoint,parent['checkpoint_sha256'],exported,dataset)
        if state['export']['status']!='PYTORCH_ONNX_PARITY_PASS':
            raise ValueError('Selected trained model failed ONNX parity')
        print('TOPFACE_MODEL_ARTIFACT_STAGE ONNX_VERIFIED',flush=True)
        if config['phase']=='formal_conversion':
            from evaluate_final import evaluate_final
            evaluated=evaluate_final(dataset,checkpoint,parent['checkpoint_sha256'],
                                     output/'SYNTHETIC_TEST.json',device='cpu')
            state['synthetic_test']={key:evaluated[key] for key in ('images','task_mAP50','task_mAP50_95',
                'class_agnostic_precision50','class_agnostic_recall50','matched_color_accuracy',
                'denominators','errors','count_groups','capacity_accuracy','over_capacity_recall',
                'end_to_end_angle_coverage','angle_diagnostics')}
        python,env=install(runtime_input,scratch/'runtime')
        env['PYTHONPATH']=str(input_root)+':'+str(scratch/'runtime/site')
        with (output/'compile.log').open('wb') as log:
            command=[str(python),str(input_root/'compile_trained_topface.py'),
                     '--export-dir',str(exported),'--dataset',str(dataset),'--output',str(output/'compiled'),
                     '--calibration-count','64','--pair-count','12']
            if ptq_backgrounds is not None:
                command.extend(['--background-bank',str(ptq_backgrounds)])
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if process.returncode:
            raise RuntimeError('Trained model compiler or Simulator failed')
        conversion=json.loads((output/'compiled/RESULT.json').read_text(encoding='utf8'))
        state.update(status='TRAINED_EXPORT_COMPILE_AND_SIMULATOR_COMPLETE',
                     identity=conversion['identity'],paired_measurements=conversion['paired_measurements'])
        if config['phase']=='formal_conversion':
            print('TOPFACE_MODEL_ARTIFACT_STAGE QUANTIZED_SYNTHETIC_TEST',flush=True)
            evaluation_budget=min(18000,int(deadline-time.monotonic()-600))
            if evaluation_budget<120:
                raise TimeoutError('No finite budget remains for the frozen quantized test')
            paired=output/'paired-synthetic-test'
            with (output/'simulator-test.log').open('wb') as log:
                process=subprocess.run([str(python),str(input_root/'simulate_dataset.py'),
                    '--dataset',str(dataset),'--model-dir',str(output/'compiled'),'--output',str(paired),
                    '--onnx',str(exported/'topface.onnx'),
                    '--max-seconds',str(evaluation_budget)],env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=evaluation_budget+30)
            if process.returncode:
                raise RuntimeError('Frozen paired synthetic-test inference failed; preserve raw outputs')
            from evaluate_saved_raw import evaluate_raw
            task=evaluate_raw(dataset,paired,exported,output/'PAIRED_SYNTHETIC_TEST.json')
            keys=('images','task_mAP50','task_mAP50_95','class_agnostic_precision50','class_agnostic_recall50',
                  'matched_color_accuracy','denominators','errors','capacity_accuracy','end_to_end_angle_coverage')
            state['paired_synthetic_test']={backend:{key:metrics[key] for key in keys}
                for backend,metrics in task['backends'].items()}
            raw_summary=json.loads((paired/'RESULT.json').read_text(encoding='utf8'))
            state['quantized_test_inference']={key:raw_summary[key] for key in ('images','completed_images',
                'scene_order_sha256','simulator_inference_seconds','raw_channel_mae','raw_channel_max','files')}
            process=subprocess.run([str(python),str(input_root/'package_trial.py'),'--compiled',str(output/'compiled'),
                '--export',str(exported),'--output',str(output/'board-package')],env=env,
                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,timeout=60)
            if process.returncode:
                raise RuntimeError('Board package construction failed')
        models=root/(run_id+'-models')
        models.mkdir()
        delivery_files=(exported/'topface.onnx',output/'compiled/topface.kmodel',
                        output/'compiled/identity.json',output/'compiled/output_contract.json')
        for path in delivery_files:
            shutil.copy2(path,models/path.name)
        state['model_files']=[{'path':(models/path.name).relative_to(root).as_posix(),
                              'bytes':path.stat().st_size,'sha256':sha(path)}
                             for path in delivery_files]
        if config['phase']=='formal_conversion':
            selected=models/'topface-best.pt'
            shutil.copy2(checkpoint,selected)
            state['model_files'].append({'path':selected.relative_to(root).as_posix(),
                                        'bytes':selected.stat().st_size,'sha256':sha(selected)})
            board_bundle=models/'topface-board-bundle.tar.gz'
            archive_directory(output/'board-package',board_bundle,deadline)
            state['model_files'].append({'path':board_bundle.relative_to(root).as_posix(),
                                        'bytes':board_bundle.stat().st_size,'sha256':sha(board_bundle)})
    except Exception as error:
        state.update(status='MODEL_ARTIFACT_BUILD_FAILED',error_type=type(error).__name__,
                     missing_module=error.name if isinstance(error,ModuleNotFoundError) else None,
                     stack_locations=[{'file':Path(frame.filename).name,'function':frame.name,'line':frame.lineno}
                                      for frame in traceback.extract_tb(error.__traceback__)[-8:]])
        print('TOPFACE_MODEL_ARTIFACT_FAILURE '+json.dumps(state),flush=True)
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'STATE.json',state)
        try:
            archive=root/(run_id+'.tar.gz')
            archive_directory(output,archive,deadline)
            state.update(archive={'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},
                         durable_status='PUBLISHER_PENDING_EXACT_SAVED_OUTPUT')
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT',publication_error_type=type(error).__name__)
        atomic_json(root/(run_id+'-RESULT.json'),state)
        print('TOPFACE_MODEL_ARTIFACT_RESULT '+json.dumps(state),flush=True)
    return state
