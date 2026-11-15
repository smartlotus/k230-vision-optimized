"""Collect complete bounded metrics from an accepted exact conversion version."""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import re

from cloud_store import atomic_json,sha
from collect_compact import download
from collect_model_files import validate_accepted_conversion

CORE={'exported/RESULT.json','compiled/RESULT.json','SYNTHETIC_TEST.json','PAIRED_SYNTHETIC_TEST.json'}


def manifest(saved):
    if saved['export']['experiment']!='formal_full':
        raise ValueError('Complete final reports require a frozen formal model')
    rows=saved['evaluation_reports'];names=[row['path'] for row in rows]
    pictures=[name for name in names if name.endswith('.jpg')]
    if (len(names)!=len(set(names)) or not CORE<=set(names) or len(names)>18 or
            len(pictures)>12 or (pictures and 'failure-previews/EVIDENCE.json' not in names) or
            sum(row['bytes'] for row in rows)>12*1024**2):
        raise ValueError('Final report inventory or bounded total differs')
    for row in rows:
        name=row['path']
        allowed=(name in CORE or name=='failure-previews/EVIDENCE.json' or
                 re.fullmatch(r'failure-previews/case-[0-9]{2}\.jpg',name))
        cap=512*1024 if name.endswith('.jpg') else 2*1024**2
        if not allowed or '..' in PurePosixPath(name).parts or not 0<row['bytes']<=cap:
            raise ValueError('Unexpected final report path or size')
    return rows


def validate_reports(directory,saved):
    exported=json.loads((directory/'exported/RESULT.json').read_text(encoding='utf8'))
    compiled=json.loads((directory/'compiled/RESULT.json').read_text(encoding='utf8'))
    reference=json.loads((directory/'SYNTHETIC_TEST.json').read_text(encoding='utf8'))
    paired=json.loads((directory/'PAIRED_SYNTHETIC_TEST.json').read_text(encoding='utf8'))
    identity=saved['identity'];expected=saved['synthetic_test']
    if (exported['onnx_sha256']!=identity['onnx_sha256'] or
            compiled['identity']!=identity or paired['identity']!=identity or
            reference['checkpoint_sha256']!=identity['checkpoint_sha256'] or
            reference['scope']!='HELD_OUT_SYNTHETIC_TEST' or paired['scope']!='HELD_OUT_SYNTHETIC_TEST' or
            reference['images']!=expected['images'] or paired['images']!=reference['images'] or
            reference['denominators']!=expected['denominators'] or
            set(paired['backends'])!={'ort','simulator'}):
        raise ValueError('Full reports differ from the exact selected model or frozen test')
    for backend,metrics in paired['backends'].items():
        compact=saved['paired_synthetic_test'][backend]
        if (metrics['images']!=reference['images'] or metrics['denominators']!=compact['denominators'] or
                metrics['dataset_audit_sha256']!=reference['dataset_audit_sha256'] or
                not metrics.get('slices') or not metrics.get('ap_by_iou')):
            raise ValueError('Per-color/sliced metrics differ from the complete test denominator')


def collect(result,publication,token_file,output,receipt):
    saved=json.loads(result.read_text(encoding='utf8'));durable=json.loads(publication.read_text(encoding='utf8'))
    validate_accepted_conversion(saved,durable);rows=manifest(saved)
    os.environ['KAGGLE_API_TOKEN']=token_file.read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    output.mkdir(parents=True,exist_ok=True);source=durable['prep']
    with api.build_kaggle_client() as client:
        for row in rows:
            target=output/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                raw=download(client,source['ref'],source['version'],saved['run_id']+'/'+row['path'],row['bytes'])
                if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                    raise ValueError('Saved report bytes differ from the published conversion')
                temporary=target.with_suffix(target.suffix+'.part');temporary.write_bytes(raw);temporary.replace(target)
            if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
                raise ValueError('Existing report bytes differ; preserve the file and investigate')
    validate_reports(output,saved)
    value={'status':'BOUNDED_COMPLETE_MODEL_REPORTS_SHA256_PASS','run_id':saved['run_id'],
        'observed_utc':datetime.now(timezone.utc).isoformat(),'source':source,
        'source_result_sha256':sha(result),'publication_receipt_sha256':sha(publication),
        'files':rows,'raw_arrays_or_dataset_downloaded_to_host':False}
    atomic_json(receipt,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('result','publication','token-file','output','receipt'):
        parser.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(collect(**vars(parser.parse_args())),indent=2))
