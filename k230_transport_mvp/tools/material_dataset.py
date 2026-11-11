"""Explicit material-pilot source registry and human-review gates; host only."""
import argparse,hashlib,json,math
from pathlib import Path
import cv2
from tools.material_preprocess import rgb565_to_rgb,model_uint8,tensor_receipt
from tools.file_integrity import sha256_file
from vision.yolo_candidates import input_contract

TASK='final_material_visible_top_pilot'


def digest(data):return hashlib.sha256(data).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def source_file(root,name):
    if not isinstance(name,str) or not name:raise ValueError('Relative source path required')
    root=Path(root).resolve();path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):raise ValueError('Source escapes project root')
    return path


def register(root,index):
    if index.get('object_set_final_confirmed_by_operator') is not True:
        raise ValueError('Operator confirmation of the final material set is required')
    records={}
    for row in index['sources']:
        if row.get('scene_group_id') is not None or row.get('split') is not None:
            raise ValueError('Register sources first; assign reviewed scene groups in the manifest')
        files={}
        for kind in ('jpeg','rgb565','metadata'):
            path=source_file(root,row[kind+'_path']);raw=path.read_bytes()
            files[kind]={'path':path.relative_to(Path(root).resolve()).as_posix(),'sha256':digest(raw)}
        metadata=json.loads(source_file(root,row['metadata_path']).read_text(encoding='utf-8'))
        if (metadata.get('source')!='REAL_CAMERA_NATIVE_LAB' or metadata.get('is_simulated') is not False
                or metadata.get('camera_id')!=2 or (metadata.get('width'),metadata.get('height'))!=(640,360)
                or type(metadata.get('frame_id')) is not int or metadata['frame_id']<1):
            raise ValueError('Explicit real CSI2 source identity and binding required')
        native=source_file(root,row['rgb565_path']).read_bytes();rgb=rgb565_to_rgb(native)
        jpeg=cv2.imread(str(source_file(root,row['jpeg_path'])))
        if jpeg is None or jpeg.shape!=(360,640,3):raise ValueError('Full640x360 JPEG required, not preview')
        key=files['rgb565']['sha256']
        candidate={'files':files,'frame_id':metadata['frame_id'],'timestamp_ms':metadata.get('timestamp_ms'),
                   'metadata_has_lab_grid':isinstance(metadata.get('lab_grid'),list),
                   'capture_series_id':row.get('capture_series_id')}
        series=candidate['capture_series_id']
        if series is not None and (not isinstance(series,str) or not series.strip()):raise ValueError('Explicit nonempty capture series identity required')
        if key in records:
            records[key]['captures'].append(candidate);continue
        records[key]={'image_id':key,'native_sha256':key,'rgb_pixel_sha256':digest(rgb.tobytes()),
            'camera_id':2,'source_wh':[640,360],'captures':[candidate],
            'scene_group_id':None,'scene_group_confirmed':False,'split':None,
            'conditions':{'camera_height_cm':None,'object_height_cm':None,'lighting':None},
            'review':{'status':'UNREVIEWED','reviewer':None,'all_visible_tops_checked':False,
                      'uncertain':False,'source_native_sha256':key,'objects':None}}
    for record in records.values():record['captures'].sort(key=canonical)
    return {'schema_version':1,'task':TASK,'classes':['visible_top'],'paper_training_enabled':False,
        'object_set_final_confirmed_by_operator':True,'images':[records[k] for k in sorted(records)],
        'input_source_entries':len(index['sources']),'unique_native_images':len(records)}


def verified_rgb(root,record):
    for capture in record['captures']:
        for entry in capture['files'].values():
            if sha256_file(source_file(root,entry['path']))!=entry['sha256']:raise ValueError('Source bytes changed')
    raw=source_file(root,record['captures'][0]['files']['rgb565']['path']).read_bytes()
    if digest(raw)!=record['native_sha256'] or record['image_id']!=record['native_sha256']:
        raise ValueError('Native identity mismatch')
    rgb=rgb565_to_rgb(raw)
    if digest(rgb.tobytes())!=record['rgb_pixel_sha256']:raise ValueError('Decoded pixel identity mismatch')
    return rgb


def review_gate(record):
    reasons=[];group=record.get('scene_group_id');review=record.get('review',{})
    if not isinstance(group,str) or not group.strip() or record.get('scene_group_confirmed') is not True:
        reasons.append('SCENE_GROUP_UNCONFIRMED')
    if record.get('split') not in ('train','val','test'):reasons.append('SPLIT_UNASSIGNED')
    if (review.get('status')!='APPROVED' or not isinstance(review.get('reviewer'),str)
            or not review['reviewer'].strip() or review.get('all_visible_tops_checked') is not True
            or not isinstance(review.get('objects'),list)):
        reasons.append('HUMAN_COMPLETE_REVIEW_REQUIRED');return reasons,[]
    if review.get('uncertain') is not False or review.get('ignore'):reasons.append('UNCERTAIN_FRAME_EXCLUDED')
    if review.get('source_native_sha256')!=record['native_sha256']:reasons.append('ANNOTATION_SOURCE_MISMATCH')
    objects=[]
    for obj in review['objects']:
        if not isinstance(obj,dict):raise ValueError('Structured object annotations required')
        if obj.get('class_id')!=0 or type(obj.get('class_id')) is not int:raise ValueError('Material stage accepts class0visible_top only')
        if obj.get('uncertain') or obj.get('ignore'):reasons.append('UNCERTAIN_FRAME_EXCLUDED')
        box=obj.get('bbox_xywh')
        if (not isinstance(box,list) or len(box)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
            raise ValueError('Finite source-pixel xywh box required')
        x,y,w,h=box
        if w<=0 or h<=0:raise ValueError('Positive box extent required')
        a,b,c,d=max(x,0),max(y,0),min(x+w,640),min(y+h,360)
        if c<=a or d<=b:raise ValueError('Box is outside source image')
        clipped=(a!=x or b!=y or c!=x+w or d!=y+h)
        if clipped and obj.get('geometry_complete_verified'):raise ValueError('Clipped box cannot certify full boundary')
        if obj.get('color_id') is not None and (type(obj['color_id']) is not int or not 1<=obj['color_id']<=6):
            raise ValueError('Optional color_id1..6 required')
        objects.append(dict(obj,bbox_xywh=[a,b,c-a,d-b],bbox_clipped=clipped))
    if len(objects)>3:reasons.append('MORE_THAN_THREE_TOPS_REQUIRES_SEPARATE_SCENE_REVIEW')
    return sorted(set(reasons)),objects


def audit(root,manifest):
    if manifest.get('task')!=TASK or manifest.get('classes')!=['visible_top'] or manifest.get('paper_training_enabled') is not False:
        raise ValueError('Material-only dataset contract required')
    identities=set();groups={};pixel_splits={};series_splits={};rows=[]
    for record in manifest['images']:
        if record['image_id'] in identities:raise ValueError('Duplicate native image identity')
        identities.add(record['image_id']);verified_rgb(root,record)
        reasons,objects=review_gate(record);split=record.get('split');group=record.get('scene_group_id')
        if group and split:
            if group in groups and groups[group]!=split:raise ValueError('Physical scene group crosses splits')
            groups[group]=split
        pixel=record['rgb_pixel_sha256']
        if split:
            if pixel in pixel_splits and pixel_splits[pixel]!=split:raise ValueError('Same source pixels cross splits')
            pixel_splits[pixel]=split
            for capture in record['captures']:
                series=capture.get('capture_series_id')
                if not series:continue
                if series in series_splits and series_splits[series]!=split:raise ValueError('Bounded capture series crosses training/test splits')
                series_splits[series]=split
        rows.append({'image_id':record['image_id'],'eligible':not reasons,'reasons':reasons,
                     'split':split,'objects':objects})
    positive=sum(len(r['objects']) for r in rows if r['eligible'] and r['split']=='train')
    split_counts={split:sum(r['eligible'] and r['split']==split for r in rows) for split in ('train','val','test')}
    return {'status':'REVIEWED_DATA_AVAILABLE' if any(r['eligible'] for r in rows) else 'WAITING_FOR_LABELS_OR_GROUPS',
        'unique_native_images':len(rows),'eligible_images':sum(r['eligible'] for r in rows),
        'train_positive_instances':positive,'eligible_split_counts':split_counts,
        'export_ready':all(split_counts.values()) and positive>0,
        'manifest_sha256':digest(canonical(manifest)),'rows':rows,
        'training_started':False,'paper_training_enabled':False}


def export_yolo(root,manifest,output):
    report=audit(root,manifest);eligible=[r for r in report['rows'] if r['eligible']]
    if not report['export_ready']:raise ValueError('Reviewed grouped train/val/test and a positive training instance required')
    output=Path(output)
    if output.exists():raise ValueError('Existing dataset output is immutable; choose a new version')
    output.mkdir(parents=True)
    records={r['image_id']:r for r in manifest['images']}
    for row in eligible:
        key=row['image_id'];split=row['split'];rgb=verified_rgb(root,records[key])
        image_dir=output/'images'/split;label_dir=output/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(image_dir/(key+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise OSError('Image export failed')
        lines=[]
        for obj in row['objects']:
            x,y,w,h=obj['bbox_xywh'];lines.append('0 %.9f %.9f %.9f %.9f'%((x+w/2)/640,(y+h/2)/360,w/640,h/360))
        (label_dir/(key+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
    # Omit an ambiguous relative `path: .`; consumers anchor this YAML to its
    # own location. The exported folder can then be moved as a unit.
    (output/'dataset.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: visible_top\n',encoding='utf-8')
    (output/'review-receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def prepare_calibration(root,manifest,output,size=320,maximum=100):
    if type(maximum) is not int or not 1<=maximum<=1000:raise ValueError('Explicit sample cap1..1000 required')
    report=audit(root,manifest);approved={r['image_id'] for r in report['rows'] if r['eligible'] and r['split']=='train'}
    selected=sorted((r for r in manifest['images'] if r['image_id'] in approved),key=lambda r:r['image_id'])[:maximum]
    if not selected:raise ValueError('No approved complete training scenes for calibration')
    output=Path(output)
    if output.exists():raise ValueError('Existing calibration output is immutable')
    output.mkdir(parents=True);rows=[]
    for record in selected:
        tensor,contract=model_uint8(verified_rgb(root,record),size)
        name=record['image_id']+'.uint8-nchw.bin';(output/name).write_bytes(tensor.tobytes())
        rows.append(dict(tensor_receipt(tensor,contract),file=name,image_id=record['image_id'],split='train'))
    result={'schema_version':1,'dataset_manifest_sha256':report['manifest_sha256'],
            'input_contract':contract,
            'samples_count':len(rows),'samples':rows,'compiler_executed':False,'native_ai2d_equivalence_verified':False}
    (output/'calibration-manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def verify_calibration(root,manifest,prepared):
    report=audit(root,manifest);prepared=Path(prepared)
    receipt=json.loads((prepared/'calibration-manifest.json').read_text(encoding='utf-8'))
    if receipt['dataset_manifest_sha256']!=report['manifest_sha256']:raise ValueError('Dataset/review/split changed after calibration preparation')
    rows=receipt['samples']
    contract=receipt['input_contract']
    if contract!=input_contract(contract['model_wh'][0]):raise ValueError('Calibration package has an inconsistent input contract')
    if not rows or receipt['samples_count']!=len(rows) or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Calibration sample count/identities inconsistent')
    approved={r['image_id'] for r in report['rows'] if r['eligible'] and r['split']=='train'}
    records={r['image_id']:r for r in manifest['images']}
    if [r['image_id'] for r in rows]!=sorted(r['image_id'] for r in rows):raise ValueError('Calibration order must be frozen')
    for row in rows:
        if row['image_id'] not in approved or row['split']!='train':raise ValueError('Calibration contains unapproved/held-out image')
        if row['contract']!=contract:raise ValueError('Mixed input contracts in calibration package')
        tensor,contract=model_uint8(verified_rgb(root,records[row['image_id']]),row['contract']['model_wh'][0])
        expected=tensor_receipt(tensor,contract)
        if any(row[key]!=value for key,value in expected.items()):raise ValueError('Calibration tensor metadata changed')
        if digest(source_file(prepared,row['file']).read_bytes())!=row['sha256']:
            raise ValueError('Calibration tensor bytes changed')
    return {'verified_host_preparation':True,'samples_count':len(rows),
            'compiler_executed':False,'native_ai2d_equivalence_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','audit','export-yolo','calibration','verify-calibration'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--size',type=int,choices=(224,320),default=320);args=parser.parse_args()
    value=json.loads(args.input.read_text(encoding='utf-8-sig'))
    if args.action not in ('audit','verify-calibration') and args.output is None:parser.error('Output path required')
    if args.action=='register':
        result=register(args.root,value)
        if args.output is None or args.output.exists():parser.error('New manifest output required')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
        print(json.dumps(audit(args.root,result),ensure_ascii=False));return
    if args.action=='audit':result=audit(args.root,value)
    elif args.action=='verify-calibration':
        if args.prepared is None:parser.error('Prepared calibration folder required')
        result=verify_calibration(args.root,value,args.prepared)
    elif args.action=='export-yolo':result=export_yolo(args.root,value,args.output)
    else:result=prepare_calibration(args.root,value,args.output,args.size)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
