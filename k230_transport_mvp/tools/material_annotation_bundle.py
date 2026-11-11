"""Exchange unapproved visible-top drafts with mature desktop annotation tools."""
import argparse,base64,copy,hashlib,json,math,os,sys
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.material_dataset import audit,canonical,source_file,verified_rgb
from tools.material_annotation_schema import SURFACE_SCHEMA,SURFACE_LABELS,LABEL_MAP,decode_surface_label,label_for


def sha(raw):return hashlib.sha256(raw).hexdigest()


def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as out:json.dump(value,out,ensure_ascii=False,indent=2)


def write_windows_launcher(root,bundle):
    """Write only the operator launcher; never regenerate images or annotations."""
    project=Path(root).resolve().parent;bundle=Path(bundle).resolve()
    try:relative=os.path.relpath(project,bundle).replace('\\','/')
    except ValueError:relative=None  # Different Windows drives need an absolute root.
    location='%~dp0'+relative if relative is not None else str(project).replace('\\','/')
    text='\r\n'.join(('@echo off','setlocal','set "PROJECT_ROOT='+location+'"',
        '"%PROJECT_ROOT%/.ops/local/venvs/kpu-label/Scripts/python.exe" '
        '"%PROJECT_ROOT%/k230_transport_mvp/tools/launch_material_labeling.py" --bundle "%~dp0."',
        'if errorlevel 1 pause',''))
    target=bundle/'start-labeling.cmd';target.write_bytes(text.encode('utf-8'))
    return target


def make_bundle(root,manifest,drafts,output,format='xanylabeling',preset='visible_top'):
    report=audit(root,manifest)
    if format not in ('xanylabeling','labelme'):raise ValueError('Explicit desktop annotation format required')
    if preset not in ('visible_top','colored_surfaces'):raise ValueError('Explicit annotation preset required')
    if preset=='colored_surfaces' and format!='xanylabeling':raise ValueError('Colored surface preset requires X-AnyLabeling')
    if drafts.get('status')!='DRAFT_ASSISTED' or drafts.get('human_approved') is not False:
        raise ValueError('Assisted drafts must remain unapproved')
    indexed={r['image_id']:r for r in drafts['images']}
    if len(indexed)!=len(drafts['images']) or not set(indexed).issubset({r['image_id'] for r in manifest['images']}):
        raise ValueError('Draft identities do not match native source manifest')
    images=[]
    for record in manifest['images']:
        rgb=verified_rgb(root,record);draft=indexed.get(record['image_id'])
        if draft and draft.get('source_native_sha256')!=record['native_sha256']:raise ValueError('Draft source hash mismatch')
        shapes=[]
        for obj in draft.get('objects',[]) if draft else []:
            if obj.get('class_id')!=0:raise ValueError('Material stage has only visible_top')
            x,y,w,h=obj['bbox_xywh'];points=[[x,y],[x+w,y+h]]
            if format=='xanylabeling':points=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            shapes.append({'label':'visible_top','points':points,'group_id':None,'shape_type':'rectangle',
                'description':'DRAFT_ASSISTED; operator review required','difficult':False,
                'flags':{'uncertain':bool(obj.get('uncertain')),'truncated':bool(obj.get('truncated'))},
                'attributes':{'color_id':str(obj['color_id'])} if obj.get('color_id') else {}})
            if preset=='colored_surfaces':
                shapes[-1]['label']=label_for(obj.get('color_id'),'top')
                shapes[-1]['attributes']={}  # Color is carried by the reviewed label.
        filename=record['image_id']+'.png'
        document={'version':'4.0.6' if format=='xanylabeling' else '5.6.1','flags':{},'checked':False,
            'shapes':shapes,'imagePath':filename,'imageData':None,'imageHeight':360,'imageWidth':640}
        ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if not ok:raise OSError('Cannot create lossless source PNG')
        images.append((record,document,png.tobytes()))
    output=Path(output)
    if output.exists():raise ValueError('Use a new annotation package version; existing edits must be preserved')
    (output/'images').mkdir(parents=True)
    package={'schema_version':1,'format':format,'dataset_manifest_sha256':report['manifest_sha256'],
        'classes':['visible_top'],'human_approved_by_export':0,'images':[]}
    if preset=='colored_surfaces':
        package.update(annotation_schema=SURFACE_SCHEMA,annotation_labels=list(SURFACE_LABELS),
                       annotation_label_map=LABEL_MAP,side_annotations_training_enabled=False)
    for record,document,png in images:
        key=record['image_id'];png_file='images/'+key+'.png';label_file='images/'+key+'.json'
        (output/png_file).write_bytes(png);save_json(output/label_file,document)
        package['images'].append({'image_id':key,'native_sha256':record['native_sha256'],
            'png_file':png_file,'png_sha256':sha(png),'label_file':label_file,
            'initial_annotation_sha256':sha((output/label_file).read_bytes()),
            'draft_shapes':len(document['shapes'])})
    save_json(output/'package.json',package)
    labels=SURFACE_LABELS if preset=='colored_surfaces' else ('visible_top',)
    (output/'labels.txt').write_text('\n'.join(labels)+'\n',encoding='utf-8')
    if format=='xanylabeling':write_windows_launcher(root,output)
    (output/'README.txt').write_text('在X-AnyLabeling中打开images目录。类别统一visible_top，仅框实际可见顶面，不含侧壁/阴影/纸靶。\n'
        '修正草稿、补漏标、删除误标后逐张保存。新采4帧没有草稿，需要补齐。\n'
        '不确定或严重截断帧先标记困难并告诉Agent排除；不要直接删掉一个真实物块后当背景。\n'
        '软件checked标记不是项目训练审批。完成后提供本目录，并说明实际核对了哪些图；Agent验证源哈希和场景分组。\n'
        '保存图像/标注不会启动训练。请不要修改或缩放原始PNG。\n',encoding='utf-8')
    if preset=='colored_surfaces':
        (output/'README.txt').write_text('此包预设六色×顶面/侧面12个中文标签。顶面和侧面都画各自可见区域的紧外接矩形。\n'
            '当前顶面任务导出为一类visible_top并保留color_id；侧面只存辅助层，不进入顶面训练。\n'
            '同物块两面可填写相同group_id，编号只在本帧使用。不必强凑不可见的面。\n'
            '所有现有框仍为辅助草稿，需人工修正/补漏；保存和checked不自动构成人工审批。\n'
            '旧包和旧窗口保留。若旧窗口有未保存编辑，先保存并告知Agent合并，不要重新生成覆盖JSON。\n',encoding='utf-8')
    return {'path':str(output.resolve()),'format':format,'images':len(images),
        'draft_boxes':sum(len(d['shapes']) for _,d,_ in images),'human_approved_by_export':0,'training_started':False,
        'annotation_preset':preset,'annotation_labels':list(labels)}


def read_rectangle(shape):
    if shape.get('label')!='visible_top' or shape.get('shape_type')!='rectangle':
        raise ValueError('Only visible_top rectangles accepted in this detector stage')
    points=shape.get('points')
    if not isinstance(points,list) or len(points) not in (2,4):raise ValueError('Rectangle needs two diagonals or four corners')
    if any(not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points):
        raise ValueError('Finite source-pixel points required')
    x0=min(p[0] for p in points);x1=max(p[0] for p in points)
    y0=min(p[1] for p in points);y1=max(p[1] for p in points)
    if not 0<=x0<x1<=640 or not 0<=y0<y1<=360:raise ValueError('Visible box must fit the source image')
    if len(points)==4 and set(map(tuple,points))!={(x0,y0),(x1,y0),(x1,y1),(x0,y1)}:
        raise ValueError('Rotated/polygon geometry must not be silently converted to a detection box')
    flags=shape.get('flags',{})
    if not isinstance(flags,dict) or any(type(v) is not bool for v in flags.values()):raise ValueError('Boolean shape flags required')
    color=shape.get('attributes',{}).get('color_id')
    if color is not None:
        if isinstance(color,bool) or str(color) not in tuple(str(i) for i in range(1,7)):raise ValueError('Invalid optional color attribute')
        color=int(color)
    return {'class_id':0,'bbox_xywh':[x0,y0,x1-x0,y1-y0],'color_id':color,
        'uncertain':bool(shape.get('difficult')) or flags.get('uncertain',False) or flags.get('ignore',False),
        'truncated':flags.get('truncated',False),'geometry_complete_verified':False}


def read_surface_rectangle(shape):
    info=decode_surface_label(shape.get('label'))
    copied=copy.deepcopy(shape);copied['label']='visible_top'
    attributes=copied.get('attributes',{})
    if not isinstance(attributes,dict):raise ValueError('Structured annotation attributes required')
    if attributes.get('color_id') is not None and str(attributes['color_id'])!=str(info['color_id']):
        raise ValueError('Surface label and explicit color attribute conflict')
    copied['attributes']=dict(attributes,color_id=info['color_id'])
    result=read_rectangle(copied)
    group=shape.get('group_id')
    if group is not None and (type(group) is not int or not 1<=group<=255):
        raise ValueError('Optional positive per-image object group1..255 required')
    result.update(surface=info['surface'],annotation_label=shape['label'],object_group_id=group)
    if info['surface']=='side':result.pop('class_id')
    return result


def check_surface_groups(rows):
    groups={}
    for row in rows:
        group=row['object_group_id']
        if group is None:continue
        state=groups.setdefault(group,{'color_id':row['color_id'],'surfaces':set()})
        if state['color_id']!=row['color_id'] or row['surface'] in state['surfaces']:
            raise ValueError('Same object group must share color and contain at most one top/side')
        state['surfaces'].add(row['surface'])


def import_bundle(root,manifest,bundle,reviewer=None,reviewed_ids=()):
    report=audit(root,manifest);bundle=Path(bundle)
    package=json.loads((bundle/'package.json').read_text(encoding='utf-8'))
    if package.get('dataset_manifest_sha256')!=report['manifest_sha256'] or package.get('classes')!=['visible_top']:
        raise ValueError('Source manifest changed after annotation export')
    schema=package.get('annotation_schema')
    if schema is not None and (schema!=SURFACE_SCHEMA or package.get('annotation_labels')!=list(SURFACE_LABELS)
                               or package.get('annotation_label_map')!=LABEL_MAP):
        raise ValueError('Unknown or changed surface annotation contract')
    records={r['image_id']:r for r in manifest['images']};entries=package['images']
    if len(entries)!=len(records) or {e['image_id'] for e in entries}!=set(records):raise ValueError('Incomplete/duplicate package image identities')
    reviewed=set(reviewed_ids)
    if not reviewed.issubset(records):raise ValueError('Human review list includes unknown images')
    if reviewed and (not isinstance(reviewer,str) or not reviewer.strip()):raise ValueError('Actual named operator review required')
    result=copy.deepcopy(manifest);updated={r['image_id']:r for r in result['images']};receipts=[]
    for entry in entries:
        key=entry['image_id'];record=records[key];verified_rgb(root,record)
        if entry.get('native_sha256')!=record['native_sha256']:raise ValueError('Package native source identity changed')
        png_path=source_file(bundle,entry['png_file']);label_path=source_file(bundle,entry['label_file'])
        if sha(png_path.read_bytes())!=entry['png_sha256']:raise ValueError('Annotation source PNG changed')
        image=cv2.imread(str(png_path))
        if image is None or image.shape!=(360,640,3) or sha(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes())!=record['rgb_pixel_sha256']:
            raise ValueError('Annotation pixels differ from native source')
        raw=label_path.read_bytes();document=json.loads(raw)
        if (document.get('imagePath')!=png_path.name or document.get('imageWidth')!=640
                or document.get('imageHeight')!=360):
            raise ValueError('Wrong annotation image binding')
        if document.get('imageData') is not None:
            import numpy as np
            encoded=document['imageData']
            if not isinstance(encoded,str):raise ValueError('Embedded annotation image must be base64 text')
            embedded=base64.b64decode(encoded,validate=True)
            decoded=cv2.imdecode(np.frombuffer(embedded,np.uint8),cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape!=(360,640,3) or sha(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB).tobytes())!=record['rgb_pixel_sha256']:
                raise ValueError('Embedded image pixels differ from native source')
        shapes=document.get('shapes')
        if not isinstance(shapes,list):raise ValueError('Missing annotation list is not an empty negative')
        if schema==SURFACE_SCHEMA:
            surfaces=[read_surface_rectangle(shape) for shape in shapes];check_surface_groups(surfaces)
            objects=[row for row in surfaces if row['surface']=='top']
            sides=[row for row in surfaces if row['surface']=='side']
        else:
            objects=[read_rectangle(shape) for shape in shapes];sides=[]
        flags=document.get('flags',{})
        if not isinstance(flags,dict) or any(type(v) is not bool for v in flags.values()):raise ValueError('Boolean image flags required')
        uncertain=bool(flags.get('ignore')) or bool(flags.get('uncertain')) or any(o['uncertain'] for o in objects+sides)
        if key in reviewed and (uncertain or len(objects)>3 or len(sides)>3):raise ValueError('Resolve uncertain/over-count frame before approving')
        approved=key in reviewed
        updated[key]['review']={'status':'APPROVED' if approved else 'UNREVIEWED',
            'reviewer':reviewer.strip() if approved else None,'all_visible_tops_checked':approved,
            'uncertain':uncertain,'source_native_sha256':record['native_sha256'],'objects':objects,
            'annotation_sha256':sha(raw),'annotation_file':label_path.name,
            'approval_basis':'OPERATOR_EXPLICIT_REVIEW_LIST' if approved else 'SOFTWARE_SAVE_OR_CHECKED_FLAG_IS_NOT_APPROVAL'}
        if schema==SURFACE_SCHEMA:
            updated[key]['side_annotations']={'annotation_schema':SURFACE_SCHEMA,'objects':sides,
                'annotation_sha256':sha(raw),'training_enabled':False,
                'reviewer':reviewer.strip() if approved else None,
                'status':'REVIEWED_AUXILIARY_ONLY' if approved else 'UNREVIEWED',
                'all_visible_sides_checked':False}
        receipts.append({'image_id':key,'annotation_sha256':sha(raw),'software_checked':bool(document.get('checked')),'human_approved':approved})
    return result,{'source_manifest_sha256':report['manifest_sha256'],'format':package['format'],
        'human_approved_images':len(reviewed),'images':receipts,'training_started':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('export','import'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--drafts',type=Path);p.add_argument('--format',choices=('xanylabeling','labelme'),default='xanylabeling')
    p.add_argument('--preset',choices=('visible_top','colored_surfaces'),default='visible_top')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--reviewer');p.add_argument('--reviewed-id',action='append',default=[]);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    if a.action=='export':
        if a.drafts is None:p.error('Assisted draft JSON required')
        print(json.dumps(make_bundle(a.root,manifest,json.loads(a.drafts.read_text(encoding='utf-8')),a.bundle,a.format,a.preset),ensure_ascii=False))
    else:
        if a.output is None or a.output.exists():p.error('New imported manifest output required')
        result,receipt=import_bundle(a.root,manifest,a.bundle,a.reviewer,a.reviewed_id)
        save_json(a.output,result);print(json.dumps(receipt,ensure_ascii=False))
