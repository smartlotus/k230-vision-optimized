"""Build an offline human review page; creating the page approves no labels."""
import argparse,base64,hashlib,json
from pathlib import Path
import cv2
from tools.material_dataset import audit,verified_rgb,canonical


def build(root,manifest,drafts,output):
    report=audit(root,manifest)
    if drafts.get('status')!='DRAFT_ASSISTED' or drafts.get('human_approved') is not False:
        raise ValueError('Only explicitly unapproved assisted drafts are accepted')
    indexed={r['image_id']:r for r in drafts['images']}
    if not set(indexed).issubset({r['image_id'] for r in manifest['images']}):raise ValueError('Drafts do not match source images')
    images=[]
    for record in manifest['images']:
        rgb=verified_rgb(root,record);ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if not ok:raise OSError('Cannot encode source image')
        draft=indexed.get(record['image_id'],{})
        if draft and draft.get('source_native_sha256')!=record['native_sha256']:raise ValueError('Draft source identity mismatch')
        images.append({'image_id':record['image_id'],'image_url':'data:image/png;base64,'+base64.b64encode(png).decode(),
                       'draft_objects':draft.get('objects',[])})
    payload={'manifest':manifest,'manifest_sha256':report['manifest_sha256'],'images':images,
             'original_human_approved_images':report['eligible_images']}
    template=Path(__file__).with_name('material_review.html').read_text(encoding='utf-8')
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    text=template.replace('/*__REVIEW_PAYLOAD__*/',encoded)
    output=Path(output)
    if output.exists():raise ValueError('Choose a new review page path; do not overwrite an existing review')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    return {'path':str(output.resolve()),'images':len(images),'assisted_boxes':sum(len(r['draft_objects']) for r in images),
            'human_approved_by_build':0,'training_started':False,'sha256':hashlib.sha256(text.encode()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--drafts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(build(a.root,json.loads(a.manifest.read_text(encoding='utf-8')),json.loads(a.drafts.read_text(encoding='utf-8')),a.output),ensure_ascii=False))
