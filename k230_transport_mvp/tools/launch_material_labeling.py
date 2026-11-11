"""Operator-invoked mature annotation tool; never captures images or trains."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'k230_transport_mvp'))
from tools.material_annotation_schema import SURFACE_SCHEMA,SURFACE_LABELS,LABEL_MAP
from tools.material_annotation_schema import decode_surface_label
from k230.block_profile import DRAW_COLORS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'k230_transport_mvp/dist/xanylabeling_material_20261109_h')
    parser.add_argument('--preset',choices=('visible_top','colored_surfaces'))
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    executable=ROOT/'.ops/local/venvs/kpu-label/Scripts/xanylabeling.exe'
    if not executable.exists():raise SystemExit('X-AnyLabeling environment missing; see ANNOTATION_WORKFLOW_20261109.md')
    folder=args.bundle.resolve()
    if not (folder/'images').is_dir():raise SystemExit('Annotation images folder missing')
    package_path=folder/'package.json'
    package=json.loads(package_path.read_text(encoding='utf-8')) if package_path.is_file() else {}
    if package.get('annotation_schema') is not None and package['annotation_schema']!=SURFACE_SCHEMA:
        raise SystemExit('Unknown annotation package schema')
    if package.get('annotation_schema')==SURFACE_SCHEMA and (package.get('annotation_labels')!=list(SURFACE_LABELS)
                                                            or package.get('annotation_label_map')!=LABEL_MAP):
        raise SystemExit('Annotation package label contract changed')
    automatic='colored_surfaces' if package.get('annotation_schema')==SURFACE_SCHEMA else 'visible_top'
    preset=args.preset or automatic
    if args.preset is not None and package_path.is_file() and preset!=automatic:
        raise SystemExit('Preset differs from annotation package contract')
    labels=list(SURFACE_LABELS) if preset=='colored_surfaces' else ['visible_top']
    config=ROOT/'.ops/local/xanylabeling-settings'/('material-surfaces-v1.yaml' if preset=='colored_surfaces' else 'material-config-h2.yaml')
    config.parent.mkdir(parents=True,exist_ok=True)
    if not config.exists():
        extra=('shape_color: manual\nlabel_colors: '+json.dumps({label:list(DRAW_COLORS[decode_surface_label(label)['color_id']]) for label in labels},ensure_ascii=False)+'\nshow_groups: true\n') if preset=='colored_surfaces' else ''
        config.write_text('language: zh_CN\nfont_family: Microsoft YaHei UI\nlabels: '+json.dumps(labels,ensure_ascii=False)+'\nvalidate_label: exact\nauto_save: false\n'
            'store_data: false\nkeep_prev: false\ndevice: CPU\nflags: [uncertain, ignore]\n'
            'auto_use_last_label: false\nauto_use_last_gid: false\ndisplay_label_popup: true\n'
            'show_texts: false\nshow_scores: false\nshow_attributes: false\n'
            +extra+
            'label_flags:\n'+''.join('  '+json.dumps(label,ensure_ascii=False)+': [uncertain, truncated]\n' for label in labels),encoding='utf-8')
    work=ROOT/'.ops/local/xanylabeling-settings'/('work-surfaces-v1' if preset=='colored_surfaces' else 'work');work.mkdir(exist_ok=True)
    cmd=[str(executable),'--no-auto-update-check','--config',str(config),'--work-dir',str(work),
         '--labels',','.join(labels),'--validatelabel','exact','--nodata','--filename',str(folder/'images'),
         '--output',str(folder/'images')]
    if args.check_only:
        print(json.dumps({'tool':'X-AnyLabeling4.0.6','bundle':str(folder),'captures_images':False,
                          'training_started':False,'human_approval_automatic':False,
                          'annotation_preset':preset,'labels':labels,'config':str(config)},ensure_ascii=False));return
    env=os.environ.copy();env['QT_API']='pyqt6'
    raise SystemExit(subprocess.call(cmd,env=env,cwd=ROOT))


if __name__=='__main__':main()
