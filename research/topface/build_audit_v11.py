"""Rebuild bounded G0/G1 evidence from the two real repositories, read-only."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

ROOT=Path(__file__).resolve().parents[2]
LIVE=Path(r'C:/Users/28389/Desktop/k230worker-live')
OUT=ROOT/'docs/topface/v11'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def text(path,value):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value+'\n',encoding='utf8')
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,encoding='utf8').strip()


def main():
    relative=Path('k230_transport_mvp/data/ai_pilot/material_manifest_reviewed_r4_20261110.json')
    r4=json.loads((ROOT/relative).read_text(encoding='utf8'))
    if sha(ROOT/relative)!=sha(LIVE/relative):raise ValueError('Original and imported r4 differ')
    approved=[i for i in r4['images'] if i['review']['status']=='APPROVED']
    objects=[o for i in r4['images'] for o in i['review']['objects']]
    approved_objects=[o for i in approved for o in i['review']['objects']]
    checks=[]
    for image in r4['images']:
        for capture in image['captures']:
            for role,item in capture['files'].items():
                path=ROOT/'k230_transport_mvp'/item['path']
                actual=sha(path)
                checks.append(dict(role=role,path=item['path'],bytes=path.stat().st_size,sha256=actual,passed=actual==item['sha256']))
    if not all(i['passed'] for i in checks):raise ValueError('Capture source hash mismatch')
    groups=defaultdict(set)
    for image in approved:groups[image['scene_group_id']].add(image['split'])
    inventory={'manifest_file':str(relative).replace('\\','/'),'manifest_bytes':(ROOT/relative).stat().st_size,
        'manifest_sha256':sha(ROOT/relative),'source_images':len(r4['images']),
        'all_top_bboxes':len(objects),'approved_images':len(approved),'approved_top_bboxes':len(approved_objects),
        'unreviewed_images':len(r4['images'])-len(approved),'uncertain_top_bboxes':sum(bool(o.get('uncertain')) for o in objects),
        'source_wh':[640,360],'source_format':'RGB565','source_frame_bytes':460800,
        'approved_color_bbox_counts':dict(Counter(o['color_id'] for o in approved_objects)),
        'all_color_bbox_counts':dict(Counter(o['color_id'] for o in objects)),
        'approved_splits':{s:{'images':sum(i['split']==s for i in approved),
            'bboxes':sum(len(i['review']['objects']) for i in approved if i['split']==s),
            'groups':len({i['scene_group_id'] for i in approved if i['split']==s})} for s in ['train','val','test']},
        'group_cross_split_count':sum(len(v)>1 for v in groups.values()),'capture_file_hashes_checked':len(checks),
        'capture_file_hash_mismatches':0,'real_ellipse_ground_truth_available':False,
        'historical_test_previously_evaluated':True,'fresh_blind_test_available':False,'capture_witnesses':checks}
    save(Path('audit/DATA_INVENTORY.json'),inventory)
    identities={}
    for name,root in [('live',LIVE),('optimized',ROOT)]:
        identities[name]={'root':git(root,'rev-parse','--show-toplevel'),'branch':git(root,'branch','--show-current'),
            'head':git(root,'rev-parse','HEAD'),'status':git(root,'status','--short','--branch'),
            'remotes':git(root,'remote','-v'),'branches':git(root,'branch','--all'),
            'recent_log':git(root,'log','-n','10','--oneline','--decorate'),
            'git_root_commits':git(root,'rev-list','--max-parents=0','HEAD')}
    imported=json.loads((ROOT/'docs/UPSTREAM_IMPORT.json').read_text(encoding='utf8'))
    identities['import']={k:v for k,v in imported.items() if k!='copied_files'}
    identities['import']['actual_manifest_file_count']=len(imported['copied_files'])
    identities['observed_utc']=datetime.now(timezone.utc).isoformat()
    save(Path('audit/REPOSITORY_IDENTITIES.json'),identities)
    contract={'schema':'topface-current-model-audit-v1.1','scope':'HISTORICAL_PILOT_NOT_ACCEPTED',
        'onnx_input':{'name':'images','shape':[1,3,224,224],'dtype':'float32','color':'RGB','layout':'NCHW','normalization':'uint8/255 once'},
        'preprocessing':{'source_wh':[640,360],'resized_wh':[224,126],'padding_tblr':[49,49,0,0],
                         'pad_value':114,'host_interpolation':'OpenCV INTER_LINEAR','board_ai2d':'tf_bilinear / half_pixel'},
        'onnx_output':{'name':'output0','shape':[1,5,1029],'encoding':'decoded cxcywh + one declared probability'},
        'kmodel_input':{'shape':[1,3,224,224],'dtype':'uint8','compiler_preprocess':True},
        'versions':{'historical_board_runtime':'2.11.0','compiler':'2.11.0','kpu_plugin':'2.11.0','current_board_readback':'UNKNOWN'},
        'evidence':['tools/material_model_contract.py:64-94','tools/material_preprocess.py:8-26',
                    'vision/yolo_candidates.py:12-23','docs/validation/frozen-model-fps-20261112/conversion.json',
                    'LIVE/k230_transport_mvp/evidence/board/kpu-green-black-20261109-e/kpu-bounded-probe-after-cleanup-native.json:67'],
        'limits':{'task_accuracy':False,'native_ai2d_pixel_equivalence':False,'live_csi2_rgb888p_capture':False,
                  'test_boxes':5,'test_tp':0,'test_fp':0,'test_fn':5,'optimizer_updates':4},
        'new_model_required':'Independent multi-color/physical-center/ellipse contract; old square single-class validator is not applicable'}
    save(Path('audit/CURRENT_MODEL_CONTRACT.json'),contract)
    text(Path('audit/PROJECT_AUDIT.md'),f'''# V1.1 项目审计（G0）

当前实验根为 `{ROOT}`；原始工程 `{LIVE}` 保持只读。两者是独立 Git 历史，非同一个 worktree。当前 hub README:10 指定 optimized 为开发入口，因此继续在其 `research/topface` 与 `docs/topface/v11` 隔离实验；不改生产源码。精确 HEAD、分支、remote、工作区与10条历史见 `REPOSITORY_IDENTITIES.json`。

optimized 的 `docs/UPSTREAM_IMPORT.json:2` 记录从旧仓库 `ops/k230-fps` 的 `0c3b8499e2877bc05df786921e199ab0533685d2` 导入，实际清单 {len(imported['copied_files'])} 项；live 当前 HEAD 为 `{identities['live']['head']}`。旧 README 的311项统计不能替代实际清单。

## 相机与输入链路

- `k230_transport_mvp/k230/camera_roles.py:1-5`：COLOR/TOP/RING→CSI2，QR/LANE→CSI1，CSI0隔离。
- `k230/camera_manager.py:65-76` 初始化 Sensor、默认 RGB565；`:106-119` 有界 snapshot、禁止自动重启和实际尺寸校验；`:126-145` 清理 sensor/display/media。
- 通用 `k230/config.py:13-14` 仍是640×480；审核真图 metadata 实际为640×360。不能修改旧配置来消除差异。
- `tools/prepare_frozen_model_fps.py:83-88` 从真实 RGB565 经主机转换后 transpose 生成691200字节RGB888P冻结文件。因此冻结文件 planar布局已验证，实时CSI2 RGB888P channel/stride/DMA寿命仍UNKNOWN。
- 历史固件传感器枚举为GC2093 CSI1/2，native1920×1080@30/60、1280×960/720@90；初始化帧率不是完整处理FPS。

## AI2D / KPU

`tools/kpu_pipeline_probe.py:137` 显式绑定存活的输入 backing；`:179-185` 配置NCHW、pad49/49、114、tf_bilinear/half_pixel；`:194-215` readinto691200→AI2D→KPU→输出拷贝；`:245-257` 清理对象与memory pool。69.29FPS历史证据是冻结文件循环且 camera/display=false，不能充当新模型端到端FPS。

历史runtime原始收据为nncase2.11.0，编译器及plugin也为2.11.0。具体证据见 CURRENT_MODEL_CONTRACT.json。新模型必须重新比较PyTorch/ORT/Simulator/KPU，不能把历史可运行性当识别通过。

## 数据与保护

r4 SHA256 `{inventory['manifest_sha256']}` 在两仓一致；重新核验 {len(checks)} 个capture文件均匹配。12图/33顶面，7批准图/17框，5困难图；实际uncertain黑框为5个（含四目标困难图），历史文字“4个”不完整。3个批准scene组无跨split。旧test已在2026-11-13被评估，不能称全新盲测。

颜色映射以 `tools/material_annotation_schema.py:3`、`data/spec/human.md:86`、`data/spec/ai01.md:75` 核对：1红、2黄、3蓝、4绿、5黑、6浅蓝。无真实椭圆真值。

UART、运动、生产Save保持关闭；CSI0隔离；CSI1不改；main.py/boot.py、r4、历史模型及用户草稿不覆盖。本轮无新的板端调用。
''')
    text(Path('audit/REUSE_MAP.md'),'''# 可复用代码地图

| 入口 | 复用范围 | 不能继承的结论 |
|---|---|---|
| k230/camera_manager.py:65-145 | 已有Sensor生命周期、尺寸校验和有界snapshot | 实时RGB888P通道/stride未验证 |
| tools/material_preprocess.py:8-26 | RGB565主机参考、letterbox、单次归一化 | 新模型尺寸/颜色/mean/std需另定 |
| tools/kpu_pipeline_probe.py:137-257 | backing存活、显式绑定、readinto、资源释放 | 冻结图FPS不等于camera-to-display |
| tools/material_dataset.py:76-196 | 审核/uncertain门、scene分组、去重、train-only PTQ | 旧单类模型形状门不可套新模型 |
| tools/probe_nncase_normalization.py | Windows plugin/DLL、CompileOptions、Simulator方法 | 小常量图不是任务模型兼容性 |
| research/topface/native_cad.py | 已验证原始单网格、轴向修正和顶面材质索引 | 原CAD几何不拆分 |
| research/topface/render_native.py | 原生bpy、多实例、相机投影校验、同次render的材质Mask | 本地通过不等于Kaggle/GPU通过 |
| research/topface/audit_v11.py | 物理中心/conic中心分离、可见框、5实例、空场、相机分组 | 合成QA不是真实识别率 |
| research/topface/cloud_store.py | Drive全字节SHA读回、唯一对象名 | CLI Save不保证运行时Secret关联 |
''')
    text(Path('audit/RISK_AND_UNKNOWN.md'),'''# 风险与未知

- UNKNOWN：当前板端身份/runtime新鲜读回、实时RGB888P channel/stride/DMA寿命、曝光/gain/WB/ISP、真实内参/畸变、近距离清晰度、工作区域毫米标定、完整链路FPS。
- 小固定镜头标称约70cm对焦，用户工作距离10–40cm；必须用真实参考图校准失焦，不能把清晰合成图当完成Sim2Real。
- 用户确认80–100°主工作范围，强透视超范围分组未声称覆盖。正常物块直立，不擅自增加侧躺训练样本。
- 当前随机背景/打印纹理/退化仍需实物校准；本地100 smoke与人工抽查是几何/外观门，不是模型质量门。
- 现有r4仅7批准图、17框，旧test已用；缺新scene/session盲测及真实椭圆/中心真值。
- NanoDet/PicoDet仅兼容性短测，不完整训练两个模型；新增六色/物理中心/椭圆head也需重新编译，原COCO图编译成功不能替代它。
- 旧Kaggle v8完成Drive115字节读回后在BlenderProc matplotlib backend安装冒烟失败；0渲染、0updates。用户已停止该旧运行，后续用新的run身份。
- 更新后cloud-direct-write允许Notebook直写或独立获授权云发布器；若选择新服务接收临时凭据，先核实该目的地授权。不得在Notebook/source/Git放凭据。
- V1.1引用的旧Execution Package ZIP曾被定位但当前路径已不可读；最新两份V1.1与原CAD包均可用，不把缺失包内容冒充已完整阅读。
''')
    previous=json.loads((ROOT/'docs/topface/hardware_profile.json').read_text())
    parameters={}
    for key,item in previous['parameters'].items():
        if key in {'model_input_width','model_input_height','preprocessing','model_color_order'}:
            key='historical_pilot_'+key
        status='UNKNOWN' if item['status']=='unknown' else ('CONFIRMED' if item['status'] in {'measured','read_from_config','from_datasheet'} else 'INFERRED')
        parameters[key]={'value':item['value'],'unit':item['unit'],'status':status,
            'source_evidence':item['source'],'confidence':item['confidence'],'verification_method':item['verification_method']}
    parameters['realtime_rgb888p_stride_bytes']={'value':None,'unit':'bytes','status':'UNKNOWN','source_evidence':'No live RGB888P channel readback','confidence':'unknown','verification_method':'Requires current board probe'}
    parameters['capture_profile']={'value':{'format':'RGB565','width':640,'height':360,'bytes':460800},'unit':None,'status':'CONFIRMED','source_evidence':'r4 capture hashes/metadata; historical','confidence':'high','verification_method':'36 source file hashes and metadata'}
    parameters['frozen_kpu_file_profile']={'value':{'format':'RGB planar uint8','shape':[1,3,360,640],'row_bytes':640,'plane_bytes':230400,'frame_bytes':691200},'unit':None,'status':'CONFIRMED','source_evidence':'prepare_frozen_model_fps.py:83-88','confidence':'high','verification_method':'Historical host-generated frozen files, not live DMA buffers'}
    parameters['new_model_input_candidates']={'value':[[224,128],[320,192],[384,224],[320,180]],'unit':'pixels W,H','status':'CONFIRMED','source_evidence':'V1.1 plan section10.2','confidence':'high','verification_method':'Experiment candidates; optional320x180 neck alignment must be checked'}
    profile={'schema':'topface-camera-profile-v1.1','fresh_board_readback_this_run':False,'parameters':parameters}
    (OUT/'camera/camera_profile.yaml').write_text(yaml.safe_dump(profile,allow_unicode=True,sort_keys=False),encoding='utf8')
    sweep={'height_above_top_mm':[100,175,250,325,400],'plane_angle_deg':[80,85,90,95,100],
        'horizontal_fov_deg':[62,65,68],'fov_status':'INFERRED around vendor nominalH65',
        'camera_motion':'vertical, user confirmed','normal_object_pose':'upright',
        'working_area_mm':None,'distortion_coefficients':None,'focus_blur_calibrated':False,
        'split_grouping':'height + plane_angle + HFOV; entire configuration group in one split'}
    (OUT/'camera/camera_sweep.yaml').write_text(yaml.safe_dump(sweep,allow_unicode=True,sort_keys=False),encoding='utf8')
    text(Path('camera/camera_profile_evidence.md'),'# 相机证据\n\n逐字段来源见 camera_profile.yaml；历史sensor与capture原始摘录及文件SHA见 ../../hardware_evidence.json，数据新鲜核验见 ../audit/DATA_INVENTORY.json。用户已确认小固定镜头、镜头到顶面100–400mm、垂直运动、80–100°。官方镜头规格：https://wiki.lckfb.com/zh-hans/lushan-pi-k230/image-recog/use-sensor.html 。标称规格不能替代内参标定。')
    text(Path('camera/calibration_gap_report.md'),'# 标定缺口\n\n尚无真实fx/fy/cx/cy及畸变系数、reprojection error、安装roll/yaw、工作区域或真实中心/椭圆真值。后续在实际CSI2分辨率/焦点拍棋盘格或ChArUco，并覆盖100–400mm。当前以可追溯FOV sweep开展仿真，不等待全部真实图完成；失焦/ISP差异仍UNKNOWN。')
    print(json.dumps({k:inventory[k] for k in ['source_images','all_top_bboxes','approved_images','approved_top_bboxes','uncertain_top_bboxes','capture_file_hashes_checked','group_cross_split_count']},indent=2))


if __name__=='__main__':main()
