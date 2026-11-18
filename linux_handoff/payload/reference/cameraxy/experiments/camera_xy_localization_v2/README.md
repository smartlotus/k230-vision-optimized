# K230 CameraXY Localization V2.0

新 SD 卡已完成持久路径验证，详见 [新卡持久路径结果](USB_NATIVE_SD_RESULT_20261118.md)。三轮共 360 帧，工程链路和 SHA 回读通过。

`FIELD_MAP` 的原点是 2400 mm 场地的几何中心。X 向官方图纸右，Y 向上。每帧的监督映射只含 `camera_x_mm` 和 `camera_y_mm`，表示 USB OV2735 光心在地面的铅垂投影。相机内参、朝向、高度、OpenCV camera-to-world 矩阵、渲染设置和随机种子在 metadata 中。相机到机器人安装外参由用户实测并在控制层使用。

## 当前结果

| 验证范围 | 结果 |
|---|---|
| 仿真与 Kaggle | G0–G5 已完成。Smoke 100、Pilot 1,000，正式训练数据 10,000 个位置、21,000 帧。Tesla T4 训练 18 轮、4,500 次更新。 |
| 主机定位 | 融合方案在空间留出集 50 mm 内 Top-1 命中 32.45%，Top-5 命中 92.05%。单点验收为 HOLD。低置信度时主 X/Y 为 null。 |
| K230 有界工程测试 | learned_map 的 USB、RGB、AI2D、KPU、解码、预览和释放通过 360 帧实测，含一次软重启。测试沿用冻结工程模型，主坐标和置信度均为空。 |
| 真实定位与部署验收 | 仍需直接量测光心的独立实拍数据、融合方案的板端资格验证和持久部署验收。真实定位误差与置信度为 UNKNOWN。 |

[正式训练与留出评估](evidence/FULL_REPORT.md) · [板端实测和工程包](USB_NATIVE_RESULT_20261117.md) · [实拍采集契约](REAL_CAPTURE.md)

## OPS 接续

任务分支为 `ops/camera-xy-localization-v2`。在仓库根目录先读 [项目入口](../../AGENTS.md)、[项目地图](../../.ops/PROJECT.md) 和 [OPS 协议](../../.ops/PROTOCOL.md)，再运行：

```powershell
python .ops/tools/ops.py status
python .ops/tools/ops.py verify --task camera-xy-localization-v2
python .ops/tools/ops.py verify --task camera-xy-native-20261117
```

- [CameraXY 研发任务](../../.ops/tasks/camera-xy-localization-v2.json)保存整体状态；[原生工程任务](../../.ops/tasks/camera-xy-native-20261117.json)保存已完成的有界验收。两份任务 JSON 的 `handoff` 字段指向各自最新的机器可核验交接。
- 本目录根部是生成、训练、推理和板端脚本，`tests/` 是主机回归，`evidence/` 是日志、报告、指纹和小型工程包。`.runtime/` 是忽略的工作目录。
- 从 [资产身份](assets.json)和 [G0 审计](evidence/G0.md)取得 Blender 场景、RoadNet 输入、TopFaceNet GLB 与源包 SHA256。渲染数据和正式权重归档按 [精确 Kaggle 版本](https://www.kaggle.com/code/lotus30/k230-camera-xy-map-full-a01/versions/1)及 [归档回执](evidence/G5-kaggle-acceptance.json)恢复，需要原私有 Dataset 和 Notebook 的访问权限。复用输入前验 SHA256。
- 原生检查用 [板端实测报告](USB_NATIVE_RESULT_20261117.md)末尾的复核命令；[输入链对照](NATIVE_PROGRESS_20261117.md)、[AI2D 留出回放](AI2D_HOLDOUT.md)和 [工程准备记录](USB_ENGINEERING_20261117.md)保留早期阶段证据。板端运行须按 OPS 协调 COM3 与 USB 相机占用。

## 复现实验

打开本目录。需要 Python 3.11+、Pillow、NumPy。生成器需要本机 Blender 5.3.0 与 G0 记录中的 RoadNet 运行版场景和 TopFaceNet GLB。主机推理另需 PyTorch 与 ONNX Runtime。

```powershell
$roadnet = '<RoadNet 运行版目录，含 blender_scene.py、sim.py 和 configs/>'
$blend = "$roadnet/.runtime/smoke-v2/field.blend"
$layout = "$roadnet/configs/field_layout.json"
$roadModel = "$roadnet/evidence/cloud/full-v3/model/roadnet.onnx"
$cad = '<TopFaceNet 或 CAD 包内的 material_block.glb>'
$blender = '<blender.exe>'
python generate.py --stage smoke --blender $blender --blend $blend --roadnet $roadnet --cad $cad --out .runtime/smoke-a01
python generate.py --stage pilot --blender $blender --blend $blend --roadnet $roadnet --cad $cad --out .runtime/pilot-a01 --predecessor .runtime/smoke-a01
```

固定种子与每个 shard 的日志保存在输出目录。Smoke 有 100 个位置，Pilot 有 1,000 个。训练和验证位置各有两帧，测试位置各有两帧基础外观和一帧配对外观。连续空间块和缓冲带冻结。留出外观与测试配对图共用相机和几何。每个阶段的 QA 校验 PNG、矩阵、投影、位置与分组、RGB/掩膜 SHA256。中断后同一命令重新验证已完成 shard，并保留失败日志和部分输出。每次只运行一个相同输出根的生成进程。

Kaggle 使用私有 Dataset 和私有已保存版本。将上传目录作为工作目录运行 Kaggle CLI，`-p .` 让 Windows 上传状态使用短路径。并行上传时使用各自的进程级临时目录。可复用当前版本的 [Pilot 输入与参数](evidence/G3-kaggle-spec.json) 和 [已回读源码身份](evidence/G3-kaggle-version.json)。需要重建输入时运行下列准备命令。为新提交使用新的私有 run ID。

```powershell
python kaggle_prepare.py --data .runtime/pilot-a01 --layout $layout --roadnet $roadModel --ort-wheel $wheel --ref OWNER/NEW-RUN-SLUG --epochs 24 --device cpu --out .runtime/new-job
```

准备程序生成 Dataset 输入和 Notebook 启动源码。`--ort-dataset` 指向已就绪的离线 ONNX Runtime wheel Dataset。wheel 必须匹配 Kaggle 容器 Python/架构和 PyPI SHA256。生成后在 `.runtime/new-job/input` 使用 `kaggle datasets create -p . -t`。输入 ready 后用 `kaggle kernels push -p .runtime/new-job/kernel -t 2400` 执行。查看指定版本的状态，取回保存源码和有界输出后运行 `verify_kaggle.py`。Kaggle 展开输入 ZIP 时，启动脚本用固定 inventory SHA256 逐个核验已挂载文件。成果归档由 Kaggle 生成，验收读取保存版本中的全部归档字节。

```powershell
$job = '.runtime/kaggle-pilot-cpu-a02'
$archiveRef = 'lotus30/k230-camera-xy-pilot-v2-cpu-a02/1'
kaggle kernels pull $archiveRef -p "$job/pulled" -m
kaggle kernels output $archiveRef -p "$job/provider" --file-pattern '^([A-Za-z0-9_-]+\.(json|zip|log))$'
python verify_kaggle.py --spec "$job/SUBMISSION.json" --output "$job/provider" --source "$job/pulled/k230-camera-xy-pilot-v2-cpu-a02.py" --version 1 --receipt .runtime/pilot-verified.json
python ambiguity_report.py --data .runtime/pilot-a01 --pairs "$job/provider/ambiguity_pairs.json" --layout $layout --out .runtime/reproduced-ambiguity
```

`infer.py` 接受从验收归档提取的同次 `*-best.pt`、`metrics.json` 和 `--image`。校准门未满足时，主坐标为 null，状态为 `unlocalized`。 `semantic_infer.py` 接受同一场地图、RoadNet ONNX 和候选集合报告，可输出 5 个候选。集合置信度只对应该仿真校准域。真实图的独立测量与评估入口见 [REAL_CAPTURE.md](REAL_CAPTURE.md)。

## 10000 位置正式仿真

```powershell
python generate.py --stage full --blender $blender --blend $blend --roadnet $roadnet --cad $cad --out .runtime/full-a01 --predecessor .runtime/pilot-a01 --pilot-selection evidence/G4-full-selection.json --shard 250
python kaggle_prepare.py --recipe xy-map-v1 --data .runtime/full-a01 --layout $layout --roadnet $roadModel --ort-wheel $wheel --ref OWNER/UNIQUE-FULL-RUN --epochs 18 --device cuda --out .runtime/full-kaggle-a01
```

用相同参数恢复分片，已验证的分片会重新核对 SHA256。仅在完整 QA 通过后准备私有 Dataset 并上传。`xy-map-v1` 的容器 `camera_xy.bin` 在提供方保持原始字节，训练入口逐一核验归档成员。使用 [正式提交规格](evidence/G5-kaggle-spec.json)和保存版本源码、输出进行独立校验：

```powershell
$job = '.runtime/kaggle-map-full-a01'
$ref = 'lotus30/k230-camera-xy-map-full-a01/1'
kaggle kernels pull $ref -p "$job/pulled" -m
kaggle kernels output $ref -p "$job/provider" --file-pattern '^([A-Za-z0-9_-]+\.(json|zip|log))$'
python verify_kaggle.py --spec evidence/G5-kaggle-spec.json --output "$job/provider" --source "$job/pulled/k230-camera-xy-map-full-a01.py" --version 1 --receipt .runtime/full-verified.json
```

若 GPU 提交被提供方拒绝且确认没有运行会话，用 `kaggle_prepare.py --reuse-input <已准备的 SUBMISSION.json> --device cpu --ref OWNER/NEW-CPU-RUN --out <新目录>` 复用原 Dataset。

`infer_map.py` 读取同次归档中的 `map-inference.pt`、`metrics.json` 和 `result.json`。默认取验证集选出的分支，可显式选择 learned_map、semantic_roadnet 或 fused_map 做研究。两个语义分支需要训练时冻结的 `--layout`、`--roadnet` 和含地图掩膜的 `--data`。置信度分别描述首个候选与五候选集合的 50 mm 命中事件。只有主机通过单点门的选中分支才可返回主坐标。

## 仿真收尾

[独立变量对照结果](evidence/G5-nuisance/SIMULATION_REPORT.md) 记录 96 个新位置和 512 对受控扰动。保持已验证的 G5 归档和原场景，令 `$job` 指向 `.runtime/kaggle-map-full-a01`。在 PowerShell 复现。

```powershell
& $blender -b $blend -t 4 --python-exit-code 1 --python render_nuisance.py -- --roadnet $roadnet --cad $cad --out .runtime/sim-nuisance-rerun *> .runtime/sim-nuisance-rerun.log
python evaluate_nuisance.py --data .runtime/sim-nuisance-rerun --full .runtime/full-a01 --layout $layout --roadnet $roadModel --verified-output "$job/provider" --out .runtime/paired-results --render-log .runtime/sim-nuisance-rerun.log
```

[G2 Smoke QA](evidence/G2-smoke-qa.json) · [G3 Pilot QA](evidence/G3-pilot-data-qa.json) · [Pilot 四路线评估](evidence/PILOT_REPORT.md) · [G4 扩展决策](evidence/G4-full-selection.json) · [仿真证据 SHA256](evidence/SHA256SUMS.json)

## 后续验收入口

真实 OV2735 采集和测量见 [实拍采集契约](REAL_CAPTURE.md)。用原图及独立量测的光心地面 X/Y 验证 Sim-to-Real，报告测量不确定度、误差分布和置信度覆盖率。

通过主机定位验收后，用 `export_k230.py` 生成固定形状 ONNX、评估量化差异，再按经审计的 nncase 2.11.0 工具链编译。学得地图头用 `--map-head`，板端解码输入为各 X/Y 锚点分数。K230 工程候选的实测记录已给出 KModel 大小、KPU 时延、各阶段时延、MMZ 与资源释放。后续还需在获准的目标方案上验收定位性能、完整系统开销和持久部署。详见 [板端实测](USB_NATIVE_RESULT_20261117.md)。

正式板端探针入口是 `board_probe.run(model_root, report_path, expected_usb_device, frames=120)`。在独立模型目录中准备 `camera_xy.kmodel` 和 `deployment.json`。探针复用 RoadNet 的 `reference.board_smoke` 和 TopFace 的 `k230.topface_adapter`、`k230.topface_service`。实测留档需核对模型与报告字节，并在同一批输入上评估 AI2D 与训练缩图的差异。
