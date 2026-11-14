# TopFaceNet 仿真训练接续

本轮目标仍是完整执行 `K230_TopFaceNet_Sim2Real_Execution_Plan_2026-11-14.md`。当前只准备了阶段 A；尚未完成云端渲染、200 张预览、批量数据、训练、迁移评估或板端部署。不要将 Kaggle Quick Save 的 COMPLETE 状态解释为任务运行成功。

## 当前可核验事实

- 分支：`ops/topfacenet-sim2real-20261114`；沿用本仓库 OPS 协议。
- 原始 CAD ZIP 的逐文件输入身份见 `INPUT_MANIFEST.json`；单一融合网格，不分拆物块。
- 本地 Blender 4.2.9 CPU 验证确认 GLB 导入后需要绕 X 轴修正 -90°。修正后的高度约 60mm、最大直径约 50mm，顶面面积约 706.85mm²，保留 9761 个顶点及 19518 个多边形。见 `local_cad_import.json`。
- 一张本地原生 Blender 640×360 RGB/顶面材质索引 Mask 已实际渲染。Mask 与解析顶面投影 IoU=0.96184。范围限本地 CAD/合成标签一致性，不能充当 BlenderProc/Kaggle 或真实识别精度验收。
- 5 项主机检查通过：投影/高度/角度约定、正圆方向无效、四实例完整保留、空场、污染 Mask 拒绝。见 `host_tests.log`。
- 原 r4 未改：12 张真实图，其中 7 张 APPROVED、17 个顶面 bbox；批准划分为 train=4、val=2、test=1。尚无真实椭圆真值。

## 相机与硬件

`hardware_profile.json` 的每个字段记录值、单位、来源、状态、置信度和核验方法；对应历史证据及哈希见 `hardware_evidence.json`。

用户确认 CSI2 为 GC2093 小镜头固定焦点版；相机垂直运动，镜头中心到物块顶面的垂直距离 100–400mm。90° 表示光轴垂直顶面，采样范围 80–100°，即相对法线偏离不超过 10°。物块保持直立，不能把相机角度变化标成物块倾斜。

[立创镜头规格](https://wiki.lckfb.com/zh-hans/lushan-pi-k230/image-recog/use-sensor.html) 的小镜头标称 EFL=4.3mm、F/2.0、H65°/V40°/D73°，描述为定焦约 70cm。这些是标称值。fx/fy/cx/cy、畸变系数、实际曝光/增益/白平衡和新现场 FPS 仍未知；100–400mm 的近距离清晰度必须由真实 CSI2 图确认。冒烟阶段使用无畸变针孔模型，不宣称已覆盖失焦/畸变域差异。

## 云端入口与人工步骤

- [私有 Notebook](https://www.kaggle.com/code/lotus30/k230-topfacenet-original-cad-drive-smoke-20261114/edit)：ID 144989064，当前 Quick Save v2，源码读回一致，尚未执行。
- [私有输入 Dataset](https://www.kaggle.com/datasets/lotus30/k230-topface-smoke-input-20261114)：v2 READY。原作者保留权利，未授予公开再分发许可。
- [K230 专属 Drive 目录](https://drive.google.com/drive/folders/1MwltNmaI24HjvcxnG1w608qui42RAPKf)：本机 `gdrive:k230-topfacenet-20261114` 访问通过，Kaggle 运行时访问尚未验证。

请刷新 Notebook，在 Add-ons → Secrets 关联已有的 `DOASTRA_GDRIVE_OAUTH_JSON`，确认 Internet 开启，再执行一次 Save Version → Save & Run All。此步骤应由用户完成；当前 Windows 浏览器控制因无法可靠识别 Chrome 地址已停止。不得向源码、Notebook、日志或 Git 写入凭据，不尝试在 CLI Save 中假定 Secret 自动关联。

阶段 A 在任何安装/渲染前校验输入 manifest、Drive 目录和一个云端生成的小对象的全字节 SHA-256 读回。之后运行官方 BlenderProc quickstart 和 1 张原 CAD RGB/顶面材质索引 Mask，执行标签审计，再在 Kaggle 创建完整归档、上传 Drive、全字节读回。GPU 用于 PyTorch 和 GPU 用于 Cycles 的证据分开。失败按阶段保留 STATE/log，不自动重跑科学工作。

## 执行后接续

1. 读取 `.ops` 最新交接并重新认领任务，检查 Notebook 的最新版本和真实执行输出。不要因状态为 COMPLETE 就判定实际运行；必须存在该次 RESULT/Drive 回执。
2. 只下载小型 `*-RESULT.json`、`*-DRIVE.json`、环境与审计摘要、必要预览图。完整归档保留在云端；核对精确版本、run_id、输入 manifest、Drive file_id/size/SHA-256。
3. 若缺 Secret，在同一 Notebook 完成关联；若已失败，先保留日志、诊断具体阶段，再用独立新 run_id 和源码版本修复。不覆盖原运行目录，不重复提交未知状态的运行。
4. 阶段 A 通过后实现/执行 200 张预览和拼图，自动检查标签，再由用户抽查。预览通过才进入 1000–3000 张数据；按场景组、种子、相机配置分组隔离，保持六色、0–4 实例、边缘/遮挡/空场完整标注。
5. 后续实现 MobileNetV2/ShuffleNetV2 TopFaceNet、checkpoint 恢复、合成/真实迁移对照、独立盲测、ONNX/nncase 早期兼容试编译与部署准备。这些尚未实现或运行，不继承旧不可用 YOLO 试点的训练/准确率结论。
6. UART、运动、生产 Save 保持关闭，CSI0 隔离，CSI1 不改。板端运行沿用已有独立测试目录授权；不得改 main.py/boot.py。

## 本地复现入口

从仓库根执行（使用已有主机 Python/OpenCV/NumPy，Kaggle 控制另需 `kaggle==2.2.5`）：

```powershell
python -m unittest discover -s research/topface -p 'test_*.py' -v
python research/topface/prepare_inputs.py --cad <原始ZIP绝对路径> --upstream <原始k230worker根目录>
python research/topface/kaggle_control.py --token-file <本机凭据文件> -- kernels status lotus30/k230-topfacenet-original-cad-drive-smoke-20261114
```

`prepare_inputs.py` 会生成新的带哈希输入；云端当前版本已冻结，修改后需创建 Dataset/Notebook 新版本，不把变动文件塞进旧身份。Kaggle CLI 在 Windows 上传时须使用本机原生绝对路径，避免其临时上传登记路径对正斜杠的兼容问题。当前 cloud_store/render/audit 代码已做主机语法和几何验证，完整云端执行仍待阶段 A 回执。
