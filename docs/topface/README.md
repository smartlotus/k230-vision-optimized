# TopFaceNet V1.1 执行入口

权威计划已切换到 `K230_TopFaceNet_Sim2Real_Project_Plan_V1.1_2026-11-14.md` 和配套 Execution AI Prompt。计划 SHA、用户相机约定、阶段和保护边界见 `execution_contract.json`。V1.0 的200张预览/MobileNetV2路线已退役；旧文件仅作历史证据，不运行 `prepare_inputs.py` 或旧 v8 Notebook。

## 当前事实（2026-11-14）

- G0/G1：真实工程、r4、相机、AI2D/KPU和原始模型审计已完成首轮，见 [审计](v11/audit/PROJECT_AUDIT.md)、[未知项](v11/audit/RISK_AND_UNKNOWN.md)。实时CSI2 RGB888P、真实内参/失焦/ISP/新现场FPS仍未确认。
- G2：本地原生Blender实际100场景/246实例，自动QA通过，用户30图抽查通过；完整归档已迁移Drive并全字节SHA读回，见 [QA](v11/reports/SIMULATION_QA.md)、[回执](v11/reports/SMOKE_DRIVE_ACCEPTANCE.json)、[预览](v11/previews/index.html)。不冒充Kaggle/GPU或真实识别精度。
- G3：两个官方COCO图均已ORT、nncase2.11编译和Simulator；NanoDet官方EMA加载及三个矩形尺寸的项目头也完成PT/ORT和编译/Simulator。新增头仍未训练，量化误差未接受，板端未运行。见 [兼容性报告](v11/g3/COMPATIBILITY.md)。
- G4：用户已启动新 [01_simulation Pilot Notebook](https://www.kaggle.com/code/lotus30/k230-topface-v11-01-simulation-pilot/edit)。实际运行 **v2、ID145002634**；v3是另一个取消的会话。v2源码已读回匹配，Drive准入和单帧资格验证通过，进入1000场景PILOT_RENDER。最新观察见 `v11/cloud/pilot-a02/OBSERVATION-v2-latest.json`。在最终RESULT与Drive归档验收前，不能称Pilot完成。
- 训练 optimizer updates=0。计划4个逻辑训练实验（overfit、20epoch预训练Pilot、同预算随机初始化对照、唯一正式全量模型），见 [训练顺序](v11/training/TRAINING_PLAN.json)。两个候选不会各做全量训练。

## 冻结身份与云存储

新run为 `k230-topface-v11-pilot-1000-a02`；私有输入 `lotus30/k230-topface-native-v11-input-20261114-a02/1`，manifest SHA `a623c1137b737420d38610793596caff281720a86321832d533f1d14299d614f`。Kaggle源码与输入读回收据见 `v11/cloud/pilot-a02/READBACK.json`、`EXECUTED_SOURCE-v2.json`。不要覆盖该输入或复用run_id重算。

新实现使用SHA固定的官方Blender4.2.9，完全移除BlenderProc安装和quickstart。先完成Drive写入及全字节读回、持久run准入，再下载和运行。GPU由PyTorch与实际Cycles分别报告；最多仅单帧运行时探测有一次CPU fallback，Pilot不会自动重跑。worker9000s、provider请求10800s，保留归档/上传预算，失败先落compact恢复结果再归档。最新API仍没有Secret绑定字段；本次由用户在编辑器关联已有Secret并启动。

Drive固定目录：[K230 TopFaceNet](https://drive.google.com/drive/folders/1MwltNmaI24HjvcxnG1w608qui42RAPKf)。新云端生成的完整数据/模型不经本机中转；本机只收有限JSON、metrics、必要预览。Notebook COMPLETE不是归档或科学质量通过，必须绑定精确版本、run_id、输入、数量、member manifest、file_id、size与SHA。

旧 v8（ID144989064）曾完成115字节Drive probe，随后因BlenderProc继承Matplotlib inline backend失败，0渲染/0updates；用户已停止，证据在 `runs/smoke-v8/`。不要重启。

## 相机、CAD和数据边界

用户确认小固定焦点GC2093，镜头到顶面垂直100–400mm，光轴与顶面80–100°，物块直立。标称H65°/EFL4.3mm/约700mm定焦只是规格，不能代替真实内参或近距离清晰度。参数状态和来源见 `v11/camera/`。

原CAD保持一体；GLB只修正坐标轴−90°，不拆分、不重画。顶面是z=60mm的Ø30平面。物理3D圆心投影、conic center与可见mask质心分别保存；新增模型也单独输出conic center偏移。见 `v11/cad/`、`research/topface/native_cad.py`。

r4两仓相同SHA且未修改：12图/33框，7批准图/17框，实际5个黑色uncertain。旧test已被使用，不能称新盲测；没有真实中心/椭圆真值。UART/运动/生产Save关闭；CSI0隔离；CSI1 QR/LANE不改；不改main.py/boot.py。

## 接续

1. 读OPS入口和当前任务，再核对原v2状态；没有最终回执不提交替代科学运行。
2. 只收小型RESULT/Drive回执；核对完整1000独立场景、label QA、分组、成员SHA及云归档全字节读回。
3. 运行独立02_dataset_audit的尺度/可见性/近重复/泄漏诊断，再进入03_training的10–20图overfit及Pilot短训。Pilot门禁通过后再准入Full10000–20000场景和唯一正式模型。
4. 完成合成/真实分开的评估、最终train-only PTQ、PT/ORT/Simulator/KPU配对及板端完整链路benchmark。尚未完成，不从历史冻结图FPS推断新模型25FPS。

本地主机检查：`python -m unittest discover -s research/topface -p 'test_*.py' -v`。当前32项通过，日志 `v11/reports/host_all_tests.log`。原生worker最新本地qualification为6图/15实例，`v11/reports/native_worker_qualification.json`。兼容性导出与nncase使用已有隔离CPU环境，不向旧工程安装依赖。
