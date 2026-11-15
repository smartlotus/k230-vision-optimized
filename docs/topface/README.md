# TopFaceNet V1.1 执行入口

权威计划为 `K230_TopFaceNet_Sim2Real_Project_Plan_V1.1_2026-11-14.md` 和配套 Execution AI Prompt。原文 SHA、用户相机约定及保护边界见 [execution_contract.json](execution_contract.json)。实现位于仓库 `research/topface/`，证据位于 `docs/topface/v11/`；原始工程与 CAD 保留。

## 当前权威记录

本页是入口索引，不替代任务控制器或接受记录。接手前读取仓库 `AGENTS.md`、`.ops/PROJECT.md`、`.ops/PROTOCOL.md` 和 OPS 当前认领，避免与现有单 writer 同时操作。

- [要求与证据覆盖](v11/reports/REQUIREMENT_COVERAGE.json)：明确已接受、待执行和 UNKNOWN，整个项目尚未完成。
- [Full 数据接受](v11/training/FULL_DATASET_ACCEPTANCE.json)：10,000 场景、24,996 个世界实例；train 7,465、val 1,601、独立 synthetic_test 934。数据索引与分片哈希见该记录及 [分片清单](v11/training/FULL_INVENTORY.json)。
- [正式训练运行记录](v11/cloud/formal-a01/HOST_PROGRESS.json) 和 [精确 v1 观察](v11/cloud/formal-a01/OBSERVATION-v1.json)：当前任务为 `lotus30/k230-topface-v11-06-formal-training`，ID 145144940、v1。查看记录的实际时间与原始会话；不要因观察连接中断重启训练。
- [正式训练配置](v11/cloud/formal-a01/FORMAL_CONFIG.json)、[完成报告](v11/training/FORMAL_TRAINING_A01.md) 和 [接受记录](v11/cloud/formal-a01/ACCEPTANCE.json)：原 v1 已完成 100 轮、23,400 次更新，第 92 轮 best 已在 Drive 全字节校验接受。实际导出、独立测试和实机验收另行核对。
- [独立运动测试接受](v11/reports/REDETECTION_SWEEP_ACCEPTANCE_A01.json)：1,872 帧、156 序列，覆盖六色、遮挡恢复、出入画和相机升降。数据生成与归档已接受，正式模型推理结果另行核对。
- 正式权重上板试用包已生成，包含 KModel、ONNX、脚本及校验清单。交付位置、文件哈希和 12 张配对结果见 [试用包接受记录](v11/reports/BOARD_PREVIEW_A01_ACCEPTANCE.json)。完整交付待正式评估归档接受，实机测试由用户择期安排。

Smoke 用户抽查、Pilot、颜色补充、组合审计及 Full 生成均已推进完成；旧版“optimizer updates=0”“等待用户启动旧审计”不再是接续指令。历史状态保留在各原始运行目录和 Git 历史。

## 原计划 17 项交付索引

表中的链接指向已有证据或当前控制入口；等待正式产物或原生测量的项目不得标为完成。完整训练 trace、best/last、逐轮验证与 TensorBoard 状态由原始云归档保留，紧凑 `RESULT.json` 不代替完整日志。

| 项 | 交付物与入口 | 当前范围 |
|---|---|---|
| 1 | [PROJECT_AUDIT.md](v11/audit/PROJECT_AUDIT.md) | 初轮工程审计 |
| 2 | [REUSE_MAP.md](v11/audit/REUSE_MAP.md) | 可复用实现地图 |
| 3 | [RISK_AND_UNKNOWN.md](v11/audit/RISK_AND_UNKNOWN.md) | 初轮风险与未知，结合当前覆盖记录阅读 |
| 4 | [camera_profile.yaml](v11/camera/camera_profile.yaml) | 历史硬件证据，当前实机标定仍未知 |
| 5 | [camera_sweep.yaml](v11/camera/camera_sweep.yaml) | 100–400 mm、80–100° 仿真范围 |
| 6 | [CAD 验证](v11/cad/CAD_VALIDATION.md)、[几何记录](v11/cad/geometry_validation.json) | 原一体 CAD 与顶面验证 |
| 7 | [可重复仿真生成器](../../research/topface/render_native.py)、[场景参数](../../research/topface/scene_config.py) | 原生 Blender 路线，运行源码身份见各冻结输入 |
| 8 | [Smoke QA](v11/reports/SIMULATION_QA.md)、[人工抽查](v11/reports/SMOKE_HUMAN_REVIEW.json)、[预览](v11/previews/index.html)、[Drive 接受](v11/reports/SMOKE_DRIVE_ACCEPTANCE.json) | 100 场景与 overlay |
| 9 | [Pilot 接受](v11/cloud/pilot-a02/ACCEPTANCE.json)、[1,240 场景组合审计](v11/cloud/audit-a02/ACCEPTANCE-v4.json)、[Pilot 指标](v11/reports/PILOT_METRICS.md) | 原 1,000 场景加颜色反例，原归档保留 |
| 10 | [Full 接受](v11/training/FULL_DATASET_ACCEPTANCE.json)、[清单](v11/training/FULL_INVENTORY.json) | 10,000 场景，完整分片在 Drive |
| 11 | [训练实现](../../research/topface/train_experiment.py)、[正式配置](v11/cloud/formal-a01/FORMAL_CONFIG.json)、[选定权重](v11/cloud/formal-a01/SELECTED_MODEL.json) | 100 轮完成，第 92 轮 best 已接受 |
| 12 | [正式训练报告](v11/training/FORMAL_TRAINING_A01.md)、[原归档接受](v11/cloud/formal-a01/ACCEPTANCE.json)、[归档与源码校验](../../research/topface/accept_full_downstream.py) | 配置、完整日志、TensorBoard、best/last 与 SHA 在已接受原归档内 |
| 13 | [仿真/真实评估覆盖](v11/reports/REQUIREMENT_COVERAGE.json)、[正式主机评估入口](../../research/topface/evaluate_formal_host.py) | 正式 934 张及四路 host 报告待模型产出；新真实盲测未执行 |
| 14 | [自动转换接续](../../research/topface/continue_formal_conversion.py) | 最终 ONNX/KModel 尚待实际导出、编译与数值测试 |
| 15 | [契约实现](../../research/topface/nanodet_topface.py) | 最终 output_contract.json 与同一转换模型一起交付 |
| 16 | [板测包组装](../../research/topface/package_formal_delivery.py)、[优先试用包](v11/reports/BOARD_PREVIEW_PROTOCOL.md)、[安装流程](v11/reports/BOARD_INSTALL_PROTOCOL.md)、[板测计划](v11/reports/BOARD_TEST_PLAN.md)、[数值配对](v11/reports/BOARD_PAIR_PROTOCOL.md) | 包依赖最终模型；原生配对、FPS 与稳定性尚未运行 |
| 17 | [当前未完成项](v11/reports/REQUIREMENT_COVERAGE.json)、[相机差距](v11/camera/calibration_gap_report.md)、[板测计划](v11/reports/BOARD_TEST_PLAN.md) | 最终结论须结合正式测试与板端证据更新 |

## 云存储与不重放原则

Drive 固定目录：[K230 TopFaceNet](https://drive.google.com/drive/folders/1MwltNmaI24HjvcxnG1w608qui42RAPKf)。完整云端数据与训练归档留在云端；本机只收有界报告、必要预览和部署文件。各接受记录中的 file ID、字节数和归档 SHA 是工件身份；接受记录自身的文件 SHA 是另一种哈希，不可混用。

目前沿用已授权的离线 Kaggle 计算与独立私有云发布器，不要求用户逐个给后续任务关联 Secret。发布器将原始保存版本归档至 Drive 并全字节读回。Notebook COMPLETE、上传成功或本地文件存在，均不能单独证明归档接受或科学质量通过。

V1.0 预览/MobileNetV2 路线及旧 v8 已退役。不要运行旧 prepare_inputs.py、重启旧 v8、覆盖已接受输入或复用 run_id 重算。网络、令牌或上传恢复需对同一保存版本、归档和预留对象做对账。

## 相机、CAD及真实测试边界

用户确认 GC2093 小固定焦点镜头；镜头中心到物块顶面 100–400 mm，光轴与顶面 80–100°，物块直立。仿真 62/65/68° 水平视场为推定值，零畸变不是实测标定。当前内参、畸变、近距离清晰度、ISP 及工作面尺度仍需真实设备证据。

原 CAD 保持一体；GLB 仅修正坐标轴 −90°，不拆分、不重画。顶面为 z=60 mm 的 Ø30 mm 平面。物理 3D 圆心投影、conic center 与可见 mask 质心分别保存；不得互相替代。

r4 保持原字节，uncertain 不自动清除。旧真实测试已经使用，不能称新盲测；真实中心/椭圆指标需要对应真值。Pilot 两张真实 val 的有限结果不证明最终模型真实泛化。UART、运动和生产 Save 关闭；CSI0 隔离，CSI1 QR/LANE 不改；保护 main.py、boot.py 及既有 SD 目录。

## 接续顺序

1. 确认现有控制器 PID、原 Kaggle ID/v1 和单 writer。当前接续链为正式训练与归档 → continue_formal_conversion.py → 六个部署文件与独立仿真报告收集。
2. 原正式会话完成后，核对完整轮数、更新数、best/last、原始归档接受及选定权重 SHA。只有实际 FORMAL_SESSION_BOUNDARY 才沿同一实验做精确 epoch 边界接续。
3. 对已接受最终模型执行 evaluate_formal_host.py 的历史真实 test 与 1,872 帧运动序列 ONNX/Simulator 评估。它们与独立 934 张静态合成测试分开，不能相互替代。
4. package_formal_delivery.py 生成早期板测包及其 SHA256SUMS.json。该校验清单只覆盖该包，不冒充整个 17 项交付已经完成。原始包保持不可变，后续报告另存。
5. 连接 K230 后按板测计划，用新 SD 目录安装已校验文件，完成固件/通道/AI2D/KPU 配对、有界性能及稳定性测试，再做冻结模型与阈值后的新真实盲测。原生结果缺失时明确 NOT_RUN/UNKNOWN，不从历史 FPS 或主机 Simulator 速度推断目标 25 FPS。
6. 独立仿真和主机评估完成后，用 [完整交付打包脚本](../../research/topface/assemble_formal_release.py) 生成模型、报告、Drive 原件索引、源代码和 SHA-256 校验表。实机结果待用户安排试跑后补充。

有改动时运行受影响检查；各历史测试记录只对其原源码、输入和环境有效，不把旧的通过数量作为当前全仓测试结论。
