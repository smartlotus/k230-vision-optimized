# K230 Vision Optimized

## 项目接续入口

先读 [OPS 协议](PROTOCOL.md)，运行 `python .ops/tools/ops.py status`，再从对应任务的 `handoff` 读取最新交接。仓库为 `smartlotus/k230-vision-optimized`。同机实时认领和同步状态以 OPS 为准。

| 模块 | 任务分支与 OPS 任务 | 模块入口 |
|---|---|---|
| CameraXY 时序定位离线 | `ops/temporal-camera-v1-20261117` · `temporal-camera-v1-20261117` | [模块索引](../docs/temporal_camera/README.md) · [离线交付与人工资料](../docs/temporal_camera/OFFLINE_HANDOVER.md) |
| CameraXY K230 短测 | `ops/temporal-camera-v1-20261117` · `temporal-native-smoke-20261117` | [板端入口](../docs/temporal_camera/native-smoke-20261117/README.md) · [实测报告](../docs/temporal_camera/native-smoke-20261117/NATIVE_REPORT.md) |
| TopFace 生产接入 | `ops/topface-production-fps-20261116` · `topface-production-fps-20261116` | [项目状态](../docs/STATUS.md) · [便携交接](../docs/HANDOFF.md) |

CameraXY 源码在 `research/temporal_camera`，Kaggle 保存版本及归档清单在 `docs/temporal_camera/cloud`。训练监督仅为 `camera_x_mm` 与 `camera_y_mm`。机械外参由用户在控制层处理。分支交接和备份按 OPS 认领进行，板端和生产代码保护边界见模块交接。

## TopFace 目标和目录

使用冻结TopFaceNet KModel完成K230物块顶面识别，并保留原视觉观察协议、稳定跟踪、标定门、active/draft及旧算法入口。
KPU TOP部署、CSI/USB画面与FPS各自验证；整车流程和科学/几何接受分开。

| 目录 | 用途 |
|---|---|
| k230_transport_mvp/k230 | 原相机、Observation和新TopFace backend/服务 |
| k230_transport_mvp/vision | 原视觉算法、共享解码 |
| k230_transport_mvp/core、ports、sim、stm32 | 原流程、通信、仿真及STM32接口 |
| research/topface | 数据、训练/导出工具、板端试验 |
| docs/topface | 正式数据/模型/离线接受证据 |
| docs/validation/topface-production-fps-20261116 | 当前部署、CSI/USB原生短测及现场记录 |
| .ops | 当前任务、不可变交接及同步协议 |

## TopFace 资源与保护边界

用户已授权模型接入、板端试验、FPS优化、USB预览以及当前私有仓库checkpoint备份。保留main.py/boot.py、旧SD目录和生产profile；UART/运动关闭。
CSI2为COLOR/TOP/RING，CSI1为QR/LANE，CSI0隔离；USB保持独立来源身份，不沿用CSI标定。
COM3和K230媒体资源单控制端。running=true时只读附着，保持预览；另一个CSI/UVC程序或上传需先处理所有权和当前运行。
闭合的失败/UNKNOWN attempt保留，不自动重放、不覆盖既有报告。新run开始前确认新报告名不存在；固定启动脚本旧报告名问题见交接。
约25FPS仍是待实现目标，按包含采集失败等待、GC和显示的墙钟计量。无标定不得输出有效机器人mm位姿。
主线集成与任务分支备份分别记录；本协议不自动授权merge/deploy范围扩张。

## TopFace 接手与验证

1. 读AGENTS.md、本页、docs/STATUS.md、docs/HANDOFF.md与.ops/PROTOCOL.md。
2. 核对Git/worktree、OPS状态和最新handoff，按自己的任务认领。
3. 实际板端状态、文件字节和运行收据高于旧“当前”文字。预览计数不是推理分母。
4. 按本次改动运行必要验证；文档刷新不重跑闭合实验。
5. checkpoint、远端SHA核对、release。

历史初始化及2026-11-12授权说明已原样保存在 [本次更新前副本](../docs/validation/topface-production-fps-20261116/project-before-status-refresh-20261116.md)。源码来源身份见 [UPSTREAM_IMPORT.json](../docs/UPSTREAM_IMPORT.json)，仓库目录地图见 [PROJECT_LAYOUT.md](../docs/PROJECT_LAYOUT.md)。
