# K230 Vision Optimized

## CameraXY temporal competition V1 (2026-11-17)

Isolated checkout: `C:/Users/28389/Desktop/k230-temporal-camera-v1`, branch `ops/temporal-camera-v1-20261117`. Module and evidence entry: [docs/temporal_camera/README.md](../docs/temporal_camera/README.md). OPS task `temporal-camera-v1-20261117`; source in `research/temporal_camera`, frozen provider state in `docs/temporal_camera/cloud`. Preserve other task pauses and accepted production code. Only camera_x_mm/camera_y_mm supervise training; no yaw, chassis labels or installation offset. Training runs on Kaggle; preserve exact original saved source/version and publisher identity. All unexecuted physical tests remain NOT RUN. This section does not change TopFace's existing ownership or board session.

## 当前接续入口

- 当前实现：`C:/Users/28389/Desktop/k230-topface-production`，分支 `ops/topface-production-fps-20261116`，私有远端 `smartlotus/k230-vision-optimized`。
- [最新项目状态](../docs/STATUS.md)记录时间戳、已完成/进行中/待验收、运行和资源证据。
- [便携交接](../docs/HANDOFF.md)记录设备查询顺序、报告路径、接续命令和已知问题。
- [经验总账](../docs/LESSONS_LEARNED.md)在板端/部署任务前阅读。
- 实时任务和认领仍以 `python .ops/tools/ops.py status` 为准；本页不维护第二份任务状态表。

## 目标和目录

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

## 资源与保护边界

用户已授权模型接入、板端试验、FPS优化、USB预览以及当前私有仓库checkpoint备份。保留main.py/boot.py、旧SD目录和生产profile；UART/运动关闭。
CSI2为COLOR/TOP/RING，CSI1为QR/LANE，CSI0隔离；USB保持独立来源身份，不沿用CSI标定。
COM3和K230媒体资源单控制端。running=true时只读附着，保持预览；另一个CSI/UVC程序或上传需先处理所有权和当前运行。
闭合的失败/UNKNOWN attempt保留，不自动重放、不覆盖既有报告。新run开始前确认新报告名不存在；固定启动脚本旧报告名问题见交接。
约25FPS仍是待实现目标，按包含采集失败等待、GC和显示的墙钟计量。无标定不得输出有效机器人mm位姿。
主线集成与任务分支备份分别记录；本协议不自动授权merge/deploy范围扩张。

## 接手与验证

1. 读AGENTS.md、本页、docs/STATUS.md、docs/HANDOFF.md与.ops/PROTOCOL.md。
2. 核对Git/worktree、OPS状态和最新handoff，按自己的任务认领。
3. 实际板端状态、文件字节和运行收据高于旧“当前”文字。预览计数不是推理分母。
4. 按本次改动运行必要验证；文档刷新不重跑闭合实验。
5. checkpoint、远端SHA核对、release。

历史初始化及2026-11-12授权说明已原样保存在 [本次更新前副本](../docs/validation/topface-production-fps-20261116/project-before-status-refresh-20261116.md)。源码来源身份见 [UPSTREAM_IMPORT.json](../docs/UPSTREAM_IMPORT.json)，仓库目录地图见 [PROJECT_LAYOUT.md](../docs/PROJECT_LAYOUT.md)。
