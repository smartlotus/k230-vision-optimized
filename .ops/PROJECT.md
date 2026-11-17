# K230 Vision Optimized

## 项目与来源

本仓库维护 K230 视觉识别、板端接入、主机验证与模型研究。原版入口和优化版入口并存，导入来源及逐文件 SHA-256 见 [UPSTREAM_IMPORT.json](../docs/UPSTREAM_IMPORT.json)。源项目为 smartlotus/k230worker-6ebd4c18，导入基线是 0c3b8499e2877bc05df786921e199ab0533685d2。

私有远端为 smartlotus/k230-vision-optimized。main 用于集成基线，ops 任务分支保存各自的源码、验收证据与交接。桌面 k230-project 汇集工作树、文档和交付物，Git 与 OPS 命令在对应工作树中执行。

## 目录与任务入口

| 路径 | 用途 |
|---|---|
| k230_transport_mvp/k230、vision | 板端适配与视觉算法 |
| k230_transport_mvp/core、ports、sim、stm32 | 任务逻辑、通信接口和仿真 |
| k230_transport_mvp/tools、tests、data、evidence | 主机工具、测试、夹具与历史证据 |
| scripts | 主机验证入口 |
| docs | 工程结构、来源和验收资料 |
| .ops | 协议、工具、任务快照和交接 |

下表的入口以各分支的仓库根为基准。用 git worktree list 定位本机目录。

| 专题 | 分支 | 资料入口 |
|---|---|---|
| TopFace 训练与离线交付 | ops/topfacenet-sim2real-20261114 | docs/topface/README.md |
| TopFace 板端接入和相机试验 | ops/topface-production-fps-20261116 | docs/STATUS.md、docs/HANDOFF.md |
| 相机及显示诊断 | ops/camera-diagnosis-20261116 | docs/validation/camera-diagnosis-20261116 |
| CameraXY Temporal V1 | ops/temporal-camera-v1-20261117 | docs/temporal_camera/README.md、docs/temporal_camera/native-smoke-20261117/README.md |
| Kaggle 与 Drive 云端归档 | ops/topface-publisher-workflow-20261114 | research/topface/publisher、任务交接 |

## 开发与交接

1. 读 [项目规则](../AGENTS.md)、[OPS 开发协议](PROTOCOL.md) 和 [目录与导入说明](../docs/PROJECT_LAYOUT.md)。
2. 执行 python .ops/tools/ops.py status。用输出中的 checkout、任务快照和最新 handoff 定位工作，再按任务执行 verify 和认领。
3. 按 [主机验证说明](../README.md#电脑端验证) 运行受影响检查。板端测试使用该任务实际授权、独立目录与实测记录。保留 active/draft、边界拒绝与标定门，UART 和运动按授权范围执行。
4. 先记录验证证据，再按协议保存 checkpoint 和释放认领。以远端 SHA 核验 GitHub 备份，以板端数据核验 FPS、精度和稳定性。大型模型与数据在交付清单或 Drive 回执中定位。

共享 OPS 登记记录当前认领和同步回执。跨机器从任务分支读取 .ops/tasks 与 .ops/handoffs，再按本机状态核验。将临时报告和草稿保存在 .ops/local。
