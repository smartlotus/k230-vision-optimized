<!-- ops-pipeline -->
## OPS Pipeline 开发交接

开发前读 `.ops/PROJECT.md`、`.ops/PROTOCOL.md`，运行 `python .ops/tools/ops.py status`。
写入前按协议认领任务；接手核验最新交接；模块结束/暂停/换 Agent 时写 checkpoint 并按已授权配置同步。
保留项目原有协议与运行状态；配置不能覆盖宿主权限和明确项目限制。
<!-- /ops-pipeline -->

## 项目边界

先读README.md和.ops/PROJECT.md；使用python .ops/tools/ops.py status确认当前任务。
新仓库独立管理源码；原始实现与*_optimized.py并存。来源身份见docs/UPSTREAM_IMPORT.json。
本轮只做电脑端；不主动连接、采样、上传、启动K230或改变SD配置。25FPS待板端验证。
保留active/draft、阈值与完整边界拒绝条件；UART默认禁用，无运动。
写入前认领任务；模块结束checkpoint/release。自动备份限当前私有远端，不扩大硬件或部署授权。
