<!-- ops-pipeline -->
## OPS Pipeline 开发交接

开发前读 `.ops/PROJECT.md`、`.ops/PROTOCOL.md`，运行 `python .ops/tools/ops.py status`。
写入前按协议认领任务；接手核验最新交接；模块结束/暂停/换 Agent 时写 checkpoint 并按已授权配置同步。
保留项目原有协议与运行状态；配置不能覆盖宿主权限和明确项目限制。
<!-- /ops-pipeline -->

## 项目边界

开发或板端操作前阅读 `docs/LESSONS_LEARNED.md` 的相关条目和当前任务交接。新根因、修复及复验结论及时补入该总账，并链接原始证据；保留失败和 UNKNOWN，不用工具 success 或 RUN_COMPLETE 代替画面、识别和标定验收。

先读README.md和.ops/PROJECT.md；使用python .ops/tools/ops.py status确认当前任务。
新仓库独立管理源码；原始实现与*_optimized.py并存。来源身份见docs/UPSTREAM_IMPORT.json。
2026-11-12用户已授权实际板端测试：允许检查COM3、独立目录安装和有界运行，保护现有草稿和旧SD配置。
不修改main.py/boot.py或保存生产参数；25FPS以原生测量为准。
保留active/draft、阈值与完整边界拒绝条件；UART默认禁用，无运动。
写入前认领任务；模块结束checkpoint/release。自动备份限当前私有远端，不扩大硬件或部署授权。
