# 项目结构和迁移边界

为保持Python导入、测试夹具路径和SD模块结构一致，源码保留在k230_transport_mvp包树。
OPS不要求把代码重命名为src；本仓库通过根入口、完整PROJECT、便携PROTOCOL、任务分支和交接来落实流程。

| 路径 | 内容 |
|---|---|
| `.ops/` | 全新项目ID、协议、工具、任务和交接 |
| `scripts/` | 从仓库根执行的主机验证入口 |
| `docs/` | 使用说明、来源清单、本仓库新验证证据 |
| `k230_transport_mvp/k230/`、`vision/` | 板端适配和算法，新旧实现并存 |
| `core/`、`ports/`、`sim/`、`stm32/` | 原有运行和验证依赖，均位于k230_transport_mvp下 |
| `tools/`、`tests/` | 主机工具与回归测试，均位于k230_transport_mvp下 |
| `data/` | 来源提交内冻结的配置、数据和测试夹具 |
| `evidence/` | 四轮优化阶段证据与测试必需的历史JSON，位于k230_transport_mvp下 |

来源为smartlotus/k230worker-6ebd4c18的0c3b8499e2877bc05df786921e199ab0533685d2。
UPSTREAM_IMPORT.json逐文件记录SHA256；导入源文件保持原样，新增根级文档和验证脚本。
原仓库历史、活跃锁、旧OPS任务、VS Code插件、VSIX、历史训练输出与未跟踪的操作者文件不在本次导入范围。
旧preservation.json的完整路径库存保留在上游；本仓库保存带上游SHA的明确摘要。
照片和标注只反映来源提交快照，不代表用户之后的修改或新认可的训练数据。

硬件限制沿用本次会话：只做电脑端，不主动连接、采样、上传或运行板卡；UART默认关闭，无运动。
当前没有板端25FPS验收。首次发布到新仓库由用户明确授权；未来主线集成和硬件部署按实际授权执行。

推荐从Python3.11环境复现。requirements-host.txt为基础主机测试依赖。
完整469项基线测试还使用ONNX1.17、PyTorch2.10、torchvision0.25、Ultralytics8.3.235及本机GCC。
NNCase编译是可选研究工具，不是普通回归的安装前提。GUI标注工具需要单独准备本机环境。
未包含训练权重；校验/门控工具继续拒绝缺失、未审核或来源不一致的输入。
