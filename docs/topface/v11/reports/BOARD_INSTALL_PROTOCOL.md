# 正式 TopFace 包的独立目录安装

## 2026-11-16 实机补充

通用经验见 [项目总账](../../../LESSONS_LEARNED.md)。本次早期未复用下述安装器，大模型经过工具文本输出被截断；最终分片重传和目标全字节 SHA 恢复。安装器的分片与回读要求继续有效。

当前 v1.8-36 的 RGBP888 直接 `to_rgb565()` 显示路径存在实测异常。修复源码保持原 NCHW 推理输入，显示端显式交错为 RGB888 后转换；当前任务分支运行参数为 `run(..., display=True, ide_preview=True)`。原发布包和历史源码不改。部署修复副本时校验修复源码自己的 SHA，不能继续使用旧包校验清单冒充一致。

以下原始安装规程及末尾“本轮仅主机验证”的叙述属于该规程建立时的历史状态；最新实机证据见 [fixed600.json](../../../validation/camera-diagnosis-20261116/fixed600.json)。

此流程仅安装已经验证的正式板测包，不启动模型推理。训练、转换和交付包仍须先完成原始接受流程。上传成功不表示 K230 兼容性、准确率或 FPS 通过。

使用仓库 `research/topface/deploy_board_trial.py`，输入由 `package_formal_delivery.py` 生成的不可变目录。它检查交付 `SHA256SUMS.json`、全部所列文件、board 成员清单、模型与契约身份，只上传九个板端文件。

当前 Codex 实例的配置不一定包含 CanMV 服务。用 `--mcp-config` 指定实际配置，不修改全局配置或 CODEX_HOME。此电脑原有配置位置为 `C:/Users/28389/.codex/config.toml`；配置内容及认证值不进入输出。连接前确认官方 CanMV MCP 服务已开启，先检查 connected、idle 和目录明确不存在。

以下命令在仓库根运行；以实际已接受的目录及本轮新路径替换示例值。第一条默认只预检，不在 SD 上写入：

```powershell
& .ops/local/topface-control-isolated-311/Scripts/python.exe research/topface/deploy_board_trial.py `
  --delivery C:/Users/28389/Desktop/k230-project/TopFaceNet-V1.1-a01 `
  --remote /sdcard/tf_a01 `
  --mcp-config C:/Users/28389/.codex/config.toml `
  --output docs/topface/v11/reports/board-preflight-a01.json
```

同一确认后执行安装。回执另存新文件，不能覆盖预检或任何旧安装回执：

```powershell
& .ops/local/topface-control-isolated-311/Scripts/python.exe research/topface/deploy_board_trial.py `
  --delivery C:/Users/28389/Desktop/k230-project/TopFaceNet-V1.1-a01 `
  --remote /sdcard/tf_a01 `
  --mcp-config C:/Users/28389/.codex/config.toml `
  --output docs/topface/v11/reports/board-install-a01.json --install
```

安装使用 262,144 字节分片，逐片读回匹配原字节。随后在板上运行一次有界文件合并脚本，关闭文件后重新读取每个合并目标，验证大小和 SHA-256；终端用本次唯一标识匹配确认，不清空旧输出。目标与分片路径均小于 64 字节。脚本不导入模型、相机、显示或 UART，也不清理旧文件。

终态 `VERIFIED_BOARD_FILES_ONLY` 只证明这次存储校验。回执始终保留 `inference_started: false`、准确率和 FPS 的 `NOT_RUN`。异常、超时或确认缺失写入 `NEEDS_RECONCILIATION`；保留同一 SD 目录和回执检查实际结果，不重新执行上传、合并，也不停止可能仍在运行的脚本。

只有安装验证完成、实际固件/runtime/CSI2 兼容性核对后，才按 [板测计划](BOARD_TEST_PLAN.md) 运行有界推理。新目录入口须显式传 root，避免使用脚本默认目录：

```python
import sys
sys.path.insert(0, '/sdcard/tf_a01')
from topface_trial import run
run(root='/sdcard/tf_a01', frames=600, display=True)
```

数值配对使用另一个全新安装目录，运行 `frames=15, display=False, paired_samples=3`，按 [配对协议](BOARD_PAIR_PROTOCOL.md) 收取同帧数据。落盘模式不测 FPS。解析 `TOPFACE_BOARD_RESULT` 的实际状态；RPC 提交成功不能代替脚本运行成功。18,000 帧稳定性和新真实盲测分别留存证据。

本轮新增路径只有主机模拟验证，未启动服务、连接设备、写 SD 或运行 KPU。原授权保护范围继续有效：不改 main.py/boot.py，不覆盖旧目录，不保存生产参数，UART 和运动关闭，CSI0/CSI1 原流程保持隔离。
