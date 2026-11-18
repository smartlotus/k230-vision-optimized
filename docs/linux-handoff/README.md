# 庐山派 K230 Linux 迁移交接

把现在的 CanMV 视觉链路迁到同一块庐山派 K230 1G 的原生 Linux，按相同模型、同一输入和同一业务规则复现。先完成 USB 摄像头、KPU 和本地输出，再接 CSI、预览、扫码、通信和长期运行。CanMV 已有结果和 Linux 验收分别记录。

- 给接手 AI 的完整任务指令见 [EXECUTION_PROMPT.md](EXECUTION_PROMPT.md)。
- 系统选型、验证步骤和回退见 [LINUX_PLATFORM.md](LINUX_PLATFORM.md)。
- 现有功能、证据与迁移验收见 [PARITY_MATRIX.md](PARITY_MATRIX.md)。
- [模型及文件 SHA-256 清单](../../linux_handoff/PAYLOAD_MANIFEST.json)包含 10 个 KModel、配对 ONNX、输入输出契约、运行配置、跨仓库源码快照与证据。TopFace 固定输入与历史帧的原始张量在 `linux_handoff/payload/evidence/canmv/`，RoadNet 实测固定输入和原始输出在 `linux_handoff/payload/evidence/roadnet-fixed/`。
- 同包的 `k230_transport_mvp/core`、`vision`、`k230`、`ports`、`stm32` 与测试为原行为参考。参考代码含 CanMV 专用模块，见矩阵确认重用方式。部分历史报告中的链接指向原仓库，完整来源由清单里的 Git SHA 定位。

解压后，从包根用 Python 3.11 或更新版本运行 `python linux_handoff/verify_payload.py`。这会核对交付模型、参考文件和证据的全量 SHA-256。离线 ZIP 还提供全包 `PACKAGE_MANIFEST.json`，可用相同脚本加 `--package` 验证。Git 检出内通常运行无参数的核验，使用 `python linux_handoff/build_bundle.py /path/to/K230_Linux_Handoff.zip` 可重建离线包。

先使用封存模型进行兼容性验证。只有证据表明现有 KModel 与 Linux runtime 不兼容时，才用绑定的 ONNX、原量化配置和训练集内校准样本生成一份新产物。ONNX 输入可能与 KModel 的外部输入不同，逐个核对契约。原训练集和云端归档不随本包复制。重新编译缺校准样本时，按模型原报告找回输入，保留兼容性失败和新旧模型对照。

当前分支交付的是迁移资料、模型和验收路线，Linux 启动、KPU 实测及摄像头迁移状态均为 `NOT_RUN`。原两张 CanMV 卡用于对照与回退，Linux 使用单独实验卡。

