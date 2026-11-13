# KPU 管线与连接中断排查

本轮使用原有4次更新模型，没有重新训练或改权重：

- checkpoint SHA256：`6de5c9b6...`
- ONNX SHA256：`df9c6390...`
- kmodel SHA256：`59fd397f...`

已确认的成功结果是：4张冻结训练图片，每张20次 KPU-only 测量，平均约 **95.42 FPS**，单次约10.48ms。
这条路径只包含准备好的224×224输入、KPU和输出读取，不包含摄像头、AI2D、显示，也不代表识别准确率。

传统 CPU 几何链路在同4张图片上约3.91–4.72 FPS；它输出椭圆材料候选，模型输出可见顶面bbox候选，任务不同。

连接中断证据：CanMV 日志记录7次预览无帧重启，并出现 `USBDBG TX_BUF_LEN`、`frame_size`、`scriptRunning` 超时和握手失败。
日志分析显示预览重试发生在部分断连之前，但这只是时间相关性；物理USB、电源、固件和运行时资源根因仍 UNKNOWN。
此前断电阶段已完成远端 checkpoint；本节记录用户随后重新连接后的有界复测。

## 重新连接后的有界复测

用户重新连接后，在不启动预览、不改原配置的条件下完成了四张冻结输入的 KPU-only 短测：每张 10 次，95.69836–95.73684 FPS，四张图均有完整结果。该数字仍不包含相机采集、AI2D、后处理、显示或准确率。

零输入 AI2D 单元测试为 0.622375 ms，约 1606.748 FPS。把同一零输入接入 AI2D→KPU 组合时，脚本没有留下阶段输出，随后 REPL 回读为 `Not connected`；单张冻结 RGB888P 测试只留下 `AI2D_ONE_BEGIN`。这两条均记为 DISCONNECTED_UNKNOWN，没有推导完整链路 FPS。

本次详细收据见 `reconnect-followup-20261112.json` 与 `reconnect-followup-20261112.log`。最新 CanMV 日志统计为 7 次预览无帧重启、28 条断连记录；预览与断连存在时间相关性，但物理 USB、电源、固件或运行时根因仍为 UNKNOWN。

## 2026-11-13 稳定重连后的模型加载边界

新鲜握手与 REPL 回读成功后，合成零输入测试依次输出 `CHAIN_BEGIN`、`KPU_OBJECT`，下一步 `load_kmodel` 没有继续输出；随后 `scriptRunning` 查询超时，停止脚本后板卡变为 `ready=false`。这把本轮断点收窄到模型加载/运行时交互附近，但仍不能判定是资源压力、USBDBG 竞争、固件还是物理 USB/供电根因。

本轮没有再次重试板卡，也没有修改原配置、UART、运动控制或旧文件。详见 `stable-reconnect-followup-20261113.json` 与 `stable-reconnect-followup-20261113.log`。

## 2026-11-13 再次分段复测

新鲜 REPL 回读成功后，单独的 `load_kmodel()` 测试完整通过，随后单独 KPU 测试 8 次完成，平均 10.71825 ms、约 93.30 FPS。把 AI2D 输出接入 KPU 后，只留下 `AK_ZERO_BEGIN`，没有出现 `AK_ZERO_MODEL_LOADED`；之后 `scriptRunning` 报 USBDBG 设备不存在。`board_info` 仍返回 `connected=true/ready=true`，说明身份缓存与实际命令通道已经不一致。

本轮结果见 `retry-followup-20261113.json` 与 `retry-followup-20261113.log`。这进一步支持“组合脚本触发运行时/USBDBG/资源状态问题”的判断，但不能据此认定物理根因。

## 硬件与代码归因阶梯

本轮按层测试：REPL、`nn.kpu()`、独立 `load_kmodel()`、独立 AI2D 张量建图与运行均通过。把 KPU 模型状态和 AI2D 张量放进同一脚本后，终端无阶段输出，板卡变为 `connected=false/ready=false`；两秒后重连仍未恢复 `ready`。

因此当前最强证据指向“代码触发的板端运行时/SDK 资源或所有权交互”，而不是模型损坏、AI2D 单独故障或摄像头故障。USB 线、电源、固件崩溃和 USBDBG 传输仍未被排除，不能把它们写成已确认根因。详见 `attribution-ladder-20261113.json` 与 `attribution-ladder-20261113.log`。

## 修复：显式绑定 AI2D 输出 tensor

已修改新版 `kpu_pipeline_probe.py` 的 AI2D 路径：不再提前调用未绑定的 `kpu.get_input_tensor(0)`，而是显式创建并保持一个 `1x3x224x224` 的 uint8 tensor，先调用 `kpu.set_input_tensor(0, destination)`，再让 AI2D 写入该 tensor。主机回归 13 项全部通过。

修复后板端复测暂未执行：COM3 仍能看到 K230 身份，但 `ready=false`，等待 3 秒仍未恢复。修复后的板端组合路径保持 UNKNOWN，不把主机 PASS 当作硬件 PASS。详见 `fix-20261113.json` 与 `fix-20261113.log`。

## 修复后的实际对照测试

显式绑定输入 tensor 的零输入组合链路已经在板端通过：模型加载、输入绑定、AI2D 建图、AI2D 运行、KPU 运行和清理全部完成，KPU 约 10.954 ms，输出形状为 `[1,5,1029]`。

随后用 `f.read()` 一次性读取 691200 字节板上 RGB888P 冻结图时，板卡掉线，后续 REPL 回读为 `Not connected`，重连 1.5 秒仍未恢复 `ready`。这支持“大块临时分配/运行时压力”可能触发复位，但它不是主机高速上传；项目探针仍保留 `image.Image + readinto` 流式读取路径。详见 `transport-memory-followup-20261113.json` 与 `transport-memory-followup-20261113.log`。

## 流式真实图复测

修复后的流式真实图路径已完成单帧收据：读取 691200 字节、显式绑定输入、AI2D、KPU、输出读取和清理全部完成，总耗时 11.6 ms（AI2D 0.634 ms，KPU 10.966 ms），输出形状 `[1,5,1029]`。后续 5 次循环没有形成终端收据，虽然板卡身份仍为 `ready=true`，因此不把它记为持续 FPS。详见 `stream-real-followup-20261113.json` 与 `stream-real-followup-20261113.log`。

后续低波特率 5 帧循环已形成完整收据：流式读入、AI2D、KPU、输出读取和轻量后处理扫描全部完成，平均 14.4312 ms、约 69.29 FPS。该结果不包含相机采集和显示，且原始分数仍有 314 个负值、最高仅 0.0123291，因此不能作为准确率结果。详见 `stream-loop-followup-20261113.json` 与 `stream-loop-followup-20261113.log`。

## 低波特率传输规避验证

项目提供 `canmv-transport-settings.json`，建议复制到工作区 `.vscode/settings.json`，将 `canmv.baudRate` 设为 `2000000`，并保持预览关闭、单一控制端。2 Mbps 下，流式真实图路径再次通过：读取 691200 字节、AI2D 0.636 ms、KPU 10.982 ms、总计 11.618 ms；等待 25 秒后仍能保持 `ready=true`，并成功回读 REPL nonce。该结果验证了低波特率工作区配置是有效规避方案，但没有修复扩展/backend 仍可能发送的 `TX_BUF_LEN` 不兼容命令。详见 `low-baud-followup-20261113.json` 与 `low-baud-followup-20261113.log`。

## 真实截图准确率复测

使用已有人工审核的 2 张验证图和 1 张测试图，共 5 个 visible_top 标注框，对当前 `material.onnx` 做主机推理：置信度阈值 0.25 下预测框为 0，TP=0、FN=5、IoU@0.5 召回率为 0；最高分只有约 0.0127。4 张训练图的最高分也只有约 0.0126。这个结果说明现有小测试模型不能作为比赛模型，详见 `real-photo-accuracy-20261113.json` 与 `real-photo-accuracy-20261113.log`。该报告是 ONNX 主机评估，不冒充 KPU 板端准确率。

另外保留了以下阶段结果：

- 原严格后处理拒绝量化输出的微小负置信度（最小约-0.00035），随后返回REPL。
- 仅用于测速的标量投影路径已通过主机测试；它把微小越界值投影到合法范围，不改变生产验证器和0.25置信度门限，也不作准确率结论。
- KPU＋轻量标量后处理曾完成一张冻结图10次，约23.6ms/次、约42FPS；该结果尚未扩展到四图。
- AI2D和完整摄像头链路多次在模型加载或首个AI2D阶段后失去通信，未生成完整FPS结果。

本轮主机回归 **486 项通过，无跳过**。原项目配置、UART、运动和生产保存均未改变。
官方 CanMV 代码建议用 AI2D 输出tensor直连KPU输入，并在 finally 中释放模型、AI2D和tensor，再调用 `gc.collect()` 与 `nn.shrink_memory_pool()`；后续恢复板卡后应按这个边界分段测试。[Kendryte AI开发文档](https://www.kendryte.com/k230_canmv/en/main/ai_dev_doc.html)
