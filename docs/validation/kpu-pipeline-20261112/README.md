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

另外保留了以下阶段结果：

- 原严格后处理拒绝量化输出的微小负置信度（最小约-0.00035），随后返回REPL。
- 仅用于测速的标量投影路径已通过主机测试；它把微小越界值投影到合法范围，不改变生产验证器和0.25置信度门限，也不作准确率结论。
- KPU＋轻量标量后处理曾完成一张冻结图10次，约23.6ms/次、约42FPS；该结果尚未扩展到四图。
- AI2D和完整摄像头链路多次在模型加载或首个AI2D阶段后失去通信，未生成完整FPS结果。

本轮主机回归 **486 项通过，无跳过**。原项目配置、UART、运动和生产保存均未改变。
官方 CanMV 代码建议用 AI2D 输出tensor直连KPU输入，并在 finally 中释放模型、AI2D和tensor，再调用 `gc.collect()` 与 `nn.shrink_memory_pool()`；后续恢复板卡后应按这个边界分段测试。[Kendryte AI开发文档](https://www.kendryte.com/k230_canmv/en/main/ai_dev_doc.html)
