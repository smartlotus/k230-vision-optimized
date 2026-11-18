# USB 与 CameraXY 工程贯通测试

后续已完成：[最终原生结果](USB_NATIVE_RESULT_20261117.md)。下文保留本轮此前的主机准备与并发暂停事实，当前状态以最终原生报告为准。

本轮用户范围：检查代码/硬件能否正常调用 USB 摄像头并执行全流程；不测定位正确率或道路检测效果。原有科学验收 HOLD、PIL/AI2D 差异及正式导出门保留。

## 当前完成状态

| 层级 | 实际结果 |
|---|---|
| 原生 USB 探测 | 已识别 `openaicam#openaicam` |
| 模式协商 | 1280×720 MJPEG，设备报告约30fps |
| 采集及 RGB ABI | 30/30 请求成功，0空帧，RGB `[360,640,3]`，清理无报错 |
| 工程候选导出 | 本线冻结 learned_map 已导出 ONNX；5输入 Torch/ONNX 最大绝对差1.52587890625e-5 |
| 工程候选编译 | nncase/插件2.11.0，KModel 2,133,296字节，SHA `ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b` |
| Simulator ABI | 5输入均输出有限值，形状 `[1,932]`；数值变化实值保留，不授予量化正确率PASS |
| 新入口板端整链 | 尚未运行；用户确认另一对话仍测试，本对话转为仅主机工作 |

32个校准输入只取冻结训练集。没有训练更新，没有按真实场景或道路真值打分。`engineering.json` 的 `native_status=NOT_RUN` 是编译时快照；后续原生运行单独留报告。

## 原生操作暂停的实际原因

本对话在 OPS 认领 COM3/USB 后，执行了30帧采集测试。读取清空后的终端时，采集结果后又出现本对话未发起的 `USB_CHAIN_C_PREFLIGHT` / `usb-chain30-c-preflight`，证据见 `evidence/USB-engineering-20261117-d/capture-and-concurrency-tools.json`。仅凭当前脚本空闲不能判断另一个会话已停止提交。

用户随后明确答复“另一对话仍在测试，本对话先做主机工作”。从发现并发起，本对话没有再做板端写入或启动，答复后也不再查询板端。其他会话的日志不计入本轮 CameraXY 验收。本轮交接释放自己的COM3/USB认领，保留主机产物。

SD卡本轮实测剩余空间为0。未删除旧数据。独立 `/tmp/xyusb1117d` 进行了8192字节写入/读回测试，可用于后续RAM暂存；持久模型和证据保存到主机。RAM目录不代表断电持久保存。

## 独立工程入口

`export_engineering.py` 复用既有 ONNX 数值和编译函数，不调用正确率统计，不修改 `export_k230.py` 的正式验收门。工件为 `engineering.kmodel`、`engineering.json`，强制 `production_eligible=false`、`require_unlocalized=true`。

`board_usb_smoke.py` 是自包含 USB → RGB → AI2D → 本线 KPU → 全输出结构解码 → 可选预览 → 资源释放入口，不引入 CSI、道路分支或生产控制。它校验模型字节与实际导入路径，每帧释放数组/张量和图像，结束清模型/AI2D、缩减运行池并停UVC/显示/媒体。主X/Y固定为null，两个置信度为None；五候选仅供结构检查。

后续由独占设备的对话先运行30帧，再按结果运行300帧与短重启测试；每次使用新报告名。启用预览时必须每帧显示成功，关闭时显式标NOT_RUN。采集返回None只称请求空帧，不称真实传感器丢帧率。

本线选中方案仍是 fused_map，本轮工程候选仅是 learned_map；工程调用通过不能代表融合方案或科学定位通过。

## 验证与归档

- 主机7项回归通过，0跳过，包括工程资格拒绝、模型身份、路径边界和统计检查。
- 独立审查修复了无帧时周期统计漏计、预览状态遗漏与帧清理异常处理。
- 原始采集报告、工程manifest、实际KModel及主机日志位于 `evidence/USB-engineering-20261117-d/`。
- `package_engineering.py` 生成的 `CameraXY_Engineering_Host_Handoff_D01.zip` 包含实际模型、两份原生源码、manifest和操作说明，逐成员字节及ZIP CRC已核验；包本身尚未上传或在板端执行。
- 原执行包禁止push，本轮沿用本地检查点，不向远端发布。
