# CameraXY USB 原生工程贯通结果

2026-11-17，北京时间。用户确认另一板端对话已结束，本轮重新认领COM3/USB并确认板端空闲。按用户范围只做代码和硬件调用测试，不计算定位正确率或道路检测效果。

## 本轮结论

最终源码的 USB → RGB → AI2D → 实际 CameraXY KPU → 全输出结构解码 → 实时预览 → 释放流程通过三轮有界实测。当前候选是 learned_map，选中的科学融合方案 fused_map 不在此次运行范围。

| 最终版本运行 | 完整处理帧/请求 | 时长 | 处理帧率 | KPU均值 |
|---|---:|---:|---:|---:|
| 首次短跑 | 30/30 | 3.914 s | 7.665 fps | 1.332 ms |
| 持续运行 | 300/300 | 36.778 s | 8.157 fps | 1.309 ms |
| MPY软重启后重新初始化 | 30/30 | 3.887 s | 7.718 fps | 1.292 ms |

最终三轮共360帧，取像请求、成功帧、AI2D、KPU、结构解码和显示计数均完整；每轮8段计时分母与请求相同。无帧返回0，清理异常0。输出为有限float32 `[1,932]`，输入为uint8 `[1,3,96,160]`。

每轮MMZ视频/模型缓冲池初始与最终占用均为0，已存在main.py SHA前后一致。这里的恢复证据是MPY软重启后重新初始化成功，不是断电恢复或长时间稳定性验收。

## 来源与范围

- 板卡 LCKFB-LSPI-K230-1G-CanMV，固件v1.8-36，nncase runtime2.11.0。
- 实际UVC设备串 `openaicam#openaicam`，1280×720 MJPEG，协商约30fps。该串绑定本轮设备，不用于推断模组厂商型号。
- 实际KModel 2,133,296字节，SHA `ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b`；沿用上一轮冻结工程候选，没有重训或重新编译。
- 最终board_usb_smoke.py SHA `476c0eba7d4fd42e8d81ae01a3fec4db335dd530df81a7d16592d477062625e6`。
- 推理仍用RGB888；显示按官方UVC示例独立转换RGB565，显示图像、推理数组和原始帧分别释放并处理别名。
- 主X/Y固定null，两个置信度固定None，状态unlocalized，production_eligible=false。正式导出门和科学HOLD未改变。

## 预览与操作者确认

用户现场确认实时预览正常，画面是真实场景。本轮撤回仅凭早期单张截图作出的异常推断，不把该场景记为故障或修复成功。

为核对图像来源，另保留同帧1280×720原始JPEG、640×360 RGB888字节及无像素改动的主机解码图。二者可正常解码。最终300帧运行中保存了当前新预览帧（frameId44，取回时age约1ms），与现场场景相符。初始化的首张预览可能为空白，不能当作持续帧或故障证据。

早期e版源码另有30/300帧API调用报告，保留为独立历史，不与最终f版360帧混算；最终结论仅取 `run-identities.json` 中三份唯一报告。

## 存储与交付

SD卡当前剩余空间0。本轮没有删除SD文件，代码和模型暂存在独立易失RAM目录 `/tmp/xyusb1117f`。全部报告已下载主机，板端打印的SHA与主机字节一致。RAM暂存不代表持久部署，断电后不能假定这些工件仍存在。

最初文件RPC不支持该临时路径；已授权实验目录的原生文件读写经实际验证后用于暂存。传输中字符串base64解析尝试失败后改为字节输入，模型按分块大小/偏移/分块SHA及全文件SHA核对。此阶段失败发生在模型运行之前，未覆盖已有报告。

## 可复核证据

- [最终聚合核验](evidence/USB-native-20261117-e/verification-final.json)：三次原生报告、源码/模型/配置SHA、完整分母、KPU ABI、工程输出边界及资源清理均通过。
- [运行身份](evidence/USB-native-20261117-e/run-identities.json)绑定板端实际打印的报告SHA；清单拒绝空、重复文件/SHA及非法帧数。
- [持续运行报告](evidence/USB-native-20261117-e/chain300-f02.json)、[软重启报告](evidence/USB-native-20261117-e/restart30-f03.json)、[实拍预览](evidence/USB-native-20261117-e/chain300-f02-preview.jpg)。
- 主机16项回归通过，0失败、0跳过；独立只读复核确认实际三次计数、字节与释放证据。
- 更新后的工程包 `evidence/USB-native-20261117-e/CameraXY_Native_Engineering_F02.zip` 含同一模型及已测源码，逐成员字节/ZIP CRC已核验。包的主机字节核验与本节独立原生验证分别记录。

精确复核命令（从实验目录执行，输出必须用新路径）：

```powershell
python verify_usb_smoke.py --evidence evidence/USB-native-20261117-e --deployment evidence/USB-engineering-20261117-d/engineering.json --identities evidence/USB-native-20261117-e/run-identities.json --out .runtime/native-verify-fresh.json
```

本轮只完成此候选的有界工程贯通，不授予真实定位准确率、道路效果、融合方案、生产启动或断电持久化验收。后续若安排这些工作，应单独确定对应输入和验收范围。
