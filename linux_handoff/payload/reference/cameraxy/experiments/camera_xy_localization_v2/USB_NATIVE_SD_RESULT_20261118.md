# CameraXY 新 SD 卡持久路径结果

2026-11-18，北京时间。新 SD 卡接入后，先在 `/sdcard/xyusb1118` 创建独立目录并写入 36 字节探针，原生写入和读回 SHA 均通过。随后把工程源码、配置和 KModel 写入新卡，所有成员逐字节回读。

## 新卡实测

| 项目 | 结果 |
|---|---|
| SD 路径 | `/sdcard/xyusb1118` |
| SD 可用空间 | 约 114.8 MB，最后回读 `114819072` 字节 |
| 板端固件 | CanMV v1.8-0，2026-07-03 构建 |
| KPU runtime | 2.11.0 |
| UVC 设备 | `openaicam#openaicam` |
| UVC 模式 | 1280×720 MJPEG 30 fps，另有 YUY2 模式 |
| 模型 | 2,133,296 字节，SHA `ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b` |

### 工程文件回读

- `board_probe.py` SHA `51aaa38e69bcba06a561236bcd4ce4d8e8479e42c9ba180348218f0c5ed710b4`
- `board_usb_smoke.py` SHA `476c0eba7d4fd42e8d81ae01a3fec4db335dd530df81a7d16592d477062625e6`
- `engineering.json` SHA `3d923840a98907c3569678b7354fd102a166162bf4a0f94e36fb3047de359792`
- `engineering.kmodel` SHA `ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b`

### 三轮持久路径运行

| 报告 | 帧数 | 时长 | 处理帧率 | 状态 |
|---|---:|---:|---:|---|
| `chain30-sd1118.json` | 30/30 | 3.801 s | 7.893 fps | `ENGINEERING_CHAIN_COMPLETE` |
| `chain300-sd1118.json` | 300/300 | 35.944 s | 8.346 fps | `ENGINEERING_CHAIN_COMPLETE` |
| `restart30-sd1118.json` | 30/30 | 3.749 s | 8.002 fps | `ENGINEERING_CHAIN_COMPLETE` |

三轮共 360 帧。每轮取像、AI2D、KPU、解码和显示计数完整，空帧为 0，清理异常为空，输入输出 ABI 与工程配置一致，主坐标保持 `null`，置信度保持 `None`。新卡没有 `/sdcard/main.py` 或 `/sdcard/boot.py`，验证器已允许两个启动文件都不存在，只检查前后集合一致。

主机聚合验证返回 `BOUNDED_NATIVE_ENGINEERING_PASS`，三轮唯一报告共 360 帧。23 项主机测试通过，包含新卡空启动文件场景。

## 解释边界

这次确认的是新 SD 卡上的持久读回和工程调用链。模型最初在 v1.8-36 环境下编译，当前板端是 v1.8-0。三轮实测证明当前固件可以加载并运行该 KModel，未重新编译模型。

用户指定的正确率和道路效果没有测试。当前运行的是 `learned_map` 工程候选，融合方案和科学定位门仍保持 HOLD。没有写入启动脚本，也没有做断电恢复测试。

## 证据

- [新卡聚合验证](evidence/USB-native-20261118-sd/verification-final-v2.json)
- [持久化预检](evidence/USB-native-20261118-sd/sd-persist-preflight.json)
- [持久化后检查](evidence/USB-native-20261118-sd/sd-postflight.json)
- [30 帧报告](evidence/USB-native-20261118-sd/chain30-sd1118.json)
- [300 帧报告](evidence/USB-native-20261118-sd/chain300-sd1118.json)
- [软重启报告](evidence/USB-native-20261118-sd/restart30-sd1118.json)
- [新卡运行身份](evidence/USB-native-20261118-sd/run-identities.json)

这些文件记录的是当前新卡和当前固件状态，后续换卡或升级固件需要重新做持久化读回和短跑验证。
