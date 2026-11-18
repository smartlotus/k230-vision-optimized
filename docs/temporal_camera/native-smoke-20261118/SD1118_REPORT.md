# 新 64G SD 卡接入结果

新卡已经被 K230 正确挂载，容量分成两个文件系统。

| 挂载点 | 总容量 | 可用容量 | 用途 |
|---|---:|---:|---|
| `/sdcard` | 约 500 MB | 约 112 MB | 启动和兼容分区，Windows 主要显示这一部分 |
| `/data` | 约 61.46 GB | 约 61.45 GB | 64G 卡的主数据分区，K230 应用和模型放这里 |

因此电脑上看到的 32G 卡只有很小容量，是因为电脑只识别到小 FAT 分区。K230 的主数据分区由系统挂载为 `/data`，不一定以普通 Windows 盘符显示。

本次已完成：

- `/data` 256 字节写入、读回和删除通过。
- 滤波定位 CNN1 测试包 9 个文件写入 `/data/k230_temporal_v1_sd1118`。
- 模型和配套文件完整 SHA256 读回通过。
- 新卡目录可读取模型和配置，运行时为 2.11.0。

首次 10 帧启动在摄像头取帧前停止。当前固件连续 5 次枚举结果均为 `UVC.probe() == (False, None)`，视频模式为空。板端没有检测到 USB 摄像头，所以没有进入推理。

下一步只需要重新插拔 USB 摄像头并确认连接到 K230 的 USB-A HOST 口。重新枚举后应看到设备 ID 和 MJPEG 模式，通常应再次出现 `openaicam#openaicam` 与 1280×720 MJPEG 30 FPS。确认后使用新报告 ID 重新运行：

```powershell
python -B -X utf8 research/temporal_camera/native_smoke.py run --remote /data/k230_temporal_v1_sd1118 --run-id sd1118_10b --frames 10 --out docs/temporal_camera/native-smoke-20261118/sd1118-10b-intent.json
python -B -X utf8 research/temporal_camera/native_smoke.py fetch --remote /data/k230_temporal_v1_sd1118 --run-id sd1118_10b --out docs/temporal_camera/native-smoke-20261118/sd1118-10b-readback.json
```

当前失败收据 `sd1118_10a` 保留，不覆盖。
