# 64 GB 新卡输出路径交接

2026-11-18，北京时间。板卡为庐山派 K230 1G CanMV，固件 `v1.8-0-gc2d1f5c`。Windows 通过 CanMV 设备读取到两个存储卷。程序卷空间较小，采集输出默认应该使用数据卷。

| 板端卷 | 总容量 | 剩余容量 |
| --- | ---: | ---: |
| `/sdcard` | 524,156,928 B | 110,231,552 B |
| `/data` | 61,456,908,288 B | 61,451,796,480 B |

| 当前入口 | 已配置的输出 |
| --- | --- |
| `/sdcard/tfp1116/run_topface_service.py` | `/data/logs/topface-<随机后缀>.json` |
| `/sdcard/xyusb1118/board_usb_smoke.py` | `/data/logs/xyusb1118/<报告名>.json` |
| `camera_snapshot_and_save.py` | `/data/captures/camera_chn_…` |
| `DataCollectionCamera.py` | `/data/captures/ai-cube/` |
| `save_image.py` | `/data/captures/save-image/` |
| `mp4muxer.py` | 两条录制入口均指向 `/data/test.mp4` |
| `audio.py` | `/data/test.wav` |

7 个修改项均做了主机 Python 语法检查和板端最终字节读回校验，CameraXY 的 23 项主机回归通过。部署前后 `main.py` 和 `boot.py` 均不存在，模型、算法和资源加载路径沿用原值。已有 MP4 编码、AI 录像和 Temporal V1 示例各自已有 `/data` 输出。RoadNet 的 `board_smoke.py` 由调用者传入 `report_path`。

用户要求停止继续改动，后续交给 VS Code 会话。这里验证到部署后的文件和目录，未运行改动后的摄像头、录像或报告生成入口。接手时以实际板端状态为准，先查看另一个会话已做的修改，核对其运行入口和输出路径。用一个有边界的运行确认新生成文件实际落在 `/data`，再核对大小及读回内容。旧报告和启动文件保持原样。

2026-11-18 bounded verification: `/data/logs/topface-verify-20261118-a.json` was created and read back at 1767 bytes. The deployed TopFace service reached initialization but returned `RUN_FAILED` with `Can not found sensor on 2` before capture; cleanup errors were empty. `/sdcard/tfp1116/production-session01.json` remained absent and startup files remained absent. This confirms the new output path and preserves the CSI2 hardware failure separately.

2026-11-18 old-card precheck: firmware v1.8-36 reported `/sdcard` free blocks `0` and `/data` total/free blocks `0`. `run_topface_service.py` now checks `os.statvfs('/data')` before camera initialization and raises `DATA_VOLUME_UNAVAILABLE` when the data volume is not mounted. It does not fall back to `/sdcard`.

板端各文件的前后哈希见 [部署收据](deploy.json)。本机原始脚本备份位于 TopFace 工作树的 `.ops/local/board-output-data-20261118/*.original`。固件示例本次直接在当前卡上调整，更换或重刷镜像后应重新核对。
