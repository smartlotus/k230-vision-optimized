# CameraXY USB 报告输出交接

2026-11-18，北京时间。使用 `/sdcard/xy…` 运行 USB 工程验证时，新报告保存到 `/data/logs/<部署目录名>/<报告名>.json`。现有 `/tmp/xy…` 调用仍在其安装目录写入报告。工程配置和 KModel 仍从传入的部署目录读取。

实际部署路径为 `/sdcard/xyusb1118/board_usb_smoke.py`，板端读回 14,055 B，SHA-256 为 `dc6e50d8c61735aa89236d45e6c31d6fcafb149a1cfb121d604015bc1ffea18f`。已创建 `/data/logs/xyusb1118`，23 项主机回归通过。旧报告维持原路径与原源码身份。

按用户要求停止板端操作，由 VS Code 会话接续。新代码尚未启动 USB 采集验证，下一次有界运行应使用新报告名并从 `/data/logs/xyusb1118` 读回。

旧 SD 卡接入复核显示固件 v1.8-36 下 `/sdcard` free blocks=0，`/data` total/free blocks=0。`board_usb_smoke.py` 现在在持久报告模式启动前检查 `/data`，未挂载时返回 `DATA_VOLUME_UNAVAILABLE`，不会尝试把报告回退写入 `/sdcard`。新卡挂载 `/data` 后继续使用原 `/data/logs/xyusb1118` 路径。
