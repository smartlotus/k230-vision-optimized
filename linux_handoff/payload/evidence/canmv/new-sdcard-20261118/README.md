# 新 SD 卡接入后的工程链路结果

日期：2026-11-18（北京时间）。新卡原有 `/sdcard/xyusb1118` 目录保持不变；当前 TopFace 和 RoadNet 分别部署到 `/sdcard/tfp1116`、`/sdcard/rn1117a`。

## 已完成

- 新 SD 可写可读，复查后仍有约 105.12 MiB 可用空间。
- 新卡固件为 `v1.8-0-gc2d1f5c`，运行时为 nncase 2.11.0。
- TopFace 模型分块写入、合并及 SHA 回读通过：`f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`。
- RoadNet 文件写入和 SHA 回读通过；模型 SHA 为 `cba11c87ce36305ffac6dee18b65b4f1e5b47070e00a2f9e706d0aee72a4980e`。
- USB 摄像头为 `openaicam#openaicam`，协商为 1280×720 MJPEG、约 30 FPS。
- 当前链路 30.008 秒内完成 191/191 帧：采集、预处理、RoadNet、TopFace、显示和释放均完成；采集失败 0，清理异常 0，工程 FPS 6.364969。
- 延长复测 120.127 秒内完成 750/750 帧；采集失败 0、清理异常 0，工程 FPS 6.243392。终端按唯一 `run_id` 提取，未重跑旧回执。
- 运行后 MMZ used=0，板卡已空闲。新卡没有 `main.py`/`boot.py`，没有写启动入口。

## 范围边界

本次只证明新卡上的代码和硬件调用链可运行。正确率、道路质量、定位、标定和长期稳定性均未测试。

结果文件：[FINAL_REPORT.json](FINAL_REPORT.json)、[new-sd-final.json](new-sd-final.json)、[usb-result.json](usb-result.json)、[usb120-result.json](usb120-result.json)、[topface-model-recovery.json](topface-model-recovery.json)、[roadnet-deployment.json](roadnet-deployment.json)。
