# TopFace 原有代码接入与 FPS

本页保留实现与短测证据索引；2026-11-16后续现场刷新见 [最新项目状态](../../STATUS.md) 和 [接续交接](../../HANDOFF.md)。本次现场记录为 [12:12–12:13快照](status-refresh-20261116T041232Z.json)，该时刻USB仍在运行，最终usb-live01.json未出现。

用户于2026-11-16完成手工观察，认可识别效果并授权部署和优化。定量准确率仍未测，用户反馈原文见 [USER_ACCEPTANCE.json](USER_ACCEPTANCE.json)。

- 同一KModel SHA `f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`，不改权重、320×192输入、0.25分数和0.5 NMS阈值。
- 显式接入原CameraManager和VisionAdapter的TOP分支，经过原StableTracker和apply_calibration；QR/LANE、COLOR/RING保留原实现。
- CSI2同时提供原RGB565通道0与模型RGBP888通道2；TOP用通道2同一帧生成检测和预览，CSI1保留QR/LANE来源。CSI0隔离。
- 模型资源由TopFaceBackend持有，运行结束显式close；active/draft和生产profile不写入，main.py/boot.py不改。
- 部署目录 `/sdcard/tfp1116`，模型从已经验证的板端工件流式复制，源码逐份回读；[部署记录](production-deployment.json)为70文件的精确SHA。原包和原SD目录保留。
- 新入口：`k230_transport_mvp/run_topface_service.py`；现有VisionAdapter可显式注入模型backend。传统调用默认行为保留。
- 试验版本1000帧为10.84199FPS，1000/1000、无采集失败、cleanup=[]，[fast1000.json](fast1000.json)。它是单相机试验，不当作集成后的双相机性能。
- [资源报告](RESOURCE_FPS_REPORT.json)为双相机集成gc4版：Python heap峰值2.990MiB、sys_heap8.046MiB/64、MMZ61.457MiB/511；sys_page与MMZ不可相加为整机占用。此前单路试验的1.742MiB/4.45MiB/18.547MiB仍在fast1000.json，仅用于该配置。CPU/KPU硬件利用率未测，KPU耗时×FPS仅为整帧时间占比估计。
- 完整640×360 RVV与原修复路径的230400个像素一致，[逐像素结果](fast-conversion-parity.json)。vector class gate只跳过已验证全部6类finite且低于原阈值的background；NaN/Inf、tie、容量、边界和NMS规则不变。
- 原代码双相机集成[1000帧基线](integrated1000.json)为9.32FPS；[gc_interval4版](integrated-gc4-1000.json)1000/1000、10.815FPS，采集失败0、cleanup=[]。最终协议状态调整用[20帧短测](final-smoke20.json)验证。真实KPU输出的优化解码与原decoder一致，见[native-decoder-parity.json](native-decoder-parity.json)。
- 生产会话[production-session01.json](production-session01.json)在用户要求切换USB时人为停止，1268成功帧、墙钟12.71FPS，但status=RUN_FAILED/IDEinterrupt，不标长稳通过。不同场景/目标数影响decode，12.71与10.82不是严格同场景对照。
- USB已枚举并协商1280x720MJPEG30FPS，显示同一模型。[600帧实测](usb-model600.json)600/600、8.4175FPS、采集错误0、cleanup=[]；输入保持16:9缩放到640x360，再转独立CHW。原600报告的device/modes受locals内省影响缺失，另存probe与actual_mode证据；最终脚本改直接变量并通过[5帧设备身份短测](usb-final-smoke5.json)。[USB实时状态](USB_PREVIEW_STATUS.json)与CSI分开，USB没有使用原CSI标定。
- 当前持续预览为USB，18,000帧上限，最终报告计划`/sdcard/tfp1116/usb-live01.json`；启动和fresh帧不算长稳通过。CPU、KPU硬件利用率未测，不能用时间份额冒充。

读取最新OPS交接和[经验总账](../../LESSONS_LEARNED.md)后再恢复。

