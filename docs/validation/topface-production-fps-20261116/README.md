# TopFace 原有代码接入与 FPS

用户于2026-11-16完成手工观察，认可识别效果并授权部署和优化。定量准确率仍未测，用户反馈原文见 [USER_ACCEPTANCE.json](USER_ACCEPTANCE.json)。

- 同一KModel SHA `f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`，不改权重、320×192输入、0.25分数和0.5 NMS阈值。
- 显式接入原CameraManager和VisionAdapter的TOP分支，经过原StableTracker和apply_calibration；QR/LANE、COLOR/RING保留原实现。
- CSI2同时提供原RGB565通道0与模型RGBP888通道2；TOP用通道2同一帧生成检测和预览，CSI1保留QR/LANE来源。CSI0隔离。
- 模型资源由TopFaceBackend持有，运行结束显式close；active/draft和生产profile不写入，main.py/boot.py不改。
- 部署目录 `/sdcard/tfp1116`，模型从已经验证的板端工件流式复制，源码逐份回读；[部署记录](production-deployment.json)为70文件的精确SHA。原包和原SD目录保留。
- 新入口：`k230_transport_mvp/run_topface_service.py`；现有VisionAdapter可显式注入模型backend。传统调用默认行为保留。
- 试验版本1000帧为10.84199FPS，1000/1000、无采集失败、cleanup=[]，[fast1000.json](fast1000.json)。它是单相机试验，不当作集成后的双相机性能。
- [资源报告](RESOURCE_FPS_REPORT.json)：Python heap峰值1.742MiB、sysheap4.45MiB/64、MMZ18.547MiB/511；sys_page与MMZ不可相加为整机占用。CPU/KPU硬件利用率未测，KPU耗时×FPS仅为整帧时间占比估计。
- 完整640×360 RVV与原修复路径的230400个像素一致，[逐像素结果](fast-conversion-parity.json)。vector class gate只跳过已验证全部6类finite且低于原阈值的background；NaN/Inf、tie、容量、边界和NMS规则不变。
- 当前集成1000帧在运行，最终收据尚未取得，正式集成/长期稳定性不提前标通过。禁止用preview帧计数冒充识别帧数。

读取最新OPS交接和[经验总账](../../LESSONS_LEARNED.md)后再恢复。

