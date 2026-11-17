# 官方初赛场地修正与双起点测试

依据用户提供的 `official-2027-task.pdf` 第 3 页图 1、第 4 页表 2 和第 7–8 页现场流程。源文件 20 页，SHA256 `1ac7fa340193591874e3ec84f9427567864c16ab39fb500db814a4d2cfc000e4`，与执行包内 PDF 逐字节相同。图 1 给出 2400 × 2400 mm 场地、400 mm 中央通道和两个 300 × 300 mm 蓝色启停区。外围灰道连接中央道路和两个蓝区。用户确认比赛从任一蓝区出发，另一蓝区作为终点，作为本轮仿真合同。PDF 的终点表述为抽签确定的启停区。

原 RoadNet `.blend` 的底板曾误用白色 Class 2。新副本将 `2.4 m square` 改为灰道材质和 Class 0。几何审计逐项核对 31 个网格的坐标、面索引和世界变换。源文件 SHA256 前后一致。原黄区、蓝区与白色站台的几何和标签逐点通过语义图检查。物料与相机渲染复用现有 CameraXY V2 的场景加载、原三段式 TopFace GLB 和标签投影计算。

- [修正副本](field_gray_v1.blend) · [俯视语义图](field_mask.png) · [场景预览](field_preview.png) · [几何指纹](CORRECTION.json)
- [100 帧 Smoke](smoke/QA.json) · [1,000 帧 Pilot](pilot/QA.json) · [双起点路径验证脚本](../../../research/temporal_camera/audit_competition.py)
- [地图参考库来源](REFERENCE.json) · [训练端实现](../../../research/temporal_camera/train_map_temporal.py) · [滤波端实现](../../../research/temporal_camera/localization_runtime.py)

启动时两个蓝区各占一半先验概率。光心初始范围采用 100 mm 仿真余量，该值未在实体安装上测量。每次更新保持全局软支持。视觉连续缺失触发 LOST，恢复时在全场搜索。没有实拍阈值时，运行端返回候选坐标和状态，可信坐标字段为空。

路径验证逐段按最长 10 mm 间隔取样，只使用灰道与蓝区的语义标签。结论覆盖模拟光心投影。场地装置实际坐标、物理安装和车体通行由现场数据验收。

1,000 帧 Pilot 分为 500 帧训练、250 帧验证、250 帧测试。40 条路径各 25 帧，每个拆分内两种出发方向等量。五种路线保持相同因果窗口和相同测试轨迹。模拟外观、速度、启动停止与转弯记录在清单中。丢帧通过删去真实输入帧注入，模糊通过图像池化注入，视觉失效通过实际黑帧注入。重定位测试在进入 LOST 后切换到远处轨迹。仿真仍缺独立的真实场地外观留出。

原 CameraXY V2 的 RGB 权重逐张量复用。用修正语义图和原参考编码器重新计算 2,304 个 50 mm 间隔的锚点，每点 12 个内部参考朝向。相机朝向不作为模型输出或监督标签。重算参考库后的冻结模型在 250 帧验证上，双区滤波初始区域判断为 5/10，中位误差 909.5 mm，P95 为 2,208.4 mm。对照单帧中位误差为 950.7 mm。这是未更新参数的诊断，距离扩量条件仍有明显差距。 [验证集回放](VAL_REPLAY.json)

修正版实际完成统一的短程 Kaggle 微调。CNN 学习单帧残差，GRU 和 TCN 学习因果时序残差。所有路线共享固定的 CameraXY 编码器和地图参考，按验证集选择权重并保留 epoch 0 候选。评估比较单帧、GRU 和 TCN 各自添加同一滤波器前后的结果。 [实测对照](../cloud/gray-map-pilot-a06/REPORT.md) · [提交身份](../cloud/gray-map-pilot-a06/SUBMISSION_INTENT.json)

实验 A 路线已实际编译出 2,638,816 字节的 KModel。完整 250 帧 nncase Simulator 回放对浮点 ONNX 的最高分锚点一致率为 71.6%。双区滤波中位误差 844.6 mm，P95 为 2,208.4 mm，启动判断 5/10。精度仍处于 HOLD，真机状态为 NOT RUN。 [指标](KMODEL_VALIDATION_COMPARISON.json) · [模型包说明](OFFLINE_MODEL_DELIVERY.md) · [模型包 SHA256](MODEL_DELIVERY_SHA256.json)
