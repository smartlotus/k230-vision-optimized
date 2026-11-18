# CanMV 基线与 Linux 迁移验收

所有 Linux 项当前均为 `NOT_RUN`。表中的帧数和 FPS 属于各自 CanMV 场景，供回归对照。

| 业务模块 | 现有可复用资产与进度 | Linux 通过条件 |
| --- | --- | --- |
| TopFace 顶部六色 | Formal a01，uint8 RGB NCHW `[1,3,192,320]`，输出 `[1,1275,50]`。阈值 0.25、NMS 0.5。固定张量与历史帧分别有 CanMV/Simulator 同输入 SHA 对照。历史 USB 单路约 8.42 FPS。源 `640×360` 等比到 `320×180`，上下各 6、填 114。 | 冻结 KModel 先通过固定输入和历史帧的输入字节、张量及解码对照。保留空场、过容量、不确定、遮挡恢复及标定门，坐标/椭圆逆映射一致。完整物块真值仍要另行实测。 |
| RoadNet | 冻结交付 KModel，uint8 RGB NCHW `[1,3,192,320]`，输出 `[1,7,192,320]`。同输入从 KPU 与 Simulator 核对 430,080 个值；与 TopFace 同帧 USB 120.127 s 计 750/750 帧、6.24 FPS，采集失败 0。仿真集见原报告。 | 保留道路/非道路、忽略标签、边界与去填充坐标映射，先固定张量对照，再在同帧采集下运行两模型；真实道路质量独立验收。 |
| CameraXY V2 | USB `learned_map` 工程候选，`[1,3,96,160]` RGB uint8；USB 30+300+30 帧。该路径强制主 XY 为 null，未校准置信度，`fused_map` 科学门仍 HOLD。 | 学习分支锚点分数和解码一致；预处理按该模型契约复现，上下各 3 的填充须核对。错误布局、分布外和无先验时保持拒绝，不输出新的有效定位。 |
| Temporal V1 CNN/TCN + CPU 滤波 | CNN1 在新卡上分别 10/10、60/60，后者约 6.58 FPS，`position_valid=false`。TCN4 为离线编译/Simulator 对照。 | 相同 timestamps、帧间隔、候选分布和丢帧在 Linux 回放中产生一致状态，复现预测、更新、LOST、重捕获。仍以相机光心 XY 和配对实拍真值验收。 |
| 时序神经网络 | TCN8、TCN16、GRU4/8 固定步展开 KModel 与 ONNX、时间配置与窗口源码。本包包含 A03/A05 与 GRU 的离线基线；其他候选与云档案见原报告。 | 先验证单模型的实际输入/输出及固定序列，再接窗口、多帧丢帧与重置。保持训练定义的帧序和配置，不将 Simulator 合格升级为 Linux 实机或定位合格。 |
| CSI、QR、LANE、显示 | 原仓库包含旧流程、相机适配和解析器，USB短测成功。CSI2 在后续一轮找不到 Sensor 2，CSI/USB并发和 Linux 驱动待验收。 | 用真实 media graph 重建职责，独立 CSI 取帧、固定 QR fixture 与真实扫码、LANE 拒绝行为通过，再测 USB/CSI 启动顺序和并发。比较框、椭圆与预览帧身份，不读旧缓存当实时。 |
| 日志、采集、录像 | CanMV 两卡已分离大容量数据卷，旧卡第三分区修复并通过写读。 | Linux 配置可切换存储根，挂载/空间守卫、独立 run ID、读回、录像可播放、断开恢复和轮转验证，避免模型卷被输出填满。 |
| STM32 / 工作台 / 业务流程 | 原代码含端口协议、任务状态机、TOP/QR/LANE、调参和工作台。实际串口联动、机器人 mm 标定和机械闭环待相应验收。 | 先跑原纯逻辑/仿真回归，接 Linux 串口回环与 STM32 收包，验证帧边界、CRC、超时和重连。GPIO、电机与机构另开物理验收，默认无运动。 |

时序神经网络的数据集和额外研究候选可由原分支 `ops/temporal-neural-camera-v1` 的 `experiments/temporal_neural_camera_localization/FINAL_OFFLINE_DELIVERY.json` 与 `OFFLINE_STATUS.json` 找回。CPU 滤波四路线材料见包内 `K230_Temporal_Offline_Delivery_R1.md`。展现竞争路线差异时使用同测试端点、相机光心真值和原冻结门控。

每次 Linux 测试写一份唯一的 receipt，至少包括镜像 SHA/SDK/内核、runtime 文件及版本、模型/配置/源码 SHA、摄像头 ID 与格式、输入字节/张量描述、每阶段与端到端耗时、失败计数和清理状态。采集时间、解码和推理时间分开。先确认正确性，再比较 FPS；25 FPS 仍是目标，实测结果按当前硬件公布。
