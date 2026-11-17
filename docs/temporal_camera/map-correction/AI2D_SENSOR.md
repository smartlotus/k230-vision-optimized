# 640×360 相机输入与 AI2D 预处理

历史 CanMV v1.8-36-gae2f20d / nncase 2.11.0 的五个无相机 AI2D 样例保留了原输入和实测输出 SHA256。现用主机实现逐字节复现这五个 640×360 RGB NCHW 输入的缩小及填边，每个输出 46,080 字节。对需要缩小的 640×360 新图，训练、浮点回放和量化校准都经过同一入口。160×90 历史数据保持相同填边与训练输入。实测的适用范围是固定四倍下采样、half-pixel bilinear、上下各三行 114。 [逐字节验证](AI2D_FIXED_RATIO_PARITY.json) · [源码](../../../research/temporal_camera/camera_preprocess.py)

将历史 PIL 与实测 AI2D 输出分别送进本项目 a06 的 KModel Simulator。细线样例改变两条路线的最高分锚点，冻结留出图改变 A 的最高分锚点。这些数据证明部署像素差异有时会影响位置输出。历史 fixture 的输入由主机生成，当前 UVC 链路、KPU 模型原生推理均为 **NOT RUN**。 [KModel 字节回放](NATIVE_PIXELS_REPLAY.json)

用经审计的灰道场景和同一 CAD 重新渲染 4 条、100 帧的 640×360 Smoke。RGB 与掩码逐文件验证，轨迹按 50/25/25 帧完整拆分。新 Smoke 的 manifest 与较早 Pilot 独立。使用 a06 冻结的两条浮点 ONNX，对同一批高分辨率 RGB 比较 PIL 与对应 AI2D 的输入。最高分锚点一致率为 CNN1 78%、TCN4 83%。该 Smoke 只有一条留出测试轨迹。新数据训练 **NOT RUN**。 [QA](AI2D_SENSOR_SMOKE_QA.json) · [同图回放](AI2D_SENSOR_SMOKE_REPLAY.json) · [压缩包 SHA256](AI2D_SMOKE_DELIVERY_SHA256.json)

紧凑存储保留源渲染的 640×360 视场和语义掩码。RGB 以 AI2D 等效缩小后的 160×90 PNG 保存，读入模型时补三行上下边。100 张紧凑 RGB 的模型输入逐像素等于对应 640×360 图像直接经过 AI2D 等效预处理的结果。紧凑 Pilot 共 1,000 帧。配对校验覆盖原 a07 的 40 条轨迹、顺序、拆分、时间戳、光心 XY、变换、灯光和外观种子。新旧图像只进行同位姿配对分析，每轮训练只使用本轮分区内图像。 [100 帧逐字节对照](AI2D_COMPACT_SMOKE_PARITY.json) · [Pilot 配对检查](AI2D_PILOT_TRAJECTORY_PARITY.json) · [Pilot 数据 QA](AI2D_PILOT_QA.json)

紧凑 Pilot 数据包已读回核验 2,008 个文件。a08 五组 Kaggle 训练已完成，Pilot 仍处于 NEEDS_IMPROVEMENT。 [数据包 SHA256](AI2D_PILOT_DATA_DELIVERY_SHA256.json) · [a08 实测报告](../cloud/gray-map-pilot-a08/REPORT.md)
