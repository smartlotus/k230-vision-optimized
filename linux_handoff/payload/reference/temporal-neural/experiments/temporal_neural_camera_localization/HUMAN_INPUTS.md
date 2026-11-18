后续实物阶段统一使用这份清单。本线离线软件、数据和仿真报告见 OFFLINE_STATUS.json。当前模型精度仍为 HOLD，不能据此验收比赛定位。

已记录比赛条件：两个启停区随机选一个作为起点，另一个作为终点。下表的起步项目只补充实际可获取的区域身份与朝向信息。

请在具备设备条件时一次提供以下信息。

| 项目 | 需要提供或完成的内容 | 用途 |
| --- | --- | --- |
| K230 与相机 | 连接 K230 和 OV2735，提供 CanMV 固件版本、USB 探测 ID 及可使用的串口 | 核对运行时和真实输入来源 |
| 实际安装 | 相机固定后拍摄安装照片，测光心离地高度、倾斜角和可用分辨率。若已有相机内参、畸变标定结果，请附文件 | 按实际视野修订仿真与预处理 |
| 场地 | 官方图对应的实际场地图、尺寸、启停区与工位位置，以及场地照片 | 核对地标和重复灰道的真实外观 |
| 起步条件 | 记录是否有指定车头或相机朝向、是否知道抽到哪个蓝区，以及从放车到出发能否拍摄连续图像 | 判断两个蓝区的启动歧义如何消除 |
| 光心 XY 真值 | 直接测相机光心在 FIELD_MAP 的 X/Y，记录测量方法、误差和同步事件。请勿用底盘中心坐标代填 | 建立独立真实精度验证集 |
| 实拍序列 | 两个蓝区分别出发，拍摄经过工位、中央十字、重复灰道、转弯和停车的连续序列。保留每帧或视频时间戳，并记录光照 | 验证时序收益和真实域差异 |
| 运动与调度 | 若已有安装偏移、相机光心速度上界、RoadNet/TopFaceNet 的调度频率和周期预算，请提供记录 | 校验采样跨度、内存与共用 KPU 周期 |

一次性接入后的执行顺序如下。

1. 核对设备身份并固定相机，完成标定或校验已有标定。
2. 在独立 SD 目录运行交付包中的 neural_board_probe，收集报告和同名 .frames.jsonl。程序不会修改 main.py 或 boot.py。
3. 按独立趟次采集两种起点的实拍和光心 XY 测量数据。每个蓝区都留出独立的验证和测试趟次，并在查看模型结果前分组。请保留原始时间戳和测量同步记录。
4. 用 `replay_capture.py --source ov2735` 回放。入口在验证趟次拟合阈值，在独立测试趟次按起点报告定位误差、接收并合格的帧数，以及记录到长间隔后的候选恢复。原生验收仍单独执行。
5. 原生测量 USB→RGB→AI2D→窗口→KPU→候选的完整周期、内存、丢帧和多任务调度，再进行有界长时验收。

`replay_capture.py` 读取 JSONL，图像路径相对清单目录。每行提供 `sample_id`、`sequence_id`、`recording_id`、`split`（`val` 或 `test`）、`start_zone_id`（1 或 2）、`timestamp_s` 和 `image_path`。`recording_id` 标记原始录像。若从一份录像切出多段，各段继续使用同一个 `recording_id` 和同一个 split。每条趟次的帧时间戳严格递增，并独立维护历史。`start_zone_id` 用于评估初始判断，推理输入是 RGB。可添加 `image_sha256` 核对图像。采集记录应保留每趟原始文件与分组，程序可检查填写的原始录像 ID 及字节完全相同的图像是否跨组。

有同步的直接实测真值时，提供 `camera_xy_target` 和 `measurement`。前者只包含相机光心 X/Y，后者记录测量基准、方法、记录编号和不确定度。没有真值的连续帧仍可用于回放，误差只统计有真值且实际推理的帧。

```json
{"sample_id":"capture-000001","sequence_id":"zone1-val-run01","recording_id":"ov2735-run01","split":"val","start_zone_id":1,"timestamp_s":0.0,"image_path":"rgb/000001.png","camera_xy_target":{"camera_x_mm":1030.0,"camera_y_mm":1010.0},"measurement":{"reference":"DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY","method":"填写实际测量及同步方法","record_id":"survey-000001","uncertainty_mm":10.0}}
```

示例值仅演示格式。解压新版回放补充包后，可在补充包根目录运行以下命令。`--manifest` 指向实际采集清单，`--out` 使用一个新文件名。补充包带有该命令所需的模型和源码，可独立回放。

```powershell
python source/experiments/temporal_neural_camera_localization/replay_capture.py --source ov2735 --manifest capture/manifest.jsonl --onnx models/geometry-a03-models/tcn8/camera_xy.onnx --profile source/experiments/temporal_neural_camera_localization/GEOMETRY_A03_RUNTIME_PROFILE.json --metrics models/geometry-a03-models/tcn8/METRICS.json --out capture-replay.jsonl
```

同名 `.report.json` 包含按趟次、验证/测试和起点划分的样本数、定位误差及恢复记录。验证集中至少需 25 个通过阈值的带真值帧，且其中至少 95% 的候选误差在 200 mm 内，才会提出阈值。阈值结果单独记录，板端和交付配置保持各自的验收流程。

要比较时序收益，在配对回放包根目录用同一个实拍清单运行下列命令。`PAIRED_TEST.json` 汇报两套训练权重在相同测试帧上的误差、逐帧胜负和大跳变次数，并分别统计全部帧、已填满时序窗口的帧以及两个起点。一次运行使用一个新的输出目录。

```powershell
python source/experiments/temporal_neural_camera_localization/compare_capture.py --source ov2735 --manifest capture/manifest.jsonl --models models/geometry-a03-models --profile source/experiments/temporal_neural_camera_localization/GEOMETRY_A03_RUNTIME_PROFILE.json --out paired-capture
```

如果采集时保存了送入 AI2D 的 640×360 RGB 中间帧，可用无损 PNG 存储这些像素，并给上述回放命令加 `--preprocessing ai2d-640`。该选项要求每张图达到这个尺寸，并在主机上复现已归档 AI2D 输出的缩放和填边。默认 `pil` 对应训练预处理。两种模式请使用独立输出目录，便于对照像素处理对结果的影响。原始相机 JPEG 到 640×360 RGB 的转换仍需在实际设备上验证。

板端试验结束后，下载同一次运行的报告及其 `.frames.jsonl`。从本源码目录运行下面的命令核对试验包模型、USB 身份、推理计数和每帧窗口内容，并整理延迟、丢帧和内存记录。参数路径可指向各自的磁盘目录。

```powershell
python verify_board_capture.py --source k230 --archive .runtime/K230_Temporal_Neural_Geometry_A03_Trial_R2.zip --receipt GEOMETRY_A03_BOARD_TRIAL_R2.json --report native.json --trace native.json.frames.jsonl --usb-id '实测的 USB ID' --out native-checked.json
```

若同一次运行有直接实测 XY，可加 `--survey board-survey.jsonl`。每行使用上面相同的 `camera_xy_target` 和 `measurement`，另外填入板端日志的 `capture_index` 和 `captured_since_start_us`。请根据采集同步记录对应到具体帧。工具只计入明确匹配的帧，报告其样本数、候选误差和最邻近秩 P95，并单独保留实机调度与像素验收状态。还没有同步测量时可以先核对板端性能报告。
