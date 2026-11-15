# K230 同输入数值配对

本协议是待接板执行路径；主机fixture和Simulator资格不能替代实际KPU证据。正式权重完成并连接开发板后再运行。原有默认benchmark入口保持 `paired_samples=0`。

1. 在全新的独立trial目录放入完整board bundle。读取实际runtime、模型SHA及output contract，再调用 `run(root="/sdcard/topface_trial_pair_001", frames=15, display=False, paired_samples=3)`。`sys.path`应包含该新目录。采样模式最多3张，显式attempt预算5–30帧；遇已存在pair目录或结果会拒绝覆盖。
2. 每张source先复制为自有数组，再由同一source生成AI2D输入。保存实际prepared后运行同一次KPU，随后在下一次运行前保存raw与实际MicroPython decoded。每个 `pair-01..03` 包含下表文件及sample.json；根索引为PAIR_CAPTURE_RESULT.json。

| 文件 | 文件契约 | 字节数 |
|---|---|---:|
| source.rgb888p | uint8 RGB NCHW `[1,3,360,640]` | 691,200 |
| prepared.rgb888p | 实际AI2D uint8 RGB NCHW `[1,3,192,320]` | 184,320 |
| kpu-raw.f32le | float32 little-endian `[1,1275,50]`，C order | 255,000 |
| decoded.json | 原生decoder完整输出 | 实际记录 |

3. 通过既有有界板端文件传输收取这些小文件，核对完整SHA，不用JPEG代替数值输入。source的原生array形状/dtype、CSI2、capture序号、ticks、runtime和model/contract SHA均记录；硬件frame ID目前无已验证API，明确记UNAVAILABLE。
4. 主机用 `research/topface/evaluate_board_pairs.py --samples <捕获目录> --model-dir <已校验模型目录> --output <新报告.json>` 评估。ORT与Simulator使用同一份实际prepared RGB；source另用于核查AI2D与主机letterbox的差异。报告raw逐通道误差、解码差异，以及真实捕获的native decoded与同raw主机decode的bbox/center/ellipse/flags/capacity对照。

数值allclose容差是显式诊断参数，不自动构成全部质量或runtime兼容验收。检测/颜色/真实几何准确率另需真值。采样模式包括磁盘I/O，FPS字段为None、timing为空；完整链路性能必须另跑默认benchmark。

目前资格：主机序列化/文件身份/旧结果保留/原生decoded字段比较fixture通过；实际Pilot KModel的一张审计train图验证新prepared直入Simulator与原路径逐字节一致。ULAb/camera/DMA/实际KPU执行均为NOT_RUN。
