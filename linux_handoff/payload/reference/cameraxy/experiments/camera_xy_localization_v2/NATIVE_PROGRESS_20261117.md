# CameraXY V2：项目地图与无摄像头实机验证

2026-11-17，北京时间。当前工作树 `k230-camera-xy-v2`，分支 `ops/camera-xy-localization-v2`。操作者已连接 K230，但明确尚未连接摄像头；本轮完成有界、无摄像头的原生输入和解码验证。定位、模型部署和完整机器人验收仍未完成。

## 各条线的关系

| 工作线 | 负责的问题 | 本轮范围 |
|---|---|---|
| CameraXY V2，本目录 | 从 USB 地面图像估计相机光心在 FIELD_MAP 的 X/Y；允许五个候选 | 本轮实际修改与实机验证 |
| Competition temporal，独立工作树 `k230-competition-temporal` | 用分组图像序列研究重复道路的定位歧义 | 发现已有活动 owner，保留其任务与模型，不接手或重跑 |
| RoadNet，独立工作树 `k230-roadnet-sim` | 场地语义/道路边界；本线复用其冻结地图、场景与 ONNX | 仿真历史资产依赖，本轮没有运行或替代其 KPU 模型 |
| TopFaceNet，独立模型/工作区 | 物料顶面检测，本线复用三段式物料 CAD 作为动态遮挡 | 本轮没有修改其模型、训练或消费者 |
| transport/live-tuning/workbench | 识别、调参、二维码、结果通信及控制衔接 | 保留既有实现、启动代码、阈值与生产配置 |

上表的相邻项目是从执行包、当前源码及 Git/OPS 登记核对的职责关系；其完整训练与实机状态未在本轮重新验收。

本线地图是 2400×2400 mm，以几何中心为原点，X 向图纸右、Y 向上。训练 target 只有 `camera_x_mm`、`camera_y_mm`，表示相机光心的地面铅垂投影。yaw、高度、内参、外参只作 metadata；相机到机器人的安装偏移属于控制层。

## 已完成的仿真与当前限制

- 正式集 10,000 位置、21,000 帧；位置 train/val/test 为 8,000/1,000/1,000。
- 已有冻结 Kaggle 保存版本训练 18 轮、4,500 次更新。本轮没有启动训练、云资源或重放该运行。
- 验证集选出的融合分支：空间留出 Top-5@50mm 为 1,841/2,000；配对外观为 751/1,000。
- 空间 Top-1 中位/P95 误差为 1059.7/1443.4 mm，单点接受覆盖率 0%。总体单点门仍 HOLD。
- 本轮原生解码只验证 `learned_map` 的冻结分数；不将其称为选中 `fused_map` 的板端推理。
- 当前导出入口要求相同 checkpoint 的主机/实测验收，且对地图头要求选中 learned 分支。本轮未绕过这些条件，未编译或运行 CameraXY KModel。

历史原始指标仍见 [FULL_REPORT](evidence/FULL_REPORT.md) 和 [受控扰动报告](evidence/G5-nuisance/SIMULATION_REPORT.md)。

## 本轮实际执行

COM3 上的 LCKFB-LSPI-K230-1G-CanMV，固件 v1.8-36，原生 nncase runtime 2.11.0。接手时查询脚本空闲；UVC 探测为 False，与操作者未接摄像头相符。

在独立 SD 目录按 a、b、c 三轮留存，不覆盖旧报告。使用冻结 G5 checkpoint `c16e0ddf9f458fb7dfbcda5902f60c09b4a9e0426bb55a7e9b3ebea3d486d989` 产生五组真实留出 logits。AI2D 运行四种确定性图样和一张冻结合成留出图，每组五次。每组输入字节、源码、输出与报告均绑定 SHA256 并下载核验。

### 已修复：原生同分排序

主机使用稳定排序；该 MicroPython 实测 `sorted(range(20), key=lambda i: 0)` 会改变原始索引顺序。对称地图恰好产生相同分数，原版 `decode_map` 因此只有 1/5 样本候选顺序与主机完全一致。

`board_probe.py` 现用 `(-score, index)` 构造显式全序，不修改主机排序、权重、候选间隔或校准阈值。c 轮结果：

- 候选坐标与顺序 **5/5 一致**。
- 主坐标 **5/5 保持 null/unlocalized**。
- 置信度最大绝对差 1.804×10⁻⁷，小于预先固定的 1×10⁻⁶ 比较容差。
- 截断张量、非有限张量 **2/2 拒绝**。
- 原生解码样本耗时 31.729–38.288 ms；这五次不是完整周期或持续帧率测量。

### 未通过：AI2D 与训练 PIL 缩图一致性

训练 `model.pixels` 使用 PIL BILINEAR 把 RGB 缩至 160×90，再上下各填充 3 行/114。现有板端 AI2D 是 640×360 → 160×96、tf_bilinear/half_pixel 加同样 padding。输出 shape/布局和五组 padding 均通过，但缩图像素并不相等。

| 输入 | 比较字节数 | 不同字节 | 最大绝对差 | 平均绝对差 |
|---|---:|---:|---:|---:|
| 常量 RGB | 46,080 | 0 | 0 | 0 |
| RGB 渐变 | 46,080 | 0 | 0 | 0 |
| 单像素细线 | 46,080 | 43,200 | 73 | 59.9473 |
| 棋盘格 | 46,080 | 6 | 1 | 0.000130 |
| 一张合成留出图 | 46,080 | 18,276 | 47 | 0.63457 |

25 次 AI2D 调用为 460–474 μs。该耗时不含原图读取、RGB 转换、模型、解码和相机采集。

将同一张留出图的原生 AI2D 输出在 CPU 上送入冻结网络，logits 最大变化为 1.38336，第五候选也发生变化。说明像素差异会影响这张图的候选输出；不能由此推断整套误差或真实域效果。

归档的合成留出图原始尺寸是 320×180，先在主机构造固定 640×360 输入。本轮只测后续 AI2D 段，未验证 UVC 解码/缩图及实际光学成像。 全部正式留出的配对主机回放见 [AI2D 预处理复核](AI2D_HOLDOUT.md)。预处理状态明确为 **MISMATCH_MEASURED**，尚未修复或授予对齐 PASS。

### 保留的测试脚本失败

a 轮新增测试工具最初把 `heldout.rgb` 同时用作输入和输出，导致原始输入被结果覆盖；b 轮复用时 SHA 校验正确拒绝，未执行后续五组解码。该错误已修为独立 `.ai2d.rgb` 输出。c 轮从 a 轮未改变的压缩输入解压到内存，按原始 691,200 字节与 SHA 验证后执行。

原始 a/b 报告保留；未将 b 轮失败计入有效解码分母。另已收紧独立目录/固定 basename、绑定实际导入模块路径，以及 CPU 回放 provider 的模型/清单/校准身份。独立只读复核已核对修复和最终所有输出 SHA。

现有 `/sdcard/main.py` 前后 SHA 相同，无清理异常。最终脚本空闲、MCP 连接保留。没有相机采集、KPU 推理、UART 或机械动作。

## 证据与接续

- [最终核验](evidence/G7-native-input-20261117/verification-final.json)：解码 PASS，预处理差异仍保留。
- [原始排序失败](evidence/G7-native-input-20261117/verification-before-fix.json)、[中间输入拒绝](evidence/G7-native-input-20261117/after-fix/native.json)。
- [最终原生报告](evidence/G7-native-input-20261117/final/native.json)、五份原生张量、PIL golden、spec 和固定输入一并保存。

可在实验目录用原有 PyTorch 环境重新核验下载工件：

```powershell
python tensor_bench.py verify --inputs evidence/G7-native-input-20261117/inputs-after --native evidence/G7-native-input-20261117/final --provider .runtime/kaggle-map-full-a01/provider --out .runtime/native-verification-new.json
```

同一报告路径不可覆盖。准备新实验必须用新的输出目录；原生根为独立 `/sdcard/xy...`，不得运行到已有报告目录。

后续独立门：先确定与训练一致的可执行预处理，或冻结新的输入版本并重新评估/校准；摄像头接上后验证原始 UVC 模式、转换及整链。真实定位仍需 [REAL_CAPTURE](REAL_CAPTURE.md) 的直接光心地面量测和不确定度；当前缺摄像头/实测集，G6 与完整 G7 继续未完成。

本实验沿用原执行包 `EXECUTION_AI_PROMPT.md` 第39行的禁止 push 约束，只保存本地 Git 检查点，未向远端推送。
