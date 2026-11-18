# 交给下一位 AI 的任务

你负责推进 **立创庐山派 K230 1G 原生 Linux 迁移线**。目标是在同一块板上复现当前 CanMV 已验证的工程链路和业务输出，然后继续补齐原来未通过的挂载、相机、通信、定位和稳定性验收。把迁移作为独立工程维护。以可复查的模型、张量、画面和报告为完成依据。

我会给你 `K230_Linux_Handoff_20261118.zip` 或公开仓库 `smartlotus/k230-vision-optimized` 的 `ops/k230-linux-20261118` 分支。先读本文件、同目录 `README.md`、`LINUX_PLATFORM.md`、`PARITY_MATRIX.md` 和 `linux_handoff/PAYLOAD_MANIFEST.json`。ZIP 解压后运行 `python linux_handoff/verify_payload.py --package`。仓库检出运行无参数的核验，再读 AGENTS.md 和 .ops 协议，检查任务所有者、当前分支和改动。Linux 代码放在自己的目录，保留已有 CanMV 基线和模型版本。任一可验收模块完成、暂停或交接时记录证据并精确提交到指定分支。按当前授权同步到已核对的远端后，验证远端 SHA。检查公开推送范围内的文件与提交历史。不要自动合并主线。

## 工作顺序

1. **能离线做的先做。** 从清单把各模型的 KModel、ONNX、编译 ABI、颜色与量化入口、类别/锚点、输入尺寸、输出张量和后处理绑定在一起。优先复用 `core`、`vision`、解析器、滤波器、时序窗口和测试夹具。给帧、推理和显示设清晰的适配接口，分离输出目录。用已有 fixture 复跑纯算法回归。先实现可读本地文件的 Linux 测试入口，便于离线验证和板端同输入测试。归档里直接 import CanMV 的代码是行为基准。
2. **先确定平台。** 目标板是 `LCKFB-LSPI-K230-1G-CanMV`，使用官方 `k230_canmv_lckfb_defconfig` 的 Debian v1.3 / nncase 2.11.0 为第一候选。镜像链接、官方校验值和固定 SDK 提交在 LINUX_PLATFORM.md。先准备主机工具和静态测试。实际烧录需要我提供单独可覆盖的 TF 卡。当前 32 GB 和 64 GB 的 CanMV 卡供对照与回退。板子、摄像头和连接有共享任务时先核对所有者，请我集中处理必要的物理接入。首次上板存完整启动日志、设备树、runtime 版本和真实磁盘挂载。
3. **先证明 KPU 再接相机。** 先跑官方 Linux KPU demo，验证实际硬件后端及依赖。用封存 KModel，做 TopFace 全 114 输入、历史平面 RGB 重放及 RoadNet 固定输入对照。核对输入/输出 dtype、形状、字节数、通道顺序和输出释义，保存准备后的输入与原始输出的 SHA-256。TopFace 样本在 `linux_handoff/payload/evidence/canmv/constant-parity-01` 和 `historical-replay-01`。RoadNet 的输入/输出二进制在 `linux_handoff/payload/evidence/roadnet-fixed/`，源与契约见 `linux_handoff/payload/reference/roadnet/`。如果同模型同输入出现差异，先记录逐元素误差、类别/坐标变化，再查 dtype、布局、预处理是否重复、runtime 与驱动。预先写下容差及依据，不因失败临时放宽。确证 Linux runtime 不兼容后，才用原 ONNX、量化配置与校准集重新编译成新身份，保留前版和差异报告。
4. **先 USB 单模型，再同帧双模型。** 摄像头是用户识别的微雪 OV2735 A，CanMV 实测 `openaicam#openaicam`、1280×720 MJPEG 30。Linux 由 V4L2 实际枚举设备与协商模式，获取原帧、序号、时间戳、stride、丢帧和颜色。先验证缩放 640×360 的 RGB，再验证模型所需的 RGB uint8 NCHW 与实测填充。分别跑 RoadNet、TopFace，然后两者同一帧输入，关联画面与结果；保存带框/椭圆预览和原图。优先短测 30 秒和 120 秒，记录实际分母、p50/p95/吞吐、失败、退出及重启。基线 120 秒双模型约 6.24 FPS，25 FPS 是优化目标。显示、解码、拷贝和 KPU 分开定位，先复现结果再优化。
5. **定位分支与可靠性。** CameraXY V2 的 USB learned-map 工程候选、Temporal V1 的 CNN1+CPU 滤波、独立 TCN/GRU 的三类验收分别运行。保留相机光心 XY 及训练定义的坐标、帧历史、采样间隔和重置规则。先复验固定样本，再验 RUNNING/LOST/恢复、黑帧、丢帧和异常间隔。无真值时保留 `null / position_valid=false` 等原边界。有了配对的真实相机光心测量再做误差和置信度接受，别把工程短测当成定位精度。
6. **CSI、扫码、工作台与通信。** 按实际设备树和 media graph，确定传感器、节点和源职责。依次验证 CSI 单源、USB+CSI 两种启动顺序、QR 正反样本及解析协议、LANE/道路拒绝、HDMI 或可用预览。将 CanMV UART/TOUCH 等接口换为 Linux 适配；先 fixture/仿真和串口回环，再 STM32 只读通信，默认禁用电机及执行器输出。不得沿用另一相机的几何标定。显示和工作台需与原协议行为、阈值、边界和草稿/生效配置相符。
7. **系统收尾。** 代码/模型与大输出分别配置根；仅在已验证的持久卷写日志、图片和录像。做缺卷、满盘、格式不匹配、摄像头拔插、SIGINT/SIGTERM、重启和资源回收。达成短测后再执行有界 30 分钟运行，分开报告瞬时错误、恢复次数与漏帧。生成可复跑的安装与回退步骤、启动命令、权限/依赖清单、原始验收报告和前后差异。写明真实可用功能、尚未实测的门与需要我处理的物理事项。

## 完成标准

先交 Linux 启动及 KPU 固定输入报告，再交 USB 同帧双模型完整链路。任务最终还要逐项处理 PARITY_MATRIX.md，含采集、预览、日志和录像、时序定位、CSI、QR、LANE、串口及工作台原行为。对原来就 HOLD 或 NOT_RUN 的项目，保留原因并按原门控验证，不以 Linux 启动成功替代。

每条报告必须有运行 ID、触发命令、SDK/镜像/runtime 版本、模型/配置/代码 SHA、数据来源、实际帧数与失败数、延迟/内存、保存文件和读回结果。先自行推进所有不依赖实物的任务，把必须由我完成的物理动作合并成一次请求。实际烧录、发布、主线集成和机械动作按各自授权执行。

