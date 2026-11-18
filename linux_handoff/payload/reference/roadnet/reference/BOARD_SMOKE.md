# K230 双摄与同帧模型实测

2026-11-17，USB 同帧双模型工程短测完成 749/749 帧，清理重启后完成 187/187 帧，最新接线复测完成 186/186 帧。最新 CSI1、CSI2 探测均未检出传感器。双摄、真实 QR、30 分钟稳定性和道路质量仍待验收。[USB 工程实测](../evidence/board-20261117/usb-engineering-v1/REPORT.md) · [最新 USB 复测](../evidence/board-20261117/usb-engineering-v1/usb-chain30-c-REPORT.md) · [CSI 探测](../evidence/board-20261117/csi-availability-v2/RESULT.json)

TopFaceNet 单模型在当前 USB 接线上完成 30.003 秒 290/290 帧，调度约 9.67 FPS，采集失败和清理错误均为 0。该结果经 RAM 回传，并回读板端启动脚本及已部署 RoadNet 文件身份。[单模型基准](../evidence/board-20261117/native-ram-v1/tf-usb30-ram-a.json) · [回传和板端复核](../evidence/board-20261117/native-ram-v1/TOPFACE_USB_BASELINE_VERIFICATION.json)

在 OPS Pipeline 中先认领 COM3、K230-media 并确认 CanMV v1.8-36 空闲。测试依赖已验证安装的 RoadNet r3、TopFaceNet 和原任务码解析器。最新测得 SD 可用空间为 0，使用下文的 RAM 回传。评估脚本仍从现有 SD 目录读取模型和依赖。

先记录固件、USBDBG 连接、USB 设备枚举及实装 CSI 接口。试验默认 USB MJPEG 1280×720@30 和 CSI1 RGB565 640×480。若协商或 Sensor 初始化失败，保存异常及启动顺序，查阅介质资源或固件日志。CSI1 是当前上游入口的预期接口，原计划的相机配置预填 CSI2。按实物接线与 CanMV 探测结果选择 `--csi-id 1` 或 `--csi-id 2`，涉及 CSI 的各轮沿用同一已核实 ID。RAM 回传会记录本次请求的 ID。

| 顺序 | mode | 参数和原始记录 |
|---|---|---|
| 1 | uvc | 10 秒，OV2735 单路与原生协商 |
| 2 | csi | 10 秒，GC2093 单路 |
| 3 | dual | 10 秒，startup_order=uvc-first |
| 4 | dual | 10 秒，startup_order=csi-first |
| 5 | dual | 60 秒，交替取帧，双路保持开启 |
| 6 | roadnet 与 topface | 分别 30 秒，单模型时延基线 |
| 7 | dual-roadnet | 30 秒，双路与 RoadNet |
| 8 | both | 60 秒，双路与两个模型复用同一 RGB 帧 |
| 9 | qr | 每种预定码与姿态单独记录，已知有效码使用 `--expected-qr` |
| 10 | both | 1800 秒，USBDBG 保留端口日志和观测时段 |

每轮使用新的 `--tag`。短测通过且资源清理完成后开始下一轮。按原尺寸打印测试页，将 A4 横放并竖直安装，记录居中 80×80mm 码的距离、偏角、位置、照度和运动。已知有效码、无效 payload、未知码和无目标分别使用新标签。无效 payload 试次省略 `--expected-qr`，查看 `qr_invalid` 拒绝计数。[80×80mm A4 测试页](../evidence/board-20261117/native-ram-v1/QR_FIXTURE_A4_80MM.pdf)

在项目根运行。确认 GC2093 和 USB 双摄短测通过后，再执行 1800 秒的 `both` 完整调度测量。示例使用 CSI1，实际接 CSI2 时将相应参数改为 `--csi-id 2`。

```powershell
python -B -X utf8 reference/native_host.py measure --mode topface --duration 30 --tag topface30-new --ram-report
python -B -X utf8 reference/native_host.py measure --mode csi --duration 10 --tag csi10-new --ram-report --csi-id 1
python -B -X utf8 reference/native_host.py measure --mode dual --order uvc-first --duration 10 --tag dual-uvcf-new --ram-report --csi-id 1
python -B -X utf8 reference/native_host.py measure --mode qr --duration 10 --tag qr-known-new --expected-qr "156+123+516+231" --ram-report --csi-id 1
python -B -X utf8 reference/native_host.py measure --mode both --duration 1800 --tag both1800-new --ram-report --csi-id 1
```

`--ram-report` 将当前评估脚本发送到板端 RAM，经 USBDBG 终端回传带本次随机标识的 JSON。运行意图、终端原文和结果保存在主机的 `evidence/board-20261117/native-ram-v1/`。省略 `--ram-report` 时沿用板端 `/sdcard/rn1117a` 内的脚本和报告路径，并要求 SD 有可用空间。

报告分别记录设备协商值、实际帧尺寸、采集请求与成功帧数、共享帧数、调度周期 FPS、QR 灰度转换和解码时间。RoadNet 与 TopFaceNet 均记录整次推理、AI2D、KPU 及输出处理延迟。直方图按 1 ms 取上界，大于 1000 ms 的样本单独计数。内存报告保留运行前后值，并每约一秒采样 MMZ、系统页、系统堆峰值和最低 Python 空闲堆。采样异常随报告保留。采集失败按 snapshot 无帧统计。硬件输出帧序号缺失时，传感器至主机的实际丢帧率仍待设备日志确认。

正常结束为 `MEASURED`，间歇性捕获失败另行标记。验收结合帧率、错误和内存变化。相机始终无帧、初始化失败或清理异常须先诊断。发起测量后若观测超时、连接中断或无法确认唯一且匹配的回报，结果为 `UNKNOWN`。保留原意图和已有日志，检查板端是否仍在运行及当前 USBDBG 输出，避免用同一标签重跑。

主机替身验证了两种启动顺序、KPU 输入所有权、同帧对象复用、采样失败、分段延迟、QR 和清理计数。[主机测量流程验证](../evidence/cloud/full-v3/BOARD_SMOKE_HOST_CHECK.json)

## 标定和地面走廊复核

板端采集时记录 OV2735 的实际分辨率、镜头畸变、相对车体位置和姿态，以及每帧的时间戳。地面标定需覆盖实际作业区域，至少保留用于拟合的非共线地面点和独立测量的复核点。同步记录已观测地面掩码与帧间车体位姿。半车宽和半车长由实车测量，安全边距及前向制动距离由实际运行工况确定。

主机 API `ground_corridor.measure_corridor` 接收 ROAD_GRAY 候选掩码、道路置信度、已知地面掩码和图像到当前车体地面的单应性。图像数组使用同一去畸变像素坐标系，移除模型上下填充。前序帧须按实测位姿配准到当前车体坐标系，x 向右、y 向前，单位毫米。API 要求连续多帧均确认道路，按车体占用和前向制动距离收缩，输出原始稳定道路轮廓、各前视线的全部道路区间、离边界距离和走廊候选。窗口不足或帧间隔超限时，走廊为空。蓝区和交叉口方向由上游状态机及定位处理，当前走廊候选仅接纳灰色道路。

按独立复核点和实拍跑动视频评估毫米误差与走廊。使用实测障碍物、白底边缘、黄色禁区和低光条件，记录误放和拒绝通行。仿真回放提供复核样例及逐窗口误差记录。 [主机复核证据](../evidence/cloud/full-v3/GROUND_CORRIDOR_HOST_CHECK.json)
