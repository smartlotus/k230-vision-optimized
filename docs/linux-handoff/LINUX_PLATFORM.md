# Linux 基线和设备接入

目标是立创庐山派标准版 K230 1G，CanMV 标识 `LCKFB-LSPI-K230-1G-CanMV`。Linux SDK 配置为 `k230_canmv_lckfb_defconfig`。按此板号和 defconfig 核对镜像、设备树和启动日志。Lite K230D、01Studio 和通用 CanMV 的镜像分别对应其他板型。

| 首选 | 备选 |
| --- | --- |
| LCKFB Debian, SDK v1.3，构建 `ac0ac1e1`，nncase 2.11.0。作为第一套可复现环境，方便用 Python 3.13 做算法对照。 | 同 defconfig 的官方 Buildroot 镜像，或为解决已复现的驱动缺陷而选定的官方日构建。每个候选都要记录完整身份和回退理由。 |

首选镜像 `CanMV-K230_LCKFB_debian_v1.3_nncase_v2.11.0_ac0ac1e1.img.gz`，官方 MD5 `9c7e9d34a57d421ff119b8f945890d43`，HTTP 元数据长度 888,193,982 字节，SDK 完整提交 `ac0ac1e1888f15ce2fe7713e6a587dc1268c57e2`。版本属于嘉楠 Linux SDK，与项目的 CanMV v1.8 编号独立。仅获取了官方元数据，尚未下载、烧录或实测该镜像。[Debian 镜像](https://download.kendryte.com/k230/release/linux_sdk_images/v1.3/k230_canmv_lckfb_defconfig/CanMV-K230_LCKFB_debian_v1.3_nncase_v2.11.0_ac0ac1e1.img.gz) · [SDK 固定提交](https://github.com/kendryte/k230_linux_sdk/tree/ac0ac1e1888f15ce2fe7713e6a587dc1268c57e2) · [官方下载中心](https://download.kendryte.com/k230/release/linux_sdk_images/daily_build/)

如果稳定镜像出现可复现的阻塞，可选择一个明确版本做 A/B。官方目录中 2026-11-17 日构建为 SDK v1.4 / `948dcad4` / nncase 2.11.0，必须保存对应 MD5 和下载时 SHA-256。一轮只切换一个变量。板端先跑官方 KPU 静态样例。官方 Debian 指南给出 Python 3.13 的 riscv64 `nncaseruntime_k230` 2.11.0 wheel，也可用同 SDK sysroot 的 C++ runtime。实际 runtime 以板端加载路径、文件哈希与样例运行证据为准。[官方 Debian 指南](https://www.kendryte.com/k230_linux/en/main/app_develop_guide/user_develop/debian_ubuntu.html) · [nncase 编译使用说明](https://github.com/kendryte/nncase/blob/master/docs/USAGE_v2.md)

## 第一次上板

1. 用户提供可覆盖的独立 TF 卡后，核对型号、容量、分区、读卡器以及镜像校验值，按官方流程烧录并完整读回。两张已有 CanMV 卡保留做 A/B 对照。Linux 第一次启动可能扩展文件系统，等启动日志稳定。[官方烧录和串口指南](https://www.kendryte.com/k230_linux/zh/main/userguide/how_to_flash.html)
2. 记录串口启动日志、`uname -a`、`/etc/os-release`、设备树 model/compatible、`lsblk -f`、`findmnt`、可用空间、可用内存和 KPU 设备/驱动。开发电脑可用 Linux 工作站或在支持的情况下使用 WSL2/Docker 构建，最终 ELF 和 wheel 必须是板端 riscv64 的 ABI。调试串口和 STM32 业务串口分别核对；按官方接线说明确认电平和引脚。
3. 插 USB 摄像头，用 `lsusb`、`v4l2-ctl --list-devices` 和 `--list-formats-ext` 核对来源，优先使用可验证的持久设备标识，记录 sysfs、VID/PID、格式和 negotiated fps。先以 V4L2 取 1 帧再连续采集短样本，比较 `1280×720 MJPEG @ 30` 的协商和实际取帧速率。Linux 节点编号动态发现。官方 USB Host 采集文档与 UVC gadget 输出文档是两个方向。[USB 摄像头采集指南](https://www.kendryte.com/k230_linux/en/main/app_develop_guide/user_develop/uvc.html)
4. 路径由配置指定。将模型/配置放在应用根，将采集、报告、录像放在独立持久目录。可用一个实际挂载的测试数据卷，例如经确认的 `/mnt/k230-data`。启动前检查设备、挂载源和自由空间，不能仅因目录存在就开始录制。写入采用同卷临时文件和关闭后的读回校验。使用磁盘预算与日志轮转，保留失败证据。

## Linux 适配边界

| CanMV 能力 | Linux 迁移面 |
| --- | --- |
| `media.uvc.UVC` | V4L2 Host / OpenCV 或绑定的 C++ 采集，记录原 MJPEG/YUY2、stride、时钟和队列行为 |
| `media.sensor`、VICAP | SDK 的 CSI、media graph、ISP 和板级传感器配置；先证明当前镜头/接线对应哪个节点 |
| `nncase_runtime`、`ai2d`、`ulab` | Linux nncase、SDK C++ 或 Debian wheel，验证 AI2D 与实际张量描述。CPU 预处理只在像素/输出匹配后用作替代 |
| `Display.VIRT/to_ide`、LCD/HDMI | 先独立 JPEG/PNG 预览能读回，再按 SDK 支持接 DRM/KMS/HDMI、屏或 RTSP；相比只需视觉等效与延迟证据 |
| `os.exitpoint`、`gc.sys_mmz` | SIGINT/SIGTERM、有界退出、V4L2 流和缓冲回收、RSS/PSS 及驱动可观测的 DMA/CMA 指标 |
| `machine.UART/Pin/TOUCH` | termios、libgpiod 和 Linux 输入设备，先用协议 fixture 与模拟端验证，最后人工批准物理输出 |

