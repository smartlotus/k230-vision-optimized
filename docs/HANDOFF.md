# K230 项目接续交接

更新：2026-11-16，北京时间；板端实查12:12–12:13。先读[最新进展](STATUS.md)、[经验总账](LESSONS_LEARNED.md)，再读 .ops/PROJECT.md、.ops/PROTOCOL.md 和 OPS 最新交接。
12:21:31复查仍running=true，usb-live01.json未生成；旧CSI报告production-session01.json于12:18:14确认存在，6026字节。本次复查没有停止或重启程序。

## 1. 接手目标与授权

用户已经授权正式KModel接入原TOP代码、板端试验、FPS优化、USB试验，以及项目记录/交接和当前私有仓库checkpoint备份。用户定性评价“识别准确率不错”不换算为准确率百分比。
当前最新硬件选择是USB摄像头实时模型预览。此交接只记录状态，不授权自动重放已结束run、购买资源、改启动文件、写生产profile或开启运动。

第一步是保留当前USB会话，读取其最终结果；不是重新训练、重新上传或直接执行启动脚本。

## 2. 权威目录与版本

| 用途 | 位置 |
|---|---|
| 用户工作区聚合入口 | C:/Users/28389/Desktop/k230-project |
| 当前实现工作树 | C:/Users/28389/Desktop/k230-topface-production |
| 聚合目录链接 | k230-project/worktrees/ops-topface-production |
| 分支 / 私有远端 | ops/topface-production-fps-20261116 / smartlotus/k230-vision-optimized |
| 原训练工作树 | C:/Users/28389/Desktop/k230-vision-optimized（ops/topfacenet-sim2real-20261114） |
| 历史相机修复工作树 | C:/Users/28389/Desktop/k230-camera-diagnosis |
| 当前板端包 | /sdcard/tfp1116 |
| 旧试验目录 | /sdcard/topface_formal_a01_20261116、/sdcard/cdiag1116，保留 |

实现提交62c62edd942cd8adff1c420f4828d83a6dcabc91；本次刷新前metadata HEAD为c09562e0922bb0f6d6481c714d54188311d4b492。
70个框架源码当前本地SHA与[最终清单](validation/topface-production-fps-20261116/FINAL_SOURCE_MANIFEST.json)一致；USB源码与保存的最终板端回读一致。本次未再次传输所有板端代码或模型，既有原生部署收据继续有效。

## 3. 接手先核对OPS和实际设备

在当前实现工作树运行：

```powershell
git status --short --branch
git rev-parse HEAD
python .ops/tools/ops.py status
python .ops/tools/ops.py verify --task status-handoff-20261116
```

根据实际状态登记并认领自己的任务，确认COM3唯一控制端。不要凭旧handoff宣称无人认领，不按锁龄抢占。
本次旧实现任务已完成交接并释放；本次文档任务在checkpoint后也会释放，最终状态以status为准。释放记录不表示板端预览已经停止。

只读设备顺序：

1. canmv_board_info：身份缓存仅作参考。
2. canmv_script_running：当前执行状态。
3. canmv_get_latest_frame(fresh=true,waitMs=4000)：核对新帧序号、年龄和画面USB标记。
4. canmv_stat_file(path="/sdcard/tfp1116/usb-live01.json")：核对最终收据是否存在。
5. 必要时canmv_terminal_output(clear=false)，保留旧输出。

本次看到板LCKFB-LSPI-K230-1G-CanMV、COM3、v1.8-36，完整固件ae2f20da0467a8e70659f032b240aae50272c49a；running=true，fresh帧29262/age0ms，后续保存帧30596/age2ms。最终usb-live01.json不存在。当前图暗/模糊、EMPTY，不作识别质量结论。

只读检查不需要停止脚本。程序运行时不启动另一条CSI/UVC程序，不重连/重启扩展来“取新画面”。本次现场记录见[机器快照](validation/topface-production-fps-20261116/status-refresh-20261116T041232Z.json)。

## 4. 当前模型、功能与限制

- KModel SHA：f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88。
- 原训练100轮/23,400更新，验证集选第92轮；不重新训练、不调低score0.25、不改NMS0.5。
- 原生模型输入uint8 RGB NCHW [1,3,192,320]；内部RGB→BGR与归一化只做一次。
- CSI2 TOP/RING/COLOR，CSI1 QR/LANE，CSI0隔离；模型TOP用CSI2通道2 RGBP888，保留通道0 RGB565。
- USB设备字符串openaicam#openaicam；用户提供硬件身份为微雪OV2735 2MP A。1280x720MJPEG30FPS，等比例转640x360 RGB888，再转独立CHW。USB不是CSI2，不使用CSI标定。
- 接入原CameraManager / VisionAdapter，模型资源由TopFaceBackend持有；TuningSession支持显式backend注入，旧工作台默认入口尚未统一切换。
- 输出六色、bbox、confidence、像素中心、椭圆及有效性。原协议用UNAVAILABLE承载OVER_CAPACITY/UNCERTAIN，原容量状态保留；无标定不输出有效机器人mm位姿。
- 同帧预览、向量背景筛选、独立packed RVV RGB565、gc_interval4/低堆触发回收已部署。相机/模型输入不被显示端原地覆盖。
- main.py/boot.py、旧SD目录和active/draft不写；UART/运动关闭。

实现入口：[TOP服务](../k230_transport_mvp/k230/topface_service.py)、[KPU与预览适配](../k230_transport_mvp/k230/topface_adapter.py)、[USB试验](../research/topface/board/usb_topface_trial.py)。

## 5. 下一步：收取USB长跑结果

本次USB预览预算18,000次循环，最终结果只在运行退出后落盘。预览frameId不是推理计数，文件暂缺不等于失败或执行停滞。

当最终文件出现时：

- 用canmv_download_file_to_host从 /sdcard/tfp1116/usb-live01.json 保存到新的本地证据文件，不覆盖旧600帧或其他收据。
- 读取status、error、requested/attempted/successful/capture_failures、wall_seconds/fps、cleanup_errors、device/modes、模型SHA和peak。
- 完整完成、含采集损失、人为IDEinterrupt、运行失败分别保留；仅报告字段支持的结论。模型精度和标定仍独立。
- 文件读取大小/哈希及当前固件、源代码版本、相机模式一起记录。记录实际读取路径；只有本机存在的资料明确标本机。
- 如果脚本停止且结果缺失，保留终端和设备状态，记UNKNOWN并排查，不能重跑覆盖。

随后更新 STATUS.md、相关原始记录索引、经验总账和OPS checkpoint。当前任务不要求等长跑结束再完成文档交接。

## 6. 只有确需新运行时才使用的启动方式

先确认原脚本已停止、硬件来源符合用户当前选择、独占COM3、源码SHA和当前模型一致。新报告名必须在开始前确认不存在。

**重要已知问题：不要直接重复执行 run_topface_service.py。** 该文件固定写production-session01.json，而这个旧文件已存在；现有“保护旧报告”检查在运行结束时发生，可能浪费整个运行预算后才报错。本次只写文档，未改代码；下列显式新文件名避开该问题。

CanMV MicroPython示例（USB新attempt；不是现在运行的指令）：

```python
import sys, os, binascii
sys.path.insert(0, "/sdcard/tfp1116")
tag = binascii.hexlify(os.urandom(8)).decode()
report = "/sdcard/tfp1116/usb-" + tag + ".json"
try:
    os.stat(report)
except OSError as error:
    if not error.args or error.args[0] != 2:
        raise
else:
    raise RuntimeError("Report already exists")
from usb_topface_trial import run
run("/sdcard/tfp1116/model", frames=600, report_path=report)
```

切回CSI也须保留当前USB结果，再在idle条件下显式调用新报告名：

```python
import sys, os, binascii
sys.path.insert(0, "/sdcard/tfp1116")
report = "/sdcard/tfp1116/csi-" + binascii.hexlify(os.urandom(8)).decode() + ".json"
try:
    os.stat(report)
except OSError as error:
    if not error.args or error.args[0] != 2:
        raise
else:
    raise RuntimeError("Report already exists")
from k230.topface_service import run
run("/sdcard/tfp1116/model", frames=1000, mode="TOP", target_id=1,
    station="BUFFER", display=True, ide_preview=True,
    gc_interval=4, report_path=report)
```

这些例子的随机报告后缀在板端协议路径长度限制内。正式新run仍应把实际报告路径和代码/输入身份写入交接。
QR/LANE、相机标定及整车流程需按自己的计划验收，不能用TOP程序运行完成替代。

## 7. 已验证与未完成

稳定短测证据：CSI1000/1000=10.81502FPS；USB600/600=8.417508FPS；零采集失败、cleanup=[]。
69项相关主机回归、230400像素显示转换一致、同一真实KPU空场tensor解码一致有效；完整source/prepared/KPU与ORT/Simulator配对没有完成。
CPU/KPU利用率未测；MMZ等内存池分开记录。25FPS目标未达到，USBJPEG解码/缩放是重要开销。
新真实定量盲测、真实中心/椭圆真值、实际镜头标定和机器人闭环未完成。用户定性接受记录不能变成百分比。

历史failed(3)与IDEinterrupt原件不改。production-session01.json为用户切换USB而中断，1268成功帧/1269attempt，status=RUN_FAILED、error=IDEinterrupt；12.71FPS不能升格长稳PASS。

## 8. 验证与Git交接

本次只改文档，不重复运行旧模型或无变化的算法测试。验证文档链接、JSON、源文件指纹、Git差异和新鲜板端状态。
Git使用现有origin和任务分支，不自动合并main、不创建新远端。代理从当前Windows系统配置读取，仅在本次Git进程设置；保留HZVPN与MCP loopback绕行，绝不把token或配置内容写进报告。

checkpoint使用本次文档任务的实际agent/session；成功后release。以OPS同步回执和最后git ls-remote核对的SHA证明备份，不把旧SHA或本地commit当新推送。
如网络失败，保留本地checkpoint并查status，再对同一任务sync --execute；不要重复生成交接或改投远端。

聚合入口中的README/PROJECT_MAP是本机链接层，代码和便携交接随当前私有仓库提交。其他工作树保留原样。

