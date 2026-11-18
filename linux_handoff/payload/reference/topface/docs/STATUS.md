# K230 项目最新进展

更新时间：2026-11-16，北京时间。板端现场快照：12:12–12:13；事实以[本次机器记录](validation/topface-production-fps-20261116/status-refresh-20261116T041232Z.json)为准。
12:21:31再次只读核对：running=true，usb-live01.json仍未生成；production-session01.json于12:18:14确认存在，大小6026字节。
接续步骤见[交接文档](HANDOFF.md)，故障预防见[经验总账](LESSONS_LEARNED.md)。

## 当前结论

正式 TopFaceNet KModel 已部署到 K230，并通过显式 backend 接入原 CameraManager / VisionAdapter 的 TOP 分支。CSI 和 USB 均完成了包含真实采集、推理和显示的短测。用户人工认可识别效果；定量准确率、几何精度与长期稳定性分别尚未验收。

本次仅刷新现场、文档和 Git 记录。没有重启预览、探测相机、重跑模型、上传板端文件或更改代码。

## 已完成

| 阶段 | 已有结果 | 证据 |
|---|---|---|
| 数据与训练 | 10,000个仿真场景；train7465 / val1601 / 独立test934；正式100轮、23,400次optimizer更新，验证集选第92轮best | [数据接受](topface/v11/training/FULL_DATASET_ACCEPTANCE.json)、[正式训练接受](topface/v11/cloud/formal-a01/ACCEPTANCE.json) |
| 导出和离线评估 | 正式PT/ONNX/KModel已交付；934张独立静态仿真、1872帧运动及1张历史实拍已评估；这些不是新真实盲测 | [TopFace入口](topface/README.md)、[主机评估](topface/v11/reports/formal-host-a01/HOST_EVALUATION_RECEIPT.json) |
| 板端部署 | /sdcard/tfp1116；70份原框架源码及同一模型复制校验；最终服务、协议和USB脚本另有字节回读 | [最终源码清单](validation/topface-production-fps-20261116/FINAL_SOURCE_MANIFEST.json) |
| TOP接入 | 同帧六色类别、置信度、框、像素中心、椭圆及有效性；空场/边界不确定/超容量；稳定跟踪、目标选择、已抓取排除、原标定门 | [实现与运行说明](validation/topface-production-fps-20261116/README.md) |
| 显示和解码修复 | 显式CHW→HWC；独立显示内存RVV转换；向量化背景筛选；保留阈值、NMS、非有限值和容量拒绝 | [230400像素一致性](validation/topface-production-fps-20261116/fast-conversion-parity.json)、[原生解码一致性](validation/topface-production-fps-20261116/native-decoder-parity.json) |
| CSI实测 | 1000/1000，零采集失败，10.81502FPS，cleanup=[] | [集成1000帧](validation/topface-production-fps-20261116/integrated-gc4-1000.json) |
| USB实测 | 1280x720 MJPEG30FPS协商；600/600，零采集失败，8.417508FPS，cleanup=[]；原600元数据缺口由独立probe/5帧记录补充 | [USB600帧](validation/topface-production-fps-20261116/usb-model600.json)、[最终5帧身份验证](validation/topface-production-fps-20261116/usb-final-smoke5.json) |
| 电脑端回归 | 69项受影响检查通过；属于实现提交时的结果，本次文档更新没有重复运行；不是全仓全部测试 | [检查范围](validation/topface-production-fps-20261116/HOST_VALIDATION.json) |

KModel SHA-256：`f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`。
输入保持 uint8 RGB NCHW [1,3,192,320]，640x360源画面等比例letterbox，score=0.25、NMS IoU=0.5。中心和椭圆是模型预测，未宣称真实几何精度。

## 进行中

- 实测 COM3 connected=true、ready=true、running=true。
- 新鲜帧序号29262、age0ms；随后保存帧30596、age2ms，画面标注USB1280x720MJPEG。当前采样图为暗/模糊视野、EMPTY，没有该场景目标真值，不作识别准确率判定。
- USB live01 的预算为18,000次循环；最终路径 `/sdcard/tfp1116/usb-live01.json` 在本次查询时不存在。它只在程序结束后落盘，因此现在不能填写成功推理帧数、长跑平均FPS或长期PASS。
- [现场图片](validation/topface-production-fps-20261116/status-20261116T041232Z-preview.jpg)是有时间戳的采样，不是持续播放器；preview frameId不能换算为模型成功帧数。

## 资源和耗时

以下来自已结束的CSI1000帧短测，不能当当前USB会话的实时资源：

| 项 | CSI集成测量 |
|---|---:|
| 墙钟有效FPS | 10.81502 |
| KPU平均 | 23.892ms |
| 解码平均 | 16.090ms |
| 显示平均 | 31.749ms |
| GC及循环间计时段平均 | 18.062ms |
| Python heap峰值 | 2.990MiB |
| sys_heap峰值 | 8.046 / 64MiB |
| MMZ峰值 | 61.457 / 511MiB |
| CPU百分比 / KPU硬件利用率 | 未测 |

USB600帧中JPEG解码/缩放平均48.430ms，推理及后处理合计29.358ms。30FPS是协商取流模式，不是模型FPS。
内存池分母不同，sys_page、MMZ、Python heap不可直接累加为整机占用。KPU耗时×FPS约25.84%只是墙钟时间份额，不是硬件计数器利用率。详见[资源报告](validation/topface-production-fps-20261116/RESOURCE_FPS_REPORT.json)。

## 尚未完成或未验收

| 项 | 当前边界 |
|---|---|
| 长期稳定性 | USB live01最终结果待收；旧采集failed(3)、IDE中断和人为切源中断全部保留 |
| 定量识别质量 | 用户定性接受已记录；新真实六色、0–5目标、空场、遮挡、恢复、距离/光照覆盖与标注未完成 |
| 几何与毫米坐标 | 实测内参/畸变/位姿、中心/椭圆真值及mm精度未接受；无标定保持UNCALIBRATED |
| 完整数值资格 | 主机夹具和真实KPU空场解码一致已过；同源source/prepared/KPU raw与ORT/Simulator的完整配对未完成 |
| 更高FPS | 约25FPS目标未达到；不同场景和目标数量的历史FPS不能作为严格加速比 |
| 入口整合 | 新显式TOP服务已可运行；旧工作台默认入口、图形化模型切换未统一 |
| 整车闭环 | QR/LANE旧实现保留；本次没有重新接受QR/LANE全流程、UART/STM32、运动及机械动作 |

当前USB不是CSI2，未沿用原CSI标定。原main.py/boot.py、生产profile及active/draft参数没有修改，UART和运动关闭。

## 版本与备份

- 当前实现工作树：`C:/Users/28389/Desktop/k230-topface-production`。
- 分支：`ops/topface-production-fps-20261116`；远端：`smartlotus/k230-vision-optimized`。
- 实现提交：`62c62edd942cd8adff1c420f4828d83a6dcabc91`；本次状态刷新前HEAD：`c09562e0922bb0f6d6481c714d54188311d4b492`（含任务释放元数据）。
- 此前实现分支已完成GitHub远端SHA核对。本次文档的最终提交/推送结果由OPS checkpoint/release同步回执和Git HEAD确定，不把提交前SHA称为本次最终备份。
- 任务分支尚未合入main；原训练工作树和历史相机修复工作树保持原样。最新OPS状态以工具输出为准。

