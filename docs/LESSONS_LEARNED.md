# K230 项目经验与故障总账

更新时间：2026-11-16（北京时间）。当前维护分支：`ops/camera-diagnosis-20261116`。

此前经验分散在板端验证目录及 TopFace 安装规程，没有独立的统一总账。本文件作为入口，原始失败、运行身份和校验收据仍保存在各证据目录，不回写历史结果。

## 使用和更新

开工先读本总账中与任务相关的条目，再读当前 OPS 交接；不要只沿聊天记录重建流程。每次确认新根因、验证修复或出现新的阻塞，在本任务 checkpoint 前更新对应条目，保留失败和 UNKNOWN。每条必须包含现象、证据、已确认原因、修复、复验及剩余边界；不能仅写“已修好”。

项目源码、经验文档及交接一起精确提交和推送当前私有任务分支；必须核对远端 SHA 后再称云备份。根 README、AGENTS 和 TopFace 入口均链接本文件。不自动覆盖原发布 ZIP、不把经验备份当主线集成。

## 本轮状态

| 层 | 2026-11-16 证据与结论 |
|---|---|
| 正式模型 | KModel 1,611,824 bytes，SHA-256 `f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`；模型和阈值未改 |
| 预览传输 | 457 帧合成色块正常；板端保存的异常转换 BMP 与主机回读 SHA 一致，横带在上行传输之前已存在 |
| 相机通道 | CSI2 / GC2093；RGB565 直出 320 帧、RGBP888 原路径 309 帧、三通道对照 429 帧，均保留原始证据 |
| 显示修复 | 同帧 CHW 显式交错为有独立内存的 HWC RGB888，再转 RGB565；保持原 NCHW 模型输入 |
| 修复后运行 | [fixed600.json](validation/camera-diagnosis-20261116/fixed600.json)：600/600 帧、0 次采集超时、cleanup_errors=[]；完整链路 7.272463 FPS，总延迟 p50/p95 159.532/169.890 ms |
| 实际识别验收 | 六色、空场、0–5 目标、遮挡、恢复、距离与光照仍需真实目标及独立标注。没有中心/椭圆真值时精度 UNKNOWN |
| USB 相机 | 一次 UVC.probe=false，电脑只见内置相机；用户随后要求先跳过。未协商视频模式、未取得 USB 相机帧，不能判定兼容性或硬件损坏 |
| 长时间运行 | [manual18000.json](validation/camera-diagnosis-20261116/manual18000.json)：请求18,000帧，第733次采集失败，实际成功732帧；RUN_FAILED，长期稳定性未通过 |
| 有界恢复复测 | [bounded1000.json](validation/camera-diagnosis-20261116/bounded1000.json)：1000次预算内，468成功/471尝试、连续3次failed(3)，按界退出；恢复未通过 |
| 逐帧释放复测 | [gc1000.json](validation/camera-diagnosis-20261116/gc1000.json)：1000/1000成功、capture_failures=0、cleanup_errors=[]；137.285秒测量窗口，墙钟有效7.284117FPS；仅该短测通过 |
| 主机中断 | [manual-gc18000.json](validation/camera-diagnosis-20261116/manual-gc18000.json)：测量0帧时IDE interrupt；扩展停用/SIGTERM日志另存。之后另启r2会话，不能把中断记录或启动ACK当稳定性通过 |

## F01：大模型经过工具文本回显后被截断

现象：本地 KModel 1,611,824 bytes，首次板端回读只有 786,444 bytes，而写工具仍返回 success。工具层取得的 base64 长度已经不足；不能把它写成已经证明的板端 USB 固定大小限制。

原因与流程问题：用命令的文本输出承载大块 base64，未在写前核对预期编码长度。仓库原本已经有 [安装规程](topface/v11/reports/BOARD_INSTALL_PROTOCOL.md) 和 [deploy_board_trial.py](../research/topface/deploy_board_trial.py)，本次早期绕过它们，属于可避免的执行错误。

修复与复验：五片重传、板端拼合后重新打开文件计算大小和 SHA，再下载完整目标核对；最终与正式模型一致。本机旧回读和部署记录位于 `C:/Users/28389/Desktop/k230-project/archives/board-readback-20261116/`，该目录只是本机可用，不冒充随仓库归档。

预防：先检索并复用现有安装器；262,144-byte 分片逐片回读，最终完整 SHA。不要把大文件放进可能被截断的工具输出。修复脚本/后续报告独立存放，发布包保持不可变。

## F02：物理屏在跑，IDE 没有画面

原因：原交付脚本设置 `Display.init(..., to_ide=False)`，只打开物理屏输出。启动 MCP 预览不能弥补固件没有输出 IDE 帧。

修复：增加显式 `ide_preview` 参数；验收调用 `display=True, ide_preview=True`。保留默认物理屏行为，并把实际选项写入运行报告。先做合成色块检查，再加载模型。

## F03：RGBP888 直接转换显示会产生横白带和噪点

证据：[三通道对照](validation/camera-diagnosis-20261116/format-comparison.jpg)、[同帧记录](validation/camera-diagnosis-20261116/same-conversion.json)、[回读校验](validation/camera-diagnosis-20261116/capture-readback-sha.json)。RGB565 直出和 RGB888 转 RGB565 正常；RGBP888 原始字节在主机可正确重建，但同一源直接 `to_rgb565()` 的板端 BMP 异常，源 SHA 转换前后不变。

源码核查：当前固件完整提交 `ae2f20da0467a8e70659f032b240aae50272c49a` 的 [py_image.c](https://github.com/kendryte/canmv_k230/blob/ae2f20da0467a8e70659f032b240aae50272c49a/port/omv/modules/py_image.c) 转换进入通用绘图路径；[draw.c](https://github.com/kendryte/canmv_k230/blob/ae2f20da0467a8e70659f032b240aae50272c49a/port/omv/imlib/draw.c) 没有 RGBP888 转换分支。实测与源码共同支持当前固件的这条转换路径不受支持。链式 `to_rgb888().to_rgb565()` 也经过该路径，本轮试验只得到黑画面，未采用。

修复：[topface_trial.py](../research/topface/board/topface_trial.py) 显示端显式 `reshape(3,H*W).transpose().copy().reshape(H,W,3)`，构造 RGB888 引用图像后转 RGB565。保留 packed 数组到转换完成，使用同一相机帧叠加结果。不要拿另一个通道的图像叠当前推理框。

复验：原生 16×16 色值 (11,79,203) 经 RGB565 显示为 (8,77,206)，符合量化；主机布局/内存所有权测试与解码器共 5 项通过；修复脚本板端回读 SHA 为 `02c4a74ebda3d1081cd4e3f14dd15da2daa30690b12f744f9a6674a2a1fa4f24`；600 帧完整运行及新鲜预览通过。显示步骤 p50=25.795 ms，其开销计入总链路。实际识别准确率及标定未因此通过。

## F04：把新文档/stub 当成当前固件兼容证明

实测 `CSC.convert(image.Image)` 报只接受 `py_video_frame_info` 或 `k_video_frame_info`；改为 dump_frame 原生对象后仍遇到 TypeError，清理也报错，未采用 CSC 方案。当前 runtime 的 AI2D 格式枚举有 `RGB_packed`，没有假设的 `NHWC_FMT`。

预防：记录真实固件/runtime、对象类型、shape/dtype 和输入格式；先核对精确源码，再做最小有界资格测试。文档主分支和缓存 stub 只是线索，不授予实机 PASS。

## F05：小型格式探针失去命令响应

一次 16×16 RGB_packed→NCHW 探针未留下最终结果，随后 USBDBG 0xA4 short read、Not connected。一次受控重连为 ready=false；用户断电重上电后，清理旧主机连接、重连，并获得本次唯一 REPL nonce，恢复 ready=true。

结论：该探针为 FAILED/UNKNOWN，底层原因未定位；没有自动重放，最终修复没有改走这条 AI2D 路径。不能把此事件扩大为相机损坏、模型损坏或已证明的 USB 线故障。cached board_info 不足以证明命令通道可用。

## F06：多会话共用 COM3，观察会串台

本对话观察到并非本对话启动的脚本与结果；用户随后明确确认其他会话已停止，由本对话独占。按 OPS 保留旧任务并恢复所有权，CSI/UVC/上传串行。子 Agent 仅读源码、证据和文档，不访问设备。

预防：资源认领必须包括 COM3 和媒体管线。RPC 返回“已有脚本”或意外新运行时，先查所有权和终端，不得把它归入自己计划的配对采集；不能为了继续而直接打断他人任务。

## F07：available=true 可能只是旧帧

本轮见到 frameId=2592 的缓存帧 ageMs=429924，而 running=false。它不是实时预览。修复验证使用 fresh=true、递增 frameId、低 ageMs，并关联当前脚本/会话；保存图片同时记录这些元数据。MCP 预览流会持续，聊天里显示的是取样帧，不把静态截图称为播放器。

## F08：MicroPython 与 CPython 的哈希接口不同

板端 `hashlib.sha256().hexdigest()` 报 AttributeError；改用 `binascii.hexlify(h.digest()).decode()`。关闭写文件后重新打开计算 SHA；工具 success 和目录中的旧 identity.json 都不能替代实际文件校验。

## F09：完成运行不等于识别验收

先前 600/600、0 超时只证明处理了帧，即使源图像的显示转换已坏也可能 RUN_COMPLETE。本轮明确分开安装校验、固件执行、画面正确性、性能、数值配对、真实识别、几何标定及长期稳定性。每个结论绑定模型/脚本 SHA、配置、实际分母和证据。

旧报告提到的“准备配对采集”没有原生配对文件，不算已采集。六色准确率需要确认场景与真值；叠加框和椭圆不证明模型已正确识别。

## F10：主机验证环境和模块路径不能凭目录名推断

首次新增测试少了 board 目录导入路径，随后补齐；普通 Python 缺 onnx，另一个既有隔离环境缺 numpy。没有自动安装依赖或把未执行项标为通过。最终受影响的 5 项布局和解码器检查在已有普通环境通过；额外的未改配对套件环境未就绪，保留记录。

## F11：采集错误未必带有 timeout 字样

修复后18,000帧尝试在732个成功测量帧后返回 `sensor(0) snapshot chn(0) failed(3)`。原脚本只将异常文本中含 timeout 的情况计为采集超时，因此报告中 `capture_timeouts=0` 不能解释为没有采集错误；status=RUN_FAILED、measurement_attempts=733 和 error 原文必须一起读。该失败完整保留，没有重写成成功短测。

精确固件 [mpp_vb_mgmt.c](https://github.com/kendryte/canmv_k230/blob/ae2f20da0467a8e70659f032b240aae50272c49a/port/core/mpp_vb_mgmt.c) 将任意 `kd_mpi_vicap_dump_frame()` 非零返回统一包装为3，底层具体码没有保留；2才是图像记录槽不足，4是格式不支持，5是mmap失败。因此3不能当作具体超时或内存不足的证据。

新版处理：保留原异常和measurement step，分别报告 timeout、native_dump_failed_3、no_frame，所有测量阶段失败进入capture_failures并保留实际attempt分母；最多两次100ms退避重取，连续三次失败终止，恢复也只标RUN_COMPLETE_WITH_CAPTURE_LOSSES。不自动重启相机、改用其他CSI或无限吞错。单独处理后续新attempt，不重放并覆盖旧失败。快照失败的物理/驱动原因仍 UNKNOWN。

该版脚本 `topface_trial_bounded.py` 回读SHA为 `825210413352175610b2cd5055b025a38bbc694ef91cff8d2dc959aa455f7097`。布局、所有权、错误分类与解码器6项主机检查通过；原生复测468成功/471尝试后连续3次失败，程序按界退出。错误统计与停止行为验证成立，采集恢复未通过。前面的600帧记录只对当时源码与配置有效，不升级成新版/长期稳定性通过。

## F12：临时图像及原生缓冲释放需要单独验证

显示修复每帧会产生交错RGB888数组、引用图像及RGB565转换图。Python引用退出局部范围不保证原生缓冲立即释放；原循环没有逐帧gc。新版只在frame=None后、下次snapshot前加入gc.collect，保留全部其他模型/采集参数，对照脚本 `topface_trial_gc.py` SHA为 `43d2653e099943295a30421614a9ff8a969717e2a5762c25fe617f1ce254ea76`。新1000次预算实测1000/1000、无采集错误、cleanup=[]，测量137.285秒，墙钟有效7.284117FPS，详见gc1000.json。此证据支持当前释放策略通过这次短测；不能声称已经证明底层内存泄漏或长期故障彻底消失。

独立审查未发现当前临时数组的悬空引用。GC清临时对象，相机原生帧仍由sensor._imgs管理；不能把两者混称。GC在shown时间点之后，阶段total和active FPS不含GC，measurement_wall_seconds与effective_camera_to_result_fps包含；实际帧率使用后者。

## F13：IDE扩展重启会中断硬件验收会话

逐帧释放版第一次长预览在测量前收到IDE interrupt（0帧）；主机日志同一窗口记录CanMV Extension Deactivated及backend SIGTERM，见 [preview-host-restart.log](validation/camera-diagnosis-20261116/preview-host-restart.log)。重连后的ready缓存和只剩soft reboot的终端不能当旧脚本仍在运行。不能凭这次中断推断模型或摄像头失效。

已保留原manual-gc18000.json，确认idle后启动独立r2会话，报告目标为 `/sdcard/cdiag1116/manual-gc18000-r2.json`。验收时保持当前IDE/扩展运行；若重启、切换控制端或更换缓存资源版本，重新查询脚本与fresh帧，保留旧attempt。公开发布最新stub的自动更新也不代表板端固件跟随升级。

## 历史经验入口

- [KPU 管线与连接故障](validation/kpu-pipeline-20261112/README.md)：旧模型/旧运行的失败和恢复证据，不覆盖为本轮结果。
- [独立安装流程](topface/v11/reports/BOARD_INSTALL_PROTOCOL.md)、[真实板测计划](topface/v11/reports/BOARD_TEST_PLAN.md)、[数值配对规程](topface/v11/reports/BOARD_PAIR_PROTOCOL.md)。
- [本轮证据目录](validation/camera-diagnosis-20261116/) 与 OPS 最新交接共同确定当前状态。

