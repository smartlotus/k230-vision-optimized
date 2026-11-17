# 滤波定位板端工程短测

日期：2026-11-17。**a06 CNN1 + CPU 时序滤波主链路已在当前 K230 与 USB 摄像头上跑通。**完整链路累计 100/100 帧，零丢帧；最终版本连续两次结束后 MMZ 占用均回到 0，设备已空闲。

本次验证取帧、预处理、KPU 推理、滤波、报告保存及读回、退出清理和重新启动。不评价定位正确率、道路检测或整车效果。`position_valid=false` 与实验精度 HOLD 保留；TCN / GRU 没有做原生模型测试。

## 实际执行与证据

链路：USB MJPEG 1280×720 → RGB888 640×360 → CHW / AI2D 160×96 → a06 CNN1 KModel → 2304 个位置 logits → CPU 网格时序滤波 → 候选 / 状态 → JSON 文件 → 电脑读回。

| 测试 | 帧数 | 结果 |
|---|---:|---|
| 初次启动 `c1_10a` | 0 | 路径检查使用了 MicroPython 不支持的 `startswith(tuple)`，进入推理前失败；原记录保留 |
| 首次完整链路 `c1_10b` | 10/10 | 无丢帧、无清理异常，4.79 FPS |
| 短测 `c1_60c` | 60/60 | 无丢帧、无清理异常，11.686 秒，5.13 FPS |
| 重新启动 `c1_10d` | 10/10 | 正常重启并完成完整链路 |
| 显式回收缓存池后 `c1_10e` | 10/10 | 无清理异常，MMZ 0 → 0 |
| 最终重启复验 `c1_10f` | 10/10 | 无清理异常，MMZ 0 → 0 |

另有两次独立 USB / AI2D 探针各 10 帧；它们不计入上述完整链路 100 帧。第二次探针检查实际输出 Tensor，其像素范围为 6..249，排除了只查看原始零数组造成的假黑画面。

- [原生结果汇总](NATIVE_SUMMARY.json)
- [60 帧完整原始报告](c1-60c-readback.json)
- [最终版本重启报告](c1-10f-readback.json)
- [结束时文件及资源核对](POSTFLIGHT.json)
- [USB 与 AI2D 原生探针](CAPTURE_AI2D_C.json)

60 帧窗口中，KPU 平均约 5.75 ms，CPU 滤波约 111.54 ms，RGB 转换及排列约 68.78 ms。这是包含采集和处理的无预览运行测量；协商的 30 FPS 不是整个定位流程 FPS。本次没有做性能优化或长时间稳定性验收。

## 本次发现和修复

1. **SD 写入只读。** `statvfs` 可用块为 0，实际写入进一步返回错误 30，不能仅解释为磁盘已满。测试在 `/tmp/k230_temporal_v1_c1r3` 独立目录进行；没有改变 SD 挂载或启动配置。SD 可写及持久部署仍待处理。
2. **大脚本传输出现 USBDBG 短读。** 64 KB 上传未得到完成标记，状态保留 UNKNOWN。单次受控重连恢复后，原生 `ubinascii` 接收 bytes，改为 8 KB 分片；探针及完整 2,646,600-byte 模型传输通过。每个完整文件重新打开计算 SHA，与主机一致。此次恢复没有证明短读的全部底层根因。
3. **临时根路径兼容性。** 将本次新增的 `startswith(tuple)` 改为两个独立字符串判断，保持路径范围约束，原失败报告不覆盖。
4. **nncase 缓存池需要显式回收。** 原短测退出后保留约 4.4 MB 缓存，调用 `nn.shrink_memory_pool()` 后回到 0。将它加入资源清理阶段，最终两次原生复验均为 0 → 0；不是把缓存驻留诊断为内存泄漏。
5. **旧备用入口缺依赖。** R2 的 `board_temporal.run()` 需要未打包的 `board_probe.py`。本次始终使用交付文档指定的 `board_multiroute.run()`，其本地依赖齐全。

原模型、训练权重、滤波参数和置信度保护不变。源码更新涉及测试入口的独立根路径、清理和主机传输工具；交付 ZIP 保持原样。

## 隔离与保留

用户先要求仅电脑端检查，随后确认其他会话结束，授权继续板端测试。期间又明确批准：仅清理已备份的 `/tmp/xyusb1117e`。全部 19 个旧临时文件先完整下载并核对大小和 SHA，再删除 e 目录内的 9 个文件；d/f 的其余 10 个文件在结束时再次核对一致。

旧文件备份位于 `C:/Users/28389/Desktop/k230-project/archives/temporal-native-storage-backup-20261117`，是本机完整备份，此目录未纳入 Git。参见 [备份清单](TMP_BACKUP_RECEIPT.json)、[批准后的清理收据](APPROVED_CLEANUP_E.json)、[存储与恢复原始观测](STORAGE_AND_RECOVERY.json)。

`/sdcard/main.py` SHA 始终为 `c0b89b0b48864ff51e0e1d9a07082d306dbaf301aa4b6c95432ea6e04dbc640c`，`boot.py` 仍不存在。结束时校验本次部署的 9 个文件，当前源码与本地一致。未运行 UART、STM32、车辆控制或其他视觉线路。测试目录保留在临时文件系统，不代表自动启动或生产部署。

## 可复用入口

[native_smoke.py](../../../research/temporal_camera/native_smoke.py) 提供 `install / patch / run / fetch`。当前临时目录使用 `--runtime-upload --current-source`；接续前检查设备所有权、文件 SHA 及新报告名。禁止覆盖旧报告、自动重放 UNKNOWN 上传、清理未获准目录或把精度 HOLD 解除。

本地 26 项回归覆盖原有 20 项逻辑，以及分片顺序、篡改 / 偏移拒绝、CanMV bytes 解码接口、测试路径范围与忙设备保护。历史电脑端结果见 [HOST_MODEL_PIPELINE.json](HOST_MODEL_PIPELINE.json)，它与上述真实原生结果分别保留。
