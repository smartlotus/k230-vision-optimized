# 新 SD 卡 + K230 工程链路观察

日期：2026-11-18（北京时间）。本轮先确认新存储介质、固件和工程调用链；没有进行正确率、道路质量、标定、QR、CSI 或控制测试。

## 当前状态

- 板卡：COM3，LCKFB-LSPI-K230-1G-CanMV，`connected=true`、`ready=true`。
- 固件：`v1.8-0-gc2d1f5c`，2026-07-03 构建；旧 `usb-chain30-c` 使用的是 v1.8-36，旧 FPS 不直接代表新卡。
- SD：215296 个可用块，每块512字节，约110 MiB 可用；`/sdcard/main.py` 和 `/sdcard/boot.py` 不存在。

## 模型身份回读

RoadNet 和 TopFace 模型身份由 nonce `rn1118-new-sd-hash-b` 绑定，SHA 与冻结包一致：

- RoadNet KModel：137208 bytes，`cba11c87ce36305ffac6dee18b65b4f1e5b47070e00a2f9e706d0aee72a4980e`。
- RoadNet identity：752 bytes，`8109a700cac1673628235050f513e5106f35a870fff84ed4eff0170140bee0ce`。
- TopFace KModel：1611824 bytes，`f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88`。
- TopFace identity：528 bytes，`a62c9e5526217edbc8df8f83857eb9e065a9f3aae0e634b05f6e8e5d6cd68d47`。
- TopFace output contract：3734 bytes，`c49c560cfce764bb0d84b5b992918644596e8c150f6793e78da138b9a6c1f94a`。

## 新卡工程运行

共享 CanMV 终端缓冲中观察到另一会话启动的 `new-sd-chain120-20261118`；本当前会话没有启动该 run，因此这里保留观察 provenance，不把它写成当前会话的启动回执。

- 固件：v1.8-0-gc2d1f5c。
- USB：`openaicam#openaicam`，1280×720 MJPEG，约30 FPS协商。
- 120.127 秒，750/750 帧。
- RoadNet 推理、完整输出结构检查、后处理、TopFace 推理和显示均为750/750。
- 采集失败0，内存采样失败0，清理错误0，结束 MMZ used=0。
- RoadNet KPU平均12.60486 ms，TopFace平均26.14054 ms，完整周期平均160.1521 ms，工程FPS 6.243392。
- 最小 Python 空闲堆 2004896 bytes，活动 MMZ 峰值 30478336 bytes。
- 正确率和真实道路质量均保持 `NOT_RUN_BY_USER_SCOPE`。

原始提取结果见 [观察记录](new-sd-20261118-OBSERVED.json)。它不是旧 v1.8-36 的 `usb-chain30-c`，两者不合并统计。

## 固件兼容性观察

历史探针在 v1.8-0 上出现过 `os.path` 不可用和直接 `import k230` 失败；后续板端脚本应使用 `os.stat`/`open`，并显式插入 `/sdcard/tfp1116` 等源码根。这些是探针兼容性问题，不代表摄像头或模型失败。实际 RAM 工程入口已经在新固件上完成120秒运行。

## 下一步

新 SD 卡、K230、USB 摄像头、RoadNet 和 TopFace 的工程链路已有一轮新固件观察证据。若继续，应由当前会话使用新的唯一 tag 启动一轮，并保存 intent、执行源码和终端；不要复用 `new-sd-chain120-20261118`。之后再单独决定是否进入正确率或道路质量测试。
