# K230 原生对照暂停交接

本轮在 COM3 重新连通后，实读 CanMV v1.8-36-gae2f20d、nncase runtime 2.11.0。预检时 MMZ 使用量为 0，USB 设备标识为 `openaicam#openaicam`。前次 CNN1 临时目录已不存在。[成功预检](PREFLIGHT.json) · [首次预检错误](PREFLIGHT_FIRST_ATTEMPT.json)

TCN4 安装校验了 R2 交付包，随后在 `/tmp/k230_temporal_v1_t4p1` 收到 59 片写入回执，覆盖模型起始 483,328 字节。下一次 CanMV 请求遇到 WinError 10054，主机进程退出。本轮没有拿到完整模型的板端 SHA256，最后一片未确认请求与临时目录现状均未知。保留安装收据及设备现场。[分片收据](TCN4_INSTALL.json) · [暂停状态](PAUSE_STATE.json)

TCN4 原生推理、优化版 CNN1 板端复测、GRU 原生运行以及长时间稳定性均为 **NOT RUN**。先前的 100 帧 A 路线短测仍可复核。[A 原生短测](../native-smoke-20261117/NATIVE_REPORT.md) · [CPU 优化主机验证](../CPU_TRACKER_PROFILE.md)

我按用户要求在本阶段结束处暂停。恢复后先核对 CanMV 连接与板卡所有权，再读取临时文件状态，根据实际大小和前缀 SHA 决定如何接续。离线交付 SHA256 清单下 20 个文件已逐字节复核，包含 9 个 ZIP。

