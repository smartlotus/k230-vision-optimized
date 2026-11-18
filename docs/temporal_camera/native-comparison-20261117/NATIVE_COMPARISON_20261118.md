# CameraXY 原生性能对比记录 2026-11-18

本记录只覆盖同一块 K230、同一固件、同一 UVC 源上的有界工程性能测量。模型准确率和视觉识别验收保持暂停，`position_valid` 全部为 false，不把本记录当作定位成功。

## 环境

- 板卡：LCKFB-LSPI-K230-1G-CanMV，COM3
- 固件：`K230-v1.8-36-gae2f20da0467a8e70659f032b240aae50272c49a`
- nncase runtime：`2.11.0`
- UVC：`openaicam#openaicam`，协商 `1280x720 MJPEG 30 FPS`
- TCN4 归档 SHA256：`b7dd989d7f1d23b74a95cbdd7528e8e47fc8e403256ed1363d66f5c4713de48b`

两个模型包都写入独立的 `/data/k230_temporal_v1_cmp1118_*` 目录，并完成板端文件字节和 SHA256 读回。当前源替换只涉及 `board_multiroute.py`、`embedded_filter.py` 和 `localization_runtime.py`，没有写入 `main.py`、`boot.py` 或生产 profile。

## 结果

| 路线 | 帧数 | 丢帧 | 状态 | camera-to-result FPS | KPU 均值 ms | AI2D 均值 ms | filter 均值 ms | cycle 均值 ms | 最小 Python 空闲字节 | 末尾 MMZ used |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CNN1 | 10 | 0 | MEASURED | 6.422607 | 5.7737 | 0.9592 | 56.2716 | 155.6961 | 2,094,024 | 0 |
| CNN1 | 60 | 0 | MEASURED | 7.111532 | 5.7527 | 0.9055 | 55.9571 | 140.5996 | 340,864 | 0 |
| TCN4 | 10 | 0 | MEASURED | 5.488474 | 18.0158 | 0.9340 | 56.0966 | 182.2354 | 2,054,976 | 0 |
| TCN4 | 60 | 0 | MEASURED | 6.105006 | 18.0121 | 0.9036 | 55.9860 | 163.8085 | 341,056 | 0 |

完整原始收据见 `cnn1-10a-readback.json`、`cnn1-60b-readback.json`、`tcn4-10a-readback.json` 和 `tcn4-60b-readback.json`。汇总见 `NATIVE_COMPARISON_20261118.json`。

## 边界

- 四次运行均为 `position_valid=false`，CNN1 为 10/10、60/60 未定位，TCN4 为 10/10、60/60 未定位。
- 四次运行均无丢帧、无清理错误，末尾 MMZ used 为 0。这证明本次有界工程链路完成并释放资源，不证明识别准确率、实景定位或长时间稳定性。
- 准确率、视觉识别、道路测试、真实标定、TCN/GRU 竞争选择和长稳验收继续保持 HOLD/NOT RUN。

