# 正式模型完成后的 K230 实测

用户于2026-11-15再次确认：正式模型训练完成后连接真实K230开发板，测试效果和准确率。当前不以Pilot的两张真实val或主机Simulator速度替代正式模型板端验收。

正式权重冻结后，先完成PyTorch/ONNX/KModel独立合成测试和量化差异报告，再交付带SHA的best.pt、ONNX、KModel、output contract和独立board trial bundle。

连接开发板后，先记录实际固件、nncase runtime、CSI2 GC2093传感器输出与输入通道顺序；在独立目录运行有界测试。沿用既有授权，不保存生产参数，不改main.py/boot.py，不启用UART或运动控制。

真实准确率使用另行拍摄并人工确认的scene/session，覆盖六色、空场、0–3目标、4/5超容量、遮挡、出框重入，以及100–400mm镜头到顶面距离、80–100°观察角。模型和阈值冻结后再解封blind test，不用同一测试集反复调参。

必须分别报告检测TP/FP/FN、precision/recall、按颜色的分母与正确数、空场和底座/夹具误检、超容量拒识和重新捕获。中心和椭圆精度只有在相应真实标注或标定存在时计算；缺真值则标UNKNOWN，不凭叠加图推断精度。

性能按capture、AI2D、KPU、decode、display和完整链路分别记录P50/P95/P99；完整FPS包含取帧失败等待时间。保留capture losses、内存及清理结果，Simulator主机延迟不当作原生KPU性能。

数值配对按 [BOARD_PAIR_PROTOCOL.md](BOARD_PAIR_PROTOCOL.md) 收取同一source/prepared/KPU raw及原生decoded，小样本落盘模式与FPS基准分开。

当前未连接板、未采集新盲测。实际日期、设备状态和验收结论以届时运行记录为准。
