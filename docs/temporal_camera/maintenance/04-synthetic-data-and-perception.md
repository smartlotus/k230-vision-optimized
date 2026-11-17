# 合成数据、相机和视觉模型仿真

## 1. Camera model

必须明确： - sensor resolution and active area； - focal length / field
of view； - principal point； - distortion model； - camera-to-world
extrinsics； - rolling/global shutter； - exposure、gain、white
balance； - frame rate、latency、motion blur； - resize、crop、letterbox
和 tensor layout。

缺少真实标定时，把参数标记为
UNKNOWN，并生成有明确来源的候选范围。不能把"相机大约在 40cm
高"推导成精确内参。

## 2. Scene-to-label consistency

所有标签从同一份渲染状态产生：bbox、mask、keypoint、center、3D
pose、orientation、depth、color/class、visibility/occlusion flag。

对图像做 crop/resize/rotation
后，必须同步变换标签。检查投影矩阵、坐标原点、y
轴方向、像素中心约定、边界闭合规则和遮挡处理。

## 3. Visual domain randomization

按真实变化范围逐项随机化：object position/rotation/scale、lighting
direction/intensity/color、background and texture、material
roughness/metallic/specular、camera
pose/intrinsics、blur/noise/compression/exposure、partial occlusion and
clutter。

视觉随机化应围绕真实相机和环境的可能变化，而不是仅追求画面丰富。

## 4. Split 策略

优先按场景、物体实例、相机、照明、日期、生成 seed
或背景分组切分。避免相邻帧泄漏。独立实拍测试集不应参与合成参数调优。

构建时序窗口前，先校验每个 trajectory/pose/appearance group 只有一个
split；仅按 `(split, trajectory_id)` 分组不能发现同一轨迹跨 split。
appearance ID 必须绑定实际渲染 scene seed、材质/光照来源，不能用新名字掩盖复用场景。

网格分类定位需要统计实际训练端点支持的网格类别，以及验证端点的覆盖比例。
不能用“全帧有覆盖”替代固定窗口最后一帧的实际监督覆盖。主验证集用于选模/校准，
未见空间和未见外观 challenge 单独冻结、报告；对从未获得正监督的类别不能声称测得正常插值性能。

时序压力测试必须修改所有路线共享的真实可见输入流：丢帧从神经网络历史移除，
失效帧实际置空或损坏；只给滤波器 quality=0 不证明时序网络经历了视觉失效。
原始与滤波结果应采用相同坐标解码；格心与亚格细化的误差分开报告。

## 5. 模型端评估

检测模型可报告 mAP、precision、recall、per-class performance、bbox
IoU、center error（pixel 和换算后的 mm）、color classification
accuracy、orientation/ellipse error、confidence calibration、inference
latency/memory/quantized accuracy。

模型输出的坐标必须定义在哪个 frame：image pixel、camera
normalized、camera metric、robot/world
frame。不能把图像中心误差直接称为物理定位精度，除非完成几何标定和换算。

## 6. Embedded deployment

部署前至少验证 model input/output
shape、RGB/BGR、NHWC/NCHW、quantization scale/zero point、operator
compatibility、memory footprint、latency/FPS、camera frame
synchronization、shared accelerator scheduling、output confidence and
invalid-frame handling。

PC 上的 PyTorch/ONNX 成功不代表 KPU/NPU
模型可运行。需要经过实际转换链、设备推理和真实相机测试。
