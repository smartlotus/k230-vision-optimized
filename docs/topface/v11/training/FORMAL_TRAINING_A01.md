# 正式训练 a01：100轮与归档接受

截至2026-11-15T15:42:16.699862+00:00，原Kaggle145144940 v1完成100轮、23,400次optimizer更新。原任务没有重启或替换。验收为FORMAL_TRAINING_COMPUTE_AND_ARCHIVE_VERIFIED，选定第92轮best，选择仅使用val。

## 实际运行与数据

- Tesla T4 / cuda:0；PyTorch 2.11.0+cu130，CUDA 13.0，cuDNN 92700。
- Python 3.13.16，NumPy 2.1.3，OpenCV 4.14.0。
- FP32、AMP/TF32关闭；batch32；320×192等比letterbox；AdamW，初始lr0.001，weight decay0.05；100更新warmup加cosine；seed20261114。
- Full10,000图：train7,465、val1,601、独立test934。数据索引SHA `0555f5e7dd9b372c51108f4482f0ca3ed8816c790eb3572266956403e645dc49`。
- TensorBoard ENABLED；训练阶段31022.5秒；峰值CUDA分配1248334848字节。
- 首轮mean loss0.443884，最后一轮mean loss0.161509。完整100轮trace、逐轮validation、TensorBoard、best/last和源码在原始Drive归档中。

## 选定best的仿真验证集结果

这些数据用于模型选择，不是独立test，也不是真实K230验收。AP采用项目101点插值规则，未宣称pycocotools COCO AP。

| 指标 | 结果与分母 |
|---|---|
| task mAP@0.5 | 99.4950% |
| task mAP@0.5:0.95 | 98.3819% |
| 运行阈值下class-agnostic precision | 99.8830% = 4268/4273，由原分母计算 |
| recall@IoU0.5 | 99.8596% = 4268/4274 |
| TP / FP / FN | 4268 / 5 / 6 |
| 匹配目标颜色正确率 | 100.0000% = 4268/4268，不含漏检 |
| 空观察误检 | 0个 / 199图 |
| 可见物理中心像素误差 | median 0.7522px，P95 4.6679px，n=4067 |
| 遮挡物理中心像素误差 | median 2.6632px，P95 12.8129px，n=175 |
| 椭圆IoU | median 0.960364，P05 0.916066，n=2294 |
| 图像方向角误差 | median 2.4098°，P95 9.4331°，n=1244 |

中心及椭圆统计条件为已匹配且具对应有效真值的目标；像素单位为原640×360图像。椭圆finite为2294/2294个匹配且geometry-valid目标，不能当全部世界实例的成功率。有效角GT共1348，预测有效角1244；本次val没有请求近圆轴比切片诊断，保留其局限，独立test会报告对应切片。没有真实中心/椭圆真值的图不能用于真实几何精度结论。

## 工件与来源

- best.pt SHA `7d7b9a332e2b30d3d09e85245d6e39fd5f9a9d52b694f4bdb0d203585b0346d7`，15093373字节。
- last.pt SHA `68078bab4f8ff2c64b2931a27896f8048ec0c87243cc6595d1654a1723f4fd84`。
- 原始归档94167738字节，SHA `692be0b4b88f5c8a3a8fca634bc5511d846696a2314b8d32341c6fd4ae99700f`。
- Drive file ID `1OQp9H_ZhqKPhDUm_DbZZwXjo0aWqSi29`；全字节读回FULL_BYTE_SHA256_PASS，独立元数据匹配，临时发布Secret已删除。
- 原保存源码SHA `7b07ecceb419e6eb385afd38e5dbdf09eaecb29a8755867f01e87cb5308c2e21`；来源ref `lotus30/k230-topface-v11-06-formal-training`，v1，ID145144940。

权威原件见 `../cloud/formal-a01/ACCEPTANCE.json`、`../cloud/formal-a01/SELECTED_MODEL.json` 和 `../cloud/publisher/k230-topface-v11-formal-a01/TRANSFER_RECEIPT.json`。这些链接的前缀按本报告位于training目录计算。

## 后续范围

原07b完整导出/934独立test及后续1872帧运动与真实图评估继续进行。用户另选07c优先板测试用包，复用本次选定best，64张train PTQ与12张train配对；试用不代替完整评估。实际ONNX/KModel、量化质量、原生K230兼容性、FPS、稳定性及新真实盲测仍待相应真实产物验证。
