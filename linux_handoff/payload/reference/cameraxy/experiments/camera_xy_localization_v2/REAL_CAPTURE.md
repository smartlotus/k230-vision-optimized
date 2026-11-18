# 真实 OV2735 定位验证

使用实际工作姿态固定 USB OV2735，把相机光心的铅垂投影定位到以 2.4 m 方形中心为原点的地面坐标系。记录铅垂定位方法、重复读数与预计测量误差。由实际光心定义量测，支架外缘、镜头外壳中心和底盘坐标均不能替代。若用棋盘格外参反算光心，请把标定板测量和独立地面控制点验证留在测量记录里。

1. 在实际使用的分辨率、裁剪和方向下独立标定 K 与畸变，保留原始标定图。标定样本与测试图独立。
2. 实测 UVC 模式、图像方向、白平衡/曝光、地面高度与相机俯仰。位置覆盖路口、不同重复直道、蓝区和暂存/粗加工附近。对同一个位置可采不同 yaw 和照明。
3. 先索引测试图及原始 SHA256。每个光心测量位置使用独立 survey_position_id。同一位置重复图共用该位置的直接测量标签。测试图冻结后再运行主机评估。
4. JSONL 每行含 sample_id、相对 image 路径、image_sha256、survey_position_id、camera_xy_target、camera_metadata 和 measurement。camera_xy_target 键只允许 camera_x_mm 和 camera_y_mm。camera_metadata.sensor 为 USB_OV2735，实测 UVC/intrinsics/图像变换写在该映射。measurement 含 reference（DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY）、method、record_id 和正数 uncertainty_mm。
5. 用冻结模型评估独立验收集。报告保留仿真置信度校准、逐帧候选误差、阈值覆盖率、已接受样本误差、地面真值不确定度和图像清单 SHA256。用于微调的实拍样本另行采集。

真实标定、合格实拍真值、真实定位误差和置信度为 UNKNOWN。[板端有界工程实测](USB_NATIVE_RESULT_20261117.md)给出了冻结 learned_map 的调用计数、时延和资源释放。机械安装外参由用户在控制层独立测量和使用。


## 从现场测量记录生成清单

可将测量整理为同一目录中的 `survey.json`，图像路径相对该目录。顶层 `camera_metadata` 含 `sensor: USB_OV2735`、`capture_source: OPERATOR_OV2735` 和现场记录。顶层 `positions` 为测量位置数组，每项含 `survey_position_id`、两坐标的 `camera_xy_target`、上述 `measurement` 与 `images`。每个 `images` 条目只含唯一 `sample_id` 和相对 `image`。实际模式、内参、裁剪和安装记录缺失时填 `UNKNOWN`，记录坐标的实际测量不确定度。

```powershell
python prepare_real.py C:/survey/survey.json C:/survey/real-xy.jsonl
```

生成程序读回原始图像计算 SHA256，拒绝底盘中心标签、重复图像字节和越界路径。每张图片应为同一实测 16:9 输入配置。重复拍摄同一个光心 X/Y 时使用同一个 `survey_position_id`。

## 冻结模型评估

单点或网格模型使用 `infer.py --real-manifest`。正式融合方案从实验目录运行下列命令，`$layout` 和 `$roadModel` 使用实验 README 中的冻结资产路径。

```powershell
$run = '.runtime/kaggle-map-full-a01/provider'
python -B -X utf8 infer_map.py --checkpoint "$run/map-inference.pt" --metrics "$run/metrics.json" --result "$run/result.json" --data .runtime/full-a01 --layout $layout --roadnet $roadModel --real-manifest C:/survey/real-xy.jsonl --out C:/survey/fused-map-report.json
```

报告 `observations` 列出每帧原图 SHA256、真值、不确定度、候选坐标及误差。`candidate_set_metrics` 给出地图头五候选在 50 mm 内的命中率、Brier 分数和风险覆盖曲线。首候选接受统计应用冻结主机资格与阈值。报告的置信度仍来自仿真校准。

工程测试夹具将 `capture_source` 设为 `SYNTHETIC_FIXTURE`，输出状态为 `SYNTHETIC_SURVEY_FIXTURE`。正式导出读取同一 checkpoint 的实拍报告，并验证已冻结的研究目标字段，包括首坐标接受覆盖率和已接受帧的中位、P95 误差。当前冻结正式方案的主机门为 HOLD。

## 主机流程复核

本机使用冻结正式数据的 4 个留出位置，各取两个外观未扰动的视角。通过原模型 CLI 验证了正式融合、学得地图、RoadNet 语义、Pilot 直接回归和网格残差五条推理路径。8 帧均使用显式合成夹具标记。22 项主机测试覆盖实拍清单、导出门及既有板端验证器。实拍精度仍为 UNKNOWN。

复现时按归档 `manifest.jsonl` 中的 `sample_id` 从正式数据定位原 PNG，将字节复制到 `survey.json` 指定的相对路径，核对原图 SHA256。从冻结 Kaggle 输入归档恢复 `layout.json` 和 `roadnet.onnx`，并以验收 `result.json` 的 `semantic_inputs` 校验。验证记录保存了执行命令、输入与输出身份。

[夹具与五路报告核验](evidence/G6-survey-pipeline-20261117/verification.json) · [22 项主机回归](evidence/G6-survey-pipeline-20261117/host-regression.log) · [证据 SHA256](evidence/G6-survey-pipeline-20261117/SHA256SUMS.json)
