# 风险与未知

- UNKNOWN：当前板端身份/runtime新鲜读回、实时RGB888P channel/stride/DMA寿命、曝光/gain/WB/ISP、真实内参/畸变、近距离清晰度、工作区域毫米标定、完整链路FPS。
- 小固定镜头标称约70cm对焦，用户工作距离10–40cm；必须用真实参考图校准失焦，不能把清晰合成图当完成Sim2Real。
- 用户确认80–100°主工作范围，强透视超范围分组未声称覆盖。正常物块直立，不擅自增加侧躺训练样本。
- 当前随机背景/打印纹理/退化仍需实物校准；本地100 smoke与人工抽查是几何/外观门，不是模型质量门。
- 现有r4仅7批准图、17框，旧test已用；缺新scene/session盲测及真实椭圆/中心真值。
- NanoDet/PicoDet 的原始图和三个输入尺寸已完成兼容性短测；实际 Pilot 已训练 NanoDet TopFaceNet，导出的50通道任务头也已编译并执行 Simulator。唯一正式架构采用 NanoDet320×192，正式 Full 训练尚待数据审计，不能把 Pilot 当成正式交付模型。
- Pilot 合成验证指标与真实验证分开。r4 的两张红色 val 图共2个GT，浮点模型2TP/3FP/0FN，量化 Simulator 为2TP/5FP/0FN；不能将合成mAP或两张图的召回率解释为六色真实准确率。[浮点原始结果](../reports/PILOT_REAL_VAL_A02.json)、[KModel原始结果](../reports/PILOT_REAL_VAL_KMODEL_A02.json)。
- 12张 train 留出图的30个ORT/Simulator检测全部匹配，但方向有效性有6次有效→无效、1次反向变化，不能判定全部几何数值验收通过。最终冻结模型需分别报告浮点和KModel的完整合成test指标；方向近圆退化和量化影响保留独立分母。[转换报告](../reports/PILOT_CONVERSION_A02.json)。
- 正式训练已准备64个真实train背景裁片，针对机械夹具/线缆误检。它们仅在仿真目标框之外增强背景；真实图的中心/椭圆仍为UNKNOWN。此策略已验证数据隔离及目标像素/标签保持，改善误检的效果尚未训练验证。[策略与证据](../training/REAL_BACKGROUND_POLICY.json)。
- 2026-11-15 的 Pilot 开发序列复测中，浮点模型在顶面完全出框、较大底座仍露出画面的4帧全部误检为顶面；同组28个可见顶面全部检出且颜色正确。此诊断序列仅175mm/90°/蓝黑两色，已用于开发评估，不能再称未见盲测；最终 Full 的独立合成test仍另行冻结。保留底座误检问题，不能假定增加训练轮数会自动消除。[原始记录](../reports/pilot-accuracy-20261115-a01/SEQUENCE_ONNX.json)。
- 正式训练的CUDA双线性插值反向存在上游非确定性；epoch边界记录RNG/optimizer不等于已验证bitwise CUDA无损恢复。计划单会话100epochs，实际停止/续接时必须保留此限制。
- 旧Kaggle v8完成Drive115字节读回后在BlenderProc matplotlib backend安装冒烟失败；0渲染、0updates。用户已停止该旧运行，后续用新的run身份。
- 用户已授权同范围独立私有GitHub发布器：仅临时短时Drive access token、签名源URL及单文件上传会话，终态删除Secret；不发送refresh token/client secret/Kaggle凭据。Full04原上传中断后已在相同文件ID/会话续传并完整读回，不重渲染；原失败记录保留在attempts目录。所有完整过程档案留云，主机收取受限报告及最终模型文件。
- V1.1引用的旧Execution Package ZIP曾被定位但当前路径已不可读；最新两份V1.1与原CAD包均可用，不把缺失包内容冒充已完整阅读。
