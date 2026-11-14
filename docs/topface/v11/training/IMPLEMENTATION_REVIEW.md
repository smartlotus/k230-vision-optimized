# Pilot 训练实现与独立审查

训练尚未执行，optimizer updates=0。本机只进行了已通过QA的73张train图/175实例的哈希、分组和prior分配检查（包括11个第五实例），以及2张真实合成图的完整模型forward/backward；所有参数梯度有限，没有optimizer.step。评估也仅在未训练模型的2张val图做接口联调，mAP/recall为0，不解释为模型质量结果。

实现入口：`research/topface/task_geometry.py`、`task_training.py`、`evaluate_task.py`、`train_experiment.py`。独立审查由只读子Agent完成，无云端或optimizer副作用。发现并修复：

1. padding空框在五目标截断前先过滤，NMS对可见内容裁切框执行，避免高分padding候选挤掉后面的真实目标。
2. GT prior分配先用最大二分匹配保证每个可表示目标至少一个主prior，再补到每GT3个；可表示目标冲突直接失败。所有目标保留在标签与分母中；不可观察或不足16px的目标单独报告，不按instance_id删目标。
3. 回归与颜色正样本按实例平衡，避免少prior的小目标权重更低。检测的六色logits使用硬标签sigmoid focal BCE；不冒称IoU质量QFL，无额外objectness。
4. Dataset构造时核对QA冻结的每个RGB/label SHA及scene/camera group；QA后漂移不能直接训练。
5. Pilot门增加有效椭圆预测覆盖率，防止99%椭圆缺失、只剩一个完美IoU而通过。近圆角度与有效角度分母分别记录，角度覆盖不足不能通过；空场FP和4/5目标完整召回也入门。
6. Overfit一旦当前模型满足全部sanity，保存该模型为best并停。不会因历史综合分数稍高却未通过sanity而保存错误checkpoint。

三个输入的精确letterbox、无半stride偏置prior、DFL l/t/r/b、物理圆心/conic center、长短轴/方向、颜色ID、flag损失mask均已显式固定。mask规则为：可见且valid的物理中心回归；隐藏中心仍保存真值并监督center_visible=false；conic/半轴按geometry_valid；方向额外按orientation_valid；geometry confidence为geometry_valid BCE，visibility为连续visible_fraction BCE。

首选Pilot320×192，640×360→320×180，再上下各6px；BGR、mean[103.53,116.28,123.675]/std[57.375,57.12,58.395]，仅一次归一化。Smoke的400mm完整目标短边中位19px（224×128约13.3px，384×224约22.8px）；该选择仍待Pilot及板端性能验证。

训练默认CUDA0、FP32，不安装/替换Kaggle已有Torch环境。16图overfit最多1000updates；两个Pilot对照各20epoch、相同数据/新头种子/超参/增强序列，前者从官方EMA初始化、后者随机；Pilot不继承overfit拟合权重。只用val选best，test不参与调参。保存best/last、optimizer/scaler、shuffle/Torch/CUDA RNG、配置/日志/TensorBoard及SHA。

当前checkpoint保存了恢复所需的部分状态，但**尚未实现并验证跨云会话的精确恢复入口**；不能称lossless resume。partial科学工作不会自动重放；若预算中断，将保留原checkpoint并单独实施/验证恢复。

02审计和03训练源码已写，准备脚本在实际Pilot compute/QA/Drive接收未通过时拒绝冻结可运行输入。运行时先Drive full-byte探测和持久run准入，再从Drive读取完整父归档、验证全部成员。完整产物留在云端，本机只收小JSON回执。尚未提交或运行02/03，不因源码存在就声称云训练完成。
