# 正式训练的完整 epoch 续跑

默认仍为一个逻辑实验、100 epochs、单会话运行。会话因运行预算在完整 epoch 后停止时，继续原实验，不新建对照实验，不重置累计轮数和 optimizer。

只有 Kaggle 原版本完成、原 archive 已通过 Drive 全字节读回，且结果为 `FORMAL_SESSION_BOUNDARY`、`resume_eligible=true` 才进入本路径。

1. 用既有 `collect_compact.py --training --publisher-route` 收取该确切版本的根 RESULT、formal_full/RESULT 和 BEST_METRICS，保留原字节。
2. 用 `accept_full_downstream.py --phase formal-session` 接受该会话。`--output` 为会话接受记录，`--model-output` 为新的 resume parent；两者使用新文件名，不能覆盖历史记录。
3. 下一次 `prepare_full_downstream.py --phase formal` 指向这个 parent，`--session` 加一，`--max-epochs` 保持原值；继续提供同一已审核真实背景 bank 和 body-only bank 及其 qualification。revision/run_id 是新的会话交易身份，原 `experiment_id` 保持不变。
4. 继续原本的私有 API Save、精确版本观察、云端发布和全字节校验。只有实际完成至少100 epochs 的 `FINITE_TRAINING_COMPLETE` 才走 `--phase formal` 接受为最终模型。

resume parent 指向 `last.pt`，并单独保存 `best.pt` 哈希。best 可以来自较早 epoch。parent.archive 保存 Kaggle 原始文件名，publication 保存 Drive 的带摘要对象名，两者不能混用。

本地 preparation 会拒绝变更原实验、数据清单、累计 epoch、背景增强、学习源码或 NanoDet 源码/预训练输入。NanoDet 源码 ZIP 固定成员时间戳，避免另一台电脑的文件时间导致无意义指纹变化。训练时另校验 schedule、runtime、模型参数、optimizer、RNG 和数据顺序；runtime 记录 PyTorch/CUDA/cuDNN、Python、NumPy、OpenCV。

`FORMAL_BUDGET_STOP_MID_EPOCH`、失败、无进展及缺完整归档的状态不能冒充干净会话边界。Working 中已保留安全 epoch 的 loose checkpoints，但强制抢占后直接从 loose 文件恢复的路径尚未验证。上游 CUDA 双线性插值反向也未证明 bitwise 可复现。这里的本地证据门测试不等于实际 GPU 续训验收。

测试证据见 [FORMAL_SESSION_QUALIFICATION.json](FORMAL_SESSION_QUALIFICATION.json)。
