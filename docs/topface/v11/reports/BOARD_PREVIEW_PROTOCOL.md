# 正式权重优先上板试用

2026-11-15 用户要求保持当前训练及原规划不变，先提供最新 checkpoint 的可用 KModel；只读探测原正式 v1 时输出列表为空、best/last 返回404。用户随后明确选择“等本轮正式权重，完成后优先生成试用包”，不使用旧 Pilot 替代。

新增独立接续 `research/topface/continue_board_preview.py`。它等待原 `formal-a01 v1` 完成、Drive 全字节接受及验证集选择权重冻结；原训练、07b 完整转换评估及现有控制器不变。新增 CPU 任务 ref 为 `lotus30/k230-topface-v11-07c-formal-board-preview`，run、输入、锁、输出和发布对象均与原任务隔离。此额外导出不进行训练。

来源必须匹配原 ref、ID145144940、v1、100轮及选定 best SHA，并复用原 formal_evidence 核对。完整数据索引和分片接受身份保持不变；只提取 train 图像，保留正式训练的背景增强及 body-only bank。使用已验证导出器和 nncase2.11，执行实际图像 PyTorch/ONNX 配对、64张train PTQ及12张train ORT/Simulator配对。

12张配对记录 raw 误差、颜色、数量、中心和椭圆变化；编译器只对 shape/finite 等合同硬失败。必须阅读实际数值，不把编译完成解释为量化精度全面通过。原定934张独立静态测试、1872帧运动评估及真实图评估继续按原流程执行。

试用包保持独立状态 `FORMAL_BOARD_PREVIEW_PACKAGE_VERIFIED`，明确 scope、完整独立测试 pending 和 native NOT_RUN。包含 best.pt、ONNX、KModel、契约、板测脚本、配对报告和逐文件SHA；另生成ZIP，回执记录ZIP的大小和SHA。原计划的最终包不会被该试用包覆盖。两次编译的KModel是否字节一致，须产出后实际比较。

默认目标：`C:/Users/28389/Desktop/k230-project/TopFaceNet-V1.1-Formal-Preview-a01` 及同名ZIP。目录只有在真实工件通过检查后才创建，不以路径预定当作已交付。

连接板后按 [独立安装流程](BOARD_INSTALL_PROTOCOL.md) 用新的 `/sdcard/tf_...` 目录。上传器接受明确标记的 preview 状态，拒绝不足100轮或隐藏独立测试状态的包。安装只校验文件，不启动模型或运动；实际KPU配对、完整FPS和新真实盲测另留证据。

当前资格仅限主机fixture和独立代码审查。见 [preview主机测试](BOARD_PREVIEW_HOST_TESTS.log)、[上传器preview测试](BOARD_DEPLOYMENT_PREVIEW_HOST_TESTS.log)。没有把fixture权重当作真实ONNX或KModel；没有访问实际硬件。实际运行结果以随后07c保存版本、Drive接受、部署包及原生报告为准。
