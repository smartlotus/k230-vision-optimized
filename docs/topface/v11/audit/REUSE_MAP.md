# 可复用代码地图

| 入口 | 复用范围 | 不能继承的结论 |
|---|---|---|
| k230/camera_manager.py:65-145 | 已有Sensor生命周期、尺寸校验和有界snapshot | 实时RGB888P通道/stride未验证 |
| tools/material_preprocess.py:8-26 | RGB565主机参考、letterbox、单次归一化 | 新模型尺寸/颜色/mean/std需另定 |
| tools/kpu_pipeline_probe.py:137-257 | backing存活、显式绑定、readinto、资源释放 | 冻结图FPS不等于camera-to-display |
| tools/material_dataset.py:76-196 | 审核/uncertain门、scene分组、去重、train-only PTQ | 旧单类模型形状门不可套新模型 |
| tools/probe_nncase_normalization.py | Windows plugin/DLL、CompileOptions、Simulator方法 | 小常量图不是任务模型兼容性 |
| research/topface/native_cad.py | 已验证原始单网格、轴向修正和顶面材质索引 | 原CAD几何不拆分 |
| research/topface/render_native.py | 原生bpy、多实例、相机投影校验、同次render的材质Mask | 本地通过不等于Kaggle/GPU通过 |
| research/topface/audit_v11.py | 物理中心/conic中心分离、可见框、5实例、空场、相机分组 | 合成QA不是真实识别率 |
| research/topface/cloud_store.py | Drive全字节SHA读回、唯一对象名 | CLI Save不保证运行时Secret关联 |
