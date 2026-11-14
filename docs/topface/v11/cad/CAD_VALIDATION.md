# 原始 CAD 验证

使用用户原始 `k230_material_block_model_bundle.zip`，原始包 SHA-256 为 `642df0a28329ebe3adab0ca96888e5078b1a005dc7b9dbd345903e10274fc7f2`。Kaggle 输入以 `cad_bundle.bin` 保存相同 ZIP 字节，避免平台自动解包改变入口。

原生 Blender 4.2.9 实际导入单一融合网格：9761 顶点、19518 多边形。原 GLB 的坐标约定导致 Blender 导入后 60 mm 高度轴沿 Y；仅绕 X 修正 −90° 并应用坐标变换，未重新建模、拆分、重拓扑或改变尺寸。实际尺寸约 49.9997 × 49.9998 × 60.0000 mm。

在 z=60 mm、法线朝 +Z、半径不超过 15.02 mm 的条件下，选出 627 个顶面面片；总面积 706.8466 mm²，接近 π×15²=706.8583 mm²。其余面不写顶面 instance ID，但仍参与 RGB、遮挡和阴影。同次 Cycles 渲染通过材质索引输出五个顶面 mask 槽；没有把上圆柱侧壁标成顶面。

数值收据见 `geometry_validation.json`，实现见 `research/topface/native_cad.py`。100 场景实际渲染及 QA、人工抽查均通过，完整数据的 Drive 全字节读回收据见 `../reports/SMOKE_DRIVE_ACCEPTANCE.json`。这些结果证明本轮 CAD/合成标签一致性，不证明真实镜头、Kaggle GPU 或识别精度。
