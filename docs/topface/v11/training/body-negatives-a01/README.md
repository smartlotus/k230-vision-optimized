# 底座可见、顶面完全出框的训练负例

Pilot 的开发序列里，4张顶面完全出框但底座可见的图全部被误报为顶面。对原Full布局的独立几何检查发现，24,996个顶面全部与画面相交，原有部分截断样本不能覆盖这一情况。

本次使用原始融合 CAD，在本机 Blender 4.2.9 的 Cycles/OPTIX 实际生成120张独立负例，seed20261123，20个原空train相机组×6色；左右上下边各30张。未改变网格拓扑或原10,000场景档案。单帧工程验证用独立seed20261122，不混入bank。

所有120张图经过独立相机重投影和原生pass审计：顶面material-index mask全零，顶面边界至少离画面4px；object-index可见像素为823–21,365。每张标注仍保留1个物理实例，bbox=None、observable top=0；几何坐标不作为可见顶面的正监督。

训练只在原始world count=0的train槽位上，以25%冻结概率选择相同camera group的负例替换。其余输入沿用真实背景裁片策略。替换时跳过真实patch，随后仍使用同一光度退化；返回有效源scene/group/seed和原槽位，记录每次替换。原训练图数7,465和100epochs不变，val/test不启用bank。

已实际检查原smoke train上的584次配对读取：40次替换；其余544次图像张量、标签和元数据完全一致。替换后保留1个world实例、0个正prior。32张实际train校准资格中2次替换，12张配对原图字节一致。Full64张校准的冻结布局/RNG预检预计2次替换，实际执行时仍须报告真实次数。

bank archive的manifest/hash贯穿正式输入、checkpoint、续跑合同、导出、PTQ。只读独立审查确认未见heldout泄露或监督错误。新增强是否降低误检，必须等正式权重训练并复测后判断，当前0新增optimizer updates。

证据为 [QUALIFICATION.json](QUALIFICATION.json)、[DATAFLOW_QUALIFICATION.json](DATAFLOW_QUALIFICATION.json) 和 [SCENE_PLAN.json](SCENE_PLAN.json)。原生渲染、完整mask/标签/审计和过程日志另作云端归档。过程档案注明源码快照是渲染后收集，不冒充启动前已捕获的manifest。
