# 风险与未知

- UNKNOWN：当前板端身份/runtime新鲜读回、实时RGB888P channel/stride/DMA寿命、曝光/gain/WB/ISP、真实内参/畸变、近距离清晰度、工作区域毫米标定、完整链路FPS。
- 小固定镜头标称约70cm对焦，用户工作距离10–40cm；必须用真实参考图校准失焦，不能把清晰合成图当完成Sim2Real。
- 用户确认80–100°主工作范围，强透视超范围分组未声称覆盖。正常物块直立，不擅自增加侧躺训练样本。
- 当前随机背景/打印纹理/退化仍需实物校准；本地100 smoke与人工抽查是几何/外观门，不是模型质量门。
- 现有r4仅7批准图、17框，旧test已用；缺新scene/session盲测及真实椭圆/中心真值。
- NanoDet/PicoDet仅兼容性短测，不完整训练两个模型；新增六色/物理中心/椭圆head也需重新编译，原COCO图编译成功不能替代它。
- 旧Kaggle v8完成Drive115字节读回后在BlenderProc matplotlib backend安装冒烟失败；0渲染、0updates。用户已停止该旧运行，后续用新的run身份。
- 更新后cloud-direct-write允许Notebook直写或独立获授权云发布器；若选择新服务接收临时凭据，先核实该目的地授权。不得在Notebook/source/Git放凭据。
- V1.1引用的旧Execution Package ZIP曾被定位但当前路径已不可读；最新两份V1.1与原CAD包均可用，不把缺失包内容冒充已完整阅读。
