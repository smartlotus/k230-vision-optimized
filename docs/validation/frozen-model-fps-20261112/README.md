# 已训练小模型的板端速度对照

使用原来四次 optimizer 更新的训练产物，没有重新训练：

- checkpoint `best.pt`: `6de5c9b6...`
- ONNX: `df9c6390...`
- 本轮转换的 kmodel: `59fd397f...`，3,284,632字节（约3.13 MiB）

四张冻结训练组原图的纯 KPU 测试均运行20次，去掉3次预热：
平均 **95.42 FPS**，单次 KPU 平均约 **10.48 ms**，输出拷贝约0.02–0.04 ms。
这项结果只表示固定224×224 uint8输入已经在KPU上运行，不包含摄像头、AI2D、后处理、显示或准确率。

同一批图用当前传统CPU几何链路逐图测试，单图约3.91–4.72 FPS；每图测量2次，产生3个材料候选。
两条路径的输出任务不同：CPU输出椭圆几何并执行纸靶检测，单类模型仅产生顶面bbox候选。
这些结果支持继续验证KPU路线，但不能据此宣称完整识别速度提高了相同倍数。

AI2D＋KPU及后处理的多次隔离测试分别停在输入读入、模型加载或AI2D首次执行之后，通信中断，均未取得完整结果。
此前关于AI2D中间tensor读回导致中断的判断只是待验证假设；不执行该读回的测试也出现中断，根因仍未知。
direct-ai2d.log、mmz-ai2d.log、prepared-postprocess-partial.log保留独立失败步骤，未回填为成功。
没有把纯KPU速度写成完整相机FPS，也没有把四张训练图上的候选写成识别准确率。
最终CanMV仅报告connected缓存为true、ready=false；当前脚本状态和内存清理未知。此后没有继续启动脚本。

下一步在板卡稳定且确认就绪/空闲后测两段：

1. 先沿用成功的224输入路径，只增加轻量后处理；再接AI2D，避免无必要的中间输入tensor CPU读回。
2. 摄像头640×360 → AI2D → KPU → 轻量后处理，关闭显示，测端到端FPS。

K230官方流程也是AI2D预处理后进入KPU，并建议复用AI2D输出tensor作为模型输入以减少中间拷贝；模型加载后需要显式释放对象和内存池。[Kendryte nncase_runtime使用说明](https://www.kendryte.com/k230_canmv/en/main/example/ai/nncase_usage.html) [K230 AI开发文档](https://www.kendryte.com/k230_canmv/en/main/ai_dev_doc.html)

主机完整回归478项通过，无跳过，见host-tests.json和host-tests.log。
本分支继承native-fps-20261112中已提交的live_optimized.py兼容修复，因此原始导入快照SHA验证会报告该文件变化。
docs/UPSTREAM_IMPORT.json保持首次导入身份不变；本次回归由tools.export_test_report执行，并记录当前测试源码SHA。
工作草稿中一度采用可变current_sha256覆盖导入校验，现已撤回，未作为已验证的来源发布。
