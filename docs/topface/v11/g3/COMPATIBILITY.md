# G3 兼容性短测

本轮均为主机工程验证，optimizer updates=0、板端未运行、任务精度未验收。编译与 Simulator 执行不能当作数值精度或 K230 FPS 通过。

## 两个官方原始 COCO 模型

| 模型 | 官方原始 ONNX SHA-256 | KModel 字节 | 编译用时 |
|---|---|---:|---:|
| NanoDet-Plus-m | `4f12723cce3d48e47ca92cb925ba74d97a965c069208edca660bbb9f7ce2c610` | 1,680,960 | 30.812 s |
| PicoDet-XS | `e83deca19863b127e702ae340ed7f7455334471886f0251413be459142a08aec` | 1,063,960 | 37.029 s |

来源：[NanoDet 官方 release](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx)、[Paddle 官方模型](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_xs_320_coco_lcnet.onnx)。两个原始模型经 ONNX checker、ORT、nncase/nncase-kpu 2.11.0 和 Simulator。Nano 输出 `[1,2125,112]`，Pico 为四组分类及四组 DFL tensor，总输出都为 952,000 字节。

保留的最终 IR 显示：Nano 98 个 K230 融合函数调用、71 个未融合 Call（主要 Slice/Reshape/Concat/GetItem/Squeeze）；Pico 41 个融合函数、8 个未融合 Call，其中 4 个 Sqrt 处理共 170000 float 元素。两者最终 IR 没有普通 Conv2D 残留，GNNEConv2D 分别 94/98。调用节点比例不是算子、MAC 或实际耗时比例。所有 compile.text.asm 均为空，mac.csv 的 Time 均为0；不能计算真实 KPU/CPU 耗时占比。

仅五种人工工程 fixture 校准下，Nano 的 Simulator relative-L2 约0.166–0.190，Pico 约0.243–1.154。没有任务数值通过结论。原始结果与可复查分区分别位于 `nanodet-original-coco/` 和 `picodet-original-coco/`。

## NanoDet 预训练及任务头

优先实现 NanoDet 项目任务头，理由是符合候选顺序，且原图没有 Pico 的 Sqrt 回退；不是最终架构冻结。使用 [NanoDet 固定 commit](https://github.com/RangiLyu/nanodet/tree/be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0) 的7个未修改 Apache-2.0 模块，来源逐文件 SHA 与 LICENSE 在 `research/topface/vendor/nanodet_core/`。

[官方 checkpoint](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320_checkpoint.ckpt) SHA-256 为 `f4c6080f3ef35a64c3030d559438b436462f227ce29700bca71327203e42dea2`。遵循官方 loader 选择 avg_model 权重，644 个目标 tensor 全部加载，243 个训练专用 tensor 明确忽略。无需安装 Lightning：受限读取器只允许已检查的5个序列化全局对象，其中历史 callback 类仅作不执行的字典键。原始80类前向与官方 ONNX 的三个输入比较通过，证明适配器与原模型相符。

项目头保留 ShuffleNetV2/GhostPAN/四级 DWConv tower/DFL，不改成 fallback 自定义 detector。输出50通道：六色质量 logits、32个bbox DFL logits、物理圆心偏移、center_valid/visible、椭圆半轴/方向、geometry/visibility quality，以及**单独的 conic center 偏移**。物理中心与 conic center 不混用。分类和几何新通道仍为随机初始化，bbox DFL 继承原权重，未训练。

224×128、320×192、384×224 的原始头与项目头均实际导出并通过 PT→ORT，最大绝对差低于7×10⁻⁵。严格320×180在 FPN 拼接出现尺寸不匹配，使用保持宽高比的320×192 letterbox；不拉伸图像。

新增头三个尺寸的 nncase 编译/Simulator 结果见各自 RESULT；使用16张已通过 QA 的 synthetic **train** 图做工程校准，另4张 train 图作配对输入。此为未训练头的可编译性检查，不是最终模型 PTQ。正式训练后必须重新用 train-only 校准、分通道数值容差和解码结果验收，随后板端实测延迟/内存再冻结最终模型。
