# 合成演示

`synthetic_input.png` 由 `tools/prepare_demo.py` 通过几何图形生成，模拟红色油相、灰色孔道与白色岩石的输入格式。它不来自实验室，不是评测样本，不用于说明模型精度。

```bash
python tools/prepare_demo.py
oilseg predict --images examples/synthetic_input.png --weights weights/attention_resunet.pth --output outputs/demo --device cpu
```

这个示例用于验证安装、模型加载、动态 padding 与导出链路。由于合成图并不代表真实实验分布，不应把其预测作为科研结论。
