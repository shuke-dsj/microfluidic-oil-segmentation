# 预训练权重

下载：`python tools/download_weights.py`，或从 [v1.0.0 Release](https://github.com/shuke-dsj/microfluidic-oil-segmentation/releases/tag/v1.0.0) 下载 `attention_resunet.pth`。

| 属性 | 内容 |
|---|---|
| 源文件 | 原研究目录 `resunet_output/best_resunet.pth` |
| 格式 | PyTorch `state_dict`；使用 `weights_only=True` 加载 |
| 大小 | 59,935,113 bytes |
| SHA-256 | `ceec24215bdc38dc0a3cffc34cc6f43a9e25701635a0d2847441f92aa2bc4d12` |
| 输入 | RGB 三色结构图，`ToTensor` 转换到 [0,1]；不额外做 ImageNet 标准化 |
| 输出 | 6 类像素 logits，类别顺序见 `oilseg.labels` |

该权重对应既有 Attention ResUNet；未重新训练。原始训练运行未保存完整环境、划分清单及权重与 CSV 的联合运行标识，因此不声称历史训练可逐位复现。模型结构保留原始 `base_model` 注册方式和未参与前向的分类头，确保 strict loading。公开权重不意味着公开训练图像或标签。
