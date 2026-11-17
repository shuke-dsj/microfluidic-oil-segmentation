# 使用与验证

## 桌面工作流

1. 安装 `.[desktop]` 并下载 Release 权重。
2. 执行 `python apps/desktop.py`。
3. 点击“导入核心模型”，选择 `weights/attention_resunet.pth`。
4. 导入经过规则预处理的三色图；需要精度评价时额外导入同尺寸 RGB 真值标签。
5. 执行单图分析或批量分析，查看油类内部相对占比和可选的 IoU；报告写到当前工作目录的 `outputs/reports/`。

`*_areas.csv` 的 `image_fraction` 分母为整张图像像素数；油类的 `oil_fraction` 分母为所有油类像素数。二者是二维像素面积占比，不代表真实三维饱和度或油体积。无油时油类内部占比留空。

预处理工具执行 `python tools/preprocess_gui.py`，先选择同一实验结构对应的基底图，再调整颜色距离阈值，选择实验图目录并批量导出。输入基底与实验图应同尺寸、已配准；该步骤不替代图像配准或标签标注。

## 注意力诊断

使用本地数据，安装 `.[analysis]` 后运行：

```bash
python tools/analyze_attention.py
```

默认读取 `data/test/images`、`data/test/masks` 和 `weights/attention_resunet.pth`，输出到 `outputs/attention_gate_analysis`。可用环境变量 `OILSEG_TEST_IMAGES`、`OILSEG_TEST_LABELS`、`OILSEG_WEIGHTS` 改写位置。该分析脚本要求图像和标签文件名（含扩展名）完全一致。它沿用历史诊断绘图逻辑，可能生成较大的 TIFF/SVG，仅在本地使用。输出包含实验图像，不应公开提交。

## 复核命令

本次对原私有目录执行的命令等价于：

```bash
oilseg evaluate --images data/test/images --masks data/test/masks --weights weights/attention_resunet.pth --output reports/reproduced --device cuda
```

默认 FP32。`--amp` 只在 CUDA 上启用混合精度，可能造成少量边界像素差异；应在报告中保留该参数。CUDA OOM 时可尝试 CPU 或减少输入尺寸，但缩放会改变实验口径，不能直接与原尺寸指标比较。

## 验证状态

本次完成：源文件编译；10 项自动测试；公开权重 strict load；私有 10 图 FP32 评估；CPU 合成推理；合成数据训练 smoke；桌面应用合成输入加载和推理检查。CI 配置随代码提交，远端状态以 GitHub Actions 实际运行结果为准。

未重跑完整 50 轮训练，未对全部 GUI 交互逐项验收，也未重新生成历史高分辨率科研图。
