# EuroSAT Remote Sensing Image Classification (MLP)

本项目是针对 **EuroSAT RGB** 数据集的遥感图像分类实现，使用纯 **NumPy** 手写构建了一个三层多层感知机（MLP）。通过网格搜索优化、Z-Score 标准化和 6 倍数据增强，模型在 10 类地物分类任务中取得了理想的性能。

## 📁 项目结构

```text
.
├── 深度学习与空间智能_HW1.pdf   # 完整实验报告 (理论推导与结果讨论)
├── code/                       # 核心源代码
│   ├── main.py                 # 项目主入口
│   ├── model.py                # MLP 模型实现 (Forward/Backward)
│   ├── train.py                # 训练循环、优化器与学习率调度
│   ├── dataloader.py           # 数据加载与增强逻辑 (旋转与翻转)
│   ├── hyperparameter_search.py # 自动网格搜索脚本
│   ├── visualize.py            # 权重可视化与错例分析
│   ├── eval.py                 # 基础评估模块
│   └── grid_visualization.py   # 网格搜索结果绘图工具
├── logs/                       # 实验日志
│   ├── grid_search_summary.csv # 54 组超参数对比汇总
│   └── detailed_curves/        # 各实验组的详细 Loss/Acc 数据
└── plots/                      # 实验可视化结果
    ├── train_loss.png          # 最终模型训练收敛曲线
    ├── grid_plot.png           # 54 组超参数对比矩阵图
    ├── confusion_matrix.png    # 最终模型混淆矩阵
    ├── weight_visualization.png # 第一层权重特征可视化
    └── error_analysis.png      # 预测错误样本分析图
```

## 🚀 核心技术实现
1. 纯 NumPy 框架实现完全不依赖 PyTorch 或 TensorFlow 等深度学习框架。手动推导并实现：前向与反向传播：基于链式法则实现三层神经网络。激活函数：支持 ReLU、Tanh 和 Sigmoid 的前馈计算与导数回传。优化器：实现带动量 (Momentum) 的 SGD，加速收敛并减少震荡。
2. 数值稳定性与预处理Z-Score 标准化：将原始图像像素从 [0, 255] 映射为均值为 0、标准差为 1 的分布，有效解决了训练初期的 Loss 爆炸问题。稳定版 Softmax：通过减去最大值防止指数运算溢出，确保交叉熵损失计算的稳定性。
3. 数据增强 (Data Augmentation)实现了 6 倍数据增强方案，包括随机旋转（90°, 180°, 270°）和水平/垂直翻转。实验结果表明，数据增强使验证集准确率提升了约 5%-8%。
4. 自动化网格搜索 (Grid Search)系统性评估了以下维度的超参数组合（共 54 组）：学习率 (LR): [0.005, 0.001, 0.0005]隐藏层维度 (HD): [512, 1024, 2048]权重衰减 (WD): [0.001, 0.0005]激活函数: [ReLU, Tanh, Sigmoid]

## 📊 实验分析摘要
1. 训练表现与收敛性在最优配置（LR=0.0005, HD=1024, ReLU）下，验证集准确率达到 73.78%。收敛特征：通过在第 20 和 40 轮执行 Step Decay（学习率减半），Loss 曲线呈现明显的阶梯式下降，证明了动态学习率对精细化权重更新的必要性。
2. 权重空间模式观察通过对第一层权重 $W_1$ 进行可视化（如 Neuron 320, 498, 94 等），发现：光谱检测：模型自发学习到了颜色过滤器。如 Neuron 320 表现出强绿色偏好（针对森林）；Neuron 656 表现出深蓝色偏好（针对水体）。几何感知：Neuron 498 展示了斜向纹理模式，反映了 MLP 具备捕捉道路、河流边缘或田地边界等线性几何特征的初步能力。
3. 错例讨论 (Error Analysis)错例主要集中在 Highway 与 River（形状相似）以及 Pasture 与 Forest（颜色相似）之间。结论：MLP 缺乏空间局部感受野，主要依赖全局光谱统计。这证明了在处理纹理相似地物时，引入具有平移不变性的卷积结构（CNN）是进一步提升性能的关键。

## 🛠️ 运行指南
1. 环境准备
  Python 3.8+
  必需库：numpy, pandas, matplotlib, pillow
2. 执行步骤B
# 切换至代码目录
cd code

# 运行完整 Pipeline (预处理 -> 网格搜索 -> 训练最优模型 -> 评估与可视化)
python main.py

# 针对已有的权重文件 (.pkl) 进行可视化分析
python visualize.py
cd code

# 运行完整 Pipeline (预处理 -> 网格搜索 -> 训练最优模型 -> 评估与可视化)
python main.py

# 针对已有的权重文件 (.pkl) 进行可视化分析
python visualize.py
📑 引用与致谢数据集: EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification课程: 深度学习与空间智能 HW1
