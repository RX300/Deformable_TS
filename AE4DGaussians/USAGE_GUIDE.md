# Static/Dynamic Gaussian Separation - Usage Guide

## ? 新系统特色

我们已经完成了一个完整的静态/动态高斯分离系统，通过编解码架构网络训练出一个 (N,1) 的二分类结果：
- **0 = 静态高斯** (不进行变形)
- **1 = 动态高斯** (应用时序变形)

## ? 核心文件结构

```
AE4DGaussians/
├── scene/gaussian_model.py          # 重新设计的高斯模型（核心）
├── gaussian_renderer/__init__.py    # 更新的渲染函数
├── utils/static_dynamic_losses.py   # 专用损失函数
├── static_dynamic_example_v2.py     # 完整训练示例
└── training_integration_v2.py       # 训练集成指南
```

## ? 快速开始

### 1. 直接运行示例
```bash
cd d:\MyProjects\Deformable_TS\AE4DGaussians
python static_dynamic_example_v2.py
```

### 2. 查看核心功能
```python
# 检查分类结果
stats = gaussians.get_static_dynamic_stats()
print(f"静态点数: {stats['static_count']}")
print(f"动态点数: {stats['dynamic_count']}")
print(f"分类熵: {stats['classification_entropy']:.3f}")

# 获取分离后的属性
deformed_attrs = gaussians.get_deformed_attributes(time=current_time)
# 自动应用: 静态点保持不变，动态点进行变形
```

## ? 系统架构详解

### StaticDynamicClassifier 网络
- **输入**: 原始高斯属性 (xyz, features, opacity, scaling, rotation)
- **输出**: (N, 1) 二分类概率
- **架构**: Encoder → Latent → Decoder → Sigmoid

### 渲染流程
1. 网络预测每个点的静态/动态概率
2. 二值化: P > 0.5 → 动态(1), P ≤ 0.5 → 静态(0)
3. 动态点应用时序变形网络
4. 静态点保持原始位置
5. 合并渲染结果

### 损失函数组合
- **分类清晰度损失**: 鼓励明确的0/1分类
- **时序一致性损失**: 防止分类结果剧烈跳变
- **运动监督损失**: 基于位置变化指导分类

## ? 训练监控

```python
# 实时统计
stats = gaussians.get_static_dynamic_stats()
print(f"""
当前分类状态:
- 静态高斯: {stats['static_count']} ({stats['static_ratio']:.1%})
- 动态高斯: {stats['dynamic_count']} ({stats['dynamic_ratio']:.1%})
- 分类确定性: {(1-stats['classification_entropy']):.1%}
""")
```

## ?? 关键参数

```python
# 分类网络参数
hidden_dim = 128           # 隐藏层维度
classification_threshold = 0.5  # 二分类阈值

# 损失权重
sparsity_weight = 0.01          # 分类稀疏性
temporal_consistency_weight = 0.05   # 时序一致性
motion_supervision_weight = 0.02     # 运动监督
```

## ? 与现有代码集成

只需要修改3个地方：

### 1. 导入新模块
```python
from utils.static_dynamic_losses import static_dynamic_classification_loss
```

### 2. 修改渲染调用
```python
# 原来
render_pkg = render(viewpoint, gaussians, pipe, background)

# 现在一样的调用，内部自动处理分离
render_pkg = render(viewpoint, gaussians, pipe, background)
classification_probs = render_pkg.get("static_weights")  # 获取分类结果
```

### 3. 添加分类损失
```python
total_loss = reconstruction_loss + 0.1 * classification_loss
```

## ? 期望效果

- **静态背景**: 分类为静态(0)，不进行变形，保持稳定
- **运动物体**: 分类为动态(1)，应用时序变形网络
- **边界清晰**: 高分类确定性，避免模糊的中间状态
- **时序稳定**: 分类结果在相邻帧间保持一致性

## ? 测试验证

运行 `static_dynamic_example_v2.py` 可以看到：
1. 网络架构初始化成功
2. 分类概率分布合理
3. 静态/动态统计符合预期
4. 训练损失收敛良好

现在您可以开始使用这个新的静态/动态分离系统了！
