# 4DGaussians AutoEncoder 架构

这个修改版本的4DGaussians使用了与AEDeformable完全相同的AutoEncoder架构，替代了原有的HexPlane和低秩分解方法。

## 主要变化

### 1. 架构变更

**原始4DGaussians方法:**
- 使用HexPlane进行4D空间时间表示
- 低秩分解获得空间特征
- 空间特征插值后送入MLP解码

**新AutoEncoder方法:**
- 直接对位置和时间进行位置编码
- 通过MLP网络学习变形映射
- 与AEDeformable完全一致的网络结构

### 2. 核心类

#### `AutoEncoderDeformNetwork`
- 核心的变形网络，处理位置编码和时间编码
- 输出位置、旋转、缩放的变化量
- 支持Blender数据集的优化设置

#### `Deformation` 
- 主要的变形模块，替代原有的HexPlane方法
- 集成AutoEncoder网络
- 保持与原始API的兼容性

#### `deform_network`
- 顶层网络包装器
- 处理位置编码和参数管理
- 与训练pipeline集成

### 3. 新增参数

在`arguments/__init__.py`的`ModelHiddenParams`中添加了以下参数：

```python
# AutoEncoder架构参数
self.latent_dim = 64          # 潜在空间维度
self.is_blender = False       # 是否使用Blender数据集设置
self.use_autoencoder = True   # 使用AutoEncoder架构
self.encoder_hidden_dim = 128 # 编码器隐藏层维度
self.decoder_hidden_dim = 128 # 解码器隐藏层维度
self.latent_reg_weight = 0.01 # 潜在空间正则化权重
self.recon_weight = 1.0       # 重建损失权重
```

### 4. 训练修改

在`train.py`中添加了AutoEncoder正则化损失：

```python
if hasattr(hyper, 'use_autoencoder') and hyper.use_autoencoder:
    # 计算AutoEncoder正则化损失
    autoencoder_reg_loss = gaussians._deformation.compute_regularization_loss(
        means3D.unsqueeze(0), 
        scales.unsqueeze(0), 
        rotations.unsqueeze(0)
    )
    loss += autoencoder_reg_loss * hyper.latent_reg_weight
```

## 使用方法

### 1. 训练

```bash
# 使用默认AutoEncoder设置训练
python train.py -s path/to/dataset

# 自定义AutoEncoder参数训练
python train.py -s path/to/dataset --latent_dim 128 --latent_reg_weight 0.02
```

### 2. 渲染

```bash
# 渲染已训练的模型
python render.py -m path/to/trained/model --skip_train
```

### 3. 测试架构

```bash
# 运行AutoEncoder架构测试
python test_autoencoder.py
```

## 与AEDeformable的对比

| 特性 | AEDeformable | 4DGaussians AutoEncoder | 
|------|--------------|-------------------------|
| 编码方式 | 高斯参数编码 | 位置时间编码 |
| 网络结构 | 编码器-潜在空间-解码器 | 直接MLP变形网络 |
| 时间处理 | 潜在空间时间变形 | 位置编码时间输入 |
| 输出 | 参数重建 | 直接变形量 |

## 优势

1. **统一架构**: 与AEDeformable使用相同的网络结构思想
2. **简化实现**: 去除了复杂的HexPlane和低秩分解
3. **更好的时间一致性**: 直接的位置时间编码学习
4. **训练稳定性**: 标准的MLP网络训练更稳定

## 参数说明

- `latent_dim`: 控制网络的表达能力，更大的值可能带来更好的质量但训练更慢
- `latent_reg_weight`: 正则化权重，防止过拟合
- `multires`: 位置编码的频率数量，影响细节表现
- `is_blender`: 针对D-NeRF数据集的优化设置

## 注意事项

1. 这个实现完全移除了HexPlane相关的grid参数
2. 所有参数现在都是MLP参数，没有额外的grid存储
3. 时间编码直接集成在网络中，不需要额外的time embedding
4. 与原始4DGaussians的checkpoint不兼容，需要重新训练

## 文件结构

```
4DGaussians/
├── scene/
│   └── deformation.py          # 修改后的变形网络
├── arguments/
│   └── __init__.py            # 新增AutoEncoder参数
├── train.py                   # 修改后的训练脚本
├── test_autoencoder.py        # 架构测试脚本
└── README_AutoEncoder.md      # 本文档
```
