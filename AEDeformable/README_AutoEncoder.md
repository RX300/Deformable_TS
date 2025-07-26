# AEDeformable - AutoEncoder Enhanced Deformable 3D Gaussians

这是对原版 Deformable 3D Gaussian Splatting 的改进版本，引入了编码器-解码器架构来学习更高效的变形表示。

## 主要改进

### 1. 自编码器架构
- **编码器**: 将高斯参数（位置、旋转、缩放）编码到低维潜在空间
- **潜在变形网络**: 在潜在空间中学习时间变形
- **解码器**: 将变形后的潜在表示解码回参数变化

### 2. 优势
- **更紧凑的表示**: 在低维潜在空间中学习变形，减少参数数量
- **更好的泛化**: 潜在表示可能捕获更高级的语义信息
- **正则化**: 通过瓶颈层减少过拟合

## 使用方法

### 基础训练（原版网络）
```bash
python train.py -s <数据集路径> -m <输出路径>
```

### 自编码器训练
```bash
python train.py -s <数据集路径> -m <输出路径> --use_autoencoder --latent_dim 64
```

### 使用便捷脚本
```bash
python train_autoencoder.py -s <数据集路径> -m <输出路径>
```

## 命令行参数

### 自编码器相关参数
- `--use_autoencoder`: 启用自编码器变形网络
- `--latent_dim`: 潜在空间维度（默认：64）

### 其他参数
- `-s, --source_path`: 数据集路径
- `-m, --model_path`: 输出模型路径
- `--iterations`: 训练迭代次数（默认：40000）
- `--warm_up`: 预热迭代次数（默认：3000）

## 网络架构详解

### 1. GaussianEncoder
```python
Input: [N, 10]  # xyz(3) + rotation(4) + scaling(3)
Hidden: [N, 128] -> [N, 128]
Output: [N, latent_dim]
```

### 2. LatentDeformNetwork
```python
Input: [N, latent_dim] + time_embedding
Hidden: MLP with skip connections
Output: [N, latent_dim]  # 变形后的潜在表示
```

### 3. GaussianDecoder
```python
Input: [N, latent_dim]
Hidden: [N, 128] -> [N, 128]
Output: [N, 10]  # d_xyz(3) + d_rotation(4) + d_scaling(3)
```

## 正则化损失

系统会自动添加两种正则化损失：

1. **潜在空间正则化**: 鼓励潜在代码接近标准正态分布
2. **重建正则化**: 确保编码-解码的一致性

## 示例结果

训练过程中会输出：
- 渲染损失（L1 + SSIM）
- 正则化损失（如果启用自编码器）
- PSNR 指标

## 注意事项

1. 自编码器模式需要更多的GPU内存
2. 训练时间可能稍长，但推理速度相似
3. 建议从较小的 `latent_dim` 开始尝试（如32或64）
4. 如果遇到内存问题，可以减少 `latent_dim` 或使用 `--load2gpu_on_the_fly`

## 对比原版

| 特性 | 原版 | 自编码器版本 |
|------|------|-------------|
| 输入 | 位置坐标 | 完整高斯参数 |
| 学习空间 | 3D坐标空间 | 潜在空间 |
| 参数数量 | 更多 | 更少（潜在空间） |
| 内存使用 | 较少 | 稍多 |
| 泛化能力 | 标准 | 可能更好 |

## 文件结构

```
AEDeformable/
├── train.py                    # 主训练脚本
├── train_autoencoder.py        # 便捷训练脚本
├── utils/
│   └── time_utils.py           # 包含所有网络定义
├── scene/
│   └── deform_model.py         # 变形模型包装器
└── arguments/
    └── __init__.py             # 命令行参数定义
```
