#!/usr/bin/env python3
"""
测试自编码器变形网络的基本功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.time_utils import (
    GaussianEncoder, 
    GaussianDecoder, 
    LatentDeformNetwork, 
    AutoEncoderDeformNetwork
)

def test_individual_components():
    """测试各个组件的基本功能"""
    print("Testing individual components...")
    
    # 模拟数据
    batch_size = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模拟的高斯参数
    xyz = torch.randn(batch_size, 3, device=device)
    rotation = torch.randn(batch_size, 4, device=device)
    scaling = torch.randn(batch_size, 3, device=device)
    time_input = torch.randn(batch_size, 1, device=device)
    
    # 测试编码器
    print("  Testing GaussianEncoder...")
    encoder = GaussianEncoder(input_dim=10, latent_dim=64).to(device)
    gaussian_params = torch.cat([xyz, rotation, scaling], dim=-1)
    latent_code = encoder(gaussian_params)
    print(f"    Input shape: {gaussian_params.shape}")
    print(f"    Output shape: {latent_code.shape}")
    assert latent_code.shape == (batch_size, 64)
    
    # 测试解码器
    print("  Testing GaussianDecoder...")
    decoder = GaussianDecoder(latent_dim=64, output_dim=10).to(device)
    decoded_params = decoder(latent_code)
    print(f"    Input shape: {latent_code.shape}")
    print(f"    Output shape: {decoded_params.shape}")
    assert decoded_params.shape == (batch_size, 10)
    
    # 测试潜在变形网络
    print("  Testing LatentDeformNetwork...")
    latent_deform = LatentDeformNetwork(latent_dim=64, is_blender=False).to(device)
    deformed_latent = latent_deform(latent_code, time_input)
    print(f"    Input shapes: {latent_code.shape}, {time_input.shape}")
    print(f"    Output shape: {deformed_latent.shape}")
    assert deformed_latent.shape == (batch_size, 64)
    
    print("  ✓ All individual components working correctly!")
    return True

def test_autoencoder_network():
    """测试完整的自编码器网络"""
    print("Testing complete AutoEncoder network...")
    
    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建网络
    ae_network = AutoEncoderDeformNetwork(latent_dim=64, is_blender=False).to(device)
    
    # 创建模拟数据
    xyz = torch.randn(batch_size, 3, device=device)
    rotation = torch.randn(batch_size, 4, device=device)
    scaling = torch.randn(batch_size, 3, device=device)
    time_input = torch.randn(batch_size, 1, device=device)
    
    # 前向传播
    d_xyz, d_rotation, d_scaling = ae_network(xyz, rotation, scaling, time_input)
    
    print(f"  Input shapes: xyz{xyz.shape}, rot{rotation.shape}, scale{scaling.shape}, time{time_input.shape}")
    print(f"  Output shapes: d_xyz{d_xyz.shape}, d_rot{d_rotation.shape}, d_scale{d_scaling.shape}")
    
    # 验证输出形状
    assert d_xyz.shape == (batch_size, 3)
    assert d_rotation.shape == (batch_size, 4)
    assert d_scaling.shape == (batch_size, 3)
    
    # 测试正则化损失计算
    print("  Testing regularization loss...")
    reg_loss = ae_network.compute_regularization_loss(xyz, rotation, scaling, time_input)
    print(f"    Regularization loss: {reg_loss.item():.6f}")
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.requires_grad
    
    print("  ✓ Complete AutoEncoder network working correctly!")
    return True

def test_gradient_flow():
    """测试梯度流"""
    print("Testing gradient flow...")
    
    batch_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建网络
    ae_network = AutoEncoderDeformNetwork(latent_dim=32, is_blender=False).to(device)
    
    # 创建模拟数据
    xyz = torch.randn(batch_size, 3, device=device, requires_grad=True)
    rotation = torch.randn(batch_size, 4, device=device, requires_grad=True)
    scaling = torch.randn(batch_size, 3, device=device, requires_grad=True)
    time_input = torch.randn(batch_size, 1, device=device)
    
    # 前向传播
    d_xyz, d_rotation, d_scaling = ae_network(xyz, rotation, scaling, time_input)
    
    # 计算一个简单的损失
    loss = (d_xyz ** 2).mean() + (d_rotation ** 2).mean() + (d_scaling ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度是否存在
    assert xyz.grad is not None
    assert rotation.grad is not None
    assert scaling.grad is not None
    
    # 检查网络参数的梯度
    param_count = 0
    grad_count = 0
    for param in ae_network.parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
    
    print(f"  Parameters with gradients: {grad_count}/{param_count}")
    assert grad_count > 0  # 至少应该有一些参数有梯度
    
    print("  ✓ Gradient flow working correctly!")
    return True

def test_memory_usage():
    """测试内存使用情况"""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("  Skipping memory test (CUDA not available)")
        return True
    
    # 记录初始内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # 创建较大的网络和数据
    batch_size = 5000
    ae_network = AutoEncoderDeformNetwork(latent_dim=128, is_blender=False).cuda()
    
    xyz = torch.randn(batch_size, 3, device="cuda")
    rotation = torch.randn(batch_size, 4, device="cuda")
    scaling = torch.randn(batch_size, 3, device="cuda")
    time_input = torch.randn(batch_size, 1, device="cuda")
    
    # 前向传播
    d_xyz, d_rotation, d_scaling = ae_network(xyz, rotation, scaling, time_input)
    
    # 记录峰值内存
    peak_memory = torch.cuda.memory_allocated()
    memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB
    
    print(f"  Memory used: {memory_used:.1f} MB for {batch_size} points")
    print(f"  Memory per point: {memory_used / batch_size * 1024:.1f} KB")
    
    # 清理
    del ae_network, xyz, rotation, scaling, time_input, d_xyz, d_rotation, d_scaling
    torch.cuda.empty_cache()
    
    print("  ✓ Memory usage test completed!")
    return True

def main():
    print("=" * 60)
    print("AutoEncoder Deform Network Test Suite")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    tests = [
        test_individual_components,
        test_autoencoder_network,
        test_gradient_flow,
        test_memory_usage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! AutoEncoder network is ready for training.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    main()
