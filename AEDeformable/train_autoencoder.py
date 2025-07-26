#!/usr/bin/env python3
"""
使用自编码器变形网络训练 Deformable 3D Gaussians 的训练脚本
"""

import subprocess
import sys
import os

def main():
    # 基础训练参数
    base_args = [
        "python", "train.py",
        "--use_autoencoder",
        "--latent_dim", "64",
        "--iterations", "20000",
        "--warm_up", "3000",
        "--test_iterations", "5000", "10000", "15000", "20000",
        "--save_iterations", "10000", "20000"
    ]
    
    # 如果有命令行参数，添加到基础参数中
    if len(sys.argv) > 1:
        base_args.extend(sys.argv[1:])
    
    print("Training with AutoEncoder Deform Network...")
    print("Command:", " ".join(base_args))
    print("-" * 50)
    
    # 运行训练
    try:
        subprocess.run(base_args, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
