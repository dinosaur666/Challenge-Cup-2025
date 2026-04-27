#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#

import torch
import numpy as np
import os

def gen_golden_data_pdist():
    # 建立输入输出目录
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # ================= 配置参数 =================
    # 可以在此处修改测试规模和数据类型
    num = 100            # 样本数量 (N)
    length = 400         # 样本长度 (D)
    p_val = 1.0        # 范数 p
    # p_val = float('inf')  # 范数 p
    dtype = torch.float32  # 数据类型
    # ===========================================

    print(f"Generating data for Pdist: num={num}, length={length}, p={p_val}, dtype={dtype}")

    # 1. 生成输入数据 (运行在CPU上)
    # 使用 torch 生成随机数据
    input_x = torch.rand([num, length], device='cpu', dtype=dtype)

    # 2. 计算真值 (Golden)
    # 为了保证精度对比的有效性，建议在 float32 下计算真值，然后再转回目标 dtype
    # torch.pdist 默认计算行向量之间的距离，返回 N*(N-1)/2 个元素
    golden = torch.pdist(input_x, p=p_val)

    # 3. 保存二进制文件
    # 将 tensor 转为 numpy 并保存为 bin 文件
    input_x_np = input_x.numpy()
    golden_np = golden.numpy()

    input_x_np.tofile("./input/input_x.bin")
    golden_np.tofile("./output/golden.bin")

    # 打印信息用于核对
    print(f"Input shape: {input_x_np.shape}")
    print(f"Output (Golden) shape: {golden_np.shape}")
    print("Data generation complete.")

if __name__ == "__main__":
    gen_golden_data_pdist()