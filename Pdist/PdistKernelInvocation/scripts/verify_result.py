#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#

import sys
import numpy as np
import math

# ================= 配置参数 =================
# for float16
# relative_tol = 1e-3
# absolute_tol = 1e-5
# error_tol = 1e-3
# dtype = np.float16

# for float32 
relative_tol = 1e-4
absolute_tol = 1e-4
error_tol = 1e-4
dtype = np.float32
# ===========================================

def get_row_indices(k, num):
    """
    根据线性索引 k 和样本数量 num，反推 Pdist 的 (i, j) 坐标。
    参考 pdist_custom_tiling.cpp 中的 GenerateTiling 逻辑。
    """
    # 对应 C++: double n2 = num - 0.5f;
    n2 = num - 0.5
    
    # 对应 C++: double n2_squared_minus_1 = n2 * n2 - 1.0f;
    n2_squared_minus_1 = n2 * n2 - 1.0
    
    # 对应 C++: i = static_cast<uint32_t>(n2-sqrt(n2_squared_minus_1 - 2 * sum));
    # 注意防止浮点误差导致 sqrt 内为负数
    val_under_sqrt = n2_squared_minus_1 - 2 * k
    if val_under_sqrt < 0:
        val_under_sqrt = 0
        
    i = int(n2 - math.sqrt(val_under_sqrt))
    
    # 对应 C++: j = sum - num * i + i * (i + 1) / 2 + i + 1;
    j = int(k - num * i + i * (i + 1) / 2 + i + 1)
    
    return i, j

def verify_result(output_file, golden_file):
    # 读取数据
    output = np.fromfile(output_file, dtype=dtype).reshape(-1)
    golden = np.fromfile(golden_file, dtype=dtype).reshape(-1)
    
    # ========================================================
    # 自动计算 Num (样本数量)
    # Pdist 输出长度 len = N * (N - 1) / 2
    # 解一元二次方程: N^2 - N - 2*len = 0  => N = (1 + sqrt(1 + 8*len)) / 2
    data_len = output.size
    num = int((1 + math.sqrt(1 + 8 * data_len)) / 2)
    print(f"[Info] Detected Sample Num: {num}, Output Length: {data_len}")
    # ========================================================

    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    
    # 打印错误详情
    print(f"\n[Error Details] Top {min(100, len(different_element_indexes))} / {len(different_element_indexes)} mismatches:")
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        
        # 计算 I 和 J
        row_i, row_j = get_row_indices(real_index, num)
        
        # 计算相对误差，处理除零情况
        diff = abs(output_data - golden_data)
        if golden_data == 0:
            rdiff = diff
        else:
            rdiff = diff / abs(golden_data)

        print(
            "Index: %06d | Row_I: %4d, Row_J: %4d | Expected: %-.9f, Actual: %-.9f, RDiff: %-.6f" %
            (real_index, row_i, row_j, golden_data, output_data, rdiff)
        )
        
        if index == 100:
            print("... (too many errors, showing first 100)")
            break
            
    error_ratio = float(different_element_indexes.size) / golden.size
    print("\nError ratio: %.4f, Tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output_bin> <golden_bin>")
        sys.exit(1)
        
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] Result Check Failed!")
        else:
            print("[PASS] Test Pass")
    except Exception as e:
        print(e)
        sys.exit(1)