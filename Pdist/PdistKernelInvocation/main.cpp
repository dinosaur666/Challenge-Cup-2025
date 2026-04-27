/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "pdist_custom_tiling.h"
#include "data_utils.h"
#include <iostream>
#include <cmath>

// 手动声明在 pdist_custom_tiling.cpp 中定义的函数
// 链接时需要确保 pdist_custom_tiling.o 被链接进来

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
// NPU模式下，编译脚本会自动生成此头文件，包含 ACLRT_LAUNCH_KERNEL 宏
#include "aclrtlaunch_pdist_custom.h"
#else
#include "tikicpulib.h"
// CPU模式下，需声明核函数原型
// 注意：pdist_custom 接收的是指针 (GM_ADDR tilingGM)
extern "C" __global__ __aicore__ void pdist_custom(GM_ADDR x, GM_ADDR z, GM_ADDR tilingGM);
#endif

int32_t main(int32_t argc, char *argv[])
{
    // ================= 1. 配置参数 =================
    // 这里的参数必须与 gen_data.py 中的配置一致
    uint32_t num = 100;         // 样本数量 N 
    uint32_t length = 400;      // 样本长度 D
    // float p = INFINITY;  // 范数 P
    float p = 1.0f;          // 范数 P
    uint32_t isHalf = 0;        // 1 代表 float16
    uint32_t blockDim = 40;      // 核数
    // ================= 2. 计算内存大小 =================
    // Pdist 输出大小为 N * (N - 1) / 2
    size_t outputDataNum = num * (num - 1) / 2;
    // float16 占 2 字节
    size_t inputByteSize, outputByteSize;
    if(isHalf){
        inputByteSize = num * length * sizeof(uint16_t); // 输入数据大小，含对齐填充
        outputByteSize = outputDataNum * sizeof(uint16_t); // 输出数据大小，含对齐填充
    }
    else{
        inputByteSize = num * length * sizeof(float); // 输入数据大小，含对齐填充
        outputByteSize = outputDataNum * sizeof(float); // 输出数据大小，含对齐填充
    }

    
    
    // Tiling 结构体大小
    size_t tilingSize = sizeof(PdistCustomTiling);

#ifdef ASCENDC_CPU_DEBUG
    // ================= 3. CPU 仿真模式 =================
    
    // 分配 Host 内存 (使用 AscendC::GmAlloc 模拟 Global Memory)
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    // 读取输入数据
    if (!ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize)) {
        std::cerr << "[ERROR] Read input_x.bin failed!" << std::endl;
        return -1;
    }

    // 计算 Tiling (核心步骤)
    // 直接调用 pdist_custom_tiling.cpp 中的逻辑
    GenerateTiling(tiling, blockDim, num, length, p, isHalf);

    // 设置运行模式为 AIV (Vector Core)
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 调用核函数
    // 你的算子定义是 pdist_custom(GM_ADDR x, GM_ADDR z, GM_ADDR tilingGM)
    // 所以这里传递 tiling 指针
    ICPU_RUN_KF(pdist_custom, blockDim, x, z, tiling);

    // 保存输出结果
    WriteFile("./output/output_z.bin", z, outputByteSize);

    // 释放内存
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);

#else
    // ================= 4. NPU 模式 (ACL) =================
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // --- Host 端内存分配与数据准备 ---
    uint8_t *xHost, *zHost, *tilingHost;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingSize));

    // 读取输入数据
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);

    // 计算 Tiling (在 Host 端计算)
    GenerateTiling(tilingHost, blockDim, num, length, p, isHalf);

    // --- Device 端内存分配 ---
    uint8_t *xDevice, *zDevice, *tilingDevice;
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // --- 数据拷贝 Host -> Device ---
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 将计算好的 Tiling 数据拷贝到 Device 侧
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // --- 启动核函数 ---
    // 使用自动生成的加载宏
    ACLRT_LAUNCH_KERNEL(pdist_custom)(blockDim, stream, xDevice, zDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // --- 数据拷贝 Device -> Host ---
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 保存输出
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    // --- 资源释放 ---
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    return 0;
}