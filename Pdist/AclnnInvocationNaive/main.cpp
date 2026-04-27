/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono> // 新增：用于计时

#include "acl/acl.h"
// 引用自动生成的算子接口头文件
#include "aclnn_pdist.h"

// ================= 用户配置区域 (修改此处参数) =================

// 1. 输入数据的 Shape 配置
const int64_t CONF_NUM    = 2024;    // 输入数据的行数 (Batch Size)
const int64_t CONF_LENGTH = 3003;    // 输入数据的列数 (Feature Dimension)

// 2. 算子属性配置
const float   CONF_P      = INFINITY;   // P范数 (例如: 1.0f, 2.0f, INFINITY)

// 3. 数据类型配置
// 设为 1 使用 float (FP32), 设为 0 使用 half (FP16)
#define USE_FP32 0

// ===============================================================

// 根据配置定义 Host 侧类型和 ACL 枚举
#if USE_FP32
    using HostType = float;
    const aclDataType ACL_TYPE = ACL_FLOAT;
#else
    using HostType = aclFloat16;
    const aclDataType ACL_TYPE = ACL_FLOAT16;
#endif

// 辅助转换函数
HostType FloatToHost(float val) {
#if USE_FP32
    return val;
#else
    return aclFloatToFloat16(val);
#endif
}

float HostToFloat(HostType val) {
#if USE_FP32
    return val;
#else
    return aclFloat16ToFloat(val);
#endif
}

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        if (deviceAddrs[i] != nullptr) aclrtFree(deviceAddrs[i]);
    }
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. 准备数据形状
    int64_t num = CONF_NUM;
    int64_t length = CONF_LENGTH;
    int64_t outputSize = (num > 1) ? (num * (num - 1)) / 2 : 0;

    std::vector<int64_t> inputXShape = {num, length};
    std::vector<int64_t> outputYShape = {outputSize};

    LOG_PRINT("\n[Config] Num=%ld, Length=%ld, OutputSize=%ld, P=%.2f, DataType=%s\n", 
              num, length, outputSize, CONF_P, USE_FP32 ? "Float32" : "Float16");

    // 3. 生成测试数据
    std::vector<HostType> inputXHostData(num * length);
    std::vector<HostType> outputYHostData(outputSize); 
    std::vector<float> goldenData(outputSize); // 存放高精度真值方便比较

    // 初始化输入 X: 随机生成 0.0 ~ 1.0 之间的数据
    srand(12345); 
    for (int i = 0; i < num * length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        inputXHostData[i] = FloatToHost(val);
    }

    // 4. 创建 Tensor
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;

    std::vector<void *> tensors(2);
    std::vector<void *> deviceAddrs(2);

    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, ACL_TYPE, &inputX);
    tensors[0] = inputX; deviceAddrs[0] = inputXDeviceAddr;
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, ACL_TYPE, &outputY);
    tensors[1] = outputY; deviceAddrs[1] = outputYDeviceAddr;
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    // 5. 获取 Workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    ret = aclnnPdistGetWorkspaceSize(inputX, CONF_P, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdistGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }

    // ================== NPU 计时开始 ==================
    auto npu_start = std::chrono::high_resolution_clock::now();

    // 执行算子
    ret = aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdist failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    auto npu_end = std::chrono::high_resolution_clock::now();
    double npu_time_ms = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();
    // ================== NPU 计时结束 ==================

    // 6. 获取结果
    ret = aclrtMemcpy(outputYHostData.data(), outputYHostData.size() * sizeof(HostType), outputYDeviceAddr,
                      outputSize * sizeof(HostType), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 7. CPU Golden 计算 & 计时
    LOG_PRINT("Computing Golden on CPU...\n");
    
    // ================== CPU 计时开始 ==================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    int idx = 0;
    for (int64_t i = 0; i < num; ++i) {
        for (int64_t j = i + 1; j < num; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < length; ++k) {
                float val_i = HostToFloat(inputXHostData[i * length + k]);
                float val_j = HostToFloat(inputXHostData[j * length + k]);
                float diff = std::abs(val_i - val_j);
                
                if (std::isinf(CONF_P)) {
                    sum = std::max(sum, diff);
                } else {
                    sum += std::pow(diff, CONF_P);
                }
            }
            float dist = std::isinf(CONF_P) ? sum : std::pow(sum, 1.0f / CONF_P);
            goldenData[idx++] = dist;
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    // ================== CPU 计时结束 ==================

    // 输出性能对比
    LOG_PRINT("\n[Performance Comparison]\n");
    LOG_PRINT("CPU Time : %10.4f ms\n", cpu_time_ms);
    LOG_PRINT("NPU Time : %10.4f ms\n", npu_time_ms);
    LOG_PRINT("Speedup  : %10.2f x\n\n", cpu_time_ms / npu_time_ms);


    // 8. 验证结果
    const float relative_tol = 1e-3f;      // rtol
    const float absolute_tol = 1e-3f;      // atol
    const float error_tol_ratio = 0.001f;  // 允许的错误率 0.1%

    LOG_PRINT("[Verify] Shape: [%ld], Dtype: %s\n", outputSize, USE_FP32 ? "Float32" : "Float16");

    int64_t diff_count = 0;
    std::vector<int64_t> error_indices;

    for (int64_t i = 0; i < outputSize; i++) {
        float out = HostToFloat(outputYHostData[i]);
        float gold = goldenData[i];
        
        // np.isclose 逻辑: absolute(a - b) <= (atol + rtol * absolute(b))
        float tolerance = absolute_tol + relative_tol * std::abs(gold);
        float diff = std::abs(out - gold);
        
        if (diff > tolerance) {
            diff_count++;
            if (error_indices.size() < 100) { 
                error_indices.push_back(i);
            }
        }
    }

    if (diff_count > 0) {
        float ratio = (float)diff_count / outputSize;
        LOG_PRINT(">>> Found %ld mismatches (%.2f%%)\n", diff_count, ratio * 100);
        LOG_PRINT("%-10s | %-15s | %-15s | %-10s\n", "Index", "Expected (CPU)", "Actual (NPU)", "Rel Diff");
        LOG_PRINT("------------------------------------------------------------\n");

        for (size_t k = 0; k < error_indices.size(); k++) {
            if (k >= 20) {
                LOG_PRINT("... and %ld more mismatches ...\n", diff_count - 20);
                break;
            }
            int64_t i = error_indices[k];
            float out = HostToFloat(outputYHostData[i]);
            float gold = goldenData[i];
            
            float rdiff = 0.0f;
            if (std::abs(gold) < 1e-9) {
                rdiff = std::abs(out - gold);
            } else {
                rdiff = std::abs(out - gold) / std::abs(gold);
            }
            
            LOG_PRINT("%-10ld | %-15.6f | %-15.6f | %-10.6f\n", i, gold, out, rdiff);
        }
    } else {
        LOG_PRINT(">>> No mismatches found. Printing first 10 results for verification:\n");
        LOG_PRINT("%-10s | %-15s | %-15s | %-10s\n", "Index", "Expected (CPU)", "Actual (NPU)", "Rel Diff");
        LOG_PRINT("------------------------------------------------------------\n");
        
        int64_t print_count = outputSize < 10 ? outputSize : 10;
        for (int64_t i = 0; i < print_count; i++) {
            float out = HostToFloat(outputYHostData[i]);
            float gold = goldenData[i];
            
            float rdiff = 0.0f;
            if (std::abs(gold) < 1e-9) {
                rdiff = std::abs(out - gold);
            } else {
                rdiff = std::abs(out - gold) / std::abs(gold);
            }
            
            LOG_PRINT("%-10ld | %-15.6f | %-15.6f | %-10.6f\n", i, gold, out, rdiff);
        }
    }

    float current_error_ratio = (float)diff_count / outputSize;
    LOG_PRINT("\n[Result] Error ratio: %.4f, Tolerance threshold: %.4f\n", current_error_ratio, error_tol_ratio);

    if (current_error_ratio > error_tol_ratio) {
        LOG_PRINT("Test Failed! Error ratio %.4f > %.4f\n", current_error_ratio, error_tol_ratio);
        ret = FAILED;
    } else {
        LOG_PRINT("Test Pass!\n");
        ret = SUCCESS;
    }

    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);
    return ret;
}