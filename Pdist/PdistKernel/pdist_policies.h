#ifndef PDIST_POLICIES_H
#define PDIST_POLICIES_H

// 依赖 Ascend C 的核心定义，所以必须包含此文件
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_BLOCK_SIZE = 32;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;

// ceil_div 也被策略类使用，所以一并移到这里
__aicore__ inline uint32_t ceil_div(uint32_t a, uint32_t b)
{
    if (b == 0) return a;
    return (a + b - 1) / b;
}


// =========================================================================
//  Pdist Reduce 策略集合
// =========================================================================

// L1 Norm (Manhattan distance) Policies
template<typename Type>
struct DistL1_S {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        WholeReduceSum<Type, true>(dst, dst, length, 1, 1, 1, withPadLength / blockSize);
    }
};

template<typename Type>
struct DistL1_M {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        const uint32_t blockNum0 = withPadLength / blockSize;
        SetMaskCount();
        SetVectorMask<Type>(0, length);
        BlockReduceSum<Type, false>(tmp, dst, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<Type>(0, blockNum0);
        WholeReduceSum<Type, false>(dst, tmp, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
    }
};

template<typename Type>
struct DistL1_L {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        const uint32_t repeatNum = ceil_div(withPadLength * sizeof(Type), REP_LEN);
        SetMaskCount();
        SetVectorMask<Type>(0, length);
        WholeReduceSum<Type, false>(tmp, dst, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<Type>(0, repeatNum);
        WholeReduceSum<Type, false>(dst, tmp, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
    }
};

template<typename Type>
struct DistL1_Ext {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        ReduceSum<Type>(dst, dst, tmp, withPadLength);
    }
};

// L-infinity Norm (Chebyshev distance) Policies
template<typename Type>
struct DistInf_S {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        WholeReduceMax<Type, true>(dst, dst, length, 1, 1, 1, withPadLength / blockSize);
    }
};

template<typename Type>
struct DistInf_M {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        const uint32_t blockNum0 = withPadLength / blockSize;
        SetMaskCount();
        SetVectorMask<Type>(0, length);
        BlockReduceMax<Type, false>(tmp, dst, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetVectorMask<Type>(0, blockNum0);
        WholeReduceMax<Type, false>(dst, tmp, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
    }
};

template<typename Type>
struct DistInf_L {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        ReduceMax<Type>(dst, dst, tmp, withPadLength);
    }
};

template<typename Type>
struct DistInf_Ext {
    __aicore__ inline static void Calculate(LocalTensor<Type>& dst, LocalTensor<Type>& tmp, uint32_t length, uint32_t withPadLength, uint32_t blockSize) {
        ReduceMax<Type>(dst, dst, tmp, withPadLength);
    }
};

#endif // PDIST_POLICIES_H