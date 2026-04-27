#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <limits>
#include "pdist_custom_tiling.h"

using namespace std;

uint32_t ceil_div(uint32_t a, uint32_t b)
{
    if (b == 0) return a;
    return (a + b - 1) / b;
}

uint32_t calculate_pattern(float p, uint32_t isHalf, uint32_t length){
    // pattern = score_p + score_type + score_length

    // --- Step A: Score P ---
    uint32_t score_p = 0;
    if (std::isinf(p)) {
        score_p = 0;
    } else if (std::abs(p - 1.0f) < 1e-6) {
        score_p = 1;
    } else if (std::abs(p - 2.0f) < 1e-6) {
        score_p = 2;
    } else {
        score_p = 2; 
    }

    // --- Step B: Score Type ---
    uint32_t score_type = 0;
    uint32_t typeSize = 0;
    if (isHalf) {
        score_type = 0;
        typeSize = 2; // sizeof(half)
    } else {
        score_type = 10;
        typeSize = 4; // sizeof(float)
    }

    // --- Step C: Score Length ---
    uint32_t totalBytes = length * typeSize;
    uint32_t score_length = 300; // 默认为最大档位 (ExtraLarge)

    if (isHalf) {
        // Half 精度下的长度分档
        if (totalBytes <= 256) {
            score_length = 0;
        } else if (totalBytes <= 4096) { // 2 * 2 * 1024
            score_length = 100;
        } else if (totalBytes <= 32768) { // 16 * 2 * 1024
            score_length = 200;
        } else {
            score_length = 300;
        }
    } else {
        // Float 精度下的长度分档
        if (totalBytes <= 256) {
            score_length = 0;
        } else if (totalBytes <= 2048) { // 0.5 * 4 * 1024
            score_length = 100;
        } else if (totalBytes <= 16384) { // 4 * 4 * 1024
            score_length = 200;
        } else {
            score_length = 300;
        }
    }
    // --- Final: 合并 ---
    uint32_t pattern = score_p + score_type + score_length;
    return pattern;
}

void GenerateTiling(uint8_t* tilingBuf, uint32_t blockDim, uint32_t num, uint32_t length, float p, uint32_t isHalf){
    PdistCustomTiling *tiling = reinterpret_cast<PdistCustomTiling *>(tilingBuf);
    
    
    // 初始化
    for(uint32_t i=0; i < blockDim; i++) tiling->ctiling[i].coreWorkNum = 0;

    // 1. Tiling 核心计算逻辑
    uint32_t returnNum = num * (num - 1) / 2;
    uint32_t blockSize = (isHalf == 1) ? 16 : 8;
    uint32_t typeSize = (isHalf == 1) ? 2 : 4;
    
    double n2 = num - 0.5f;
    double n2_squared_minus_1 = n2 * n2 - 1.0f;
    uint32_t total_blocks = returnNum / blockSize;
    uint32_t remainder = returnNum % blockSize; 
    uint32_t avg_blocks = total_blocks / blockDim; 
    int32_t extra_blocks = total_blocks % blockDim; 
    uint32_t realBlockDim = total_blocks >= blockDim ? blockDim : total_blocks+(remainder == 0 ? 0 : 1);
    
    // 2. 基础信息赋值
    uint32_t totalNum = num * length;
    uint32_t withPadLength = ceil_div(length, blockSize) * blockSize;
    uint32_t withPadLengthBytes = withPadLength * typeSize;
    uint32_t copyLen = length * typeSize;
    uint32_t rightPadding = withPadLength - length;
    uint32_t pattern = calculate_pattern(p, isHalf, length);

    for(uint32_t i=0, sum=0; i < realBlockDim; i++, extra_blocks--){
        tiling->ctiling[i].coreIndexI = static_cast<uint32_t>(n2-sqrt(n2_squared_minus_1 - 2 * sum));
        tiling->ctiling[i].coreIndexJ = sum - num * tiling->ctiling[i].coreIndexI + tiling->ctiling[i].coreIndexI * (tiling->ctiling[i].coreIndexI + 1) / 2 + tiling->ctiling[i].coreIndexI + 1;
        tiling->ctiling[i].coreWorkNum = (avg_blocks+(extra_blocks>0 ? 1 : 0))*blockSize;
        if(i+1 == realBlockDim) tiling->ctiling[i].coreWorkNum += remainder;
        tiling->ctiling[i].coreIndexStart = sum;
        sum += tiling->ctiling[i].coreWorkNum;
        tiling->ctiling[i].coreBlockNum = ceil_div(tiling->ctiling[i].coreWorkNum, blockSize);
        // 每个core基础信息赋值
        tiling->ctiling[i].num = num;
        tiling->ctiling[i].length = length;
        tiling->ctiling[i].totalNum = totalNum;
        tiling->ctiling[i].totalSize = totalNum * typeSize;
        tiling->ctiling[i].isSmall = (totalNum * typeSize <= 160*1024) && (length * typeSize <= 8*1024) ? 1 : 0;
        tiling->ctiling[i].returnNum = returnNum;
        tiling->ctiling[i].blockSize = blockSize;
        tiling->ctiling[i].pattern = pattern;
        tiling->ctiling[i].withPadLength = withPadLength;
        tiling->ctiling[i].withPadLengthBytes = withPadLengthBytes;
        tiling->ctiling[i].copyLen = copyLen;
        tiling->ctiling[i].rightPadding = rightPadding;
    }
}