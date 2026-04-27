#ifndef PDIST_TILING_H
#define PDIST_TILING_H
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {

// 保持 CoreTiling 定义不变，供 Host 和 Kernel 代码使用
struct CoreTiling {
    uint32_t num;
    uint32_t length;
    uint32_t totalNum;
    uint32_t totalSize;
    uint32_t isSmall;
    uint32_t returnNum;
    uint32_t blockSize;
    uint32_t pattern;
    uint32_t withPadLength;
    uint32_t withPadLengthBytes;
    uint32_t copyLen;
    uint32_t rightPadding;
    uint32_t coreWorkNum;
    uint32_t coreBlockNum;
    uint32_t coreIndexI;
    uint32_t coreIndexJ;
    uint32_t coreIndexStart;
};

BEGIN_TILING_DATA_DEF(PdistTilingData)
  // 修改处：不要直接使用 CoreTiling 类型
  // CoreTiling 有 17 个 uint32 成员，共 40 个核
  // 17 * 40 = 680
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 680, ctilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}

#endif // PDIST_TILING_H