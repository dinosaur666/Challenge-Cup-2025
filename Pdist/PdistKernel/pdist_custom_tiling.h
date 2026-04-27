#ifndef PDIST_CUSTOM_TILING_H
#define PDIST_CUSTOM_TILING_H
#include <cstdint>

struct CoreTiling {
    uint32_t num;           // 样本数量
    uint32_t length;        // 样本长度
    uint32_t totalNum;      // 总数据大小
    uint32_t totalSize;     // 总数据字节大小
    uint32_t isSmall;       // 是否使用全部数据载入L2 cache
    uint32_t returnNum;     // 返回元素个数
    uint32_t blockSize;     // copy的block元素量
    uint32_t pattern;       // 编码后的模式：p + type + length
    uint32_t withPadLength; // 对齐后的长度
    uint32_t withPadLengthBytes; // 对齐后的长度（字节数）
    uint32_t copyLen;       // 实际拷贝长度
    uint32_t rightPadding;   // 右侧填充字节数
    uint32_t coreWorkNum;   // 每个核的工作量分配
    uint32_t coreBlockNum;  // 每个核需要处理的block数量
    uint32_t coreIndexI;    // 每个核的起始i索引
    uint32_t coreIndexJ;    // 每个核的起始j索引
    uint32_t coreIndexStart; // 每个核的起始返回索引
};

// 定义新的 Tiling 结构体
struct PdistCustomTiling {
    CoreTiling ctiling[40];    // 基础 Tiling 信息
};

void GenerateTiling(uint8_t* tilingBuf, uint32_t blockDim, uint32_t num, uint32_t length, float p, uint32_t isHalf);

// =========================================================================
// Pattern 宏定义生成
// 规则: pattern = score_p + score_type + score_length
// P: Inf=0, L1=1, L2=2
// Type: Half=0, Float=10
// Length: Small=0, Medium=100, Large=200, ExtraLarge=300
// =========================================================================

// --- Half (Type Score = 0) ---
// Length Small (Score = 0)
#define PATTERN_HALF_INF_SMALL    0   // 0 + 0 + 0
#define PATTERN_HALF_L1_SMALL     1   // 1 + 0 + 0
#define PATTERN_HALF_L2_SMALL     2   // 2 + 0 + 0
// Length Medium (Score = 100)
#define PATTERN_HALF_INF_MED      100 // 0 + 0 + 100
#define PATTERN_HALF_L1_MED       101 // 1 + 0 + 100
#define PATTERN_HALF_L2_MED       102 // 2 + 0 + 100
// Length Large (Score = 200)
#define PATTERN_HALF_INF_LARGE    200 // 0 + 0 + 200
#define PATTERN_HALF_L1_LARGE     201 // 1 + 0 + 200
#define PATTERN_HALF_L2_LARGE     202 // 2 + 0 + 200
// Length ExtraLarge (Score = 300)
#define PATTERN_HALF_INF_XLARGE   300 // 0 + 0 + 300
#define PATTERN_HALF_L1_XLARGE    301 // 1 + 0 + 300
#define PATTERN_HALF_L2_XLARGE    302 // 2 + 0 + 300

// --- Float (Type Score = 10) ---
// Length Small (Score = 0)
#define PATTERN_FLOAT_INF_SMALL   10  // 0 + 10 + 0
#define PATTERN_FLOAT_L1_SMALL    11  // 1 + 10 + 0
#define PATTERN_FLOAT_L2_SMALL    12  // 2 + 10 + 0
// Length Medium (Score = 100)
#define PATTERN_FLOAT_INF_MED     110 // 0 + 10 + 100
#define PATTERN_FLOAT_L1_MED      111 // 1 + 10 + 100
#define PATTERN_FLOAT_L2_MED      112 // 2 + 10 + 100
// Length Large (Score = 200)
#define PATTERN_FLOAT_INF_LARGE   210 // 0 + 10 + 200
#define PATTERN_FLOAT_L1_LARGE    211 // 1 + 10 + 200
#define PATTERN_FLOAT_L2_LARGE    212 // 2 + 10 + 200
// Length ExtraLarge (Score = 300)
#define PATTERN_FLOAT_INF_XLARGE  310 // 0 + 10 + 300
#define PATTERN_FLOAT_L1_XLARGE   311 // 1 + 10 + 300
#define PATTERN_FLOAT_L2_XLARGE   312 // 2 + 10 + 300

#endif // PDIST_CUSTOM_TILING_H