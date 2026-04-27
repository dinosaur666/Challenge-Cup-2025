#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace optiling {

const uint32_t BLOCK_DIM = 40;

// 辅助函数：向上取整除法
inline uint32_t ceil_div(uint32_t a, uint32_t b)
{
    if (b == 0) return a;
    return (a + b - 1) / b;
}

// 辅助函数：计算 Pattern (完全复用 pdist_custom_tiling.cpp 的逻辑)
uint32_t calculate_pattern(float p, uint32_t isHalf, uint32_t length) {
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
        score_p = 2; // Default to L2
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
    uint32_t score_length = 300; // Default ExtraLarge

    if (isHalf) {
        if (totalBytes <= 256) score_length = 0;
        else if (totalBytes <= 4096) score_length = 100;
        else if (totalBytes <= 32768) score_length = 200;
        else score_length = 300;
    } else {
        if (totalBytes <= 256) score_length = 0;
        else if (totalBytes <= 2048) score_length = 100;
        else if (totalBytes <= 16384) score_length = 200;
        else score_length = 300;
    }
    
    return score_p + score_type + score_length;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PdistTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    
    // 1. 获取输入参数
    uint32_t num = static_cast<uint32_t>(x1_shape->GetOriginShape().GetDim(0));
    uint32_t length = static_cast<uint32_t>(x1_shape->GetOriginShape().GetDim(1));
    
    // 获取 Attrs
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* p_ptr = attrs->GetFloat(0);
    float p = (p_ptr != nullptr) ? *p_ptr : 2.0f;

    // 获取数据类型
    auto input_dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t isHalf = (input_dtype == ge::DT_FLOAT16) ? 1 : 0;
    uint32_t typeSize = (isHalf == 1) ? 2 : 4;

    // 2. 基础计算
    uint32_t returnNum = (num > 1) ? (num * (num - 1) / 2) : 0;
    uint32_t blockSize = (isHalf == 1) ? 16 : 8; // 一个 Block (32B) 能存多少个元素
    
    uint32_t totalNum = num * length;
    uint32_t withPadLength = ceil_div(length, blockSize) * blockSize;
    uint32_t withPadLengthBytes = withPadLength * typeSize;
    uint32_t copyLen = length * typeSize;
    uint32_t rightPadding = withPadLength - length; // 元素个数差值
    
    // 计算 Pattern
    uint32_t pattern = calculate_pattern(p, isHalf, length);

    // 计算 isSmall (是否全部载入 L2 Cache)
    // 逻辑：总大小 <= 160KB 且 单条数据 <= 8KB
    uint32_t isSmall = (totalNum * typeSize <= 160 * 1024) && (length * typeSize <= 8 * 1024) ? 1 : 0;

    // 3. 任务分配逻辑 (核心 Tiling)
    double n2 = num - 0.5f;
    double n2_squared_minus_1 = n2 * n2 - 1.0f;
    
    uint32_t total_blocks = returnNum / blockSize;
    uint32_t remainder = returnNum % blockSize; 
    
    uint32_t avg_blocks = total_blocks / BLOCK_DIM; 
    int32_t extra_blocks = total_blocks % BLOCK_DIM; 
    uint32_t realBlockDim = total_blocks >= BLOCK_DIM ? BLOCK_DIM : total_blocks + (remainder == 0 ? 0 : 1);

    // 获取结构体数组的指针
    // TILING_DATA_FIELD_DEF_ARR 会生成 get_ctiling() 方法，返回数组首地址
    CoreTiling ctilingArray[BLOCK_DIM]; 
    memset(ctilingArray, 0, sizeof(ctilingArray));

    // 4. 填充每个 Core 的 Tiling 数据
    uint32_t sum = 0;
    for(int32_t i = 0; i < realBlockDim; i++, extra_blocks--) {
        CoreTiling& coreTiling = ctilingArray[i];

        // 4.1 计算索引 (i, j)
        coreTiling.coreIndexI = static_cast<uint32_t>(n2 - sqrt(n2_squared_minus_1 - 2 * sum));
        coreTiling.coreIndexJ = sum - num * coreTiling.coreIndexI + coreTiling.coreIndexI * (coreTiling.coreIndexI + 1) / 2 + coreTiling.coreIndexI + 1;
        
        // 4.2 计算工作量
        coreTiling.coreWorkNum = (avg_blocks + (extra_blocks > 0 ? 1 : 0)) * blockSize;
        if (i + 1 == realBlockDim) {
            coreTiling.coreWorkNum += remainder;
        }
        
        // 4.3 记录起始位置
        coreTiling.coreIndexStart = sum;
        sum += coreTiling.coreWorkNum;
        
        // 4.4 计算 Block 数量
        coreTiling.coreBlockNum = ceil_div(coreTiling.coreWorkNum, blockSize);

        // 4.5 填充通用参数 (每个核都一样)
        coreTiling.num = num;
        coreTiling.length = length;
        coreTiling.totalNum = totalNum;
        coreTiling.totalSize = totalNum * typeSize;
        coreTiling.isSmall = isSmall;
        coreTiling.returnNum = returnNum;
        coreTiling.blockSize = blockSize;
        coreTiling.pattern = pattern;
        coreTiling.withPadLength = withPadLength;
        coreTiling.withPadLengthBytes = withPadLengthBytes;
        coreTiling.copyLen = copyLen;
        coreTiling.rightPadding = rightPadding; 
    }
    if (tiling.get_ctilingData() != nullptr) {
        memcpy(tiling.get_ctilingData(), ctilingArray, sizeof(ctilingArray));
    }

    // 5. 设置 Context 和保存 Tiling
    context->SetBlockDim(realBlockDim);
    
    // 序列化 TilingData
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // 设置 workspace 大小 (如果不使用，置0)
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS; 
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    int64_t num = x_shape->GetDim(0);
    int64_t out_dim = 0;
    if (num > 1) out_dim = num * (num - 1) / 2;
    *y_shape = {out_dim};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        // 注意：根据实际情况可能还需要添加 "ascend910c" 等配置
    }
};

OP_ADD(Pdist);
}