#include "kernel_operator.h"
#include "pdist_custom_tiling.h"
#include "pdist_policies.h"

using namespace AscendC;

__aicore__ inline void CopyTiling(CoreTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM) + GetBlockIdx() * sizeof(CoreTiling) / sizeof(uint32_t);
    for (int i = 0; i < sizeof(CoreTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template<typename Type, typename DistPolicy> 
class KernelPdist {
protected:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<TPosition::VECCALC> diffBuf, workBuf;

    GlobalTensor<Type> xGm;
    GlobalTensor<Type> zGm;
    uint32_t coreIdx, coreNum, blockNum, offset, withPadLength;
    CoreTiling tiling;
    uint32_t copyLen;
    uint8_t rightPadding;

public:
    __aicore__ inline KernelPdist() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, CoreTiling tiling)
    {
        this->tiling = tiling;
        this->offset = 0;
        this->coreNum = GetBlockNum();
        this->coreIdx = GetBlockIdx();
        this->blockNum = tiling.coreBlockNum;
        this->withPadLength = tiling.withPadLength;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling.withPadLengthBytes);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tiling.withPadLengthBytes);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, DATA_BLOCK_SIZE);
        pipe.InitBuffer(diffBuf, tiling.withPadLengthBytes);
        pipe.InitBuffer(workBuf, tiling.withPadLengthBytes);
        xGm.SetGlobalBuffer((__gm__ Type *)x, tiling.totalNum);
        zGm.SetGlobalBuffer((__gm__ Type *)z, tiling.returnNum);
        this->copyLen = tiling.copyLen;
        this->rightPadding = static_cast<uint8_t>(tiling.rightPadding);
    }
 
    __aicore__ inline void Process()
    {
        if (tiling.coreWorkNum == 0) return;
        CopyIn(tiling.coreIndexI, tiling.coreIndexJ);
        LocalTensor<Type> xLocal = inQueueX.DeQue<Type>(); 

        uint32_t i = tiling.coreIndexI;
        uint32_t j = tiling.coreIndexJ;
        uint32_t workSize_sub_1 = tiling.coreWorkNum - 1;
        offset = 0;

        for(uint32_t blockid = 0; blockid < this->blockNum; blockid++){
            LocalTensor<Type> zLocal = outQueueZ.AllocTensor<Type>();
            for(uint32_t k = 0; k < tiling.blockSize && offset < tiling.coreWorkNum; k++, offset++){
                bool switch_x = false;
                if (offset < workSize_sub_1) {
                    if(j+1 < tiling.num){
                        ++j;
                        CopyInJustY(j);
                    }
                    else if(j+1 == tiling.num){
                        j = ++i+1;
                        CopyIn(i, j);
                        switch_x = true;  
                    }
                }
                Compute(zLocal, xLocal, k);
                if (switch_x) {
                    inQueueX.FreeTensor(xLocal);
                    xLocal = inQueueX.DeQue<Type>();
                }
            }
            DataCopy(zGm[tiling.coreIndexStart + blockid * tiling.blockSize], zLocal, tiling.blockSize);
            outQueueZ.FreeTensor(zLocal);
        }
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Process_L2()
    {
        if (tiling.coreWorkNum == 0) return;
        CopyIn(tiling.coreIndexI, tiling.coreIndexJ);
        LocalTensor<Type> xLocal = inQueueX.DeQue<Type>(); 
        uint32_t i = tiling.coreIndexI;
        uint32_t j = tiling.coreIndexJ;
        uint32_t workSize_sub_1 = tiling.coreWorkNum - 1;
        offset = 0;
        LocalTensor<Type> zLocal = outQueueZ.AllocTensor<Type>();
        for(uint32_t blockid = 0; blockid < this->blockNum; blockid++){
            for(uint32_t k = 0; k < tiling.blockSize && offset < tiling.coreWorkNum; k++, offset++){
                bool switch_x = false;
                if (offset < workSize_sub_1) {
                    if(j+1 < tiling.num){
                        ++j;
                        CopyInJustY(j);
                    }
                    else if(j+1 == tiling.num){
                        j = ++i+1;
                        CopyIn(i, j);
                        switch_x = true;  
                    }
                }
                Compute_L2(zLocal, xLocal, k);
                if (switch_x) {
                    inQueueX.FreeTensor(xLocal);
                    xLocal = inQueueX.DeQue<Type>();
                }
            }
            Sqrt(zLocal, zLocal, tiling.blockSize);
            DataCopy(zGm[tiling.coreIndexStart + blockid * tiling.blockSize], zLocal, tiling.blockSize);
        }
        outQueueZ.FreeTensor(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

protected:
    __aicore__ inline void CopyIn(uint32_t rowI, uint32_t rowJ)
    {
        LocalTensor<Type> xLocal = inQueueX.AllocTensor<Type>();
        LocalTensor<Type> yLocal = inQueueY.AllocTensor<Type>();
        DataCopyExtParams copyParams = {1, copyLen, 0, 0, 0};
        DataCopyPadExtParams<Type> padParams = {false, 0, rightPadding, 0};
        DataCopyPad<Type>(xLocal, xGm[rowI * tiling.length], copyParams, padParams);
        DataCopyPad<Type>(yLocal, xGm[rowJ * tiling.length], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyInJustY(uint32_t rowJ)
    {
        LocalTensor<Type> yLocal = inQueueY.AllocTensor<Type>();
        DataCopyExtParams copyParams = {1, copyLen, 0, 0, 0};
        DataCopyPadExtParams<Type> padParams = {false, 0, rightPadding, 0};
        DataCopyPad<Type>(yLocal, xGm[rowJ * tiling.length], copyParams, padParams);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(LocalTensor<Type>& zLocal, LocalTensor<Type>& xLocal, uint32_t index)
    {
        LocalTensor<Type> yLocal = inQueueY.DeQue<Type>();
        LocalTensor<Type> diffLocal = diffBuf.Get<Type>();
        LocalTensor<Type> workLocal = workBuf.Get<Type>();
        Sub(diffLocal, xLocal, yLocal, withPadLength);
        Abs(diffLocal, diffLocal, withPadLength);
        DistPolicy::Calculate(diffLocal, workLocal, tiling.length, withPadLength, tiling.blockSize);
        zLocal.SetValue(index, diffLocal.GetValue(0));
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void Compute(LocalTensor<Type>& zLocal, LocalTensor<Type>& xLocal, LocalTensor<Type>& yLocal, uint32_t index)
    {
        LocalTensor<Type> diffLocal = diffBuf.Get<Type>();
        LocalTensor<Type> workLocal = workBuf.Get<Type>();
        Sub(diffLocal, xLocal, yLocal, withPadLength);
        Abs(diffLocal, diffLocal, withPadLength);
        DistPolicy::Calculate(diffLocal, workLocal, tiling.length, withPadLength, tiling.blockSize);
        zLocal.SetValue(index, diffLocal.GetValue(0));
    }

    __aicore__ inline void Compute_L2(LocalTensor<Type>& zLocal, LocalTensor<Type>& xLocal, uint32_t index)
    {
        LocalTensor<Type> yLocal = inQueueY.DeQue<Type>();
        LocalTensor<Type> diffLocal = diffBuf.Get<Type>();
        LocalTensor<Type> workLocal = workBuf.Get<Type>();
        Sub(diffLocal, xLocal, yLocal, withPadLength);
        Mul(diffLocal, diffLocal, diffLocal, withPadLength);
        DistPolicy::Calculate(diffLocal, workLocal, tiling.length, withPadLength, tiling.blockSize);
        zLocal.SetValue(index, diffLocal.GetValue(0));
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute_L2(LocalTensor<Type>& zLocal, LocalTensor<Type>& xLocal, LocalTensor<Type>& yLocal, uint32_t index)
    {
        LocalTensor<Type> diffLocal = diffBuf.Get<Type>();
        LocalTensor<Type> workLocal = workBuf.Get<Type>();
        Sub(diffLocal, xLocal, yLocal, withPadLength);
        Mul(diffLocal, diffLocal, diffLocal, withPadLength);
        DistPolicy::Calculate(diffLocal, workLocal, tiling.length, withPadLength, tiling.blockSize);
        zLocal.SetValue(index, diffLocal.GetValue(0));
    }
};

// 子类 KernelPdistSmall
template<typename Type, typename DistPolicy>
class KernelPdistSmall : public KernelPdist<Type, DistPolicy> {
private:
    TQue<TPosition::VECIN, 1> inQueueData;

public:
    __aicore__ inline KernelPdistSmall() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, CoreTiling tiling)
    {
        this->tiling = tiling;
        this->offset = 0;
        this->coreNum = GetBlockNum();
        this->coreIdx = GetBlockIdx();
        this->blockNum = tiling.coreBlockNum;
        this->withPadLength = tiling.withPadLength;
        this->copyLen = tiling.copyLen;
        this->rightPadding = static_cast<uint8_t>(tiling.rightPadding);

        this->xGm.SetGlobalBuffer((__gm__ Type *)x, tiling.totalNum);
        this->zGm.SetGlobalBuffer((__gm__ Type *)z, tiling.returnNum);
        this->pipe.InitBuffer(this->inQueueData, 1, tiling.num * tiling.withPadLength * sizeof(Type));
        this->pipe.InitBuffer(this->outQueueZ, BUFFER_NUM, DATA_BLOCK_SIZE);
        this->pipe.InitBuffer(this->diffBuf, tiling.withPadLengthBytes);
        this->pipe.InitBuffer(this->workBuf, tiling.withPadLengthBytes);
        LocalTensor<Type> allData = inQueueData.template AllocTensor<Type>();
        uint32_t singleDataBytes = tiling.length * sizeof(Type);

        // 判断：单条数据是否满足 32 Byte 对齐
        if (singleDataBytes % 32 == 0) DataCopy(allData, this->xGm[0], tiling.totalNum);
        else {
            DataCopyExtParams copyParams = {1, this->copyLen, 0, 0, 0};
            DataCopyPadExtParams<Type> padParams = {false, 0, this->rightPadding, 0};
            for(uint32_t i = 0; i < tiling.num; i++) {
                DataCopyPad<Type>(allData[i * tiling.withPadLength], 
                    this->xGm[i * tiling.length], 
                    copyParams, 
                    padParams
                );
            }
        }
        inQueueData.EnQue(allData);
    }

    __aicore__ inline void Process()
    {
        if (this->tiling.coreWorkNum == 0) return;
        LocalTensor<Type> allData = this->inQueueData.template DeQue<Type>();
        
        uint32_t i = this->tiling.coreIndexI;
        uint32_t j = this->tiling.coreIndexJ;
        this->offset = 0;
        for(uint32_t blockid = 0; blockid < this->blockNum; blockid++){
            LocalTensor<Type> zLocal = this->outQueueZ.template AllocTensor<Type>();
            for(uint32_t k = 0; k < this->tiling.blockSize && this->offset < this->tiling.coreWorkNum; k++, this->offset++){
                LocalTensor<Type> xLocal = allData[i * this->withPadLength];
                LocalTensor<Type> yLocal = allData[j * this->withPadLength];
                this->Compute(zLocal, xLocal, yLocal, k);
                if(j+1 < this->tiling.num) ++j;
                else if(j+1 == this->tiling.num) j = ++i+1;
            }
            DataCopy(this->zGm[this->tiling.coreIndexStart + blockid * this->tiling.blockSize], zLocal, this->tiling.blockSize);
            this->outQueueZ.FreeTensor(zLocal);
        }
        this->inQueueData.FreeTensor(allData);
    }

    __aicore__ inline void Process_L2()
    {
        if (this->tiling.coreWorkNum == 0) return;
        LocalTensor<Type> allData = this->inQueueData.template DeQue<Type>();
        
        uint32_t i = this->tiling.coreIndexI;
        uint32_t j = this->tiling.coreIndexJ;
        uint32_t workSize_sub_1 = this->tiling.coreWorkNum - 1;
        this->offset = 0;
        for(uint32_t blockid = 0; blockid < this->blockNum; blockid++){
            LocalTensor<Type> zLocal = this->outQueueZ.template AllocTensor<Type>();
            for(uint32_t k = 0; k < this->tiling.blockSize && this->offset < this->tiling.coreWorkNum; k++, this->offset++){
                LocalTensor<Type> xLocal = allData[i * this->withPadLength];
                LocalTensor<Type> yLocal = allData[j * this->withPadLength];
                this->Compute_L2(zLocal, xLocal, yLocal, k);
                if(j+1 < this->tiling.num) ++j;
                else if(j+1 == this->tiling.num) j = ++i+1;
            }
            Sqrt(zLocal, zLocal, this->tiling.blockSize);
            DataCopy(this->zGm[this->tiling.coreIndexStart + blockid * this->tiling.blockSize], zLocal, this->tiling.blockSize);
            this->outQueueZ.FreeTensor(zLocal);
        }
        this->inQueueData.FreeTensor(allData);
    }
};

// ================= 入口函数 =================
#define CALL_KERNEL(KClass, TType, DPolicy, IsL2) \
    KClass<TType, DPolicy<TType>> op; \
    op.Init(x, z, tiling); \
    if (IsL2) op.Process_L2(); else op.Process();

#define DISPATCH_CASE(Pattern, TType, DPolicy, IsL2) \
    case Pattern: { \
        if (tiling.isSmall) { \
            CALL_KERNEL(KernelPdistSmall, TType, DPolicy, IsL2) \
        } else { \
            CALL_KERNEL(KernelPdist, TType, DPolicy, IsL2) \
        } \
        break; \
    }

extern "C" __global__ __aicore__ void pdist_custom(GM_ADDR x, GM_ADDR z, GM_ADDR tilingGM)
{
    CoreTiling tiling;
    CopyTiling(&tiling, tilingGM);
    
    switch (tiling.pattern) {
        // Half, L1 (P=1)
        DISPATCH_CASE(PATTERN_HALF_L1_SMALL, half, DistL1_S, false)
        DISPATCH_CASE(PATTERN_HALF_L1_MED, half, DistL1_M, false)
        DISPATCH_CASE(PATTERN_HALF_L1_LARGE, half, DistL1_L, false)
        DISPATCH_CASE(PATTERN_HALF_L1_XLARGE, half, DistL1_Ext, false)
        
        // Float, L1 (P=1)
        DISPATCH_CASE(PATTERN_FLOAT_L1_SMALL, float, DistL1_S, false)
        DISPATCH_CASE(PATTERN_FLOAT_L1_MED, float, DistL1_M, false)
        DISPATCH_CASE(PATTERN_FLOAT_L1_LARGE, float, DistL1_L, false)
        DISPATCH_CASE(PATTERN_FLOAT_L1_XLARGE, float, DistL1_Ext, false)

        // Half, L2 (P=2) -> isL2=true
        DISPATCH_CASE(PATTERN_HALF_L2_SMALL, half, DistL1_S, true)
        DISPATCH_CASE(PATTERN_HALF_L2_MED, half, DistL1_M, true)
        DISPATCH_CASE(PATTERN_HALF_L2_LARGE, half, DistL1_L, true)
        DISPATCH_CASE(PATTERN_HALF_L2_XLARGE, half, DistL1_Ext, true)

        // Float, L2 (P=2) -> isL2=true
        DISPATCH_CASE(PATTERN_FLOAT_L2_SMALL, float, DistL1_S, true)
        DISPATCH_CASE(PATTERN_FLOAT_L2_MED, float, DistL1_M, true)
        DISPATCH_CASE(PATTERN_FLOAT_L2_LARGE, float, DistL1_L, true)
        DISPATCH_CASE(PATTERN_FLOAT_L2_XLARGE, float, DistL1_Ext, true)

        // Half, Inf (P=inf)
        DISPATCH_CASE(PATTERN_HALF_INF_SMALL, half, DistInf_S, false)
        DISPATCH_CASE(PATTERN_HALF_INF_MED, half, DistInf_M, false)
        DISPATCH_CASE(PATTERN_HALF_INF_LARGE, half, DistInf_L, false)
        DISPATCH_CASE(PATTERN_HALF_INF_XLARGE, half, DistInf_Ext, false)

        // Float, Inf (P=inf)
        DISPATCH_CASE(PATTERN_FLOAT_INF_SMALL, float, DistInf_S, false)
        DISPATCH_CASE(PATTERN_FLOAT_INF_MED, float, DistInf_M, false)
        DISPATCH_CASE(PATTERN_FLOAT_INF_LARGE, float, DistInf_L, false)
        DISPATCH_CASE(PATTERN_FLOAT_INF_XLARGE, float, DistInf_Ext, false)

        default:
            break;
    }
}