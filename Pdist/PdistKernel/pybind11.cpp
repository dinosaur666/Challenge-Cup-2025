#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acl/acl.h"

#include "aclrtlaunch_pdist_custom.h"
#include "pdist_custom_tiling.h"

extern void GenerateTiling(uint8_t* tilingBuf, uint32_t blockDim, uint32_t num, uint32_t length, float p, uint32_t isHalf);

namespace my_pdist{

at::Tensor run_pdist_custom(const at::Tensor &x, float p){
    //获取tiling相关参数
    TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D (num, length)");
    uint32_t num = x.size(0);
    uint32_t length = x.size(1);
    at::Tensor z = at::empty({num*(num-1)/2}, x.options());
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = 40;
    uint32_t isHalf;
    if (x.scalar_type() == at::ScalarType::Float) {
        isHalf = 0;  // float32
    } else if (x.scalar_type() == at::ScalarType::Half) {
        isHalf = 1;  // float16(half)
    }
    //为tiling分配显存
    size_t tilingFileSize = sizeof(PdistCustomTiling);
    uint8_t *tiling = nullptr;
    uint8_t *tilingDevice = nullptr;
    aclrtMallocHost((void **)(&tiling), tilingFileSize);
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    GenerateTiling(tiling, blockDim, num, length, p, isHalf);
    aclrtMemcpy(tilingDevice, tilingFileSize, tiling, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ACLRT_LAUNCH_KERNEL(pdist_custom)
    (blockDim, acl_stream,   
     const_cast<void *>(x.storage().data()),  // GM_ADDR x
     const_cast<void *>(z.storage().data()),  // GM_ADDR z
     tilingDevice
    );
    aclrtFreeHost(tiling);

    return z;
}
}

PYBIND11_MODULE(pdist_custom, m)
{
    m.doc() = "pdist_custom pybind11 interfaces"; // 模块文档
    m.def("run_pdist_custom", &my_pdist::run_pdist_custom, "Run pdist custom operator (p=1)");
}