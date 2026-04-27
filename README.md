# Challenge-Cup-Huawei-2025 | BUPT-ParCIS

![Platform](https://img.shields.io/badge/platform-Ascend%20NPU-blue)
![CANN](https://img.shields.io/badge/CANN-8.1+-orange)
![Language](https://img.shields.io/badge/language-C%2B%2B%20%2F%20Python-green)

华为"挑战杯"参赛项目 —— 基于昇腾 Ascend NPU 的**高性能 Pdist（Pairwise Distance）自定义算子**设计与实现。

---

## 项目简介

本项目面向华为昇腾 910B4 NPU，基于 Ascend C 编程模型，实现了一个高度优化的 Pairwise Distance 算子。算子支持多种距离度量（L1、L2、Linf）和双精度模式（FP32/FP16），通过自适应策略选择、负载均衡、流水线优化等技术，在大规模数据场景下实现了相比 CPU 实现 **数千至数万倍** 的加速比。

## 目录结构

```
Challenge-Cup-Huawei-2025/
├── Pdist/                           # 算子核心实现
│   ├── PdistKernel/                 # PyTorch 扩展（快速验证）
│   ├── PdistKernelInvocation/       # 独立调用程序（性能分析）
│   ├── PdistFramework/              # 算子框架与构建部署
│   ├── AclnnInvocationNaive/        # ACLNN API 验证程序
│   ├── images/                      # 性能可视化图片
│   └── README.md                    # 算子详细文档（设计/优化/实验）
├── BUPT-ParCIS-答辩PPT.pptx         # 答辩演示文稿
├── 挑战杯指导文档.pdf                 # 挑战杯指导文档
└── 挑战杯-答辩材料.pdf               # 挑战杯答辩材料
```

## 核心特性

- **多距离度量**：支持 L1（曼哈顿）、L2（欧几里得）、Linf（切比雪夫）三种距离计算
- **双精度支持**：FP32 高精度 / FP16 高性能
- **自适应策略**：根据数据规模自动选择最优 Reduce 策略（WholeReduce / BlockReduce / 通用 Reduce）
- **负载均衡**：按输出元素数均匀分配至最多 40 个 AI Core，保证对齐约束
- **流水线优化**：Double Buffer + 行向量最大复用，IO 与计算重叠
- **内存优化**：小数据场景下一次性加载至 L2 Cache，减少反复 IO

## 性能亮点

| 数据形状 | p | 精度 | CPU 耗时 (ms) | NPU 耗时 (ms) | 加速比 |
|---|---|---|---|---|---|
| [100, 400] | 2.0 | FP32 | 80.81 | 0.78 | **103x** |
| [100, 400] | 2.0 | FP16 | 99.55 | 0.62 | **161x** |
| [2024, 3000] | 2.0 | FP32 | 262,417 | 28.60 | **9,176x** |
| [2024, 3000] | 2.0 | FP16 | 316,423 | 20.65 | **15,325x** |
| [2024, 3003] | 1.0 | FP16 | 301,563 | 20.63 | **14,617x** |

> 更多实验数据见 [Pdist/README.md](Pdist/README.md)。

## 环境要求

| 组件 | 版本 |
|------|------|
| 硬件 | 华为昇腾 NPU (Ascend 910B4) |
| CANN | 8.1+ |
| CMake | 3.16+ |
| Python | 3.9+ |
| PyTorch | 需安装 torch_npu |

## 快速开始

### 方法一：PyTorch 扩展（快速验证）

```bash
cd Pdist/PdistKernel
# 在 pdist_custom_test.py 中配置 shape / p / dtype
bash run.sh -v Ascend910B4
```

### 方法二：独立调用（性能分析）

```bash
cd Pdist/PdistKernelInvocation
# 在 main.cpp 和 scripts/gen_data.py 中配置参数
bash run.sh -v Ascend910B4 -r npu
```

### 方法三：算子包构建与 ACLNN 验证

```bash
cd Pdist/PdistFramework
bash build.sh
./build_out/custom_opp_hce_aarch64.run

cd ../AclnnInvocationNaive
# 在 main.cpp 中配置参数
bash run.sh
```

## 算子架构

```
┌──────────────────────────────────────────────┐
│             Application Layer                │
│   PyTorch App │ ACLNN App │ Standalone App   │
├──────────────────────────────────────────────┤
│          PyTorch Extension Layer              │
│        PdistKernel (Pybind11 + Ascend C)     │
├──────────────────────────────────────────────┤
│         Operator Framework Layer              │
│   PdistFramework (Host + Kernel + Plugin)    │
├──────────────────────────────────────────────┤
│        Hardware Abstraction Layer             │
│         Ascend C / CANN (NPU Hardware)       │
└──────────────────────────────────────────────┘
```

## 关键优化技术

1. **Pattern 编码系统**：将距离类型 + 数据类型 + 数据规模编码为单一 Pattern 值，实现策略的自动分派
2. **并行负载均衡**：按输出元素数划分任务，保证每个 AI Core 返回 32B 对齐的数据块
3. **双缓冲流水线**：IO 与计算重叠执行，配合行向量复用最大化 AI Core 利用率
4. **非对齐处理**：DataCopyPad 零填充 + 尾核截断，适配任意输入长度
5. **小数据全载入**：整体数据 < 160KB 时一次性加载至 L2 Cache，避免逐条 IO 开销

## 详细文档

算子的完整设计文档（Tiling 切分设计、策略类设计、流水线图解、全部实验数据）请参阅 [Pdist/README.md](Pdist/README.md)。

## 团队

**BUPT-ParCIS** — 北京邮电大学

## License

本项目仅供学习与竞赛交流使用。
