# Challenge-Cup-2025 | BUPT-ParCIS

![Platform](https://img.shields.io/badge/platform-Ascend%20NPU-blue)
![CANN](https://img.shields.io/badge/CANN-8.1.RC1-orange)
![Model](https://img.shields.io/badge/model-Qwen2.5--3B--Instruct-green)
![Framework](https://img.shields.io/badge/framework-vLLM%20%2B%20vllm--ascend-purple)

“挑战杯”2025年度青年科技创新揭榜挂帅擂台赛 — 基于**昇腾 NPU** 的**大语言模型推理优化**，聚焦模型训练调优与性能加速，助力全栈自主 AI。

## 项目简介

本项目参加华为"揭榜挂帅"赛道，赛题要求在昇腾 NPU 硬件上对 3B 及以下参数量大语言模型进行推理优化，在**精度、性能、格式合规**三个维度上取得高分。

- **基座模型**：Qwen2.5-3B-Instruct（3B 参数量）
- **推理框架**：vLLM + vllm-ascend（昇腾后端插件）
- **硬件平台**：华为昇腾 NPU (Ascend 910B)
- **开发环境**：ModelArts + CANN 8.1.RC1

## 赛题任务

模型需要处理四类任务：

| 任务类型 | 说明 | 评估指标 |
|---------|------|---------|
| **math** | 数学推理求解 | 精确匹配（LaTeX boxed 答案） |
| **code-generate** | 根据函数签名和文档实现代码 | pass@3 |
| **choice** | 中英文多项选择题（ABCD） | 选项匹配 |
| **generic-generate** | 中英文通用问答 | 内容匹配 |

评分公式：**总分 = 0.4 x 精度分 + 0.4 x 性能分 + 0.2 x 格式分**

## 目录结构

```
Challenge-Cup-Huawei-2025/
├── 阶段A/                                    # A榜阶段
│   ├── competitioin_submission-JYC-1/         # 参赛提交包
│   │   ├── competition_model.py               # 核心推理代码（Competition类）
│   │   ├── prompt.py                          # Prompt模板与系统提示词
│   │   ├── data/                              # 训练/测试数据
│   │   │   ├── A-data.jsonl                   # A榜数据集
│   │   │   ├── few_shot_2.jsonl               # Few-shot示例
│   │   │   └── test.jsonl                     # 测试数据
│   │   ├── dependencies/                      # 离线依赖包
│   │   ├── custom_kernels/                    # 自定义算子目录
│   │   ├── requirements.txt                   # Python依赖列表
│   │   └── Qwen2.5-3B-Instruct/              # 模型文件
│   ├── 挑战杯指导文档.md                       # A榜完整指导文档
│   ├── 7月9日  华为茶思会会议纪要.md             # 专家指导会议纪要
│   ├── SH-05华为技术有限公司-推理大模型的训练调优与性能加速助力全栈自主AI比赛方案(2).pdf
│   └── 华为云Ascend C算子开发环境搭建手册S5赛季.docx
│
├── 阶段B/                                    # B榜阶段
│   ├── B-参考文档1.md                          # B榜指导文档
│   ├── B-参考文档2.md                          # B榜补充材料与QA
│   ├── 吴小鱼-赛题&评分标准解读.pdf
│   ├── 揭榜挂帅华为赛题算子示例.xlsx
│   └── 李大帅-算子解读  .pdf
│
├── 决赛答辩/                                  # 决赛材料
│   ├── BUPT-ParCIS-答辩PPT.pptx
│   └── 挑战杯-答辩材料.pdf
│
└── README.md
```

## 技术方案

### 推理流程

```
输入 (jsonl) → 按任务类型分组 → Prompt构建（含Few-shot） → vLLM批量推理 → 结果格式化输出
```

### 核心实现

- **批量推理**：按任务类型分组批处理，同类型共享 SamplingParams，充分利用 vLLM 并行能力
- **Prompt 工程**：针对每种任务设计独立的 System Prompt + 2-shot 示例模板，输出格式为 `<answer>` 标签包裹
- **差异化采样策略**：
  - `choice`：max_tokens=2048, temperature=0.8
  - `code-generate`：n=3（pass@3 评估）, max_tokens=2048
  - `generic-generate`：max_tokens=2048
  - `math`：max_tokens=2048
- **格式合规**：输出严格遵循 `...<answer>...</answer>` 格式，支持数学 LaTeX boxed 答案

### 环境要求

| 组件 | 版本 |
|------|------|
| 硬件 | 华为昇腾 NPU (Ascend 910B) |
| OS | EulerOS 2.10.7 |
| CANN | 8.1.RC1 |
| Python | 3.9.10 |
| PyTorch | 2.5.1 + torch-npu 2.5.1 |
| vLLM | 0.8.4 |
| vllm-ascend | 0.8.4rc2 |

## 提交说明

提交包结构（`competition_submission.zip`，不超过 10GB）：

```
competition_submission.zip
├── competition_model.py          # 入口：Competition类 + get_results方法
├── requirements.txt              # 依赖列表
├── dependencies/                 # 离线依赖包
├── custom_kernels/               # 自定义算子（.run文件）
└── Qwen2.5-3B-Instruct/          # 模型文件
```

判题流程：执行 `.run` 算子 → 安装依赖 → 调用 `get_results()` 推理判分。

## 团队

**BUPT-ParCIS** — 北京邮电大学

## License

本项目仅供学习与竞赛交流使用。
