[ 英文 ](README.md) | [ 中文 ](README.zh-CN.md)

# 学术论文专业摘要生成 · 集合

本仓库是一个**集合**，包含多种学术论文专业摘要生成的方法和实现，为研究机构打造。它包含四个不同版本（V1-V4），展示了从提示工程到高级MCP工作流的演进过程。

- 任务：每周对大量论文产出约25词的学术摘要，用于 weekly Research alert。

## 核心成果

- 任务花费时间：（程序完成前）每周40 小时 → （程序完成后）每周2 小时内
- 质量保障：方法、主题准确，风格一致
- 可复用：MCP 流程化 + 修改过程数据

## 背景与目标

- 时间限制：论文具有时效性，需在 4 天内完成总结。

### 现实挑战

- 论文数量大、时效性强
- 主题广、方法多，理解成本高
- 需反复根据内部标准修改，沟通成本高

## 版本演进

### V1 · Prompt Engineering
- 框架：https://github.com/B-Snowii/research-paper-summary-v1-prompt-engineering

- 尝试对象：GPT‑4.1、DS‑V3、DS‑R1、NotebookLM、Flan‑T5、T5‑Large 等，尝试不同的 shot 数量。
- 结果：在严格长度、方法准确性、学术特征等约束下，产出内容只适用大众阅读，而不适用于学术范畴。

### V2 · 预训练（Pretraining）
- 框架：https://github.com/B-Snowii/research-paper-summary-v2-pretraining

- 尝试：多种权重初始化（包含 downscaling 与 upscaling）。
- 难点：
  - 整理学术论文训练集成本极高，且难覆盖所有主题。  
  - 工作论文具新颖性，缺少足够同类材料进行有效预训练。

### V3 · 微调（Fine‑tuning / PEFT）
- 框架：https://github.com/B-Snowii/research-paper-summary-v3-finetuning

- 方案：全量微调与 PEFT（Adapter、LoRA、P‑tuning）。
- 受限：无可用 GPU 进行训练，仅能评测他人开源微调模型。

### V4 · MCP 工作流（过程监督与 CoT）
- 流水线：主题/方法识别 → CoT → 草案 → 约束核查 → 风格重写 → 自评打分
- 模型：Llama3、Deepseek‑R1
- 相关开源框架：https://github.com/B-Snowii/ai-academic-summary-mcp

### V5 · RLHF 去毒化
- 框架：https://github.com/B-Snowii/ai-summary-detoxification-RLHF
- 方法：基于人类反馈的强化学习进行内容去毒化
- 模型：FLAN-T5-base配合PPO算法和Facebook RoBERTa毒性检测
- 重点：确保AI生成的摘要安全无毒，同时保持质量

## 示例摘要

- Example 1: This paper examines how belief vs taste drivers shape early-stage ESG collaboration, via randomized experiments with founders and VCs; methods: randomized assignment and survey-based measures.
- Example 2: This paper uses a calibrated life-cycle model to value reductions in health risks and quantify insurance, financial, and fiscal impacts; methods: structural modeling with parameter calibration.
- Example 3: This paper exploits staggered adoption of hospital pay-transparency laws to study effects on patient satisfaction; methods: panel data analysis with staggered policy timing and fixed effects.

## 许可证

- 本项目采用 MIT 许可证。

## 相关仓库

本集合包含以下独立仓库，每个代表不同的方法：

- **V1：Prompt Engineering** — https://github.com/B-Snowii/research-paper-summary-v1-prompt-engineering
- **V2：Pretraining** — https://github.com/B-Snowii/research-paper-summary-v2-pretraining
- **V3：Fine‑tuning / PEFT** — https://github.com/B-Snowii/research-paper-summary-v3-finetuning
- **V4：MCP 流程实现** — https://github.com/B-Snowii/ai-academic-summary-mcp
- **V5：RLHF 去毒化** — https://github.com/B-Snowii/ai-summary-detoxification-RLHF