[ En ](README.md) | [ 中文 ](README.zh-CN.md)

# Academic Paper Summary Generator · Showcase
This repository showcases a workflow and key implementation for generating professional around 25-word summaries for academic papers, built for a research institute.

- Task: Produce around 25-word academic summaries for many papers weekly, used in weekly Research alerts.

## Key Outcomes
- Time cost: Before over 40 hrs/week → After ≤2 hrs/week
- Quality assurance: Method/topic accuracy and consistent style
- Reusability: MCP pipeline + edits data

## Background & Goals
- Time constraint: Summaries finalized within 4 days due to time sensitivity.

### Practical Challenges
- High volume and urgency
- Broad topics and methods; high comprehension cost
- Repeated internal edits and communication

## Version Progression
### V1 · Prompt Engineering
- Tried: GPT‑4.1, DS‑V3, DS‑R1, NotebookLM, Flan‑T5, T5‑Large; varied shot counts.
- Result: Under strict length, method accuracy, and academic tone constraints, outputs suited general readership rather than academic domain needs.

### V2 · Pretraining
- Tried: Multiple weight initializations (including downscaling and upscaling).
- Challenges:
  - High-cost curation of domain-fit corpora; hard to cover all topics.
  - Novelty of working papers; insufficient similar materials for effective pretraining.

### V3 · Fine‑tuning / PEFT
- Plan: Full fine-tuning and PEFT (Adapter, LoRA, P‑tuning).
- Constraint: No available GPU for training; only evaluation of community fine-tuned models.

### V4 · MCP workflow (process supervision + CoT)
- Pipeline: Topic/Method Identification → CoT → Draft → Constraint Checks → Style Rewrite → Self-Scoring
- Models: Llama3, Deepseek‑R1
- Framework: https://github.com/B-Snowii/ai-academic-summary-mcp

## Sample Summaries
- Example 1: This paper examines how belief vs taste drivers shape early-stage ESG collaboration, via randomized experiments with founders and VCs; methods: randomized assignment and survey-based measures.
- Example 2: This paper uses a calibrated life-cycle model to value reductions in health risks and quantify insurance, financial, and fiscal impacts; methods: structural modeling with parameter calibration.
- Example 3: This paper exploits staggered adoption of hospital pay-transparency laws to study effects on patient satisfaction; methods: panel data analysis with staggered policy timing and fixed effects.

## License
- MIT License

---

#
