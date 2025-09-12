# AI摘要去毒化RLHF项目

[English](README.md) | **中文**

## 项目简介

本项目实现了一个基于强化学习人类反馈（RLHF）的对话摘要去毒化系统。通过使用PPO（近端策略优化）算法和毒性检测模型，对FLAN-T5模型进行微调，使其生成的摘要更加安全、无毒。

## 为什么需要去毒化？

面对传统过滤方法的误杀、无法识别语义伪装等缺陷，AI系统需要：

- **输入净化**：过滤用户生成内容中的仇恨/歧视性语言（如变体攻击）
- **输出防护**：确保生成内容符合伦理规范（医疗/法律等敏感场景）
- **合规要求**：动态适配多国监管标准

本项目采用RLHF（强化学习人类反馈）方法，能够理解语言的含义和意图，在保持内容质量的同时确保安全性。

## 项目成果

### 📊 去毒化效果

- **毒性分数改善**：
  - **平均毒性改善：+9.08%** ✅
  - **标准差改善：+28.75%** ✅

### 🎯 技术特点

- 使用FLAN-T5-base作为基础模型
- 集成Facebook RoBERTa毒性检测模型
- 采用PPO算法进行强化学习训练
- 支持LoRA（低秩适应）高效微调

## 实验结果展示

### 样本对比结果

以下是部分样本的去毒化前后对比：

| 样本 | 原始摘要 | 去毒化后摘要 | 奖励分数提升 |
|------|----------|-------------|-------------|
| 1 | <pad> Li Hong's sorry that Alice can't see a class this morning.</s> | <pad> Alice isn't allowed to visit Mrs. Brown because Alice's mother is ill. They give her a recommendation.</s> | +0.558 |
| 2 | <pad> #Person1# tells #Person2# #Person1# is forming a music band and has a man who plays guitars and bass. #Person2# tells #Person1# he and #Person1#'s singer won't answer have enough room. They already have heard of the other members of the band's name but #Person1# is not so sure. *Person2# will audition this weekend so #Person1# can practice with the drummers.</s> | <pad> #Person1# suggests #Person2# to form a music band and find the members of the group who are funny to play guitar, bass, guitar, and a singer.</s> | +0.507 |
| 3 | <pad> #Person1# says #Person2# works hard on the paper and says it was worth the time. #Person2# gives #Person1# the teacher's approval.</s> | <pad> #Person1#'s paper has been proofreadized by her mom and #Person1# praises her hard work. #Person1# then attempts to say something positive in the meeting. #Person1# agrees with the teacher.</s> | +0.478 |
| 4 | <pad> #Person2# likes the restaurant but #Person2# conveniently abandons it because it's a new restaurant but it's not the kind of restaurant. #Person2# thinks that the service isn't good and wants to eat at another restaurant.</s> | <pad> #Person1# and #Person2# are helping to find out what's been different about the restaurant but gazes at the food. #Person1# agrees that #Platinor accidentally gets in and all the guards were good.</s> | +0.386 |
| 5 | <pad> Judy and #Person1# are surprised when they hear Richard sack someone by his manager. Judy agrees the person had been fired. They are surprised that everybody in the company thinks it's true.</s> | <pad> Ellen and Judy are talking about a fire at the company. While Judy is joking about the news. Judy isn't surprised and asks Judy about it.</s> | +0.379 |

## 核心组件

### 📊 数据集
- **数据集**：来自Hugging Face的`knkarthick/dialogsum`数据集
- **数据特点**：包含多轮对话和对应的摘要，对话长度在200-1000字符之间
- **数据分割**：80%训练集，20%测试集，使用固定随机种子确保可重现性
- **预处理**：将对话转换为摘要任务提示格式

### 🏗️ 模型架构
- **基础模型**：FLAN-T5-base（250M参数）
- **微调方法**：LoRA（低秩适应）
- **毒性检测**：Facebook RoBERTa毒性检测模型

### ⚙️ 训练配置
- **算法**：PPO（近端策略优化）
- **奖励函数**：基于毒性检测模型的logits
- **训练轮数**：可配置的训练步数
- **学习率**：自适应学习率调整

## 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/B-Snowii/ai-summary-detoxification-RLHF.git
   cd ai-summary-detoxification-RLHF
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载预训练模型**
   
   项目需要下载预训练的PEFT模型检查点。该模型是经过对话摘要指令微调的FLAN-T5模型。
   
   ```bash
   # 创建模型目录
   mkdir peft-dialogue-summary-checkpoint-from-s3
   
   # 使用AWS CLI下载预训练模型（推荐方式）
   aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/
   ```

## 许可证

本项目采用MIT许可证。