# AI Summary Detoxification RLHF Project

**English** | [中文](README.zh-CN.md)

## Project Overview

This project implements a dialogue summary detoxification system based on Reinforcement Learning from Human Feedback (RLHF). By using PPO (Proximal Policy Optimization) algorithm and toxicity detection models, it fine-tunes the FLAN-T5 model to generate safer, non-toxic summaries.

## Why Detoxification is Needed?

Facing the limitations of traditional filtering methods such as false positives and inability to recognize semantic disguises, AI systems need:

- **Input Sanitization**: Filter hate/discriminatory language in user-generated content (e.g., variant attacks)
- **Output Protection**: Ensure generated content complies with ethical standards (medical/legal sensitive scenarios)
- **Compliance Requirements**: Dynamically adapt to multi-national regulatory standards

This project adopts RLHF (Reinforcement Learning from Human Feedback) methods, which can understand the meaning and intent of language while ensuring safety while maintaining content quality.

## Project Achievements

### 📊 Detoxification Effectiveness

- **Toxicity Score Improvement**:
  - **Average Toxicity Improvement: +9.08%** ✅
  - **Standard Deviation Improvement: +28.75%** ✅

### 🎯 Technical Features

- Uses FLAN-T5-base as the base model
- Integrates Facebook RoBERTa toxicity detection model
- Employs PPO algorithm for reinforcement learning training
- Supports LoRA (Low-Rank Adaptation) efficient fine-tuning

## Experimental Results

### Sample Comparison Results

The following shows a comparison of samples before and after detoxification:

| Sample | Original Summary | Detoxified Summary | Reward Score Improvement |
|--------|------------------|-------------------|-------------------------|
| 1 | <pad> Li Hong's sorry that Alice can't see a class this morning.</s> | <pad> Alice isn't allowed to visit Mrs. Brown because Alice's mother is ill. They give her a recommendation.</s> | +0.558 |
| 2 | <pad> #Person1# tells #Person2# #Person1# is forming a music band and has a man who plays guitars and bass. #Person2# tells #Person1# he and #Person1#'s singer won't answer have enough room. They already have heard of the other members of the band's name but #Person1# is not so sure. *Person2# will audition this weekend so #Person1# can practice with the drummers.</s> | <pad> #Person1# suggests #Person2# to form a music band and find the members of the group who are funny to play guitar, bass, guitar, and a singer.</s> | +0.507 |
| 3 | <pad> #Person1# says #Person2# works hard on the paper and says it was worth the time. #Person2# gives #Person1# the teacher's approval.</s> | <pad> #Person1#'s paper has been proofreadized by her mom and #Person1# praises her hard work. #Person1# then attempts to say something positive in the meeting. #Person1# agrees with the teacher.</s> | +0.478 |
| 4 | <pad> #Person2# likes the restaurant but #Person2# conveniently abandons it because it's a new restaurant but it's not the kind of restaurant. #Person2# thinks that the service isn't good and wants to eat at another restaurant.</s> | <pad> #Person1# and #Person2# are helping to find out what's been different about the restaurant but gazes at the food. #Person1# agrees that #Platinor accidentally gets in and all the guards were good.</s> | +0.386 |
| 5 | <pad> Judy and #Person1# are surprised when they hear Richard sack someone by his manager. Judy agrees the person had been fired. They are surprised that everybody in the company thinks it's true.</s> | <pad> Ellen and Judy are talking about a fire at the company. While Judy is joking about the news. Judy isn't surprised and asks Judy about it.</s> | +0.379 |

## Core Components

### 📊 Dataset
- **Dataset**: DialogSum dataset from Hugging Face `knkarthick/dialogsum`
- **Data Characteristics**: Contains multi-turn dialogues and corresponding summaries, dialogue length between 200-1000 characters
- **Data Split**: 80% training set, 20% test set, using fixed random seed for reproducibility
- **Preprocessing**: Converts dialogues to summary task prompt format

### 🏗️ Model Architecture
- **Base Model**: FLAN-T5-base (250M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Toxicity Detection**: Facebook RoBERTa toxicity detection model

### ⚙️ Training Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Reward Function**: Based on toxicity detection model logits
- **Training Rounds**: Configurable training steps
- **Learning Rate**: Adaptive learning rate adjustment

## Installation Steps

1. **Clone the Project**
   ```bash
   git clone https://github.com/B-Snowii/ai-summary-detoxification-RLHF.git
   cd ai-summary-detoxification-RLHF
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Model**
   
   The project requires downloading pre-trained PEFT model checkpoints. This model is a FLAN-T5 model fine-tuned for dialogue summarization.
   
   ```bash
   # Create model directory
   mkdir peft-dialogue-summary-checkpoint-from-s3
   
   # Download pre-trained model using AWS CLI (recommended method)
   aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/
   ```

## License

This project is licensed under the MIT License.