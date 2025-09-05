import os
import warnings
warnings.filterwarnings('ignore')
#%pip install -U pip
#%pip install "torch>=2.7,<2.9"
#%pip install "transformers>=4.45,<5" "tokenizers>=0.20.0" "datasets>=2.19" "evaluate>=0.4.2" "rouge_score>=0.1.2" "loralib>=0.1.1" "peft>=0.12.0"
#%pip install "torchdata>=0.8"
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
dataset
model_name='google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\n all model parameters: {all_model_params}\npercentage of trainable model parameters:{100 * trainable_model_params / all_model_params:.2f}%"
print(print_number_of_trainable_model_parameters(original_model))

index = 200
dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.
{dialogue}
summary:"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'*99
print(dash_line)
#print(f'input prompt:{prompt}')
print(f'baseline human summary:{summary}')
print(f'model generation - zero shot:\n{output}')


def tokenize_function(example):
    start_prompt = "summarize the following conversation."
    end_prompt = 'summary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id','topic', 'dialogue', 'summary',])
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index %100 ==0, with_indices=True)#only those 100 multiple id reserved

print(f"shapes of the datasets:")
print(f"training:{tokenized_datasets['train'].shape}")
print(f"validation:{tokenized_datasets['validation'].shape}")
print(f"test:{tokenized_datasets['test'].shape}")

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,#how many times the training algorithm will work through the entire training dataset.
    weight_decay=0.01,#prevent overfitting
    logging_steps=1,#log metrics every single step, only for output
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# trainer.train()
# trainer.save_model()
# print(f"Model saved to: {output_dir}")
# checkpoint_model: truocpham/flan-dialogue-summary-checkpoint(Hugging Face)

#############################after fine-tuned###############
instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']
prompt = f"summarize the following conversation. {dialogue} summary:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
#num_beams: beam searching for the num of possible next tokens
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')

rouge = evaluate.load('rouge')
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']
original_model_summaries = []
instruct_model_summaries = []
for _, dialogue in enumerate(dialogues):#_ is a index, but don't care the value
    prompt = f"summarize the following conversation. {dialogue} summary:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)#append: add the new item to the end of the list

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))
#zip used to combine lists together
df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries'])
df

original_model_results = rouge.compute(
    predictions=original_model_summaries,#generated by model
    references=human_baseline_summaries[0:len(original_model_summaries)],#generated by human
    use_aggregator=True,#give overall score, avg score across all summary pairs
    use_stemmer=True,#stem words: playing/played-->play
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print(f'original model: {original_model_results}')
print(f'instruct model: {instruct_model_results}')

dataset = load_dataset("nash5657/dialogue-summary-training-results")
human_baseline_summaries = dataset['train']['human_baseline_summaries']
original_model_summaries = dataset['train']['original_model_summaries']
instruct_model_summaries = dataset['train']['instruct_model_summaries']

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print(f'original model:{original_model_results}')
print(f'instruct model:{instruct_model_results}')

print("absolute percentage improvement of instruct model over original model")
improvement = (np.array(list(instruct_model_results.values()))) - np.array(list(original_model_results.values()))
for key, value in zip(instruct_model_results.keys(), improvement):
    print(f'{key}:{value*100:2f}%')

