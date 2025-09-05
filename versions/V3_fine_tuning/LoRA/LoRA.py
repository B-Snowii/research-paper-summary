from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import pandas as pd
import evaluate
import numpy as np
import os
import argparse

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

os.environ.setdefault("HF_HOME", "./.hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "./.hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "./.hf/datasets")
os.environ.setdefault("HF_HUB_CACHE", "./.hf/hub")

def get_compute_dtype(force_cpu: bool = False):
    if not force_cpu and torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
parser.add_argument("--max_steps", type=int, default=1)
parser.add_argument("--num_eval_samples", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--force_cpu", action="store_true")
cli = parser.parse_args()

model_name = cli.model_name
compute_dtype = get_compute_dtype(force_cpu=cli.force_cpu)
device_str = "cuda" if (torch.cuda.is_available() and not cli.force_cpu) else "cpu"
print(f"Using device: {device_str}, dtype: {compute_dtype}")

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name)

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

def tokenize_function(example):
    start_prompt = "summarize the following conversation."
    end_prompt = 'summary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True).input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True).input_ids
    return example


huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
dataset

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id','topic', 'dialogue', 'summary',])
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index %100 ==0, with_indices=True)#only those 100 multiple id reserved

def train_and_eval_lora(variant_name: str, lora_r: int, lora_alpha: int, num_eval_samples: int, max_new_tokens: int):
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q","v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model_local = get_peft_model(base_model.clone() if hasattr(base_model, 'clone') else AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=compute_dtype), lora_cfg)

    out_dir = f"./{variant_name}-dialogue-summary-training-{str(int(time.time()))}"
    train_args = TrainingArguments(
        output_dir=out_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=1,
        logging_steps=1,
        max_steps=cli.max_steps,
        dataloader_pin_memory=False,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model_local,
        args=train_args,
        train_dataset=tokenized_datasets["train"],
    )

    print(f"[Train] Start {variant_name} (r={lora_r})...")
    trainer.train()
    print(f"[Train] Done {variant_name}.")

    dialogues_local = dataset['test'][0:num_eval_samples]['dialogue']
    human_refs_local = dataset['test'][0:num_eval_samples]['summary']
    preds_local = []
    for dlg in dialogues_local:
        prompt_local = f"summarize the following conversation {dlg} summary:"
        input_ids_local = tokenizer(prompt_local, return_tensors="pt").input_ids
        gen_out = model_local.generate(
            input_ids=input_ids_local,
            generation_config=GenerationConfig(max_new_tokens=max_new_tokens)
        )
        text_out = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        preds_local.append(text_out)

    try:
        rouge_metric = evaluate.load('rouge')
        rouge_results_local = rouge_metric.compute(
            predictions=preds_local,
            references=human_refs_local[0:len(preds_local)],
            use_aggregator=True,
            use_stemmer=True,
        )
    except Exception as e:
        print(f"[Eval] ROUGE failed for {variant_name}: {e}")
        rouge_results_local = None

    print(f"{variant_name} trainable params info:")
    print(print_number_of_trainable_model_parameters(model_local))
    print(f"{variant_name} ROUGE:")
    print(rouge_results_local)
    return rouge_results_local

def print_improvements(base_name, base_results, other_name, other_results):
    if base_results is None or other_results is None:
        print(f"Skip diff: {other_name} vs {base_name} (missing results)")
        return
    print(f"{other_name} over {base_name}")
    for key in base_results.keys():
        diff = (other_results[key] - base_results[key]) * 100
        print(f"{key}: {diff:.2f}%")

# Run variants: r in [4,8,16,32,64], baseline = r4
print("\n=== Training and evaluating LoRA variants (r=4/8/16/32/64) ===")
results = {}
for r in [4, 8, 16, 32, 64]:
    results[r] = train_and_eval_lora(f"LoRA-{r}", lora_r=r, lora_alpha=r, num_eval_samples=cli.num_eval_samples, max_new_tokens=cli.max_new_tokens)

print("\n=== Improvements over r=4 (baseline) ===")
base = results.get(4)
for r in [8, 16, 32, 64]:
    print_improvements("LoRA-4", base, f"LoRA-{r}", results.get(r))

# Save results to CSV
try:
    rows = []
    base_metrics = results.get(4) or {}
    for r, metrics in results.items():
        if metrics is None:
            continue
        row = {"rank": r}
        row.update(metrics)
        if base_metrics:
            for k in metrics.keys():
                row[f"impr_vs_r4_{k}"] = (metrics[k] - base_metrics[k]) * 100
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(".", "results_lora_ranks.csv")
        df.sort_values("rank").to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
except Exception as e:
    print(f"[Save] Failed to write CSV: {e}")

# Exit early to avoid running the legacy block below
raise SystemExit

#peft_trainer.train()
#peft_model_path="./peft-dialogue-summary-checkpoint-local"
#peft_trainer.model.save_pretrained(peft_model_path)
#tokenizer.save_pretrained(peft_model_path)
instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
peft_model = PeftModel.from_pretrained(peft_model_base, 
'./peft-dialogue-summary-checkpoint-from-s3/', 
torch_dtype=torch.bfloat16, is_trainable=False)
#is_trainable=False -- ONLY INFERENCE, NOT train, so the percentage of trainable model parameters: 0.00%
print(print_number_of_trainable_model_parameters(peft_model))


index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']
prompt = f"summarize the following conversation.{dialogue} summary:"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

dash_line = '-' * 99
print(dash_line) 
print(f'Baseline Human Summary: \n {human_baseline_summary}')
print(dash_line)
print(f'Original Model: \n {original_model_text_output}')
print(dash_line)
print(f'Instruct Model: \n{instruct_model_text_output}')
print(dash_line)
print(f'PEFT Model: {peft_model_text_output}')

dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']
original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"summarize the following conversation {dialogue} summary:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    human_baseline_text_output = human_baseline_summaries[idx]
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)#use append to add the value to the list
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))

    df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries','original_model_summaries','instruct_model_summaries','peft_model_summaries'])
    df

rouge = evaluate.load('rouge')

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

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)
print('PEFT MODEL:')
print(peft_model_results)

results = load_dataset("nash5657/dialogue-summary-training-results", split="train")
human_baseline_summaries = results['human_baseline_summaries']
original_model_summaries = results['original_model_summaries']
instruct_model_summaries = results['instruct_model_summaries']
peft_model_summaries = results['peft_model_summaries']

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

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)
print('PEFT MODEL:')
print(peft_model_results)

print("over original model")
improvement = (np.array(list(peft_model_results.values()))) - (np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}:{value*100:.2f}%')

print("over instruct model")
improvement = (np.array(list(peft_model_results.values()))) - (np.array(list(instruct_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}:{value*100:.2f}%')