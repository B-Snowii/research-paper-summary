#tentative
#Token indices sequence length is longer than the specified maximum sequence length for this model (521 > 512). Running this sequence through the model will result in indexing errors

#original
import os #interact with the operating system and files
#pip install torch #pip install torchdata 
# #pip install --disable-pip-version-check  #check install and version 2.6.0 & 0.11.0
#pip install transformers (4.49.0)

#I used pip at first, but not worked. So i got pip for terminal and %pip for notebook
# %pip install -U datasets
# %pip install --disable-pip-version-check torch torchdata
# %pip install transformers
import sys
print(sys.executable)


#################################use data from huggingface#############
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
example_indices = [666] #pick up some samples for testing
# original: dash_line = '-'.join('' for x in range (50))
dash_line =  '-' * 99



##############################encoder &decoder -Flan-T5##############
model_name='google/long-t5-tglobal-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#use the trained transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
sentence = "What is the time, Amy?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')#encode token to num, output PyTorch: nput_ids and attention_mask
sentence_decoded = tokenizer.decode(
    sentence_encoded["input_ids"][0],
    skip_special_tokens=True
)#decode num into token


for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt', max_length=1024, truncation=True)
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,#control output length, avoid cutoff issues, and efficiency. adjustable
        )[0],#[0] extract the first and only one output
        skip_special_tokens=True #remove special characters like<\s>
    )

def make_prompt(example_indices_full, example_index_to_summarize):#define function_name(parameters)
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt += f"""
Dialogue:
{dialogue}
What was going on?
{summary}
"""#provide one example
    # Only add the 'to summarize' dialogue ONCE at the end
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prompt += f"""
Dialogue:
{dialogue}
What was going on?
"""#provide sth that wanting for result
    return prompt


##########few-shot-10shots
example_indices_full = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,160,240,320,400,480,560,640,720,800,880,960,1040,1120,1200,1280,1360,1440,1450,1460]#pick a sample
example_index_to_summarize = 666#pick one that waiting for solution
instruction = (
    "Below are several examples of ideal summaries from previous data. "
    "Please carefully study these samples and generate a new summary for the given dialogue, "
    "emulating the style, tone, and quality of the former examples. "
    "Your summary must be clear, formal, and concise."
)
few_shot_prompt10 = instruction + '\n' + make_prompt(example_indices_full, example_index_to_summarize)
#print(few_shot_prompt10) --too long

summary = dataset['test'][example_index_to_summarize]['summary']

generation_config = GenerationConfig(max_new_tokens=10,do_sample=True,top_k=5)

inputs = tokenizer(few_shot_prompt10, return_tensors='pt')

output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
        )[0],
        skip_special_tokens=True
)
print(dash_line)
print(f'baseline human summary:\n{summary}')
print(dash_line)
print(f'model generation - few shot10:\n{output}\n')

# ================= DeepSeek API Integration =================
import requests

def deepseek_generate_summary(prompt, api_url, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Set your DeepSeek API endpoint and key
api_url = "https://api.deepseek.com/v1/chat/completions"
api_key = ""  # Provide your API key via environment variable or config

# Use the same prompt as for the local model
prompt = few_shot_prompt10

deepseek_summary = deepseek_generate_summary(prompt, api_url, api_key)

print(dash_line)
print("DeepSeek model generation - few shot10:")
print(deepseek_summary)
print(dash_line)









