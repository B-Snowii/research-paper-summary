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


#################################use data from huggingface#############
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
example_indices = [666] #pick up some samples for testing
# original: dash_line = '-'.join('' for x in range (50))
dash_line =  '-' * 99
for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example', i+1)
    print(dash_line)
    print('ID')
    print(dataset['test'][index]['id'])
    print('Original Version')
    print(dataset['test'][index]['dialogue'])#train, test, validation
    print('Summary')
    print(dataset['test'][index]['summary'])
    print('Topic')
    print(dataset['test'][index]['topic'])
    print(dash_line)
    print()#new line \n


##############################encoder &decoder -Flan-T5##############
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#use the trained transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
sentence = "What is the time, Amy?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')#encode token to num, output PyTorch: nput_ids and attention_mask
sentence_decoded = tokenizer.decode(
    sentence_encoded["input_ids"][0],
    skip_special_tokens=True
)#decode num into token
print('encoded')
print(sentence_encoded["input_ids"][0])#dedimension,needs 1D
print()
print('decoded')
print(sentence_decoded)

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,#control output length, avoid cutoff issues, and efficiency. adjustable
        )[0],#[0] extract the first and only one output
        skip_special_tokens=True #remove special characters like<\s>
    )
    print(dash_line)
    print('Example',i+1)
    print(dash_line)
    print(f'input:\n{dialogue}') #f:embed variables inside a string
    print(dash_line)
    print(f'summary:\n{summary}')
    print(dash_line)
    print(f"example{i+1}:")
    print(dash_line)
    print(f'model generation - without prompt:\n{output}\n')#it will create the 'next discussion', but for example it only provided a summary that sb will go
#the print root must inside the for loop, or it will only print the last result

######################################Zero-shot########################
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt=f"""
summarize the conversations.
{dialogue}
summary:
""" #if we want to generate multiple lines, then we use f & triple "
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True #remove special characters like<\s>
    )                
    print(dash_line)
    print('Example',i+1)
    print(dash_line)
    print(f'input:\n{prompt}') #f:embed variables inside a string
    print(dash_line)
    print(f'summary:\n{summary}')
    print(dash_line)
    print(f'model generation - zero shot:\n{output}\n')


#######choose training data
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
        dialogue = dataset['test'][example_index_to_summarize]['dialogue']
        prompt += f"""
Dialogue:
{dialogue}
What was going on?
"""#provide sth that wanting for result
    return prompt


########one-shot
example_indices_full = [80]#pick a sample
example_index_to_summarize = 666#pick one that waiting for solution
one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
#print(one_shot_prompt) -- too long

summary = dataset['test'][example_index_to_summarize]['summary']

generation_config = GenerationConfig(max_new_tokens=50)#do_sample=true(use the distribution instead of the most possible one)

inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        )[0],
        skip_special_tokens=True
)
print(dash_line)
print(f'baseline human summary:\n{summary}')
print(dash_line)
print(f'model generation - one shot:\n{output}\n')
##########few-shot-5shots
example_indices_full = [80,160,240,320,400]#pick a sample
example_index_to_summarize = 666#pick one that waiting for solution
few_shot_prompt5 = make_prompt(example_indices_full, example_index_to_summarize)
#print(few_shot_prompt5) --too long

summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(few_shot_prompt5, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        )[0],
        skip_special_tokens=True
)
print(f'raw:\n{dataset["test"][example_index_to_summarize]["dialogue"]}') #f:embed variables inside a string
print(dash_line)
print(f'baseline human summary:\n{summary}')
print(dash_line)
print(f'model generation - few shot5:\n{output}\n')


#sample=True tells the model to randomly sample the next word/token based on their probabilities, rather than always picking the most probable one 

#few_shot_prompt10 = make_prompt(example_indices_full, example_index_to_summarize)
#print(few_shot_prompt10) --too long

#summary = dataset['test'][example_index_to_summarize]['summary']

#generation_config = GenerationConfig(max_new_tokens=50,do_sample=True)





