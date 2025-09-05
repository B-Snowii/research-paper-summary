import datasets
from transformers import AutoTokenizer
import numpy as np
####################################loading model and dataset##########
dataset = datasets.load_dataset(
    "parquet",
    data_files="./data/preprocessed_dataset.parquet",
    split="train"
)
#print(dataset)

#use shard to cut them into 10 parts
dataset = dataset.shard(num_shards=10, index=0)#index = 0 means the first shred is using
#print(dataset)
model_path_or_name = "upstage/SOLAR-10.7B-V1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name,
    use_fast=False
)
#testing the model
out = tokenizer.tokenize("Hello, world! Here's a long sentence")
out

##################################convert words to numbers##########
def tokenization(example):
    tokens = tokenizer.tokenize(example["text"])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]#mark before and after
    example["input_ids"] = token_ids
    example["num_tokens"] = len(token_ids)
    return example

dataset = dataset.map(tokenization, load_from_cache_file=False)
print(dataset)

#test parts of the dataset
sample = dataset[3]
print("text", sample["text"][:30]) # 
print("\ninput_ids", sample["input_ids"][:30])
print("\nnum_tokens", sample["num_tokens"])
total_tokens = np.sum(dataset["num_tokens"])
print(f"Total number: {total_tokens}")

####################################cut the dataset into equal parts########
input_ids = np.concatenate(dataset["input_ids"])#connect datas by their ids
max_seq_length = 32#make sure each part is 32 tokens
total_length = len(input_ids) - len(input_ids) % max_seq_length#calculate the length
print(total_length)
input_ids = input_ids[:total_length]#throw aways those extra tokens (out of divided range)
print(input_ids.shape)
input_ids_reshaped = input_ids.reshape(-1,max_seq_length).astype(np.int32)#-1=totle/max_length, automatically calculate the total number of parts(32 each), np.32 to save spaces
input_ids_reshaped.shape
type(input_ids_reshaped)

####################################save the revised dataset#########
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids":input_ids_list}#define the column name
)
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrained_dataset.parquet")

