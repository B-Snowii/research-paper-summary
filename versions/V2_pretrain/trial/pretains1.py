#import warnings
#warnings.filterwarnings("ignore") #what will happen if not ignore?
import datasets
import os
import requests
import heapq
import re
import urllib
import langid
pretraining_dataset = datasets.load_dataset(
    "vilm/RedPajama-v2-small",#searched redpajama on the dataset, download a small one
    split="train",
)
#print(pretraining_dataset)#num_rows: 500000
#pretraining_dataset = pretraining_dataset.select_columns(["text"])
#print(pretraining_dataset[0]["text"][:500])#in this dataset it only includes text
instruction_dataset = datasets.load_dataset(
    "c-s-ale/alpaca-gpt4-data",#where can i find this dataset?
    split="train"
)
#print(instruction_dataset)
#i=0
#print("Instruction:"+instruction_dataset[i]["instruction"]
    #+"\nInput:"+ instruction_dataset[i]["input"]
    #+"\nOutput:"+ instruction_dataset[i]["output"]
#)

code_dir="./code"#relative path, so i generate a file called code
urls = [
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/double_linear_search_recursion.py",
    "https://raw.githubusercontent.com/KosingZhu/tensorflow/master/tensorflow/python/tools/module_util.py",
    "https://raw.githubusercontent.com/EricRemmerswaal/tensorflow/master/tensorflow/python/distribute/distribute_coordinator_context.py",
    "https://raw.githubusercontent.com/computationalartist/tensorflow/master/tensorflow/python/ops/numpy_ops/integration_test/benchmarks/numpy_mlp.py",
    "https://raw.githubusercontent.com/Van-an/tensorflow/master/tensorflow/python/distribute/coordinator/values.py",
    "https://raw.githubusercontent.com/nkgwer/tensorflow/master/tensorflow/lite/tools/visualize.py",
    "https://raw.githubusercontent.com/gitblazer/youtube-dl/master/youtube_dl/version.py",
    "https://raw.githubusercontent.com/Joshua-Barawa/My-Photos/master/venv/lib/python3.8/site-packages/django/contrib/messages/__init__.py",
    "https://raw.githubusercontent.com/PaliC/pytorch/master/test/fx/test_subgraph_rewriter.py"
]

#for url in urls:
#    print(f"working now:{url}")
#    response = requests.get(url)
#    file_name = os.path.basename(url)
#    file_path = os.path.join(code_dir, file_name)

    #with open(file_path,"wb") as file:
        #file.write(response.content)

#files = os.listdir(code_dir)
#for file in files:
    #print(file)#confirm download

code_dataset = []
for file in os.listdir(code_dir):
    code_dataset.append(
        {'text':open(os.path.join(code_dir,file),'r').read()}
    )
code_dataset = datasets.Dataset.from_list(code_dataset)
#print(code_dataset)
dataset = datasets.concatenate_datasets(
    [pretraining_dataset, code_dataset]
)
#print(dataset)

def paragraph_length_filter(x):
    """Returns False if a page has too few lines or lines are too short."""
    lines = x['text'].split("\n")#count the number of line break
    if(
        len(lines)<3
        or min(heapq.nlargest(3,[len(line) for line in lines])) <3
    ):
        return False
    return True

dataset = dataset.filter(
    paragraph_length_filter,
    load_from_cache_file=False
)
dataset.num_rows

def find_duplicates(paragraphs):#calculate the repeted sentence num and their length(number,length)
    """find the number of repetitions in the paragraphs"""
    unique_x=set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in paragraphs:
        if element in unique_x:
            duplicate_chars +=len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars

def paragraph_repetition_filter(x):
    """return false if a page hass too many repetitions"""
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False
    if char_duplicates / len(text)>0.2:
        return False
    return True

dataset = dataset.filter(
    paragraph_repetition_filter,
    load_from_cache_file=False#ask it to apply the function again
)
dataset.num_rows

def dedup(ds):
    def dedup_func(x):
        """use this to remove duplicates"""
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True
    unique_text = set()
    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    return ds
dataset = dedup(dataset)

#Function	Removes...	Example scenario it catches
#find_duplicates	Repeated paragraphs within a document	"A\n\nB\n\nA" #help to measure the proportion in next step
#paragraph_repetition_filter	Documents with excessive internal repetition	"A\n\nA\n\nA\n\nA\n\nA" (too many repeats)
#dedup	Duplicate documents (identical "text")	Two rows: both "X\n\nY\n\nZ" #repeated records

def english_filter(ds):
    def is_eng(x):
        lang, _ = langid.classify(x['text'])
        return lang == 'en'
    ds = ds.filter(is_eng, load_from_cache_file=False, num_proc=1)
    return ds
dataset = english_filter(dataset)
file_path = "./data/preprocessed_dataset.parquet"#Fast, compressed, typed but Not human-readable
dataset. to_parquet(file_path)
