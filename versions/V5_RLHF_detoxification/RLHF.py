import os
import platform

# Expose a reusable initializer for Interactive Window usage
def init_env():
    _expected_types_env = os.environ.get('EXPECTED_INSTANCE_TYPES')
    _expected_type_single = os.environ.get('EXPECTED_INSTANCE_TYPE')
    _expected_types_raw = _expected_types_env or _expected_type_single or 'DELL,DESKTOP-FPSG7R5'
    allowed = [s.strip().lower() for s in _expected_types_raw.split(',') if s.strip()]

    current = (
        os.environ.get('HOSTNAME')
        or os.environ.get('COMPUTERNAME')
        or platform.node()
        or ''
    )

    print(f'Expected-instance types:{allowed}')
    print(f'currently chosen instance type:{current or "(unknown)"}')

    skip_check_local = os.environ.get('SKIP_INSTANCE_CHECK', '').lower() in {'1', 'true', 'yes'}
    if not skip_check_local:
        assert current, 'Error. HOSTNAME/COMPUTERNAME is empty; cannot validate instance type.'
        current_lower_local = current.lower()
        assert any(expected in current_lower_local for expected in allowed), (
            f'Error. You selected {current}, please select one of {allowed} instead'
        )
        print('Instance type has been chosen successfully')
    else:
        print('Instance type check skipped by SKIP_INSTANCE_CHECK')

    return allowed, current

# Allow multiple acceptable instance types via env.
# EXPECTED_INSTANCE_TYPES takes precedence (comma-separated),
# fallback to EXPECTED_INSTANCE_TYPE, then sensible defaults.
_expected_types_env = os.environ.get('EXPECTED_INSTANCE_TYPES')
_expected_type_single = os.environ.get('EXPECTED_INSTANCE_TYPE')
_expected_types_raw = _expected_types_env or _expected_type_single or 'DELL,DESKTOP-FPSG7R5'
allowed_instance_types = [s.strip().lower() for s in _expected_types_raw.split(',') if s.strip()]

instance_type_current = (
    os.environ.get('HOSTNAME')
    or os.environ.get('COMPUTERNAME')
    or platform.node()
    or ''
)

print(f'Expected-instance types:{allowed_instance_types}')
print(f'currently chosen instance type:{instance_type_current or "(unknown)"}')

skip_check = os.environ.get('SKIP_INSTANCE_CHECK', '').lower() in {'1', 'true', 'yes'}

if not skip_check:
    assert instance_type_current, 'Error. HOSTNAME/COMPUTERNAME is empty; cannot validate instance type.'
    current_lower = instance_type_current.lower()
    assert any(expected in current_lower for expected in allowed_instance_types), (
        f'Error. You selected {instance_type_current}, please select one of {allowed_instance_types} instead'
    )
    print('Instance type has been chosen successfully')
else:
    print('Instance type check skipped by SKIP_INSTANCE_CHECK')

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

#trl: transformer reinforcement learning
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate
import numpy as np
import pandas as pd

from tqdm import tqdm#visible progress
tqdm.pandas()

model_name="google/flan-t5-base"

#################################Preparing dataset######################

huggingface_dataset_name = "knkarthick/dialogsum"

dataset_original = load_dataset(huggingface_dataset_name)

dataset_original

def build_dataset(model_name, dataset_name, input_min_text_length, input_max_text_length):
    """Prepare dataset for training and testing.
    Parameters:
    -model_name:Tokenizer(str).
    -dataset_name:Dataset to load(str).
    -input_min_text_length: minimum length of the dialogues(int).
    -input_max_text_length: maximum

    Returns:
    - dataset_splits(datasets.dataset_dict.DatasetDict): dataset with train and test parts.
    """

    dataset = load_dataset(dataset_name, split="train")
    #dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False)
    def keep(sample):
        L = len(sample["dialogue"])
        return(L > input_min_text_length and L <= input_max_text_length)
    dataset = dataset.filter(keep, batched=False)
    #batched=False:progress one sample at a time, return boolean that determines whether the sample is kept

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(sample):
        prompt = f"Summarize the following conversation.\n\n{sample['dialogue']}\n\nSummary:"
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
    #20%test-80%train, shuffle: in sorted order; seed: for reproducibility(only be able when shuffle = True)
    return dataset_splits

dataset = build_dataset(model_name=model_name,
                        dataset_name=huggingface_dataset_name,
                        input_min_text_length=200,
                        input_max_text_length=1000)
print(dataset)


################################Loading Pre-trained model#####################
checkpoint_dir = r"./peft-dialogue-summary-checkpoint-from-s3"
assert os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Configure LoRA and load the adapter in trainable mode
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

peft_model = PeftModel.from_pretrained(
    base_model,
    checkpoint_dir,
    lora_config=lora_config,
    is_trainable=True,
)
peft_model.to("cpu")

def trainable_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable:{trainable_model_params}, all:{all_model_params},percentage of trainable model paramters: {100*trainable_model_params/all_model_params}"

print(trainable_parameters(peft_model))

################################PPO Model##########################
# Add a Value Head on top of the PEFT model for PPO training
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model,
    torch_dtype=torch.float32,
    is_trainable=True,
)

print(f'PPO model parameters to be updated: \n{trainable_parameters(ppo_model)}')
print(ppo_model.v_head)

###############################Reference Model#######################
ref_model = create_reference_model(ppo_model)
print(f'Reference model parameters to be updated \n {trainable_parameters(ref_model)}')

################################PRE-Detoxify the summaries################
toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)
toxicity_model.to("cpu")
print(toxicity_model.config.id2label)

###non-toxic_sample
non_toxic_text = "#Person 1# tells Tommy that he dislike the movie."
toxicity_input_ids = toxicity_tokenizer(non_toxic_text, return_tensors="pt").input_ids
logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [not hate, hate]:{logits.tolist()[0]}')

probabilities = logits.softmax(dim=1).tolist()[0]
print(f'probabilities[not hate, hate]:{probabilities}')
#reward: logit for "hate" (we want to minimize toxicity, so use hate logit as negative reward)
hate_index = 1
hate_reward = (logits[:,hate_index]).tolist()
print(f'reward: {hate_reward}')

##toxic_sample
toxic_text = "#Person 1# tells Tommy that this movie was terrible, dumb and stupid"
toxicity_input_ids = toxicity_tokenizer(toxic_text, return_tensors="pt").input_ids
logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [not hate, hate]:{logits.tolist()[0]}')

probabilities = logits.softmax(dim=1).tolist()[0]
print(f'probabilities[not hate, hate]:{probabilities}')
#reward: logit for "hate" (we want to minimize toxicity, so use hate logit as negative reward)
hate_index = 1
hate_reward = (logits[:,hate_index]).tolist()
print(f'reward: {hate_reward}')

###############################Toxicity evaluation: manually, logits and softmax##########################
device = "cpu"#= 0 if torch.cuda.is_available() else "cpu" for NYIDIA GPU

sentiment_pipe = pipeline("sentiment-analysis", model=toxicity_model, tokenizer=toxicity_tokenizer, device=-1)

reward_logits_kwargs = {
    "top_k": None, #return all scores
    "function_to_apply": "none",#retrieve raw logits-->PPO will use this one
    "batch_size":16
}

reward_probabilities_kwargs = {
    "top_k": None,
    "function_to_apply": "softmax",#range[0,1], more readable
    "batch_size":16
}

print(f"Reward model output: \n For non-toxic text\n{sentiment_pipe(non_toxic_text, **reward_logits_kwargs)}")
print(f"For non-toxic text\n{sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs)}")
print(f"For toxic text\n{sentiment_pipe(toxic_text, **reward_logits_kwargs)}")
print(f"For toxic text\n{sentiment_pipe(toxic_text, **reward_probabilities_kwargs)}")

####################################Toxicity evaluation (no evaluate.load)###########################
def compute_toxicity(texts):
    """Return list of hate probabilities in [0,1] using the sentiment pipeline."""
    results = sentiment_pipe(
        texts,
        top_k=None,
        function_to_apply="softmax",
        batch_size=16,
    )
    hate_probs = []
    for res in results:
        # res: list of {label, score}
        hate_prob = 0.0
        for item in res:
            if item.get("label") == "hate":
                hate_prob = float(item.get("score", 0.0))
                break
        hate_probs.append(hate_prob)
    return hate_probs

## quick check on samples
toxicity_score = compute_toxicity([non_toxic_text])
print(f"Toxicity score (non-toxic text):{toxicity_score}")
toxicity_score = compute_toxicity([toxic_text])
print(f"Toxicity score (toxic text):{toxicity_score}")

##Use this evaluator to evaluate our dataset
def evaluate_toxicity(model, tokenizer, dataset, num_samples):
    """Preprocess the dataset and split it into train and test parts
    
    Parameters:
    -model(trl model): Actor(model to be evaluated)
    -toxicity_evaluator: use the library tool
    -tokenier: Tokenizer to be used
    -dataset: the prepared dataset(length selection)
    -num_samples(int): maximum num of evaluation samples
    
    Returns:
    tuple: A tuple contains two numpy.float64 values:
    -mean: Mean of the sample toxicity
    -std: Standard deviation of the sample toxicity
    """

    max_new_tokens=100
    toxicities = []
    input_texts = []
    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break

        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        #padding: pad the input_ids to the same length, or pt will raise an error
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, top_k=0.0, top_p=1.0, do_sample=True)
        #random sample(unuse top p and top k)
        response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        toxicity_score = compute_toxicity([(input_text + " " + generated_text)])
        toxicities.extend(toxicity_score)
    mean = np.mean(toxicities)
    std = np.std(toxicities)

    return mean, std

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Safely set pad_token_id to be compatible across TRL versions
def set_model_pad_token(model, tok):
    try:
        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tok.pad_token_id
            return
    except Exception:
        pass
    try:
        if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "config"):
            model.pretrained_model.config.pad_token_id = tok.pad_token_id
            return
    except Exception:
        pass

set_model_pad_token(ppo_model, tokenizer)
set_model_pad_token(ref_model, tokenizer)


# Ensure generation_config exists on models (needed by some TRL versions)
def ensure_generation_config(model, tokenizer, fallback_model_name: str):
    try:
        has_gc = hasattr(model, "generation_config") and model.generation_config is not None
    except Exception:
        has_gc = False
    if not has_gc:
        try:
            gen_cfg = GenerationConfig.from_pretrained(fallback_model_name)
        except Exception:
            gen_cfg = GenerationConfig()
        if getattr(tokenizer, "eos_token_id", None) is not None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
        try:
            model.generation_config = gen_cfg
        except Exception:
            pass

ensure_generation_config(ppo_model, tokenizer, model_name)
ensure_generation_config(ref_model, tokenizer, model_name)
mean_before_detoxification, std_before_detoxication = evaluate_toxicity(model=ref_model, tokenizer=tokenizer, dataset=dataset["test"],num_samples=10)

print(f'toxicity [mean,std] before detox: [{mean_before_detoxification}, {std_before_detoxication}]')

###################################Detoxify the summaries##########################
##################Collate function#########
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
test_data = [
    {"key1": "v1a", "key2": "v2a", "key3": "v3a"},
    {"key1": "v1b", "key2": "v2b", "key3": "v3b"},
]
#list[dict] → dict[list]
#Before: '
# batch = [
#  {"text": "hello world", "label": 0, "length": 2},
#  {"text": "goodbye",     "label": 1, "length": 1},
#]
#After: 
#{
#  "text":   ["hello world", "goodbye"],
#  "label":  [0, 1],
#  "length": [2, 1],
#}
print(f"Collator input: {test_data}; Collator output: {collator(test_data)}")

######################PPO TRAINER
learning_rate=1.41e-5
max_ppo_epochs=1#one pass through the dataset, recommended [1,4], use the sample selected by 'step' and determine the training progress
mini_batch_size=4#mini-batch, recommended [1,8]
batch_size=16

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)


ppo_trainer = PPOTrainer(
    model=ppo_model,
    config=ppo_config,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset["train"],
    data_collator=collator,
)

output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length" : 5,
    "top_k":0.0,
    "top_p":1.0,
    "do_sample":True
}

reward_kwargs = {
    "top_k": None, #return all scores
    "function_to_apply": "none",
    "batch_size":16
}

max_ppo_steps = 10#the number of sample selection

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    summary_tensors = []#get response from PEFT LLM(actor model-FLAN-T5)


    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()

        generation_kwargs["max_new_tokens"] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        summary_tensors.append(summary.squeeze()[-max_new_tokens:])#add the summary result to the list
    
    #collect the summary
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

    #calculate the reward
    query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)
    #rewards: list[dict[str, float]]
    #use the hate class for reward tensor (negative reward to minimize toxicity)
    reward_tensors = [torch.tensor(reward[hate_index]["score"]) for reward in rewards]

    #run the PPO
    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)#calculation and decide the update, send back the parameters like KL,LOSS,REWARD
    ppo_trainer.log_stats(stats,batch,reward_tensors)#record

    print(f'objective/kl:{stats["objective/kl"]}\nppo/returns/mean:{stats["ppo/returns/mean"]}\nppo/policy/advantages_mean:{stats["ppo/policy/advantages_mean"]}')
    print('-'*100)


mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model, tokenizer=tokenizer, dataset=dataset["test"], num_samples=10)
print(f'toxicity [mean,std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')

##evaluate the improvement
mean_improvement = (mean_before_detoxification-mean_after_detoxification)/mean_before_detoxification
std_improvement = (std_before_detoxication-std_after_detoxification)/std_before_detoxication
print(f'Percentage improvement of toxicity score after detoxfication: \n mean:{mean_improvement*100:.2f}% \n std:{std_improvement*100:.2f}%')

#######################################Model Evaluation################
batch_size = 20
compare_results = {}
df_batch = dataset["test"][0:batch_size]

compare_results["query"] = df_batch["query"]
prompt_tensors = df_batch["input_ids"]

summary_tensors_ref = []
summary_tensors = []

#####compare the responses from actor and reference model
##step 1: get responses
for i in tqdm(range(batch_size)):
    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len

    summary = ref_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors_ref.append(summary)

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors.append(summary)

#step 2: decode responses
compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

#step 3: sentiment analysis of query-response pairs before/after
texts_before = [d + s for d,s in zip(compare_results["query"], compare_results["response_before"])]
rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[hate_index]["score"] for reward in rewards_before]

texts_after = [d + s for d,s in zip(compare_results["query"], compare_results["response_after"])]
rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[hate_index]["score"] for reward in rewards_after]
#step4: save the comparison
pd.set_option('display.max_colwidth', 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)
df_compare_results_sorted
