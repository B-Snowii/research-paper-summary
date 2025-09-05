import warnings
import torch, torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig,TextStreamer, AutoConfig,AutoModelForCausalLM, TextStreamer#compare with LlamaTokenizer
from copy import deepcopy
import gc
warnings.filterwarnings('ignore')


def fix_torch_seed(seed: int = 66):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_torch_seed()

config = LlamaConfig(
    num_hidden_layers=16,#we changed the layer here
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    use_cache=False
)
print(config)
model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)


model_name_or_path= "upstage/TinySolar-248m-4k"
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


print(model)
def print_nparams(model):
    nparams = sum(p.numel() for p in model.parameters())
    print(f"the total number of parameter is {nparams}")
print_nparams(pretrained_model)

layers = model.model.layers
new_layers= deepcopy(pretrained_model.model.layers[:-4]) + deepcopy(pretrained_model.model.layers[4:])#= (N-4) + (N-4) = 2N - 8--->2(12-4)=16
model.model.layers = nn.ModuleList(list(new_layers))
model.config.num_hidden_layers = len(model.model.layers)
model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)
model.lm_head = deepcopy(pretrained_model.lm_head)
print(model.config)
print_nparams(model)

prompt = "I am an engineer. I love"
inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer,skip_prompt=True, skip_special_tokens=True)


outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    )

model.save_pretrained('../data/TinySolar-308m-4k-init')
