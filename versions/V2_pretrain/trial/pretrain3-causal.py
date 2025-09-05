import warnings
import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import gc
warnings.filterwarnings('ignore')


def fix_torch_seed(seed: int = 66):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_torch_seed()

config = LlamaConfig()
#print(config)

config.num_hidden_layers = 32
config.hidden_size = 1024
config.intermediate_size = 4096
config.num_key_value_heads=8
config.torch_dtype = "bfloat16"
config.use_cache = False
#print(config)

model = LlamaForCausalLM(config)
#print(model)

def print_nparams(model):
    """calculate the total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"the total number of parameter is {nparams}")#use {} to insert the value of variable
print_nparams(model)

#layer_name = "model.layers.0.self_attn.q_proj.weight"
#for name,param in model.named_parameters():
#    if name == layer_name:
#        print(f"First 30 weights of layer '{layer_name}':")
#        print(param.data.view(-1)[:30])
#        break
model_dir = "upstage/SOLAR-10.7B-V1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

prompt = "I am an engineer. I love"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)
outputs = model.generate(
    **inputs,#to name the parameters
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False
)
del model
del streamer
del outputs
gc.collect()
