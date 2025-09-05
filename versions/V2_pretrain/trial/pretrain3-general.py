import warnings
import torch
from transformers import AutoTokenizer, AutoConfig,AutoModelForCausalLM, TextStreamer#compare with LlamaTokenizer
import gc
warnings.filterwarnings('ignore')


def fix_torch_seed(seed: int = 66):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_torch_seed()


model_name_or_path= "upstage/TinySolar-248m-4k"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

def print_nparams(model):
    nparams = sum(p.numel() for p in model.parameters())
    print(f"the total number of parameter is {nparams}")
print_nparams(model)

prompt = "I am an engineer. I love"
inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer,skip_prompt=True, skip_special_tokens=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        use_cache=True,
        max_new_tokens=128,
        do_sample=False,
    )

del outputs, model
gc.collect()
