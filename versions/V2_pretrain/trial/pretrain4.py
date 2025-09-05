import warnings
import torch
from transformers import default_data_collator,AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, AutoTokenizer, TextStreamer
import datasets
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import transformers
warnings.filterwarnings('ignore')

#####################load model####################
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "./data/TinySolar-308m-4k-init",
    device_map="cpu",
    torch_dtype=torch.float32,
    use_cache=False,
)
pretrained_model
#####################Load dataset####################
class CustomDataset(Dataset):
    def __init__(self,args,split="train"):
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )
    def __len__(self):
        "Return the num of samples in the dataset"
        return len(self.dataset)
    def __getitem__(self,idx):
        """Retrieve a single data sample from the dataset at the specified index"""
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])
        return{"input_ids":input_ids,"labels":labels}

@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(
        default="./data/packaged_pretrained_dataset.parquet")
    num_proc: int = field(default=1)
    max_seq_length: int = field(default=32)

    seed : int = field(default=0)
    optim: str = field(default="adamw_torch") #min loss function
    max_steps: int = field(default=10000)#smoke test
    per_device_train_batch_size: int = field(default=2)
    
    learning_rate: float = field(default=5e-5)#(1e-5~5e-5, warmup, )
    weight_decay: float = field(default=0)#get rid of overfitting
    warmup_steps: int = field(default=10)#(testing)
    lr_scheduler_type: str = field(default="linear")#(linear/consine/constant)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    bf16: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)

    logging_steps: int = field(default=3)
    report_to: str = field(default="none")
# int: counts/steps; float: magnitude/ratio; bool: switches; str: strategy/name.

save_strategy: str = field(default="steps")
save_steps: int = field(default=1000)        # Save every 1000 steps to create checkpoint-1000/2000/.../10000
save_total_limit: int = field(default=2)     # Keep only the latest 2 checkpoints to avoid disk bloat

parser = transformers.HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(
    args=["--output_dir","output","--dataset_name",r"C:\Users\HKUBS\Desktop\llm_course_bing\course_1\Bing-homeworks_week2\pretrain\data\packaged_pretrained_dataset.parquet"]
)#compare agrs and args,(a package)
train_dataset = CustomDataset(args=args)

class LossLoggingCallback(TrainerCallback):
    def on_log(self,args,state,control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)
    def __init__(self):
        self.logs = []
trainer = Trainer(
    model=pretrained_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=default_data_collator,
    callbacks=[LossLoggingCallback()]
)
trainer.train()
