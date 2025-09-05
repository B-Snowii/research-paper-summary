import os
import argparse
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    TrainingArguments,
)
class CompatibleSeq2SeqTrainer(Seq2SeqTrainer):
    # Compatible with older/custom Trainers that pass num_items_in_batch into inputs
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        # Remove keys not accepted by the model forward
        if hasattr(inputs, "to_dict"):
            safe_inputs = {k: v for k, v in inputs.to_dict().items() if k != "num_items_in_batch"}
        else:
            safe_inputs = {k: v for k, v in dict(inputs).items() if k != "num_items_in_batch"}

        outputs = model(**safe_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):  # type: ignore[override]
        processed = super()._prepare_inputs(inputs)
        if isinstance(processed, dict):
            processed.pop("num_items_in_batch", None)
        return processed



def select_device_and_dtype() -> Tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    return device, dtype


def get_dataset_and_fields(dataset_name: str, data_dir: Optional[str] = None):
    dataset_name = dataset_name.lower()

    # Prefer local JSON if provided (useful when Hub is inaccessible)
    if data_dir:
        if dataset_name == "samsum":
            data_files = {
                "train": os.path.join(data_dir, "train.json"),
                "validation": os.path.join(data_dir, "validation.json"),
                "test": os.path.join(data_dir, "test.json"),
            }
            dataset = load_dataset("json", data_files=data_files)
            return dataset, "dialogue", "summary"
        elif dataset_name == "dialogsum":
            data_files = {
                "train": os.path.join(data_dir, "train.json"),
                "validation": os.path.join(data_dir, "validation.json"),
                "test": os.path.join(data_dir, "test.json"),
            }
            dataset = load_dataset("json", data_files=data_files)
            return dataset, "dialogue", "summary"
        else:
            raise ValueError("Unsupported dataset for local JSON. Use 'dialogsum' or 'samsum'.")

    # Remote Hub loading
    if dataset_name == "dialogsum":
        dataset = load_dataset("knkarthick/dialogsum")
        return dataset, "dialogue", "summary"
    elif dataset_name == "samsum":
        dataset = load_dataset("samsum")
        return dataset, "dialogue", "summary"
    else:
        raise ValueError("Unsupported dataset. Use 'dialogsum' or 'samsum'.")


def tokenize_datasets(
    dataset,
    tokenizer,
    source_field: str,
    target_field: str,
    max_source_length: int,
    max_target_length: int,
):
    def preprocess_function(examples):
        inputs = examples[source_field]
        targets = examples[target_field]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train an Adapter (no LoRA/PEFT) for T5/FLAN-T5 on DialogSum/SAMSum")
    parser.add_argument("--base_model", type=str, default="t5-large", help="Base model, e.g., t5-large or flan-t5-base")
    parser.add_argument("--dataset", type=str, default="dialogsum", choices=["dialogsum", "samsum"])
    parser.add_argument("--adapter_config", type=str, default="pfeiffer", choices=["pfeiffer", "houlsby"], help="Adapter architecture")
    parser.add_argument("--adapter_name", type=str, default="summarization", help="Name for the adapter to add/train")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "adapter_training"))
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0, help="If > 0, cap number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="If > 0, cap number of eval samples")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--push_to_hub", action="store_true", help="If set, push the trained adapter to the Hub")
    parser.add_argument("--hub_adapter_repo", type=str, default=None, help="HF repo id for the adapter, e.g., username/my-t5-sum-adapter")
    parser.add_argument("--data_dir", type=str, default=None, help="Local dataset directory containing train.json/validation.json/test.json")
    parser.add_argument("--hf_endpoint", type=str, default=None, help="Optional HF mirror endpoint, e.g., https://hf-mirror.com")
    args = parser.parse_args()

    device, dtype = select_device_and_dtype()
    print(f"device: {device}, dtype: {dtype}")

    # Optional mirror endpoint
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # Load dataset (local JSON takes precedence if provided)
    raw_dataset, src_field, tgt_field = get_dataset_and_fields(args.dataset, data_dir=args.data_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Tokenize
    tokenized = tokenize_datasets(
        raw_dataset,
        tokenizer,
        source_field=src_field,
        target_field=tgt_field,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    # Subset sampling for low-resource training
    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation")
    if args.max_train_samples and args.max_train_samples > 0:
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples and args.max_eval_samples > 0:
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(args.max_eval_samples, len(eval_dataset))))

    # Model with adapter support
    # Standard Transformers model + adapters.init
    import adapters  # type: ignore
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
    adapters.init(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure and add adapter
    try:
        from transformers.adapters import AdapterConfig  # type: ignore
    except Exception:
        try:
            from adapters import AdapterConfig  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Adapters support not found. Please install: pip install -U adapter-transformers"
            ) from e

    adapter_config = AdapterConfig.load(args.adapter_config)
    # Safely add adapter; if it already exists, ignore the error
    try:
        model.add_adapter(args.adapter_name, config=adapter_config)
    except Exception:
        pass
    # Activate immediately
    try:
        model.set_active_adapters(args.adapter_name)
    except Exception:
        pass

    # Train only the adapter parameters
    model.train_adapter(args.adapter_name)
    model.set_active_adapters(args.adapter_name)

    # Ensure adapter is active before building collator/trainer
    model.set_active_adapters(args.adapter_name)

    # Data collator with dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            predict_with_generate=False,
            fp16=(dtype == torch.float16),
            bf16=(dtype == torch.bfloat16),
        )
    except TypeError:
        # Fallback for older Transformers versions with fewer arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
        )

    # Ensure compatibility with older Seq2SeqTrainer expecting args.generation_config
    try:
        getattr(training_args, "generation_config")
    except Exception:
        try:
            from transformers import GenerationConfig  # type: ignore
            setattr(training_args, "generation_config", GenerationConfig())
        except Exception:
            setattr(training_args, "generation_config", None)

    trainer = CompatibleSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter locally
    save_dir = os.path.join(args.output_dir, f"{args.adapter_name}_saved")
    os.makedirs(save_dir, exist_ok=True)
    model.save_adapter(save_dir, args.adapter_name)
    print(f"Adapter saved to: {save_dir}")

    # Optionally push to hub (adapter weights only)
    if args.push_to_hub:
        if not args.hub_adapter_repo:
            raise ValueError("--push_to_hub requires --hub_adapter_repo, e.g., username/my-t5-sum-adapter")
        print(f"Pushing adapter to Hub repo: {args.hub_adapter_repo}")
        model.push_adapter_to_hub(args.hub_adapter_repo, args.adapter_name)
        print("Pushed adapter to Hub.")


if __name__ == "__main__":
    main()


