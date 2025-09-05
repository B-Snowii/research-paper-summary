import os
import argparse
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import evaluate


def select_device_and_dtype() -> Tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    return device, dtype


def try_load_pt_model(
    pt_model_id_or_path: str,
    base_model_name: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    # First try as PEFT adapter over base model
    try:
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, pt_model_id_or_path, torch_dtype=dtype, is_trainable=False)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return model, tokenizer
    except Exception:
        pass

    # Fallback: load as a full model
    model = AutoModelForSeq2SeqLM.from_pretrained(pt_model_id_or_path, torch_dtype=dtype)
    model.to(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(pt_model_id_or_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer


def load_base_model(base_model_name: str, device: str, dtype: torch.dtype) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer


def prepare_dialogsum(tokenizer, max_source_length: int, max_target_length: int):
    dataset = load_dataset("knkarthick/dialogsum")

    def tokenize_function(example):
        start_prompt = "summarize: "
        end_prompt = ""
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example["input_ids"] = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        example["labels"] = tokenizer(
            example["summary"],
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return example

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["id", "topic", "dialogue", "summary"])
    return dataset, tokenized


def generate_summary(
    model,
    tokenizer,
    inputs_text: str,
    device: str,
    max_new_tokens: int = 128,
    num_beams: int = 5,
    length_penalty: float = 0.9,
    no_repeat_ngram_size: int = 3,
) -> str:
    input_ids = tokenizer(inputs_text, return_tensors="pt").input_ids.to(device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    outputs = model.generate(input_ids=input_ids, generation_config=gen_cfg)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_rouge_on_dialogsum(
    model,
    tokenizer,
    raw_dataset,
    device: str,
    sample_count: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
):
    dialogues = raw_dataset["test"][0:sample_count]["dialogue"]
    references = raw_dataset["test"][0:sample_count]["summary"]

    predictions = []
    for d in dialogues:
        prompt = f"summarize: {d}"
        pred = generate_summary(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        predictions.append(pred)

    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references, use_aggregator=True, use_stemmer=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate P-Tuning model vs base on DialogSum")
    parser.add_argument("--pt_model_id", required=True, help="HF repo id or local dir of P-Tuning model/adapter")
    parser.add_argument("--base_model", default="t5-large")
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.9)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = parser.parse_args()

    device, dtype = select_device_and_dtype()
    print(f"device: {device}, dtype: {dtype}")

    # Load models
    pt_model, pt_tokenizer = try_load_pt_model(
        pt_model_id_or_path=args.pt_model_id,
        base_model_name=args.base_model,
        device=device,
        dtype=dtype,
    )
    base_model, base_tokenizer = load_base_model(args.base_model, device=device, dtype=dtype)

    # Prepare data once (use base tokenizer for shaping, content strings come from raw dataset)
    raw_dataset, _ = prepare_dialogsum(base_tokenizer, args.max_source_length, args.max_target_length)

    # Evaluate
    base_scores = evaluate_rouge_on_dialogsum(
        base_model,
        base_tokenizer,
        raw_dataset,
        device=device,
        sample_count=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print("BASE MODEL ROUGE:")
    print(base_scores)

    pt_scores = evaluate_rouge_on_dialogsum(
        pt_model,
        pt_tokenizer,
        raw_dataset,
        device=device,
        sample_count=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print("P-TUNING MODEL ROUGE:")
    print(pt_scores)

    # Relative improvement over base
    base_vals = list(base_scores.values())
    pt_vals = list(pt_scores.values())
    keys = list(pt_scores.keys())
    print("over base model")
    for k, dv in zip(keys, [p - b for p, b in zip(pt_vals, base_vals) ]):
        print(f"{k}:{dv * 100:.2f}%")


if __name__ == "__main__":
    main()


