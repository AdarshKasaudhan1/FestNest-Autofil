"""
FestNest AI AutoFill — Phase 5: LoRA Fine-Tuning
Fine-tunes Qwen2.5-1.5B-Instruct using LoRA / QLoRA (PEFT + TRL / Transformers)
on the prepared dataset in data/dataset/.
Designed to complete on a free Google Colab T4 GPU in under 15-20 minutes (< 2 hours target).
Saves the fine-tuned LoRA adapter to models/qwen2.5-1.5b-festnest-lora.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

try:
    from trl import SFTTrainer, SFTConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    SFTTrainer = None
    SFTConfig = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune_lora")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_DIR = Path("data/dataset")
DEFAULT_OUTPUT_ADAPTER_DIR = Path("models/qwen2.5-1.5b-festnest-lora")
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B-Instruct using LoRA.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face base model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help=f"Path to dataset directory containing train.jsonl and validation.jsonl (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ADAPTER_DIR),
        help=f"Directory to save the fine-tuned LoRA adapter (default: {DEFAULT_OUTPUT_ADAPTER_DIR})",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Directory to store intermediate training checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch size: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA parameters (default: 2e-4)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1536,
        help="Maximum input sequence length in tokens (default: 1536)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit NormalFloat4 QLoRA quantization (requires CUDA)",
    )
    parser.add_argument(
        "--no_4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit quantization (e.g., when training on CPU or full precision)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 1 optimization step to test data loading and model initialization",
    )
    return parser.parse_args()


def load_tokenized_dataset(dataset_dir: Path, tokenizer, max_seq_length: int):
    """
    Loads train.jsonl and validation.jsonl and verifies entries.
    """
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "validation.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}. Run scripts/03_prepare_dataset.py first.")

    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)

    logger.info(f"Loading dataset files: {data_files}")
    raw_dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Loaded dataset: {raw_dataset}")

    # Ensure each sample has a valid 'text' field (ChatML format)
    def ensure_text(example):
        if "text" in example and example["text"]:
            return {"text": example["text"]}
        elif "messages" in example and example["messages"]:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        elif "instruction" in example and "output" in example:
            text = (
                f"<|im_start|>user\n{example['instruction']}\n\n{example.get('input', '')}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
            )
            return {"text": text}
        return {"text": ""}

    processed_dataset = raw_dataset.map(ensure_text)
    return processed_dataset


def build_lora_config(args) -> LoraConfig:
    """
    Constructs LoRA configuration targeting all linear attention and MLP projections
    for Qwen2.5 causal language models.
    """
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_adapter_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    logger.info("=" * 65)
    logger.info("FestNest AI AutoFill — Phase 5: LoRA Fine-Tuning")
    logger.info(f"Base Model:      {args.model_id}")
    logger.info(f"Dataset Dir:     {dataset_dir}")
    logger.info(f"Output Adapter:  {output_adapter_dir}")
    logger.info(f"Epochs:          {args.epochs}")
    logger.info(f"Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum})")
    logger.info(f"Learning Rate:   {args.lr}")
    logger.info(f"LoRA Rank (r):   {args.lora_r} (alpha={args.lora_alpha})")
    logger.info(f"CUDA Available:  {torch.cuda.is_available()}")
    logger.info("=" * 65)

    has_cuda = torch.cuda.is_available()
    use_4bit = args.load_in_4bit and has_cuda

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Configure Quantization & Model Loading
    quantization_config = None
    torch_dtype = torch.float32

    if use_4bit:
        logger.info("Configuring 4-bit QLoRA (BitsAndBytes NormalFloat4)...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = compute_dtype
    elif has_cuda:
        torch_dtype = torch.float16

    device_map = "auto" if has_cuda else "cpu"

    logger.info(f"Loading base model (device_map={device_map}, dtype={torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 3. Apply PEFT / LoRA
    peft_config = build_lora_config(args)
    model = get_peft_model(model, peft_config)

    logger.info("LoRA trainable parameter overview:")
    model.print_trainable_parameters()

    # 4. Load & Process Dataset
    dataset = load_tokenized_dataset(dataset_dir, tokenizer, args.max_seq_length)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", None)

    # 5. Training Setup
    max_steps = 1 if args.dry_run else -1
    num_epochs = 1 if args.dry_run else args.epochs

    common_train_args = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": has_cuda and not torch.cuda.is_bf16_supported(),
        "bf16": has_cuda and torch.cuda.is_bf16_supported(),
        "logging_steps": 1 if args.dry_run else 5,
        "save_strategy": "epoch" if not args.dry_run else "no",
        "eval_strategy": "epoch" if (eval_dataset is not None and not args.dry_run) else "no",
        "save_total_limit": 2,
        "report_to": "none",
    }

    if HAS_TRL:
        logger.info("Initializing TRL SFTTrainer...")
        sft_config_kwargs = common_train_args.copy()
        sft_config_kwargs["dataset_text_field"] = "text"

        # Dynamically set max sequence length parameter across TRL versions
        if hasattr(SFTConfig, "max_length"):
            sft_config_kwargs["max_length"] = args.max_seq_length
        else:
            sft_config_kwargs["max_seq_length"] = args.max_seq_length

        training_args = SFTConfig(**sft_config_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    else:
        logger.info("TRL not found; using standard Transformers Trainer...")
        training_args = TrainingArguments(**common_train_args)
        # Tokenize dataset for standard Trainer
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
        tokenized_train = train_dataset.map(tokenize_fn, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True) if eval_dataset else None

        from transformers import Trainer, DataCollatorForLanguageModeling
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    # 6. Execute Fine-Tuning
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    # 7. Save LoRA Adapter & Tokenizer
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving fine-tuned LoRA adapter to {output_adapter_dir}...")
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    # 8. Save Training Metadata
    meta_path = output_adapter_dir / "training_meta.json"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.model_id,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "quantization": "4bit-nf4" if use_4bit else "none",
        },
        "train_metrics": train_result.metrics if hasattr(train_result, "metrics") else {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training metadata saved to {meta_path}")
    logger.info("=" * 65)
    logger.info("Phase 5 LoRA fine-tuning script executed successfully!")


if __name__ == "__main__":
    main()
