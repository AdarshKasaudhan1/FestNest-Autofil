"""
FestNest AI AutoFill — Phase 4: Dataset Preparation
Loads labeled OCR/JSON pairs from data/labeled/, formats them for instruction-tuning,
and splits into Train (80%), Validation (10%), and Test (10%) splits with a fixed seed.
Outputs dataset in Hugging Face datasets-compatible format:
  - data/dataset/train.jsonl
  - data/dataset/validation.jsonl
  - data/dataset/test.jsonl
  - data/dataset/hf_dataset (Hugging Face DatasetDict saved to disk)
  - data/dataset/dataset_info.json (Metadata, field schemas, token/character statistics)
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

try:
    import datasets
    from datasets import Dataset, DatasetDict
except ImportError:
    datasets = None
    Dataset = None
    DatasetDict = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")

DEFAULT_LABELED_DIR = Path("data/labeled")
DEFAULT_OUTPUT_DIR = Path("data/dataset")
TARGET_SCHEMA_FIELDS = [
    "event_name",
    "event_type",
    "date",
    "time",
    "venue",
    "organizer",
    "prize_pool",
    "registration_deadline",
    "contact",
    "description",
]

INSTRUCTION_TEXT = "Extract event details as JSON from this OCR text."
SYSTEM_PROMPT = (
    "You are an AI assistant that extracts structured event information from poster OCR text "
    "into JSON format matching FestNest schemas."
)


def load_labeled_data(labeled_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads and validates all labeled JSON files from data/labeled/.
    Filters out non-JSON files such as .gitkeep.
    """
    if not labeled_dir.exists():
        raise FileNotFoundError(f"Labeled directory not found: {labeled_dir}")

    files = sorted([f for f in labeled_dir.iterdir() if f.is_file() and f.suffix.lower() == ".json"])
    logger.info(f"Found {len(files)} JSON candidate file(s) in {labeled_dir}")

    valid_records = []
    skipped_count = 0

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ocr_text = data.get("ocr_text", "").strip()
            json_output = data.get("json_output")

            if not ocr_text:
                logger.warning(f"Skipping {file_path.name}: empty or missing 'ocr_text'")
                skipped_count += 1
                continue

            if not isinstance(json_output, dict):
                logger.warning(f"Skipping {file_path.name}: 'json_output' is not a dictionary")
                skipped_count += 1
                continue

            # Ensure all schema keys exist in json_output
            cleaned_json_output = {}
            for field in TARGET_SCHEMA_FIELDS:
                val = json_output.get(field, "")
                cleaned_json_output[field] = str(val) if val is not None else ""

            valid_records.append({
                "source_file": data.get("source_file", file_path.name),
                "ocr_text": ocr_text,
                "json_output": cleaned_json_output,
            })
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            skipped_count += 1

    logger.info(f"Successfully loaded {len(valid_records)} valid labeled records ({skipped_count} skipped).")
    return valid_records


def format_instruction_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a single labeled record into standard instruction-tuning representations:
      - Alpaca format: instruction, input, output
      - Qwen2.5 ChatML messages: [{'role': 'system', ...}, {'role': 'user', ...}, {'role': 'assistant', ...}]
      - Rendered ChatML text string: <|im_start|>...<|im_end|>
      - Rendered Alpaca text string
    """
    ocr_text = record["ocr_text"]
    json_output_str = json.dumps(record["json_output"], indent=2, ensure_ascii=False)

    # Standard Alpaca fields (per Phase 4 specification)
    instruction = INSTRUCTION_TEXT
    inp = ocr_text
    output = json_output_str

    # ChatML messages format (native for Qwen2.5-Instruct fine-tuning)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"{instruction}\n\n{inp}",
        },
        {
            "role": "assistant",
            "content": output,
        },
    ]

    # Pre-rendered ChatML text for fast tokenizer ingestion
    chatml_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{inp}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )

    # Alpaca formatted single text prompt
    alpaca_text = (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{output}"
    )

    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "messages": messages,
        "text": chatml_text,
        "alpaca_text": alpaca_text,
        "source_file": record["source_file"],
        "schema_data": record["json_output"],
    }


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Shuffles records deterministically with seed and splits into train/val/test sets.
    Ensures minimum count of at least 1 in val and test when dataset size allows.
    """
    total = len(records)
    if total == 0:
        return [], [], []

    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    if total < 3:
        # Minimal edge case for tiny test datasets
        train_set = shuffled[:1]
        val_set = shuffled[1:2]
        test_set = shuffled[2:]
        return train_set, val_set, test_set

    num_test = max(1, int(round(total * test_ratio)))
    num_val = max(1, int(round(total * val_ratio)))
    num_train = total - num_val - num_test

    if num_train < 1:
        num_train = 1
        num_val = max(1, (total - 1) // 2)
        num_test = total - num_train - num_val

    train_set = shuffled[:num_train]
    val_set = shuffled[num_train : num_train + num_val]
    test_set = shuffled[num_train + num_val :]

    logger.info(
        f"Dataset split (Total: {total}, Seed: {seed}): "
        f"Train: {len(train_set)} ({len(train_set)/total*100:.1f}%), "
        f"Validation: {len(val_set)} ({len(val_set)/total*100:.1f}%), "
        f"Test: {len(test_set)} ({len(test_set)/total*100:.1f}%)"
    )

    return train_set, val_set, test_set


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary character length statistics across the input and output texts.
    """
    if not records:
        return {}

    input_lens = [len(r["input"]) for r in records]
    output_lens = [len(r["output"]) for r in records]

    return {
        "count": len(records),
        "input_char_min": min(input_lens),
        "input_char_max": max(input_lens),
        "input_char_avg": round(sum(input_lens) / len(input_lens), 1),
        "output_char_min": min(output_lens),
        "output_char_max": max(output_lens),
        "output_char_avg": round(sum(output_lens) / len(output_lens), 1),
    }


def save_dataset(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    Saves formatted records as:
      1. JSONL files: train.jsonl, validation.jsonl, test.jsonl
      2. Hugging Face DatasetDict saved to disk via datasets.DatasetDict.save_to_disk()
      3. dataset_info.json containing split metadata and statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }

    # 1. Write JSON Lines files
    for split_name, items in splits.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in items:
                # Store instruction tuning structure
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(items)} examples to {jsonl_path}")

    # 2. Write Hugging Face DatasetDict (if datasets package is installed)
    hf_dataset_dir = output_dir / "hf_dataset"
    if datasets is not None:
        try:
            hf_dict = DatasetDict({
                split_name: Dataset.from_list(items)
                for split_name, items in splits.items()
            })
            hf_dict.save_to_disk(str(hf_dataset_dir))
            logger.info(f"Saved Hugging Face DatasetDict to {hf_dataset_dir}")
        except Exception as e:
            logger.warning(f"Could not save HF DatasetDict to disk: {e}")
    else:
        logger.warning("'datasets' package not found; skipped DatasetDict.save_to_disk.")

    # 3. Write metadata / stats JSON
    info_path = output_dir / "dataset_info.json"
    stats = {
        "created_at": datetime.now().isoformat(),
        "random_seed": seed,
        "schema_fields": TARGET_SCHEMA_FIELDS,
        "splits": {
            "train": compute_statistics(train_records),
            "validation": compute_statistics(val_records),
            "test": compute_statistics(test_records),
            "total_samples": len(train_records) + len(val_records) + len(test_records),
        },
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "test": str(output_dir / "test.jsonl"),
            "hf_dataset": str(hf_dataset_dir),
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved dataset metadata and statistics to {info_path}")


def verify_dataset(output_dir: Path) -> bool:
    """
    Validates that the generated dataset can be successfully reloaded
    and conforms to the expected structure.
    """
    logger.info("Running verification on generated dataset...")
    train_jsonl = output_dir / "train.jsonl"
    val_jsonl = output_dir / "validation.jsonl"
    test_jsonl = output_dir / "test.jsonl"

    for p in [train_jsonl, val_jsonl, test_jsonl]:
        if not p.exists():
            logger.error(f"Missing expected output file: {p}")
            return False

    # Check JSONL parsing
    sample_first = None
    with open(train_jsonl, "r", encoding="utf-8") as f:
        line = f.readline()
        if line:
            sample_first = json.loads(line)

    if not sample_first:
        logger.error("Train dataset is empty!")
        return False

    required_keys = ["instruction", "input", "output", "messages", "text"]
    for k in required_keys:
        if k not in sample_first:
            logger.error(f"Sample record missing key: '{k}'")
            return False

    # Verify reloading via Hugging Face datasets if available
    if datasets is not None:
        try:
            hf_ds = datasets.load_from_disk(str(output_dir / "hf_dataset"))
            logger.info(f"HF load_from_disk verification passed: {hf_ds}")
        except Exception as e:
            logger.warning(f"Note: HF reload check encountered: {e}")

    logger.info("Verification check PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare instruction-tuning dataset for Qwen2.5-1.5B.")
    parser.add_argument(
        "--labeled_dir",
        type=str,
        default=str(DEFAULT_LABELED_DIR),
        help=f"Directory containing labeled JSON files (default: {DEFAULT_LABELED_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for prepared dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.80,
        help="Train split ratio (default: 0.80)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="Test split ratio (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for splitting (default: 42)",
    )
    args = parser.parse_args()

    labeled_path = Path(args.labeled_dir)
    output_path = Path(args.output_dir)

    logger.info("=" * 65)
    logger.info("FestNest AI AutoFill — Phase 4: Dataset Preparation")
    logger.info("=" * 65)

    # 1. Load labeled pairs
    raw_records = load_labeled_data(labeled_path)
    if not raw_records:
        logger.error(f"No valid labeled pairs found in {labeled_path}. Please complete Phase 3 first.")
        sys.exit(1)

    # 2. Format into instruction-tuning examples
    formatted_records = [format_instruction_record(r) for r in raw_records]

    # 3. Split dataset
    train_records, val_records, test_records = split_dataset(
        formatted_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 4. Save to Hugging Face compatible formats
    save_dataset(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        output_dir=output_path,
        seed=args.seed,
    )

    # 5. Verify saved dataset
    success = verify_dataset(output_path)
    if not success:
        logger.error("Dataset verification failed!")
        sys.exit(1)

    # 6. Display a preview of a training record
    sample = train_records[0]
    logger.info("=" * 65)
    logger.info(f"Sample Training Record Preview (Source: {sample['source_file']}):")
    logger.info(f"Instruction: {sample['instruction']}")
    logger.info(f"Input OCR snippet: {sample['input'][:150]}...")
    logger.info(f"Target Output JSON:\n{sample['output']}")
    logger.info("=" * 65)
    logger.info("Phase 4 Dataset Preparation completed successfully!")


if __name__ == "__main__":
    main()
