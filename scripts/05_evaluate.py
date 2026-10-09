"""
FestNest AI AutoFill — Phase 6: Model Evaluation
Evaluates both the base Qwen2.5-1.5B-Instruct model and the fine-tuned LoRA version
on the held-out test dataset.
Computes field-level accuracy (exact match per JSON field, averaged across all fields)
for each model and outputs:
  - Before/After comparison table across all 10 schema fields
  - 5 side-by-side example outputs (OCR Input, Base Output, Fine-Tuned Output, Ground Truth)
  - Saves full evaluation artifacts to data/evaluation_results.json and EVALUATION_REPORT.md
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = Path("models/qwen2.5-1.5b-festnest-lora")
DEFAULT_TEST_FILE = Path("data/dataset/test.jsonl")
DEFAULT_VAL_FILE = Path("data/dataset/validation.jsonl")
DEFAULT_TRAIN_FILE = Path("data/dataset/train.jsonl")
DEFAULT_OUTPUT_JSON = Path("data/evaluation_results.json")
DEFAULT_OUTPUT_MD = Path("EVALUATION_REPORT.md")

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

SYSTEM_PROMPT = (
    "You are an AI assistant that extracts structured event information from poster OCR text "
    "into JSON format matching FestNest schemas."
)


def normalize_value(val: Any) -> str:
    """
    Normalizes a field value for robust exact match evaluation:
    converts to string, strips whitespace, converts to lowercase,
    and collapses internal whitespace.
    """
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        s = json.dumps(val, ensure_ascii=False)
    else:
        s = str(val)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_token_f1(pred: str, truth: str) -> float:
    """
    Computes token-level F1 overlap score between predicted and ground-truth strings.
    """
    p_tokens = set(pred.split())
    t_tokens = set(truth.split())
    if not t_tokens and not p_tokens:
        return 1.0
    if not t_tokens or not p_tokens:
        return 0.0
    common = p_tokens.intersection(t_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(t_tokens)
    return (2 * prec * rec) / (prec + rec)


def extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts a JSON dictionary from model outputs,
    handling markdown code blocks, partial prefixes, and raw JSON strings.
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direct parse attempt
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Check if nested under "event" or similar
            if "event" in obj and isinstance(obj["event"], dict):
                return obj["event"]
            return obj
    except Exception:
        pass

    # 2. Markdown fenced code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                if "event" in obj and isinstance(obj["event"], dict):
                    return obj["event"]
                return obj
        except Exception:
            pass

    # 3. Outermost curly braces {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                if "event" in obj and isinstance(obj["event"], dict):
                    return obj["event"]
                return obj
        except Exception:
            pass

    return None


def load_evaluation_examples(
    test_path: Path,
    val_path: Optional[Path] = None,
    train_path: Optional[Path] = None,
    target_count: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads all held-out test examples.
    If the test set has fewer than target_count examples, supplements with validation/train
    examples to reach target_count for side-by-side illustration while keeping official test metrics separate.
    """
    test_examples = []
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_examples.append(json.loads(line))

    logger.info(f"Loaded {len(test_examples)} official held-out test examples from {test_path}")

    # Build side-by-side demonstration pool
    demo_examples = list(test_examples)
    if len(demo_examples) < target_count and val_path and val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    demo_examples.append(json.loads(line))
                    if len(demo_examples) >= target_count:
                        break

    if len(demo_examples) < target_count and train_path and train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    demo_examples.append(json.loads(line))
                    if len(demo_examples) >= target_count:
                        break

    demo_examples = demo_examples[:target_count]
    logger.info(f"Prepared {len(demo_examples)} examples for side-by-side output inspection.")
    return test_examples, demo_examples


def format_prompt(tokenizer, ocr_text: str) -> str:
    """
    Formats the OCR input text into standard ChatML conversation prompt for Qwen2.5.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract event details as JSON from this OCR text:\n\n{ocr_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 200,
) -> str:
    """
    Generates text from the model using greedy decoding for deterministic, reproducible evaluation.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_tokens = output_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    predictions: List[str],
) -> Dict[str, Any]:
    """
    Computes field-level exact match accuracy and token F1 overlap across all 10 schema fields.
    """
    total_samples = len(examples)
    if total_samples == 0:
        return {}

    field_exact_matches = {f: 0 for f in TARGET_SCHEMA_FIELDS}
    field_f1_scores = {f: 0.0 for f in TARGET_SCHEMA_FIELDS}
    valid_json_count = 0
    parsed_predictions = []

    for ex, pred_text in zip(examples, predictions):
        ground_truth = ex.get("schema_data") or json.loads(ex.get("output", "{}"))
        parsed_json = extract_json_dict(pred_text)

        if parsed_json is not None:
            valid_json_count += 1
            parsed_predictions.append(parsed_json)
        else:
            parsed_json = {}
            parsed_predictions.append({})

        for field in TARGET_SCHEMA_FIELDS:
            truth_val = normalize_value(ground_truth.get(field, ""))
            pred_val = normalize_value(parsed_json.get(field, ""))

            if truth_val == pred_val:
                field_exact_matches[field] += 1

            f1 = compute_token_f1(pred_val, truth_val)
            field_f1_scores[field] += f1

    field_accuracies = {
        f: round((field_exact_matches[f] / total_samples) * 100, 2)
        for f in TARGET_SCHEMA_FIELDS
    }
    field_avg_f1 = {
        f: round((field_f1_scores[f] / total_samples) * 100, 2)
        for f in TARGET_SCHEMA_FIELDS
    }
    macro_accuracy = round(sum(field_accuracies.values()) / len(TARGET_SCHEMA_FIELDS), 2)
    macro_f1 = round(sum(field_avg_f1.values()) / len(TARGET_SCHEMA_FIELDS), 2)
    valid_json_rate = round((valid_json_count / total_samples) * 100, 2)

    return {
        "total_samples": total_samples,
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "macro_exact_accuracy": macro_accuracy,
        "macro_token_f1": macro_f1,
        "field_accuracies": field_accuracies,
        "field_f1_scores": field_avg_f1,
        "parsed_predictions": parsed_predictions,
    }


def generate_markdown_report(
    test_metrics_base: Dict[str, Any],
    test_metrics_ft: Dict[str, Any],
    side_by_side_samples: List[Dict[str, Any]],
) -> str:
    """
    Renders the formatted markdown evaluation report with comparison table and 5 side-by-side examples.
    """
    md = []
    md.append("# 📊 FestNest AI AutoFill — Phase 6: Model Evaluation Report")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  ")
    md.append(f"**Fine-Tuned Model:** `Qwen2.5-1.5B-Instruct + LoRA (models/qwen2.5-1.5b-festnest-lora)`  ")
    md.append(f"**Evaluation Split:** Held-out Test Set ({test_metrics_ft['total_samples']} samples)\n")

    md.append("## 1. Before vs. After Field-Level Comparison Table\n")
    md.append("| Field | Base Model Accuracy | Fine-Tuned Accuracy | Improvement (Delta) | Base F1 Score | Fine-Tuned F1 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for f in TARGET_SCHEMA_FIELDS:
        base_acc = test_metrics_base["field_accuracies"].get(f, 0.0)
        ft_acc = test_metrics_ft["field_accuracies"].get(f, 0.0)
        delta_acc = ft_acc - base_acc
        delta_str = f"+{delta_acc:.1f}%" if delta_acc >= 0 else f"{delta_acc:.1f}%"

        base_f1 = test_metrics_base["field_f1_scores"].get(f, 0.0)
        ft_f1 = test_metrics_ft["field_f1_scores"].get(f, 0.0)

        md.append(f"| **`{f}`** | {base_acc:.1f}% | **{ft_acc:.1f}%** | `{delta_str}` | {base_f1:.1f}% | **{ft_f1:.1f}%** |")

    md.append("| " + "-" * 75 + " |")
    delta_macro = test_metrics_ft["macro_exact_accuracy"] - test_metrics_base["macro_exact_accuracy"]
    delta_macro_str = f"+{delta_macro:.1f}%" if delta_macro >= 0 else f"{delta_macro:.1f}%"
    md.append(
        f"| **Overall Macro Accuracy** | **{test_metrics_base['macro_exact_accuracy']:.1f}%** | "
        f"**{test_metrics_ft['macro_exact_accuracy']:.1f}%** | **`{delta_macro_str}`** | "
        f"**{test_metrics_base['macro_token_f1']:.1f}%** | **{test_metrics_ft['macro_token_f1']:.1f}%** |"
    )

    delta_json = test_metrics_ft["valid_json_rate"] - test_metrics_base["valid_json_rate"]
    delta_json_str = f"+{delta_json:.1f}%" if delta_json >= 0 else f"{delta_json:.1f}%"
    md.append(
        f"| **Valid JSON Syntax Rate** | {test_metrics_base['valid_json_rate']:.1f}% | "
        f"**{test_metrics_ft['valid_json_rate']:.1f}%** | `{delta_json_str}` | — | — |"
    )
    md.append("\n---\n")

    md.append("## 2. Five Side-by-Side Example Outputs\n")
    for idx, sample in enumerate(side_by_side_samples, start=1):
        md.append(f"### Example {idx}: `{sample['source_file']}`\n")
        ocr_snippet = sample['ocr_text'].strip()
        if len(ocr_snippet) > 280:
            ocr_snippet = ocr_snippet[:280] + "..."

        md.append(f"**1. Raw OCR Input Snippet:**\n```text\n{ocr_snippet}\n```\n")

        truth_json = json.dumps(sample["ground_truth"], indent=2, ensure_ascii=False)
        base_json = sample.get("base_model_output", "").strip()
        ft_json = sample.get("fine_tuned_output", "").strip()

        md.append("**2. Ground Truth vs. Predictions:**\n")
        md.append("#### Ground Truth JSON:")
        md.append(f"```json\n{truth_json}\n```\n")
        md.append("#### Fine-Tuned Model Prediction:")
        md.append(f"```json\n{ft_json}\n```\n")
        md.append("#### Base Model Prediction:")
        md.append(f"```json\n{base_json}\n```\n")
        md.append("\n" + "=" * 50 + "\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs. Fine-Tuned Qwen2.5 on FestNest event schema.")
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=str(DEFAULT_ADAPTER_PATH),
        help=f"Path to saved LoRA adapter directory (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=str(DEFAULT_TEST_FILE),
        help=f"Path to test.jsonl (default: {DEFAULT_TEST_FILE})",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(DEFAULT_VAL_FILE),
        help=f"Path to validation.jsonl (default: {DEFAULT_VAL_FILE})",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(DEFAULT_TRAIN_FILE),
        help=f"Path to train.jsonl (default: {DEFAULT_TRAIN_FILE})",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help=f"Path to save evaluation results JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=str(DEFAULT_OUTPUT_MD),
        help=f"Path to save markdown report (default: {DEFAULT_OUTPUT_MD})",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=140,
        help="Max generated new tokens (default: 140)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of examples for side-by-side demo inspection (default: 2)",
    )
    args = parser.parse_args()

    test_path = Path(args.test_file)
    val_path = Path(args.val_file)
    train_path = Path(args.train_file)
    adapter_path = Path(args.adapter_path)
    output_json_path = Path(args.output_json)
    output_md_path = Path(args.output_md)

    logger.info("=" * 65)
    logger.info("FestNest AI AutoFill — Phase 6: Model Evaluation")
    logger.info(f"Base Model:      {args.base_model}")
    logger.info(f"LoRA Adapter:    {adapter_path}")
    logger.info(f"Test File:       {test_path}")
    logger.info(f"Num Examples:    {args.num_examples}")
    logger.info("=" * 65)

    if not adapter_path.exists():
        logger.error(f"LoRA adapter path not found: {adapter_path}. Please complete Phase 5 first.")
        sys.exit(1)

    # 1. Load Evaluation Data
    test_examples, demo_examples = load_evaluation_examples(
        test_path=test_path,
        val_path=val_path,
        train_path=train_path,
        target_count=args.num_examples,
    )

    if not test_examples and not demo_examples:
        logger.error("No evaluation samples found. Run scripts/03_prepare_dataset.py first.")
        sys.exit(1)

    # All unique examples to evaluate across both sets
    eval_set = demo_examples if len(demo_examples) >= len(test_examples) else test_examples
    logger.info(f"Total evaluation pool: {len(eval_set)} sample(s)")

    # 2. Hardware and Environment Setup
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        # Utilize all available CPU threads for fast inference
        cpu_threads = os.cpu_count() or 4
        torch.set_num_threads(cpu_threads)
        logger.info(f"CUDA not available; set PyTorch CPU thread count to {cpu_threads}")

    # 3. Load Tokenizer & Base Model
    logger.info(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading base model {args.base_model} on {device}...")
    model_dtype = torch.float16 if has_cuda else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=model_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.config.use_cache = True

    # 4. Run Inference with Base Model
    logger.info("Running inference with Base Qwen2.5-1.5B-Instruct...")
    base_predictions = []
    for idx, ex in enumerate(eval_set, start=1):
        prompt = format_prompt(tokenizer, ex["input"])
        logger.info(f"  [Base Model] Processing example {idx}/{len(eval_set)}: {ex['source_file']}")
        t0 = time.time()
        resp = generate_response(base_model, tokenizer, prompt, device, args.max_new_tokens)
        t_elapsed = time.time() - t0
        logger.info(f"  [Base Model] Completed {idx}/{len(eval_set)} in {t_elapsed:.1f}s")
        base_predictions.append(resp)

    # 5. Attach Fine-Tuned LoRA Adapter
    logger.info(f"Attaching LoRA adapter from {adapter_path}...")
    ft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    ft_model.eval()

    # 6. Run Inference with Fine-Tuned Model
    logger.info("Running inference with Fine-Tuned LoRA Model...")
    ft_predictions = []
    for idx, ex in enumerate(eval_set, start=1):
        prompt = format_prompt(tokenizer, ex["input"])
        logger.info(f"  [Fine-Tuned Model] Processing example {idx}/{len(eval_set)}: {ex['source_file']}")
        t0 = time.time()
        resp = generate_response(ft_model, tokenizer, prompt, device, args.max_new_tokens)
        t_elapsed = time.time() - t0
        logger.info(f"  [Fine-Tuned Model] Completed {idx}/{len(eval_set)} in {t_elapsed:.1f}s")
        ft_predictions.append(resp)

    # 7. Compute Official Metrics on Held-Out Test Set
    num_test = len(test_examples)
    test_base_preds = base_predictions[:num_test]
    test_ft_preds = ft_predictions[:num_test]

    logger.info("Computing metrics on held-out test split...")
    metrics_base = evaluate_predictions(test_examples, test_base_preds)
    metrics_ft = evaluate_predictions(test_examples, test_ft_preds)

    # 8. Prepare Side-by-Side Comparison for 5 Samples
    side_by_side = []
    for i, ex in enumerate(eval_set[:5]):
        side_by_side.append({
            "source_file": ex["source_file"],
            "ocr_text": ex["input"],
            "ground_truth": ex.get("schema_data") or json.loads(ex.get("output", "{}")),
            "base_model_output": base_predictions[i],
            "fine_tuned_output": ft_predictions[i],
        })

    # 9. Generate Reports
    md_report = generate_markdown_report(metrics_base, metrics_ft, side_by_side)

    # Print Comparison Table to stdout
    print("\n" + "=" * 70)
    print("PHASE 6: EVALUATION BEFORE/AFTER COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Field':<24} | {'Base Model':<12} | {'Fine-Tuned':<12} | {'Delta':<8}")
    print("-" * 70)
    for f in TARGET_SCHEMA_FIELDS:
        b_acc = metrics_base["field_accuracies"].get(f, 0.0)
        ft_acc = metrics_ft["field_accuracies"].get(f, 0.0)
        delta = ft_acc - b_acc
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        print(f"{f:<24} | {b_acc:>10.1f}% | {ft_acc:>10.1f}% | {delta_str:>7}")
    print("-" * 70)
    delta_macro = metrics_ft["macro_exact_accuracy"] - metrics_base["macro_exact_accuracy"]
    delta_macro_str = f"+{delta_macro:.1f}%" if delta_macro >= 0 else f"{delta_macro:.1f}%"
    print(f"{'OVERALL MACRO ACCURACY':<24} | {metrics_base['macro_exact_accuracy']:>10.1f}% | {metrics_ft['macro_exact_accuracy']:>10.1f}% | {delta_macro_str:>7}")
    print(f"{'VALID JSON SYNTAX RATE':<24} | {metrics_base['valid_json_rate']:>10.1f}% | {metrics_ft['valid_json_rate']:>10.1f}% | {'--':>7}")
    print("=" * 70 + "\n")

    # 10. Save Artifacts
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "test_metrics_base": metrics_base,
        "test_metrics_fine_tuned": metrics_ft,
        "side_by_side_examples": side_by_side,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation metrics JSON to {output_json_path}")

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Saved full markdown report to {output_md_path}")
    logger.info("Phase 6 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
