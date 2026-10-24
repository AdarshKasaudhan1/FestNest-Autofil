"""
scripts/evaluate_accuracy.py
Evaluates extraction accuracy on test.jsonl and validation.jsonl:
Compares:
  - Baseline Heuristics (Phase A alone)
  - Progressive Hybrid Pipeline (Phase A + Phase B Merged LoRA Refinement)
Outputs field-level exact match and token F1 comparison tables to verify zero regression.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure app.py functions can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import extract_event_data, TARGET_SCHEMA_FIELDS, clean_poster_text


def normalize_value(val: Any) -> str:
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


def evaluate_dataset(split_name: str, filepath: Path) -> Dict[str, Any]:
    print(f"\n==================== EVALUATING {split_name.upper()} SET ({filepath.name}) ====================")
    with open(filepath, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} samples from {filepath}")

    heur_exact = {f: 0 for f in TARGET_SCHEMA_FIELDS}
    heur_f1 = {f: 0.0 for f in TARGET_SCHEMA_FIELDS}

    hybrid_exact = {f: 0 for f in TARGET_SCHEMA_FIELDS}
    hybrid_f1 = {f: 0.0 for f in TARGET_SCHEMA_FIELDS}

    for idx, sample in enumerate(samples, 1):
        raw_text = sample.get("input", "")
        gt = sample.get("schema_data") or json.loads(sample.get("output", "{}"))
        filename = sample.get("source_file", f"sample_{idx}")

        print(f"\n--- Sample {idx}/{len(samples)}: {filename} ---")

        # 1. Baseline Heuristics (Phase A alone)
        heur_data, heur_conf, _, _ = extract_event_data(raw_text, filename, run_phase_b=False)

        # 2. Hybrid Pipeline (Phase A + Phase B Refinement)
        hybrid_data, hybrid_conf, _, t_llm = extract_event_data(raw_text, filename, run_phase_b=True)

        for f in TARGET_SCHEMA_FIELDS:
            gt_val = normalize_value(gt.get(f, ""))
            h_val = normalize_value(heur_data.get(f, ""))
            hyb_val = normalize_value(hybrid_data.get(f, ""))

            if h_val == gt_val:
                heur_exact[f] += 1
            heur_f1[f] += compute_token_f1(h_val, gt_val)

            if hyb_val == gt_val:
                hybrid_exact[f] += 1
            hybrid_f1[f] += compute_token_f1(hyb_val, gt_val)

            if f in ["event_name", "organizer", "date", "venue"]:
                print(f"  {f:<14} | GT: {gt_val[:28]:<28} | Heur: {h_val[:24]:<24} | Hybrid: {hyb_val[:28]}")

    n = len(samples)
    field_results = []
    print("\n" + "=" * 80)
    print(f"{'Field':<24} | {'Phase A (Heuristics) F1':<25} | {'Hybrid (Phase A+B) F1':<25}")
    print("-" * 80)
    for f in TARGET_SCHEMA_FIELDS:
        h_f1_score = round((heur_f1[f] / n) * 100, 2)
        hyb_f1_score = round((hybrid_f1[f] / n) * 100, 2)
        h_acc = round((heur_exact[f] / n) * 100, 2)
        hyb_acc = round((hybrid_exact[f] / n) * 100, 2)
        field_results.append({
            "field": f,
            "heuristics_exact": h_acc,
            "heuristics_f1": h_f1_score,
            "hybrid_exact": hyb_acc,
            "hybrid_f1": hyb_f1_score,
        })
        print(f"{f:<24} | {h_f1_score:<25.1f} | {hyb_f1_score:<25.1f}")
    print("=" * 80)

    avg_h_f1 = round(sum(r["heuristics_f1"] for r in field_results) / len(field_results), 2)
    avg_hyb_f1 = round(sum(r["hybrid_f1"] for r in field_results) / len(field_results), 2)
    print(f"MACRO F1 OVERALL: Heuristics = {avg_h_f1}% | Hybrid (Phase A+B) = {avg_hyb_f1}%")

    return {
        "split": split_name,
        "sample_count": n,
        "macro_heuristics_f1": avg_h_f1,
        "macro_hybrid_f1": avg_hyb_f1,
        "fields": field_results,
    }


def main():
    test_res = evaluate_dataset("test", Path("data/dataset/test.jsonl"))
    val_res = evaluate_dataset("validation", Path("data/dataset/validation.jsonl"))

    full_results = {
        "test": test_res,
        "validation": val_res,
    }

    out_file = Path("data/accuracy_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nAccuracy report written to {out_file.resolve()}")


if __name__ == "__main__":
    main()
