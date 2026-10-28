"""
scripts/03_prepare_dataset.py
FestNest AI AutoFill — Phase 4: Dataset Preparation (Improved)
Loads verified records from data/verified_ground_truth.json and data/labeled/,
formats them for instruction-tuning with Qwen2.5 ChatML format,
and performs strict DOCUMENT-LEVEL GROUPED SPLITTING:
- Train Split: ~70% of documents (19 documents)
- Validation Split: ~15% of documents (4 documents)
- Held-Out Test Split: ~15% of documents (4 documents)
Guarantees NO document leakage between train, validation, and test sets.
Prevents target JSON truncation by formatting clean, focused context.
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

DEFAULT_GT_PATH = Path("data/verified_ground_truth.json")
DEFAULT_OUTPUT_DIR = Path("data/dataset")

TARGET_SCHEMA_FIELDS = [
    "event_name",
    "event_type",
    "event_category",
    "date",
    "event_dates",
    "time",
    "event_time",
    "venue",
    "organizer",
    "prize_pool",
    "registration_deadline",
    "registration_link",
    "contact",
    "contact_details",
    "description",
    "event_description",
]

INSTRUCTION_TEXT = "Extract event details as JSON matching FestNest schemas from this poster text."
SYSTEM_PROMPT = (
    "You are an AI assistant that extracts structured event information from poster text "
    "into JSON format matching FestNest schemas."
)

# Fixed Document-Level Splits to Guarantee Zero Document Leakage
HELD_OUT_TEST_DOCS = {
    "BUHONO-26",           # Scanned school quiz contest (2 pages)
    "CHEMOVISTA-2K26",     # Technical national hackathon (2 pages)
    "SARANSH-2026",        # Academic book review competition (2 pages)
    "Anchorage-Shark-Tank" # CUSAT guidelines competition (5 pages)
}

VALIDATION_DOCS = {
    "ACM-MarketSphere",    # Single-page vector finance hackathon (1 page)
    "IDEATHON-4",          # Single-page university ideathon (1 page)
    "QUIZ-DOM-3",          # Multi-page scanned college quiz (2 pages)
    "DSBA-BATTALIA"        # Multi-page sports tournament (2 pages)
}


def load_verified_dataset(gt_path: Path) -> List[Dict[str, Any]]:
    """Loads verified records, filtering out corrupted/invalid files."""
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)
        
    valid_records = []
    for r in all_records:
        if not r.get("is_valid_event", True):
            logger.info(f"Skipping invalid/corrupted document: {r.get('source_file')} ({r.get('provenance')})")
            continue
        valid_records.append(r)
        
    logger.info(f"Loaded {len(valid_records)} valid event records from {len(all_records)} total entries.")
    return valid_records


def build_compact_input_text(event: Dict[str, Any], raw_texts: Dict[str, str]) -> str:
    """
    Constructs clean, informative input text for instruction tuning without exceeding
    token limits (keeping input under ~1000 tokens so output JSON is never truncated).
    """
    src = event["source_file"]
    full_text = raw_texts.get(src, "")
    
    # If sub-event, focus on sub-event relevant text
    if event.get("is_sub_event"):
        ev_name = event["event_name"].split("-")[-1].strip().lower()
        sub_lines = []
        for line in full_text.splitlines():
            l_str = line.strip()
            if not l_str:
                continue
            if ev_name in l_str.lower() or any(k in l_str.lower() for k in ["prize", "fee", "date", "time", "venue", "contact", "guidelines", "rules"]):
                sub_lines.append(l_str)
        if len(sub_lines) > 5:
            return "\n".join(sub_lines)[:2500]
            
    # For full brochures, prioritize top lines and key metadata lines
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    if len("\n".join(lines)) <= 2500:
        return "\n".join(lines)
        
    header = lines[:20]
    tail = lines[20:]
    keywords = ["date", "time", "venue", "prize", "register", "contact", "phone", "organis", "fee", "rules", "cash"]
    filtered_tail = [l for l in tail if any(k in l.lower() for k in keywords) or any(c.isdigit() for c in l)][:40]
    
    return "\n".join(header + ["\n--- Highlights ---"] + filtered_tail)[:2500]


def format_instruction_record(record: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """Formats event into Alpaca and ChatML format."""
    canonical_json = {
        "event_name": record.get("event_name", ""),
        "event_type": record.get("event_category", ""),
        "date": record.get("event_dates", ""),
        "time": record.get("event_time", ""),
        "venue": record.get("venue", ""),
        "organizer": record.get("organizer", ""),
        "prize_pool": record.get("prize_pool", ""),
        "registration_deadline": record.get("registration_deadline", ""),
        "registration_link": record.get("registration_link", ""),
        "contact": record.get("contact_details", ""),
        "description": record.get("event_description", ""),
    }
    
    json_output_str = json.dumps(canonical_json, indent=2, ensure_ascii=False)
    
    chatml_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{INSTRUCTION_TEXT}\n\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{json_output_str}<|im_end|>"
    )
    
    return {
        "document_id": record["document_id"],
        "source_file": record["source_file"],
        "instruction": INSTRUCTION_TEXT,
        "input": input_text,
        "output": json_output_str,
        "text": chatml_text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION_TEXT}\n\n{input_text}"},
            {"role": "assistant", "content": json_output_str}
        ],
        "schema_data": canonical_json
    }


def split_by_document(
    formatted_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records strictly by source document group."""
    train_records = []
    val_records = []
    test_records = []
    
    for r in formatted_records:
        doc_id = r["document_id"]
        if doc_id in HELD_OUT_TEST_DOCS:
            test_records.append(r)
        elif doc_id in VALIDATION_DOCS:
            val_records.append(r)
        else:
            train_records.append(r)
            
    logger.info("=" * 60)
    logger.info(f"Strict Document-Level Split:")
    logger.info(f"  Train:      {len(train_records)} examples across {len(set(r['document_id'] for r in train_records))} documents")
    logger.info(f"  Validation: {len(val_records)} examples across {len(set(r['document_id'] for r in val_records))} documents ({sorted(VALIDATION_DOCS)})")
    logger.info(f"  Test:       {len(test_records)} examples across {len(set(r['document_id'] for r in test_records))} documents ({sorted(HELD_OUT_TEST_DOCS)})")
    logger.info("=" * 60)
    
    return train_records, val_records, test_records


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    in_lens = [len(r["input"]) for r in records]
    out_lens = [len(r["output"]) for r in records]
    return {
        "count": len(records),
        "input_char_min": min(in_lens),
        "input_char_max": max(in_lens),
        "input_char_avg": round(sum(in_lens) / len(in_lens), 1),
        "output_char_min": min(out_lens),
        "output_char_max": max(out_lens),
        "output_char_avg": round(sum(out_lens) / len(out_lens), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default=str(DEFAULT_GT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw texts
    raw_texts = {}
    raw_path = Path("data/all_29_extracted_raw.json")
    if raw_path.exists():
        docs = json.load(open(raw_path, encoding="utf-8"))
        for d in docs:
            raw_texts[d["filename"]] = "\n\n--- Page Break ---\n\n".join(p["text"] for p in d["pages"])
            
    # Load verified ground truth
    events = load_verified_dataset(Path(args.gt_path))
    
    # Format records
    formatted = []
    for ev in events:
        inp = build_compact_input_text(ev, raw_texts)
        rec = format_instruction_record(ev, inp)
        formatted.append(rec)
        
    train_set, val_set, test_set = split_by_document(formatted)
    
    # Save JSONL files
    for split_name, s_data in [("train", train_set), ("validation", val_set), ("test", test_set)]:
        p = output_dir / f"{split_name}.jsonl"
        with open(p, "w", encoding="utf-8") as out:
            for item in s_data:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(s_data)} records to {p}")
        
    # Save info
    info = {
        "created_at": datetime.now().isoformat(),
        "splitting_strategy": "document_grouped_strict",
        "held_out_test_docs": sorted(list(HELD_OUT_TEST_DOCS)),
        "validation_docs": sorted(list(VALIDATION_DOCS)),
        "schema_fields": TARGET_SCHEMA_FIELDS,
        "splits": {
            "train": compute_statistics(train_set),
            "validation": compute_statistics(val_set),
            "test": compute_statistics(test_set),
            "total_samples": len(formatted)
        },
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "test": str(output_dir / "test.jsonl")
        }
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as inf:
        json.dump(info, inf, indent=2)
    logger.info(f"Saved dataset metadata to {output_dir / 'dataset_info.json'}")

if __name__ == "__main__":
    main()
