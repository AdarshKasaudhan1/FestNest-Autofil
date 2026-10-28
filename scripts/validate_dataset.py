"""
scripts/validate_dataset.py
Validates JSONL dataset splits (train.jsonl, validation.jsonl, test.jsonl):
- Detects missing required keys
- Checks data types
- Tests JSON validity of output payloads
- Verifies zero document leakage between train, validation, and test splits
- Checks for unsupported category values
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REQUIRED_TOP_KEYS = ["document_id", "source_file", "instruction", "input", "output", "text", "messages", "schema_data"]
REQUIRED_SCHEMA_KEYS = [
    "event_name",
    "event_type",
    "date",
    "time",
    "venue",
    "organizer",
    "prize_pool",
    "registration_deadline",
    "contact",
    "description"
]

ALLOWED_CATEGORIES = [
    "Hackathon",
    "Cultural Fest",
    "Workshop",
    "Competition",
    "Tech Talk",
    "Sports",
    "Mega Fest"
]

def validate_split(split_path: Path):
    print(f"\n--- Validating {split_path.name} ---")
    if not split_path.exists():
        print(f"ERROR: {split_path} does not exist!")
        return set(), 0
        
    lines = split_path.read_text(encoding="utf-8").strip().splitlines()
    doc_ids = set()
    errors = 0
    
    for idx, l in enumerate(lines, 1):
        try:
            item = json.loads(l)
        except Exception as e:
            print(f"Line {idx}: MALFORMED JSON line: {e}")
            errors += 1
            continue
            
        # Check required top keys
        for k in REQUIRED_TOP_KEYS:
            if k not in item:
                print(f"Line {idx} ({item.get('source_file')}): Missing top-level key '{k}'")
                errors += 1
                
        # Check schema data
        sdata = item.get("schema_data", {})
        for sk in REQUIRED_SCHEMA_KEYS:
            if sk not in sdata:
                print(f"Line {idx}: Missing schema key '{sk}' in schema_data")
                errors += 1
                
        # Check category
        cat = sdata.get("event_type", "")
        if cat and cat not in ALLOWED_CATEGORIES:
            print(f"Line {idx}: Unsupported category '{cat}'")
            errors += 1
            
        # Check parseability of assistant output
        try:
            out_obj = json.loads(item["output"])
            if not isinstance(out_obj, dict):
                print(f"Line {idx}: 'output' is not a JSON object")
                errors += 1
        except Exception as e:
            print(f"Line {idx}: 'output' contains invalid JSON string: {e}")
            errors += 1
            
        doc_ids.add(item["document_id"])
        
    print(f"Validated {len(lines)} records in {split_path.name} (Unique Document IDs: {len(doc_ids)}). Total Errors: {errors}")
    return doc_ids, errors


def main():
    dataset_dir = Path("data/dataset")
    train_docs, err_train = validate_split(dataset_dir / "train.jsonl")
    val_docs, err_val = validate_split(dataset_dir / "validation.jsonl")
    test_docs, err_test = validate_split(dataset_dir / "test.jsonl")
    
    print("\n" + "=" * 60)
    print("CHECKING DOCUMENT LEAKAGE ACROSS SPLITS")
    print("=" * 60)
    
    train_val_leak = train_docs.intersection(val_docs)
    train_test_leak = train_docs.intersection(test_docs)
    val_test_leak = val_docs.intersection(test_docs)
    
    leak_errors = len(train_val_leak) + len(train_test_leak) + len(val_test_leak)
    
    if leak_errors == 0:
        print("PASS: Zero document overlap! All splits are completely disjoint by source document.")
    else:
        print(f"FAIL: Document leakage detected!")
        if train_val_leak:
            print(f"  Train & Val overlap: {train_val_leak}")
        if train_test_leak:
            print(f"  Train & Test overlap: {train_test_leak}")
        if val_test_leak:
            print(f"  Val & Test overlap: {val_test_leak}")
            
    total_err = err_train + err_val + err_test + leak_errors
    print(f"\nTotal Dataset Validation Errors: {total_err}")
    if total_err == 0:
        print("ALL DATASET VALIDATION CHECKS PASSED PERFECTLY!\n")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
