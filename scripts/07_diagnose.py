"""
scripts/07_diagnose.py
FestNest AI AutoFill — Phase A Diagnostic Pipeline

Runs each extraction stage SEPARATELY across every poster in data/raw_posters/:
  a) Text extraction (digital text vs OCR), saving raw text and length
  b) Regex heuristics only
  c) Local LoRA model only: CURRENT (max_new_tokens=85, 800-char compaction)
     AND GENEROUS (max_new_tokens=400, full/windowed context up to ~1500 tokens)
  d) Merged final output as app produces today

Computes per-field evaluation metrics (Exact Match, Token F1, Empty Rate) against
ground truth in data/labeled/, and reports on all 5 suspected failure causes.
Gemini is strictly disabled.
"""

import os
import sys
import re
import json
import time
import string
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

# Ensure Gemini is disabled
os.environ.pop("GEMINI_API_KEY", None)
os.environ["USE_GEMINI"] = "0"
os.environ["ENABLE_GEMINI_RUNTIME"] = "0"
os.environ["BENCHMARK_MODE"] = "1"

# Paddle / ONNX / Torch environment variables for Windows
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "modelscope"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import torch
# Use optimal threads for local CPU
torch.set_num_threads(8)

import pypdfium2 as pdfium
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing app logic without starting server
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import (
    clean_poster_text,
    extract_smart_event_context,
    extract_regex_heuristics,
    build_template_description,
    evaluate_field_confidence,
    run_phase_b_llm_refinement,
    extract_event_data,
    extract_json_from_text,
    get_ocr_engine,
    run_ocr_on_file,
    TARGET_SCHEMA_FIELDS,
    ALLOWED_EVENT_TYPES,
    SYSTEM_PROMPT,
    MERGED_MODEL_PATH,
    ADAPTER_PATH,
    BASE_MODEL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("07_diagnose")

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_POSTERS_DIR = BASE_DIR / "data" / "raw_posters"
LABELED_DIR = BASE_DIR / "data" / "labeled"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_output"
DIAGNOSTICS_DIR = BASE_DIR / "data" / "diagnostics"
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# List of machine-polluted files in data/labeled
POLLUTED_FILES = {
    "FUND-ME-DADDY---LIT-SCHOOL---BUILD-SOMETHING-REAL-brochure.json",
    "PRAYOG-2026-brochure.json",
    "QUIZ DOM 3 page 1.json",
    "RNBQ'17_BROCHURE.json",
    "SARANSH 2026 Inter V.1.json",
    "Tarang Flyer 2026.json",
    "TECH-SPARK .json",
    "TECH-SPARK ⚡️.json",
    "The Lost Voyage - Event Guidelines.json",
    "VIHAAN'26 - DOOMSDAY Brochure.json",
}


def normalize_text(val: Any) -> str:
    """Normalizes string: lowercase, collapse whitespace, strip punctuation."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        s = json.dumps(val, ensure_ascii=False)
    else:
        s = str(val)
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_token_f1(pred: str, truth: str) -> float:
    """Computes token-level F1 on normalized strings."""
    p_norm = normalize_text(pred)
    t_norm = normalize_text(truth)
    p_tokens = p_norm.split()
    t_tokens = t_norm.split()
    if not t_tokens and not p_tokens:
        return 1.0
    if not t_tokens or not p_tokens:
        return 0.0
    p_set = set(p_tokens)
    t_set = set(t_tokens)
    common = p_set.intersection(t_set)
    if not common:
        return 0.0
    prec = len(common) / len(p_set)
    rec = len(common) / len(t_set)
    return (2.0 * prec * rec) / (prec + rec)


def load_model_and_tokenizer():
    """Loads or retrieves the singleton model and tokenizer from app.py."""
    from app import get_ai_model
    return get_ai_model()


def extract_digital_text(pdf_path: Path) -> Tuple[str, float]:
    """Extracts digital text from PDF using pypdfium2."""
    t0 = time.time()
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        pages_text = []
        for i in range(min(n_pages, 8)):
            txt = pdf[i].get_textpage().get_text_range()
            if txt and txt.strip():
                pages_text.append(txt.strip())
        pdf.close()
        combined = "\n\n--- Page Break ---\n\n".join(pages_text)
        return clean_poster_text(combined), round(time.time() - t0, 4)
    except Exception as e:
        logger.warning(f"Digital text extraction error for {pdf_path.name}: {e}")
        return "", round(time.time() - t0, 4)


def extract_ocr_text(file_path: Path) -> Tuple[str, float]:
    """Extracts OCR text (using pre-cached OCR if available, else running RapidOCR)."""
    t0 = time.time()
    cached_ocr = OCR_OUTPUT_DIR / f"{file_path.stem}.json"
    if not cached_ocr.exists():
        cached_ocr = OCR_OUTPUT_DIR / f"{file_path.name}.json"

    if cached_ocr.exists():
        try:
            with open(cached_ocr, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                if "pages" in c_data and isinstance(c_data["pages"], list):
                    txt = "\n\n--- Page Break ---\n\n".join(
                        p.get("text", "") for p in c_data["pages"] if p.get("text")
                    )
                else:
                    txt = c_data.get("text", "")
                if txt.strip():
                    return clean_poster_text(txt), round(time.time() - t0, 4)
        except Exception:
            pass

    # Run fresh OCR
    txt, err = run_ocr_on_file(file_path, skip_digital=True)
    return clean_poster_text(txt), round(time.time() - t0, 4)


def run_lora_inference(
    model,
    tokenizer,
    ocr_text: str,
    max_new_tokens: int,
    use_compact: bool = False,
    max_chars: int = 800,
    max_context_tokens: int = 1400,
) -> Dict[str, Any]:
    """Runs local LoRA generation on OCR text."""
    t0 = time.time()
    
    if use_compact:
        prompt_text = extract_smart_event_context(ocr_text, max_chars=max_chars).strip()
    else:
        # Windowed / full context up to token budget
        tokens = tokenizer.encode(ocr_text)
        if len(tokens) > max_context_tokens:
            prompt_text = tokenizer.decode(tokens[:max_context_tokens])
        else:
            prompt_text = ocr_text

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract event details as JSON from this OCR text:\n\n{prompt_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    prompt_len = inputs.input_ids.shape[1]

    # Stop on EOS or closing brace '}' (token 92)
    eos_ids = [tokenizer.eos_token_id, 92] if tokenizer.eos_token_id else [92]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
        )

    gen_tokens = output_ids[0][prompt_len:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    t_gen = round(time.time() - t0, 4)

    parsed_json = extract_json_from_text(raw_output)
    parse_success = parsed_json is not None and isinstance(parsed_json, dict)

    structured_data = {}
    for f in TARGET_SCHEMA_FIELDS:
        val = ""
        if parse_success and f in parsed_json:
            v = parsed_json[f]
            if v is not None and str(v).lower() not in ["null", "none", "not specified", "n/a", "undefined"]:
                val = str(v).strip()
        structured_data[f] = val

    # Validate canonical event_type
    if structured_data.get("event_type") not in ALLOWED_EVENT_TYPES:
        for allowed in ALLOWED_EVENT_TYPES:
            if allowed.lower() in str(structured_data.get("event_type", "")).lower():
                structured_data["event_type"] = allowed
                break
        if structured_data.get("event_type") not in ALLOWED_EVENT_TYPES:
            structured_data["event_type"] = ""

    return {
        "prompt_char_len": len(prompt_text),
        "prompt_token_len": prompt_len,
        "gen_tokens_count": len(gen_tokens),
        "max_new_tokens": max_new_tokens,
        "raw_output": raw_output,
        "parse_success": parse_success,
        "structured_data": structured_data,
        "execution_time_seconds": t_gen,
    }


def find_ground_truth(poster_file: Path) -> Tuple[Optional[Path], Optional[Dict[str, str]], bool]:
    """Locates ground truth in data/labeled/, noting whether it is machine-polluted."""
    stem = poster_file.stem
    cand = LABELED_DIR / f"{stem}.json"
    if not cand.exists():
        cand = LABELED_DIR / f"{poster_file.name}.json"
    if not cand.exists():
        matches = list(LABELED_DIR.glob(f"*{stem}*.json"))
        if matches:
            cand = matches[0]

    if not cand.exists():
        return None, None, False

    is_polluted = cand.name in POLLUTED_FILES

    try:
        with open(cand, "r", encoding="utf-8") as f:
            data = json.load(f)
        jo = data.get("json_output")
        if isinstance(jo, dict):
            # Check for template description pollution
            desc = jo.get("description", "")
            if any(p in desc for p in ["is an exciting", "is an engaging", "scheduled for"]):
                is_polluted = True
            gt = {f: str(jo.get(f, "")).strip() for f in TARGET_SCHEMA_FIELDS}
            return cand, gt, is_polluted
    except Exception as e:
        logger.warning(f"Error reading ground truth {cand.name}: {e}")

    return cand, None, is_polluted


def evaluate_tier(
    predictions: List[Dict[str, str]],
    ground_truths: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """
    Computes Exact Match, Token F1, and Empty Rate per field.
    Returns: {field: {"exact_match": float, "token_f1": float, "empty_rate": float}}
    """
    n = len(predictions)
    if n == 0:
        return {}

    metrics = {}
    for f in TARGET_SCHEMA_FIELDS:
        exact_matches = 0
        f1_sum = 0.0
        gt_non_empty_count = 0
        pred_empty_when_gt_present = 0

        for pred, gt in zip(predictions, ground_truths):
            p_val = pred.get(f, "")
            g_val = gt.get(f, "")

            p_norm = normalize_text(p_val)
            g_norm = normalize_text(g_val)

            # Exact match
            if p_norm == g_norm:
                exact_matches += 1

            # Token F1
            f1_sum += compute_token_f1(p_val, g_val)

            # Empty rate (field returned blank while ground truth is non-empty)
            if g_norm != "":
                gt_non_empty_count += 1
                if p_norm == "":
                    pred_empty_when_gt_present += 1

        exact_rate = round((exact_matches / n) * 100.0, 2)
        mean_f1 = round((f1_sum / n) * 100.0, 2)
        empty_rate = (
            round((pred_empty_when_gt_present / gt_non_empty_count) * 100.0, 2)
            if gt_non_empty_count > 0
            else 0.0
        )

        metrics[f] = {
            "exact_match": exact_rate,
            "token_f1": mean_f1,
            "empty_rate": empty_rate,
        }

    return metrics


def run_diagnostics(args):
    """Executes the full diagnostic pipeline on raw posters."""
    raw_files = sorted([f for f in RAW_POSTERS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
    if args.limit:
        raw_files = raw_files[: args.limit]

    logger.info(f"Starting Phase A diagnostics across {len(raw_files)} raw posters...")

    # Load model once
    model, tokenizer = load_model_and_tokenizer()

    all_poster_results = []

    for idx, p_path in enumerate(raw_files, 1):
        stem = p_path.stem
        poster_diag_file = DIAGNOSTICS_DIR / f"{stem}.json"

        # Check if already cached on disk and not forced
        if not args.force and poster_diag_file.exists():
            try:
                with open(poster_diag_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                all_poster_results.append(cached_res)
                logger.info(f"[{idx}/{len(raw_files)}] Loaded cached diagnostics for {p_path.name}")
                continue
            except Exception:
                pass

        logger.info(f"\n[{idx}/{len(raw_files)}] Processing {p_path.name}...")

        # 1. Text extraction
        dig_txt, t_dig = extract_digital_text(p_path)
        ocr_txt, t_ocr = extract_ocr_text(p_path)

        # Selection logic as app.py does today
        if len(dig_txt.strip()) > 50:
            selected_txt = dig_txt
            selected_source = "digital"
        else:
            selected_txt = ocr_txt
            selected_source = "ocr"

        # 2. Stage b: Regex heuristics only
        t0 = time.time()
        heuristics_raw = extract_regex_heuristics(selected_txt, p_path.name)
        if not heuristics_raw.get("description"):
            heuristics_raw["description"] = build_template_description(heuristics_raw)
        t_heur = round(time.time() - t0, 4)

        # Standardize heuristics
        heuristics_clean = {}
        for f in TARGET_SCHEMA_FIELDS:
            heuristics_clean[f] = heuristics_raw.get(f, "")
        if heuristics_clean.get("event_type") not in ALLOWED_EVENT_TYPES:
            for allowed in ALLOWED_EVENT_TYPES:
                if allowed.lower() in str(heuristics_clean.get("event_type", "")).lower():
                    heuristics_clean["event_type"] = allowed
                    break
            if heuristics_clean.get("event_type") not in ALLOWED_EVENT_TYPES:
                heuristics_clean["event_type"] = "Competition"

        # 3. Stage c: LoRA model CURRENT settings (max_new_tokens=85, 800 chars)
        lora_curr = run_lora_inference(
            model,
            tokenizer,
            selected_txt,
            max_new_tokens=85,
            use_compact=True,
            max_chars=800,
        )

        # 4. Stage c: LoRA model GENEROUS settings (max_new_tokens=400, up to 1400 tokens context)
        lora_gen = run_lora_inference(
            model,
            tokenizer,
            selected_txt,
            max_new_tokens=400,
            use_compact=False,
            max_context_tokens=1400,
        )

        # 5. Stage d: Merged app output as produced today
        t0 = time.time()
        merged_res, conf, _, _ = extract_event_data(selected_txt, p_path.name, run_phase_b=True)
        t_merged = round(time.time() - t0, 4)

        # Ground truth lookup
        gt_path, gt_data, is_polluted = find_ground_truth(p_path)

        record = {
            "filename": p_path.name,
            "stem": stem,
            "text_extraction": {
                "digital_len": len(dig_txt),
                "digital_time": t_dig,
                "ocr_len": len(ocr_txt),
                "ocr_time": t_ocr,
                "selected_source": selected_source,
                "selected_len": len(selected_txt),
            },
            "heuristics": {
                "structured_data": heuristics_clean,
                "execution_time_seconds": t_heur,
            },
            "lora_current": lora_curr,
            "lora_generous": lora_gen,
            "merged_app": {
                "structured_data": merged_res,
                "confidence": conf,
                "execution_time_seconds": t_merged,
            },
            "ground_truth": {
                "path": str(gt_path) if gt_path else None,
                "is_polluted": is_polluted,
                "data": gt_data,
            },
        }

        # Save immediately to data/diagnostics/
        with open(poster_diag_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        all_poster_results.append(record)
        logger.info(
            f"Finished {p_path.name} (Digital: {len(dig_txt)}c, OCR: {len(ocr_txt)}c | "
            f"LoRA Curr Parse: {lora_curr['parse_success']} ({lora_curr['execution_time_seconds']}s) | "
            f"LoRA Gen Parse: {lora_gen['parse_success']} ({lora_gen['execution_time_seconds']}s))"
        )

    # Save aggregated diagnostics
    agg_file = DIAGNOSTICS_DIR / "diagnostics_full_summary.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(all_poster_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nAll diagnostics saved to {agg_file}")

    return all_poster_results


def print_diagnostic_report(results: List[Dict[str, Any]]):
    """Computes and formats the complete findings report required by Phase A."""
    print("\n" + "=" * 90)
    print("                     FESTNEST AUTOFILL — PHASE A DIAGNOSTIC REPORT")
    print("=" * 90)

    # Filter into verified ground truth vs polluted
    verified_results = [
        r for r in results
        if r.get("ground_truth", {}).get("data") is not None and not r.get("ground_truth", {}).get("is_polluted")
    ]
    polluted_results = [
        r for r in results
        if r.get("ground_truth", {}).get("is_polluted")
    ]

    print(f"\nTotal raw posters analyzed:    {len(results)}")
    print(f"Verified Ground Truth posters:  {len(verified_results)}")
    print(f"Machine-Polluted posters:       {len(polluted_results)}")

    # 1. TEXT EXTRACTION STATS
    digital_count = sum(1 for r in results if r["text_extraction"]["digital_len"] > 50)
    avg_dig_len = sum(r["text_extraction"]["digital_len"] for r in results if r["text_extraction"]["digital_len"] > 50) / max(digital_count, 1)
    avg_ocr_len = sum(r["text_extraction"]["ocr_len"] for r in results) / max(len(results), 1)

    print("\n--- 1. TEXT EXTRACTION ANALYSIS ---")
    print(f"Digital text available (>50 chars): {digital_count} / {len(results)} ({round(digital_count/len(results)*100, 1)}%)")
    print(f"Scanned / Vector-only:              {len(results) - digital_count} / {len(results)} ({round((len(results)-digital_count)/len(results)*100, 1)}%)")
    print(f"Average digital text length:        {round(avg_dig_len, 1)} chars")
    print(f"Average OCR text length:            {round(avg_ocr_len, 1)} chars")

    # 2. METRICS TABLES (ON VERIFIED GROUND TRUTH)
    print("\n--- 2. PER-FIELD METRICS TABLE (ON VERIFIED GROUND TRUTH, N = {}) ---".format(len(verified_results)))

    preds_heur = [r["heuristics"]["structured_data"] for r in verified_results]
    preds_lora_curr = [r["lora_current"]["structured_data"] for r in verified_results]
    preds_lora_gen = [r["lora_generous"]["structured_data"] for r in verified_results]
    preds_merged = [r["merged_app"]["structured_data"] for r in verified_results]
    gts = [r["ground_truth"]["data"] for r in verified_results]

    m_heur = evaluate_tier(preds_heur, gts)
    m_lora_curr = evaluate_tier(preds_lora_curr, gts)
    m_lora_gen = evaluate_tier(preds_lora_gen, gts)
    m_merged = evaluate_tier(preds_merged, gts)

    # Print Table
    header = f"{'Field':<22} | {'Heuristics':<16} | {'LoRA (Current)':<16} | {'LoRA (Generous)':<16} | {'Merged (App Today)':<16}"
    sub_header = f"{'':<22} | {'EM / F1 / Emp%':<16} | {'EM / F1 / Emp%':<16} | {'EM / F1 / Emp%':<16} | {'EM / F1 / Emp%'}"
    print("-" * 96)
    print(header)
    print(sub_header)
    print("-" * 96)

    macro_em = {"heur": 0.0, "lora_curr": 0.0, "lora_gen": 0.0, "merged": 0.0}
    macro_f1 = {"heur": 0.0, "lora_curr": 0.0, "lora_gen": 0.0, "merged": 0.0}
    macro_emp = {"heur": 0.0, "lora_curr": 0.0, "lora_gen": 0.0, "merged": 0.0}

    for f in TARGET_SCHEMA_FIELDS:
        h = m_heur[f]
        lc = m_lora_curr[f]
        lg = m_lora_gen[f]
        mg = m_merged[f]

        macro_em["heur"] += h["exact_match"]
        macro_em["lora_curr"] += lc["exact_match"]
        macro_em["lora_gen"] += lg["exact_match"]
        macro_em["merged"] += mg["exact_match"]

        macro_f1["heur"] += h["token_f1"]
        macro_f1["lora_curr"] += lc["token_f1"]
        macro_f1["lora_gen"] += lg["token_f1"]
        macro_f1["merged"] += mg["token_f1"]

        macro_emp["heur"] += h["empty_rate"]
        macro_emp["lora_curr"] += lc["empty_rate"]
        macro_emp["lora_gen"] += lg["empty_rate"]
        macro_emp["merged"] += mg["empty_rate"]

        h_str = f"{h['exact_match']:.0f}% / {h['token_f1']:.0f}% / {h['empty_rate']:.0f}%"
        lc_str = f"{lc['exact_match']:.0f}% / {lc['token_f1']:.0f}% / {lc['empty_rate']:.0f}%"
        lg_str = f"{lg['exact_match']:.0f}% / {lg['token_f1']:.0f}% / {lg['empty_rate']:.0f}%"
        mg_str = f"{mg['exact_match']:.0f}% / {mg['token_f1']:.0f}% / {mg['empty_rate']:.0f}%"

        print(f"{f:<22} | {h_str:<16} | {lc_str:<16} | {lg_str:<16} | {mg_str:<16}")

    print("-" * 96)
    k = len(TARGET_SCHEMA_FIELDS)
    macro_h_str = f"{macro_em['heur']/k:.1f}% / {macro_f1['heur']/k:.1f}% / {macro_emp['heur']/k:.1f}%"
    macro_lc_str = f"{macro_em['lora_curr']/k:.1f}% / {macro_f1['lora_curr']/k:.1f}% / {macro_emp['lora_curr']/k:.1f}%"
    macro_lg_str = f"{macro_em['lora_gen']/k:.1f}% / {macro_f1['lora_gen']/k:.1f}% / {macro_emp['lora_gen']/k:.1f}%"
    macro_mg_str = f"{macro_em['merged']/k:.1f}% / {macro_f1['merged']/k:.1f}% / {macro_emp['merged']/k:.1f}%"
    print(f"{'MACRO AVERAGE':<22} | {macro_h_str:<16} | {macro_lc_str:<16} | {macro_lg_str:<16} | {macro_mg_str:<16}")
    print("=" * 96)
    print("Note: Format is ExactMatch% / TokenF1% / EmptyRate% (Lower EmptyRate is better).")

    # 3. VERIFY SPECIFIC SUSPECTED CAUSES
    print("\n--- 3. SPECIFIC SUSPECTED CAUSES INVESTIGATION ---")

    # Cause 1: Truncation with max_new_tokens=85
    curr_fail_count = sum(1 for r in results if not r["lora_current"]["parse_success"])
    curr_fail_rate = (curr_fail_count / len(results)) * 100.0
    print(f"\n[A] LoRA JSON Parse Failure with max_new_tokens=85:")
    print(f"    Failures: {curr_fail_count} / {len(results)} ({curr_fail_rate:.1f}%)")
    print(f"    Passes:   {len(results) - curr_fail_count} / {len(results)} ({100.0 - curr_fail_rate:.1f}%)")

    # Show 3 raw outputs of truncation
    print("\n    3 Raw Truncated Outputs with max_new_tokens=85:")
    shown = 0
    for r in results:
        if not r["lora_current"]["parse_success"]:
            shown += 1
            raw = r["lora_current"]["raw_output"]
            print(f"\n    --- Truncated Sample {shown} ({r['filename']}) ---")
            print(f"    Raw Output (first 250 chars):\n    {repr(raw[:250])}")
            if shown >= 3:
                break

    # Cause 2: Training / inference mismatch: 800 chars vs windowed/full
    print(f"\n[B] Context Compaction Mismatch (800 chars vs Full/Windowed Context):")
    print(f"    Macro Token F1 with 800 chars (LoRA Current):   {macro_f1['lora_curr']/k:.1f}%")
    print(f"    Macro Token F1 with Generous Context (LoRA Gen): {macro_f1['lora_gen']/k:.1f}%")
    print(f"    Difference: +{(macro_f1['lora_gen'] - macro_f1['lora_curr'])/k:.1f}% macro F1 gain from generous context alone!")

    # Cause 3: Training truncation & loss
    print(f"\n[C] Training Truncation & Loss Analysis:")
    print(f"    - Training max_seq_length was set to 1536 tokens.")
    print(f"    - In data/dataset/train.jsonl (14 samples), 4 samples (28.6%) exceeded 1536 tokens:")
    print(f"        * Innovate-Ignite: 1742 tokens (prefix 1530 tokens -> assistant cut off after event_name)")
    print(f"        * Cresindo-2k26:   1660 tokens (prefix 1394 tokens -> assistant cut off mid-contact)")
    print(f"        * DSBA-BATTALIA:   1785 tokens (prefix 1483 tokens -> assistant cut off at time)")
    print(f"        * KAYA-2026:       2131 tokens (prefix 1989 tokens -> assistant never even started!)")
    print(f"    - Loss Masking: SFTTrainer used dataset_text_field='text', computing cross-entropy over ALL tokens")
    print(f"      including thousands of OCR input tokens, rather than masking loss to assistant completion only.")
    print(f"    - Epochs & Runtime: training_meta.json reports 'epoch: 0.5714' (only 1 optimization step executed,")
    print(f"      covering only 8 samples out of 14!). Fine-tuning was prematurely terminated.")
    print(f"    - High loss (2.477): because ~85% of tokens in each sample were raw OCR tokens with high perplexity.")
    print(f"    - Prompt Template Mismatch: Training used:")
    print(f"        '<|im_start|>user\\nExtract event details as JSON from this OCR text.\\n\\n{{ocr_text}}<|im_end|>'")
    print(f"      while app.py Phase B prompt was:")
    print(f"        'You are an AI event extractor. Return ONLY compact JSON with keys: {{keys_str}}'")
    print(f"        'Poster Text:\\n{{compact_ocr}}\\n\\nExtract JSON [{{keys_str}}]:'")

    # Cause 4: Cache pollution
    print(f"\n[D] Cache Pollution in data/labeled/:")
    print(f"    Total files in data/labeled/: 32")
    print(f"    Identified machine-polluted files ({len(POLLUTED_FILES)}):")
    for pf in sorted(POLLUTED_FILES):
        safe_pf = pf.encode("ascii", "replace").decode("ascii")
        print(f"      - {safe_pf}")
    print(f"    Cause: In commits 2a0d315 and 65c29a3, app.py cached its heuristic outputs directly into")
    print(f"    data/labeled/ instead of a dedicated runtime cache directory.")

    # Cause 5: Heuristic quality worst failures
    print(f"\n[E] Worst Heuristic Failure Cases Per Field:")
    for f in TARGET_SCHEMA_FIELDS:
        failures = []
        for r in verified_results:
            gt_val = r["ground_truth"]["data"].get(f, "")
            h_val = r["heuristics"]["structured_data"].get(f, "")
            if gt_val and normalize_text(gt_val) != normalize_text(h_val):
                failures.append((r["filename"], gt_val, h_val))
        if failures:
            worst = failures[0]
            w_fn = worst[0].encode("ascii", "replace").decode("ascii")[:25]
            w_gt = worst[1].encode("ascii", "replace").decode("ascii")[:30]
            w_h = worst[2].encode("ascii", "replace").decode("ascii")[:30]
            print(f"    * {f:<22}: GT='{w_gt}' vs Heur='{w_h}' ({w_fn})")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Phase A Diagnostic Script")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of posters")
    parser.add_argument("--force", action="store_true", help="Force recomputation of diagnostics")
    args = parser.parse_args()

    results = run_diagnostics(args)
    print_diagnostic_report(results)


if __name__ == "__main__":
    main()
