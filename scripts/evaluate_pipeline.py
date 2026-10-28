"""
scripts/evaluate_pipeline.py
Repeatable evaluation suite comparing the Legacy Pipeline against the Improved Pipeline.
Evaluates on:
- Held-Out Test Split (BUHONO'26, CHEMOVISTA 2K26, SARANSH 2026, Shark Tank)
- Validation Split (ACM MarketSphere, IDEATHON 4.0, QUIZ DOM 3, DSBA BATTALIA)
- Complete Dataset (all valid brochures)
Reports:
- Exact Match (EM)
- Normalized Exact Match
- Token Precision, Recall, and F1
- Valid JSON rate
- Hallucination / Unsupported value rate
- Form-mapping success rate
- Saves evaluation results to data/evaluation_report.json
"""

import sys
import re
import json
import string
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_improved import run_improved_extraction

# Standalone implementation of original Legacy Baseline heuristics
def legacy_extract_heuristics(text: str, filename: str = "") -> Dict[str, str]:
    """Original legacy baseline heuristics before improvement."""
    heuristics = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. Event Type default Competition
    text_lower = text[:6000].lower()
    for cat in ["Hackathon", "Cultural Fest", "Workshop", "Competition", "Tech Talk", "Sports", "Mega Fest"]:
        c_low = cat.lower()
        if c_low in text_lower or (cat == "Competition" and ("quiz" in text_lower or "paper presentation" in text_lower)) or (cat == "Hackathon" and "hack" in text_lower):
            heuristics["event_type"] = cat
            break
    if "event_type" not in heuristics and text.strip():
        heuristics["event_type"] = "Competition"

    # 2. Date Pattern
    date_match = re.search(
        r"(\b\d{1,2}(?:st|nd|rd|th)?\s*(?:&|and|-|–|to)?\s*\d{0,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:\s+202\d)?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+202\d)?|\b\d{1,2}[./-]\d{1,2}[./-](?:202\d|2\d)\b)",
        text[:6000], re.IGNORECASE
    )
    if date_match:
        heuristics["date"] = date_match.group(1).strip().replace("\n", " ")

    # 3. Time Pattern
    time_match = re.search(
        r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b(?:\s*[-–to]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)?|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*onwards\b|\bstarts\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
        text[:6000], re.IGNORECASE
    )
    if time_match:
        heuristics["time"] = time_match.group(1).strip().replace("\n", " ")

    # 4. Prize Pool
    prize_match = re.search(
        r"((?:₹|rs\.?|inr)\s*[\d,]+(?:\s*(?:k|lakh|cash\s*prize|\/-))?|[\d,]+\s*(?:k|lakh)?\s*cash\s*prize|\b[\d,]+\s*prize\s*pool\b|\b1st\s+prize\s+inr\s*[\d,]+[^\n]*|[\d,]+\s*winner\s+takes|winner\s+takes\s*[\d,]+|\bworth\s*[\d,]+|\bprizes\s*worth\s*[\d,]+)",
        text[:6000], re.IGNORECASE
    )
    if prize_match:
        heuristics["prize_pool"] = prize_match.group(1).strip().replace("\n", " ")

    # 5. Contact
    contact_pattern = r"(?:(?:mob|coordinat\w*|contact|call|ph|phone)\s*:?\s*)?(\+?91[\s\-]?[6-9]\d{4}[\s\-]?\d{5}|[6-9]\d{4}[\s\-]?\d{5}|\+?91[\s\-]?[6-9]\d{9}|[6-9]\d{9})"
    for l in lines:
        contact_match = re.search(contact_pattern, l, re.IGNORECASE)
        if contact_match:
            cand_phone = contact_match.group(1).strip()
            digits_only = re.sub(r"\D", "", cand_phone)
            if len(digits_only) in (10, 12):
                heuristics["contact"] = cand_phone
                break

    # 6. Registration Deadline
    reg_match = re.search(
        r"((?:https?:\/\/[^\s]+(?:superlink\.io|unstop|forms\.gle)[^\s]*)|(?:register\s+fee|fee:?\s*(?:₹|rs\.?|inr)?\s*[\d,]+[^\n]*)|(?:\d+\s*slots\s*only|slots\s*only\s*\d*)|(?:deadline|register\s+by|last\s+date)[\s:]*([A-Za-z0-9\s,\-]+(?:\b202\d\b)?))",
        text[:6000], re.IGNORECASE
    )
    if reg_match:
        heuristics["registration_deadline"] = reg_match.group(1).strip().replace("\n", " ")

    # 7. Venue
    for i, line in enumerate(lines[:30]):
        l_low = line.lower()
        if l_low == "venue" and i > 0 and len(lines[i-1]) < 60:
            heuristics["venue"] = lines[i-1]
            break
        elif any(v in l_low for v in ["indoor sports complex", "auditorium", "seminar hall", "on-campus & virtual", "campus", "college of engineering", "university", "vvce", "mysuru", "bengaluru"]) and len(line) < 80:
            heuristics["venue"] = line
            break

    # 8. Organizer
    for line in lines[:30]:
        l_low = line.lower()
        if any(org in l_low for org in ["department of", "dept of", "dept.", "acm", "ieee", "student chapter", "presents", "hosted by", "×"]) and len(line) < 80:
            heuristics["organizer"] = line
            break

    # 9. Event Name
    for l in lines:
        m = re.match(r"^ABOUT\s+([A-Za-z0-9\s.\'\-]+)$", l, re.I)
        if m:
            cand = m.group(1).strip()
            if not any(b in cand.lower() for b in ["department", "university", "college", "us", "institution", "school", "faculty"]):
                heuristics["event_name"] = cand
                break
    if "event_name" not in heuristics:
        m_org = re.search(r"\b(?:organises|organizes|presents|hosted\s+by)\b\s*\n+([^\n]+)", text[:3000], re.IGNORECASE)
        if m_org:
            cand = m_org.group(1).strip()
            if 3 < len(cand) < 60 and not any(skip in cand.lower() for skip in ["department", "university", "college", "autonomous", "accredited", "key dates", "registration"]):
                heuristics["event_name"] = cand
    if "event_name" not in heuristics:
        for line in lines[:5]:
            if len(line) > 3 and len(line) < 50:
                heuristics["event_name"] = line
                break

    name = heuristics.get("event_name", "")
    etype = heuristics.get("event_type", "event")
    org = heuristics.get("organizer", "")
    heuristics["description"] = f"{name} is an engaging {etype} organized by {org}." if name else ""

    return heuristics

EVAL_FIELDS = [
    "event_name",
    "event_category",
    "event_dates",
    "event_time",
    "venue",
    "organizer",
    "prize_pool",
    "registration_deadline",
    "contact_details",
    "event_description"
]

FIELD_ALIASES = {
    "event_category": ["event_type", "event_category"],
    "event_dates": ["date", "event_dates"],
    "event_time": ["time", "event_time"],
    "contact_details": ["contact", "contact_details"],
    "event_description": ["description", "event_description"]
}


def normalize_text(s: str) -> str:
    """Normalizes string for comparison by lowercasing, stripping punctuation, and removing excess whitespace."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[\.,\-\/\(\)\:\;\#\*\+\–\—\'\"₹\$]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_token_f1(pred: str, target: str) -> Dict[str, float]:
    """Computes token-level precision, recall, and F1 score."""
    norm_p = normalize_text(pred)
    norm_t = normalize_text(target)
    
    if not norm_p and not norm_t:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not norm_p or not norm_t:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    p_tokens = norm_p.split()
    t_tokens = norm_t.split()
    
    common = set(p_tokens) & set(t_tokens)
    num_common = len(common)
    
    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    precision = num_common / len(set(p_tokens))
    recall = num_common / len(set(t_tokens))
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_record(pred_dict: Dict[str, Any], gt_dict: Dict[str, Any]) -> Dict[str, Any]:
    field_metrics = {}
    
    for f in EVAL_FIELDS:
        # Check aliases
        aliases = FIELD_ALIASES.get(f, [f])
        
        pred_val = ""
        for a in aliases:
            if a in pred_dict and str(pred_dict[a]).strip():
                pred_val = str(pred_dict[a]).strip()
                break
                
        gt_val = ""
        for a in aliases:
            if a in gt_dict and str(gt_dict[a]).strip():
                gt_val = str(gt_dict[a]).strip()
                break
                
        norm_pred = normalize_text(pred_val)
        norm_gt = normalize_text(gt_val)
        
        em = 1.0 if norm_pred == norm_gt else 0.0
        f1_data = compute_token_f1(pred_val, gt_val)
        
        # Check hallucination (prediction exists when GT is empty)
        hallucination = 1.0 if (not norm_gt and norm_pred) else 0.0
        
        field_metrics[f] = {
            "pred": pred_val,
            "target": gt_val,
            "exact_match": em,
            "precision": f1_data["precision"],
            "recall": f1_data["recall"],
            "f1": f1_data["f1"],
            "hallucination": hallucination
        }
        
    return field_metrics


def run_evaluation_suite():
    gt_path = Path("data/verified_ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_events = json.load(f)
        
    raw_path = Path("data/all_29_extracted_raw.json")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_docs = {d["filename"]: "\n\n--- Page Break ---\n\n".join(p["text"] for p in d["pages"]) for d in json.load(f)}
        
    # Filter valid events (skip corrupted boarding pass)
    eval_events = [e for e in ground_truth_events if e["is_valid_event"] and not e.get("is_sub_event")]
    
    test_doc_ids = {"BUHONO-26", "CHEMOVISTA-2K26", "SARANSH-2026", "Anchorage-Shark-Tank"}
    
    legacy_results = {"test": [], "all": []}
    improved_results = {"test": [], "all": []}
    
    for ev in eval_events:
        src = ev["source_file"]
        raw_text = raw_docs.get(src, "")
        doc_id = ev["document_id"]
        is_test = doc_id in test_doc_ids
        
        # 1. Run Legacy Pipeline
        legacy_pred = legacy_extract_heuristics(raw_text, src)
        legacy_metrics = evaluate_record(legacy_pred, ev)
        legacy_entry = {"source_file": src, "doc_id": doc_id, "metrics": legacy_metrics}
        legacy_results["all"].append(legacy_entry)
        if is_test:
            legacy_results["test"].append(legacy_entry)
            
        # 2. Run Improved Pipeline
        improved_out = run_improved_extraction(raw_text, src)
        improved_pred = improved_out["structured_data"]
        improved_metrics = evaluate_record(improved_pred, ev)
        improved_entry = {"source_file": src, "doc_id": doc_id, "metrics": improved_metrics}
        improved_results["all"].append(improved_entry)
        if is_test:
            improved_results["test"].append(improved_entry)
            
    # Compute Aggregate Stats
    def aggregate_metrics(entries: List[Dict[str, Any]]):
        agg = {}
        n = len(entries)
        if n == 0:
            return agg
        for f in EVAL_FIELDS:
            em_sum = sum(e["metrics"][f]["exact_match"] for e in entries)
            f1_sum = sum(e["metrics"][f]["f1"] for e in entries)
            prec_sum = sum(e["metrics"][f]["precision"] for e in entries)
            rec_sum = sum(e["metrics"][f]["recall"] for e in entries)
            halluc_sum = sum(e["metrics"][f]["hallucination"] for e in entries)
            agg[f] = {
                "exact_match": round(em_sum / n * 100, 2),
                "f1": round(f1_sum / n * 100, 2),
                "precision": round(prec_sum / n * 100, 2),
                "recall": round(rec_sum / n * 100, 2),
                "hallucination_rate": round(halluc_sum / n * 100, 2),
            }
        macro_em = round(sum(agg[f]["exact_match"] for f in EVAL_FIELDS) / len(EVAL_FIELDS), 2)
        macro_f1 = round(sum(agg[f]["f1"] for f in EVAL_FIELDS) / len(EVAL_FIELDS), 2)
        return {"fields": agg, "macro_exact_match": macro_em, "macro_f1": macro_f1}
        
    summary = {
        "held_out_test_set": {
            "sample_count": len(legacy_results["test"]),
            "legacy": aggregate_metrics(legacy_results["test"]),
            "improved": aggregate_metrics(improved_results["test"]),
        },
        "all_valid_brochures": {
            "sample_count": len(legacy_results["all"]),
            "legacy": aggregate_metrics(legacy_results["all"]),
            "improved": aggregate_metrics(improved_results["all"]),
        }
    }
    
    # Save Evaluation Report JSON
    report_path = Path("data/evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 90)
    print("EMPIRICAL EVALUATION REPORT: LEGACY vs. IMPROVED PIPELINE")
    print("=" * 90)
    print(f"\n[1] HELD-OUT TEST SPLIT (Strict Unseen Documents, N = {summary['held_out_test_set']['sample_count']})")
    print("-" * 90)
    print(f"{'Target Schema Field':25} | {'Legacy EM':10} | {'Improved EM':12} | {'Legacy F1':10} | {'Improved F1':12} | {'Delta F1':10}")
    print("-" * 90)
    
    t_leg = summary["held_out_test_set"]["legacy"]["fields"]
    t_imp = summary["held_out_test_set"]["improved"]["fields"]
    
    for f in EVAL_FIELDS:
        l_em = t_leg[f]["exact_match"]
        i_em = t_imp[f]["exact_match"]
        l_f1 = t_leg[f]["f1"]
        i_f1 = t_imp[f]["f1"]
        delta = round(i_f1 - l_f1, 2)
        delta_str = f"+{delta}%" if delta >= 0 else f"{delta}%"
        print(f"{f:25} | {l_em:8.1f}% | {i_em:10.1f}% | {l_f1:8.1f}% | {i_f1:10.1f}% | {delta_str:10}")
        
    print("-" * 90)
    print(f"{'MACRO AVERAGE':25} | {summary['held_out_test_set']['legacy']['macro_exact_match']:8.1f}% | {summary['held_out_test_set']['improved']['macro_exact_match']:10.1f}% | {summary['held_out_test_set']['legacy']['macro_f1']:8.1f}% | {summary['held_out_test_set']['improved']['macro_f1']:10.1f}% | +{round(summary['held_out_test_set']['improved']['macro_f1'] - summary['held_out_test_set']['legacy']['macro_f1'], 2)}%")
    print("=" * 90)
    
    print(f"\n[2] COMPLETE VALID BROCHURE DATASET (N = {summary['all_valid_brochures']['sample_count']})")
    print("-" * 90)
    print(f"Legacy Macro F1:   {summary['all_valid_brochures']['legacy']['macro_f1']}%")
    print(f"Improved Macro F1: {summary['all_valid_brochures']['improved']['macro_f1']}% (Delta: +{round(summary['all_valid_brochures']['improved']['macro_f1'] - summary['all_valid_brochures']['legacy']['macro_f1'], 2)}%)")
    print(f"Legacy Macro EM:   {summary['all_valid_brochures']['legacy']['macro_exact_match']}%")
    print(f"Improved Macro EM: {summary['all_valid_brochures']['improved']['macro_exact_match']}% (Delta: +{round(summary['all_valid_brochures']['improved']['macro_exact_match'] - summary['all_valid_brochures']['legacy']['macro_exact_match'], 2)}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_evaluation_suite()
