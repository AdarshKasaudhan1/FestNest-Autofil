"""
scripts/verify_unseen_posters.py

Step 4 Verification:
Tests IMPERIUM 2.0, 5 other image-only PDFs, and 3 digital PDFs not in data/raw_posters.
Reports OCR engine, text length, first 200 chars, per-stage timings, and filled fields.
Computes p50 and p95 total latencies for image-only PDFs.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Configure UTF-8 for Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure BENCHMARK_MODE is enabled
os.environ["BENCHMARK_MODE"] = "1"
os.environ["USE_GEMINI"] = "0"

import app as app_mod
from app import app, warmup_server


def run_verification():
    print("=" * 80)
    print("STEP 4: UNSEEN POSTER VERIFICATION (WARM PIPELINE)")
    print("=" * 80)

    # 1. Warm up server pipeline (OCR + LLM)
    print("\n[Warmup] Pre-warming OCR engine and LLM...")
    t0 = time.time()
    warmup_server()
    print(f"[Warmup] Complete in {round(time.time() - t0, 2)}s. OCR Engine: {app_mod._OCR_ENGINE_NAME}, AI Backend: {app_mod._ACTIVE_BACKEND}")

    client = app.test_client()

    image_only_files = [
        Path("data/raw_posters/IMPERIUM 2.0 - Brochure_.pdf"),
        Path("data/raw_posters/BUHONO'26 (3).pdf"),
        Path("data/raw_posters/CHEMOVISTA-2K26-brochure.pdf"),
        Path("data/raw_posters/IDEATHON-4-0-brochure.pdf"),
        Path("data/raw_posters/INDUX-5-0---2026-brochure.pdf"),
        Path("data/raw_posters/KALARAV-2026-brochure.pdf"),
    ]

    digital_files = [
        Path("data/uploads/1792826477_Agora Brochure'26 FINAL (1).pdf"),
        Path("data/uploads/1791540955_Theervu-athon-2026-brochure.pdf"),
        Path("data/uploads/1791540555_PRAYOG-2026-brochure.pdf"),
    ]

    results = {"image_only": [], "digital": []}

    print("\n" + "=" * 80)
    print("PART 1: IMAGE-ONLY / SCANNED PDFS (6 SAMPLES)")
    print("=" * 80)

    scanned_totals = []
    scanned_ocr_times = []

    for pdf_path in image_only_files:
        if not pdf_path.exists():
            print(f"[SKIP] File not found: {pdf_path}")
            continue

        clean_name = pdf_path.name
        print(f"\n--- Testing Scanned PDF: {clean_name} ---")

        with open(pdf_path, "rb") as f:
            t_req_0 = time.time()
            res = client.post("/api/autofill?sync=1", data={"file": (f, clean_name)})
            req_time = round(time.time() - t_req_0, 4)

        if res.status_code not in (200, 201):
            print(f"[ERROR] HTTP {res.status_code}: {res.get_json()}")
            continue

        data = res.get_json()
        ocr_text = data.get("ocr_text", "")
        timings = data.get("timings", {})
        structured = data.get("structured_data", {})
        engine = app_mod._OCR_ENGINE_NAME if timings.get("ocr", 0) > 0 else "pdfium_direct"

        scanned_totals.append(timings.get("total", req_time))
        scanned_ocr_times.append(timings.get("ocr", 0.0))

        item_res = {
            "filename": clean_name,
            "engine": engine,
            "ocr_length": len(ocr_text),
            "first_200": ocr_text[:200].replace("\n", " "),
            "timings": timings,
            "fields": {k: v for k, v in structured.items() if v and str(v).strip()},
        }
        results["image_only"].append(item_res)

        safe_first_200 = ocr_text[:200].replace(chr(10), ' ').encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"  Engine:            {engine}")
        print(f"  OCR Text Length:   {len(ocr_text)} characters")
        print(f"  First 200 Chars:   {safe_first_200}")
        print(f"  Timings:           upload={timings.get('upload_save')}s, direct_text={timings.get('text_extract')}s, ocr={timings.get('ocr')}s, heur={timings.get('heuristics')}s, llm={timings.get('llm')}s -> TOTAL={timings.get('total')}s")
        print(f"  Filled Fields ({len(item_res['fields'])}/10):")
        for k, v in item_res["fields"].items():
            print(f"    - {k}: {repr(v)}")

        # For IMPERIUM 2.0, also test forced pure scanned OCR path
        if clean_name == "IMPERIUM 2.0 - Brochure_.pdf":
            print(f"\n--- Testing Scanned PDF (Forced Scanned OCR): IMPERIUM 2.0 ---")
            t_ocr_0 = time.time()
            from app import run_ocr_on_file, extract_event_data
            scan_txt, scan_err = run_ocr_on_file(pdf_path, skip_digital=True)
            t_ocr_val = round(time.time() - t_ocr_0, 4)
            t_ext_0 = time.time()
            scan_struct, scan_conf, t_h, t_l = extract_event_data(scan_txt, clean_name, run_phase_b=True)
            total_scan = round(t_ocr_val + (time.time() - t_ext_0), 4)

            scanned_totals.append(total_scan)
            scanned_ocr_times.append(t_ocr_val)

            scan_res = {
                "filename": "IMPERIUM 2.0 (Forced Scanned OCR)",
                "engine": app_mod._OCR_ENGINE_NAME,
                "ocr_length": len(scan_txt),
                "first_200": scan_txt[:200].replace("\n", " "),
                "timings": {
                    "upload_save": 0.0,
                    "text_extract": 0.0,
                    "ocr": t_ocr_val,
                    "heuristics": t_h,
                    "llm": t_l,
                    "total": total_scan,
                },
                "fields": {k: v for k, v in scan_struct.items() if v and str(v).strip()},
            }
            results["image_only"].append(scan_res)
            safe_scan_200 = scan_txt[:200].replace(chr(10), ' ').encode('ascii', errors='backslashreplace').decode('ascii')
            print(f"  Engine:            {app_mod._OCR_ENGINE_NAME}")
            print(f"  OCR Text Length:   {len(scan_txt)} characters")
            print(f"  First 200 Chars:   {safe_scan_200}")
            print(f"  Timings:           ocr={t_ocr_val}s, heur={t_h}s, llm={t_l}s -> TOTAL={total_scan}s")
            print(f"  Filled Fields ({len(scan_res['fields'])}/10):")
            for k, v in scan_res["fields"].items():
                print(f"    - {k}: {repr(v)}")

    print("\n" + "=" * 80)
    print("PART 2: UNSEEN DIGITAL PDFS (NOT IN RAW_POSTERS)")
    print("=" * 80)

    digital_totals = []

    for pdf_path in digital_files:
        if not pdf_path.exists():
            print(f"[SKIP] File not found: {pdf_path}")
            continue

        clean_name = pdf_path.name
        print(f"\n--- Testing Digital PDF: {clean_name} ---")

        with open(pdf_path, "rb") as f:
            t_req_0 = time.time()
            res = client.post("/api/autofill?sync=1", data={"file": (f, clean_name)})
            req_time = round(time.time() - t_req_0, 4)

        if res.status_code not in (200, 201):
            print(f"[ERROR] HTTP {res.status_code}: {res.get_json()}")
            continue

        data = res.get_json()
        ocr_text = data.get("ocr_text", "")
        timings = data.get("timings", {})
        structured = data.get("structured_data", {})

        digital_totals.append(timings.get("total", req_time))

        item_res = {
            "filename": clean_name,
            "engine": "pdfium_direct",
            "ocr_length": len(ocr_text),
            "first_200": ocr_text[:200].replace("\n", " "),
            "timings": timings,
            "fields": {k: v for k, v in structured.items() if v and str(v).strip()},
        }
        results["digital"].append(item_res)

        safe_digital_200 = ocr_text[:200].replace(chr(10), ' ').encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"  Engine:            pdfium_direct")
        print(f"  Text Length:       {len(ocr_text)} characters")
        print(f"  First 200 Chars:   {safe_digital_200}")
        print(f"  Timings:           upload={timings.get('upload_save')}s, direct_text={timings.get('text_extract')}s, ocr={timings.get('ocr')}s, heur={timings.get('heuristics')}s, llm={timings.get('llm')}s -> TOTAL={timings.get('total')}s")
        print(f"  Filled Fields ({len(item_res['fields'])}/10):")
        for k, v in item_res["fields"].items():
            print(f"    - {k}: {repr(v)}")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("MEASURED LATENCY SUMMARY")
    print("=" * 80)

    if scanned_totals:
        p50_scanned = round(float(np.percentile(scanned_totals, 50)), 2)
        p95_scanned = round(float(np.percentile(scanned_totals, 95)), 2)
        p50_ocr = round(float(np.percentile(scanned_ocr_times, 50)), 2)
        p95_ocr = round(float(np.percentile(scanned_ocr_times, 95)), 2)
        print(f"Scanned/Image-Only PDFs (n={len(scanned_totals)}):")
        print(f"  Total Pipeline Latency: p50 = {p50_scanned}s, p95 = {p95_scanned}s (Min: {min(scanned_totals)}s, Max: {max(scanned_totals)}s)")
        print(f"  OCR Stage Only:         p50 = {p50_ocr}s, p95 = {p95_ocr}s")
        if p50_ocr > 5.0 or p50_scanned > 5.0:
            print("  Limiting Stage: OCR (Rasterization + RapidOCR ONNX inference on CPU takes ~2-5s per page).")
    if digital_totals:
        p50_digital = round(float(np.percentile(digital_totals, 50)), 2)
        p95_digital = round(float(np.percentile(digital_totals, 95)), 2)
        print(f"Digital PDFs (n={len(digital_totals)}):")
        print(f"  Total Pipeline Latency: p50 = {p50_digital}s, p95 = {p95_digital}s (Min: {min(digital_totals)}s, Max: {max(digital_totals)}s)")

    # Save to data/step4_verification_report.json
    out_path = Path("data/step4_verification_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    run_verification()
