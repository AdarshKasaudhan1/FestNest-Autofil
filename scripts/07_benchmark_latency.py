"""
scripts/07_benchmark_latency.py
Runs the end-to-end FestNest AutoFill pipeline on all PDFs in data/raw_posters/
under BENCHMARK_MODE=1 against a warm server.
Measures per-stage timings:
  - upload_save
  - text_extract
  - ocr
  - heuristics
  - llm
  - total
Computes p50, p95, mean, min, max, and writes results to data/latency_report.json.
"""

import os
import sys
import json
import time
import statistics
import requests
from pathlib import Path
from typing import Dict, Any, List

# Ensure safe console output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")
RAW_POSTERS_DIR = Path("data/raw_posters")
OUTPUT_REPORT_PATH = Path("data/latency_report.json")


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Waits until the server is reachable and reports ready via /api/health."""
    print(f"Checking server availability and readiness at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"{url}/api/health", timeout=2)
            if res.status_code == 200:
                data = res.json()
                if data.get("ready"):
                    print(f"Server is UP and ready! Active backend: {data.get('active_backend')}")
                    return True
                else:
                    print(f"Server is warming up... (backend: {data.get('active_backend')})")
            elif res.status_code == 503:
                data = res.json()
                print(f"Server is warming up (503)... backend: {data.get('active_backend')}")
        except Exception:
            pass
        time.sleep(1)
    return False


def warmup_server(url: str, sample_pdf: Path) -> None:
    """Warms up server caches and models with one request."""
    print(f"Warming up server using {sample_pdf.name}...")
    try:
        payload = {"sample": sample_pdf.name}
        res = requests.post(
            f"{url}/api/autofill",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if res.status_code == 200:
            print("Server warmup completed successfully.")
        else:
            print(f"Warmup returned status {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"Warmup warning: {e}")


def run_benchmark() -> Dict[str, Any]:
    if not wait_for_server(SERVER_URL):
        print(f"Error: Server at {SERVER_URL} not reachable. Start app.py first!")
        sys.exit(1)

    all_pdfs = sorted(list(RAW_POSTERS_DIR.glob("*.pdf")))
    if not all_pdfs:
        print(f"Error: No PDFs found in {RAW_POSTERS_DIR.resolve()}")
        sys.exit(1)

    print(f"Discovered {len(all_pdfs)} PDFs in {RAW_POSTERS_DIR.resolve()}")

    # Warmup on the first PDF
    warmup_server(SERVER_URL, all_pdfs[0])

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 105)
    print(f"{'Filename':<36} | {'Type':<8} | {'Extract':<8} | {'OCR':<8} | {'Phase A':<8} | {'LLM':<8} | {'Total':<8}")
    print("-" * 105)

    for pdf_file in all_pdfs:
        filename = pdf_file.name
        safe_filename = filename.encode("ascii", errors="replace").decode("ascii")
        file_size_kb = round(pdf_file.stat().st_size / 1024, 1)

        t_client_start = time.time()
        try:
            payload = {"sample": filename}
            resp = requests.post(
                f"{SERVER_URL}/api/autofill?sync=1",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=45,
            )
            client_elapsed = round(time.time() - t_client_start, 4)

            if resp.status_code == 200:
                data = resp.json()
                timings = data.get("timings", {})
                upload_t = timings.get("upload_save", 0.0)
                text_extract_t = timings.get("text_extract", 0.0)
                ocr_t = timings.get("ocr", 0.0)
                heuristics_t = timings.get("heuristics", 0.0)
                phase_a_t = timings.get("phase_a", round(text_extract_t + ocr_t + heuristics_t, 4))
                llm_t = timings.get("llm", 0.0)
                total_t = timings.get("total", client_elapsed)

                doc_type = "Digital" if ocr_t == 0.0 else "Scanned"

                record = {
                    "filename": filename,
                    "file_size_kb": file_size_kb,
                    "status": "success",
                    "doc_type": doc_type,
                    "timings": {
                        "upload_save": upload_t,
                        "text_extract": text_extract_t,
                        "ocr": ocr_t,
                        "heuristics": heuristics_t,
                        "phase_a": phase_a_t,
                        "llm": llm_t,
                        "total": total_t,
                        "client_roundtrip": client_elapsed,
                    },
                    "structured_data": data.get("structured_data", {}),
                }
                results.append(record)

                print(
                    f"{safe_filename[:36]:<36} | {doc_type:<8} | "
                    f"{text_extract_t:<8.3f} | {ocr_t:<8.3f} | "
                    f"{phase_a_t:<8.3f} | {llm_t:<8.3f} | {total_t:<8.3f}"
                )
            else:
                print(f"{safe_filename[:36]:<36} | ERROR    | Status: {resp.status_code}")
                results.append({
                    "filename": filename,
                    "file_size_kb": file_size_kb,
                    "status": "error",
                    "status_code": resp.status_code,
                    "error": resp.text[:200],
                })
        except Exception as e:
            client_elapsed = round(time.time() - t_client_start, 4)
            print(f"{safe_filename[:38]:<38} | TIMEOUT  | Error: {str(e)[:40]}")
            results.append({
                "filename": filename,
                "file_size_kb": file_size_kb,
                "status": "exception",
                "error": str(e),
                "client_roundtrip": client_elapsed,
            })

    print("=" * 95 + "\n")

    # Aggregate statistics
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        print("No successful responses to compute stats.")
        return {"results": results}

    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        s_vals = sorted(values)
        idx_p95 = int(len(s_vals) * 0.95)
        p95 = s_vals[min(idx_p95, len(s_vals) - 1)]
        return {
            "mean": round(statistics.mean(values), 4),
            "p50": round(statistics.median(values), 4),
            "p95": round(p95, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    all_totals = [r["timings"]["total"] for r in successful]
    all_extract = [r["timings"]["text_extract"] for r in successful]
    all_ocr = [r["timings"]["ocr"] for r in successful]
    all_heur = [r["timings"]["heuristics"] for r in successful]
    all_phase_a = [r["timings"].get("phase_a", r["timings"]["text_extract"] + r["timings"]["ocr"] + r["timings"]["heuristics"]) for r in successful]
    all_llm = [r["timings"]["llm"] for r in successful]

    digital_records = [r for r in successful if r["doc_type"] == "Digital"]
    scanned_records = [r for r in successful if r["doc_type"] == "Scanned"]

    digital_totals = [r["timings"]["total"] for r in digital_records]
    digital_phase_a = [r["timings"].get("phase_a", r["timings"]["text_extract"] + r["timings"]["heuristics"]) for r in digital_records]
    scanned_totals = [r["timings"]["total"] for r in scanned_records]

    stats_summary = {
        "sample_count": len(results),
        "success_count": len(successful),
        "digital_count": len(digital_records),
        "scanned_count": len(scanned_records),
        "overall_total": calc_stats(all_totals),
        "phase_a_overall": calc_stats(all_phase_a),
        "phase_a_digital": calc_stats(digital_phase_a),
        "text_extract": calc_stats(all_extract),
        "ocr": calc_stats(all_ocr),
        "heuristics": calc_stats(all_heur),
        "llm": calc_stats(all_llm),
        "digital_total": calc_stats(digital_totals),
        "scanned_total": calc_stats(scanned_totals),
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_mode": True,
        "summary": stats_summary,
        "items": results,
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("--- BENCHMARK LATENCY SUMMARY (BENCHMARK_MODE=1) ---")
    print(f"Total Tested: {len(results)} | Success: {len(successful)} (Digital: {len(digital_records)}, Scanned: {len(scanned_records)})")
    print(f"OVERALL TOTAL   : p50 = {stats_summary['overall_total']['p50']}s | p95 = {stats_summary['overall_total']['p95']}s | mean = {stats_summary['overall_total']['mean']}s")
    if digital_totals:
        print(f"DIGITAL Phase A : p50 = {stats_summary['phase_a_digital']['p50']}s | p95 = {stats_summary['phase_a_digital']['p95']}s | mean = {stats_summary['phase_a_digital']['mean']}s")
        print(f"DIGITAL Total   : p50 = {stats_summary['digital_total']['p50']}s | p95 = {stats_summary['digital_total']['p95']}s | mean = {stats_summary['digital_total']['mean']}s")
    if scanned_totals:
        print(f"SCANNED Total   : p50 = {stats_summary['scanned_total']['p50']}s | p95 = {stats_summary['scanned_total']['p95']}s | mean = {stats_summary['scanned_total']['mean']}s")
    print(f"STAGES (p50): Extract = {stats_summary['text_extract']['p50']}s | OCR = {stats_summary['ocr']['p50']}s | Heuristics = {stats_summary['heuristics']['p50']}s | LLM = {stats_summary['llm']['p50']}s")
    print(f"\nFull report written to: {OUTPUT_REPORT_PATH.resolve()}")

    return report


if __name__ == "__main__":
    run_benchmark()
