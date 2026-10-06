"""
FestNest AI AutoFill — Phase 3: Bootstrap Labeling
Calls the Gemini API (using GEMINI_API_KEY from .env) to convert extracted OCR text
into ground-truth JSON schema pairs for LoRA fine-tuning.
Saves pairs in data/labeled/<stem>.json as {"ocr_text": ..., "json_output": {...}}
Respects free-tier rate limits with retry and delay.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap_label")

OCR_OUTPUT_DIR = Path("data/ocr_output")
LABELED_DIR = Path("data/labeled")

VALID_EVENT_TYPES = [
    "Hackathon",
    "Cultural Fest",
    "Workshop",
    "Competition",
    "Tech Talk",
    "Sports",
    "Mega Fest",
]

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

SYSTEM_PROMPT = f"""You are an event information extraction assistant for FestNest.
Given the following raw OCR text extracted from an event poster or brochure, extract all details into a structured JSON object matching the exact schema.

Schema specification:
{{
  "event_name": "Exact official name of the event",
  "event_type": "Must be exactly one of: {', '.join(VALID_EVENT_TYPES)}",
  "date": "Date or date range of the event",
  "time": "Timing of the event if available",
  "venue": "Campus venue or college location or 'Online'",
  "organizer": "Organizing club, department, or college",
  "prize_pool": "Prize money, cash pool, or awards mentioned",
  "registration_deadline": "Last registration date if mentioned, else empty string",
  "contact": "Contact phone numbers or emails of coordinators",
  "description": "Concise 1-3 sentence summary of what this event is about"
}}

Rules:
1. Return ONLY the JSON object. Do not wrap in markdown quotes, backticks, or other text.
2. If a specific detail cannot be determined from the text, use an empty string "" for that field.
3. `event_type` MUST be strictly one of: {', '.join(VALID_EVENT_TYPES)}. Select the best matching type.
"""

# Ordered candidate models for API call
MODEL_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]


def call_gemini_api(api_key: str, ocr_text: str, max_retries: int = 4) -> Dict[str, Any]:
    """
    Calls the Gemini REST API with rate-limit retries and returns parsed JSON.
    """
    user_prompt = f"{SYSTEM_PROMPT}\n\nOCR Text:\n\"\"\"\n{ocr_text}\n\"\"\""

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    last_error = None

    for model_name in MODEL_CANDIDATES:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=25)

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError("Gemini returned empty candidate list.")
                    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(raw_text)
                    return parsed

                elif resp.status_code in (429, 503):
                    # Rate limit or high demand - back off and retry
                    wait_time = 6 * attempt
                    logger.warning(
                        f"Model {model_name} returned {resp.status_code}. Retrying in {wait_time}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    continue

                elif resp.status_code == 404:
                    # Model not available, try next model candidate
                    logger.debug(f"Model {model_name} not found (404). Trying next candidate.")
                    last_error = f"404: {resp.text}"
                    break

                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    logger.warning(f"Error calling {model_name}: {last_error}")
                    break

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error calling {model_name}: {e}. Retrying...")
                time.sleep(4 * attempt)
                last_error = str(e)

    raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_error}")


def sanitize_extracted_schema(extracted: Dict[str, Any]) -> Dict[str, str]:
    """
    Ensures all 10 schema fields exist, values are strings, and event_type is valid.
    """
    sanitized = {}
    for field in TARGET_SCHEMA_FIELDS:
        val = extracted.get(field, "")
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        sanitized[field] = str(val or "").strip()

    # Normalize event_type if needed
    current_type = sanitized["event_type"]
    if current_type not in VALID_EVENT_TYPES:
        matched = False
        for valid in VALID_EVENT_TYPES:
            if valid.lower() in current_type.lower():
                sanitized["event_type"] = valid
                matched = True
                break
        if not matched:
            sanitized["event_type"] = "Competition"

    return sanitized


def main():
    parser = argparse.ArgumentParser(description="FestNest AI AutoFill — Phase 3 Bootstrap Labeling")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of OCR files to process")
    parser.add_argument("--delay", type=float, default=6.0, help="Delay in seconds between requests (~10 req/min)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already labeled files")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env. Please set it before running Phase 3.")
        sys.exit(1)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    ocr_files = sorted([f for f in OCR_OUTPUT_DIR.glob("*.json") if f.name != ".gitkeep"])
    if not ocr_files:
        logger.warning(f"No OCR json files found in {OCR_OUTPUT_DIR}. Run Phase 2 first.")
        return

    if args.limit:
        ocr_files = ocr_files[:args.limit]

    logger.info(f"Starting bootstrap labeling for {len(ocr_files)} OCR output file(s)...")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, ocr_file in enumerate(ocr_files, start=1):
        target_path = LABELED_DIR / ocr_file.name

        if target_path.exists() and not args.overwrite:
            logger.info(f"[{idx}/{len(ocr_files)}] Skipping (already labeled): {ocr_file.name}")
            skipped += 1
            succeeded += 1
            continue

        logger.info(f"[{idx}/{len(ocr_files)}] Labeling: {ocr_file.name}")

        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            ocr_text = ocr_data.get("ocr_text", "").strip()
            if not ocr_text:
                logger.warning(f"  -> Skipping {ocr_file.name}: empty ocr_text")
                failed += 1
                continue

            extracted = call_gemini_api(api_key, ocr_text)
            json_output = sanitize_extracted_schema(extracted)

            labeled_record = {
                "source_file": ocr_file.name,
                "ocr_text": ocr_text,
                "json_output": json_output,
            }

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(labeled_record, f, indent=2, ensure_ascii=False)

            logger.info(
                f"  -> Successfully labeled {ocr_file.name}: "
                f"event_name='{json_output['event_name']}', type='{json_output['event_type']}'"
            )
            succeeded += 1

            # Respect free tier rate limits (~10 requests/min)
            time.sleep(args.delay)

        except Exception as e:
            logger.error(f"  -> Failed to label {ocr_file.name}: {e}")
            failed += 1
            continue

    logger.info("=" * 60)
    logger.info(
        f"Phase 3 Complete: {succeeded} succeeded ({skipped} cached), {failed} failed."
    )
    logger.info(f"Labeled dataset saved in: {LABELED_DIR.resolve()}")


if __name__ == "__main__":
    main()
