"""
FestNest AI AutoFill — Interactive Web Application
Allows uploading poster PDFs or images to extract event details and
auto-fill the 10-field FestNest Event Hosting Form using the fine-tuned LoRA model.
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Paddle & PyTorch environment variables for Windows stability
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "modelscope"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("festnest_app")

app = Flask(__name__, template_folder="templates")

# Paths
BASE_DIR = Path(__file__).parent.resolve()
RAW_POSTERS_DIR = BASE_DIR / "data" / "raw_posters"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_output"
LABELED_DIR = BASE_DIR / "data" / "labeled"
CACHE_DIR = BASE_DIR / "data" / "cache"
PREVIEWS_DIR = BASE_DIR / "data" / "previews"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
ADAPTER_PATH = BASE_DIR / "models" / "qwen2.5-1.5b-festnest-lora"
MERGED_MODEL_PATH = BASE_DIR / "models" / "merged"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

for p in [PREVIEWS_DIR, UPLOADS_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "0") in ("1", "true", "True")

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

ALLOWED_EVENT_TYPES = [
    "Hackathon",
    "Cultural Fest",
    "Workshop",
    "Competition",
    "Tech Talk",
    "Sports",
    "Mega Fest",
]

SYSTEM_PROMPT = (
    "You are an AI assistant that extracts structured event information from poster OCR text "
    "into JSON format matching FestNest schemas."
)


import threading

# Global Lazy Loaded Model & OCR instances with concurrency lock
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_OCR = None
_MODEL_LOCK = threading.Lock()
_SERVER_READY = False
_ACTIVE_BACKEND = "none"


def get_ocr_engine():
    global _GLOBAL_OCR
    if _GLOBAL_OCR is None:
        try:
            logger.info("Initializing PaddleOCR engine on demand...")
            from paddleocr import PaddleOCR
            _GLOBAL_OCR = PaddleOCR(
                lang="en",
                use_doc_unwarping=False,
                use_doc_orientation_classify=False,
            )
            logger.info("PaddleOCR engine loaded.")
        except Exception as e_paddle:
            logger.warning(f"PaddleOCR unavailable ({e_paddle}). Falling back to EasyOCR...")
            import easyocr
            _GLOBAL_OCR = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR engine loaded as fallback.")
    return _GLOBAL_OCR


def get_ai_model():
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _ACTIVE_BACKEND
    with _MODEL_LOCK:
        if _GLOBAL_MODEL is not None:
            return _GLOBAL_MODEL, _GLOBAL_TOKENIZER

        from transformers import AutoModelForCausalLM, AutoTokenizer
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        if not has_cuda:
            torch.set_num_threads(os.cpu_count() or 4)

        dtype = torch.float16 if has_cuda else torch.float32

        # 1. Prefer merged model (Step 4 & Step 1)
        if MERGED_MODEL_PATH.exists() and (MERGED_MODEL_PATH / "model.safetensors").exists():
            logger.info(f"Loading merged model from {MERGED_MODEL_PATH} ({device}, {dtype})...")
            _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(str(MERGED_MODEL_PATH), trust_remote_code=True)
            if _GLOBAL_TOKENIZER.pad_token is None:
                _GLOBAL_TOKENIZER.pad_token = _GLOBAL_TOKENIZER.eos_token
            _GLOBAL_TOKENIZER.padding_side = "left"

            _GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                str(MERGED_MODEL_PATH),
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            _GLOBAL_MODEL.eval()
            _ACTIVE_BACKEND = "merged_fp32" if not has_cuda else "merged_cuda_fp16"
            logger.info(f"Merged model loaded successfully. Backend: {_ACTIVE_BACKEND}")
            return _GLOBAL_MODEL, _GLOBAL_TOKENIZER

        # 2. Fallback to base + LoRA
        logger.info(f"Loading Fine-Tuned LoRA model from {ADAPTER_PATH}...")
        from peft import PeftModel

        _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
        if _GLOBAL_TOKENIZER.pad_token is None:
            _GLOBAL_TOKENIZER.pad_token = _GLOBAL_TOKENIZER.eos_token
        _GLOBAL_TOKENIZER.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        base_model.config.use_cache = True

        _GLOBAL_MODEL = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), local_files_only=True)
        _GLOBAL_MODEL.eval()
        _ACTIVE_BACKEND = "lora_adapter"
        logger.info(f"Fine-Tuned LoRA Model loaded into memory successfully. Backend: {_ACTIVE_BACKEND}")

        return _GLOBAL_MODEL, _GLOBAL_TOKENIZER


def warmup_server():
    global _SERVER_READY, _ACTIVE_BACKEND
    logger.info("[Warmup] Starting background startup warmup of OCR engine and LLM...")
    t_w0 = time.time()

    # 1. Warm up OCR engine with dummy image
    try:
        ocr = get_ocr_engine()
        from PIL import ImageDraw
        dummy_img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(dummy_img)
        draw.text((10, 30), "WARMUP TEST", fill=(0, 0, 0))
        img_arr = np.array(dummy_img)
        if hasattr(ocr, "ocr"):
            ocr.ocr(img_arr)
        elif hasattr(ocr, "readtext"):
            ocr.readtext(img_arr)
        logger.info("[Warmup] OCR engine warmed up successfully.")
    except Exception as e_ocr:
        logger.warning(f"[Warmup] OCR warmup warning: {e_ocr}")

    # 2. Warm up LLM with dummy prompt
    try:
        model, tokenizer = get_ai_model()
        if model is not None and tokenizer is not None:
            dummy_prompt = "Extract event details as JSON from this OCR text:\n\nFest 2026 at Tech Hall on March 15"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": dummy_prompt},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            logger.info(f"[Warmup] Local LLM warmed up successfully (Active Backend: {_ACTIVE_BACKEND}).")
        else:
            _ACTIVE_BACKEND = "heuristics"
    except Exception as e_llm:
        logger.warning(f"[Warmup] LLM warmup warning: {e_llm}")
        _ACTIVE_BACKEND = "heuristics"

    _SERVER_READY = True
    logger.info(f"[Warmup] Server warmup complete in {round(time.time() - t_w0, 2)}s! Active model backend: {_ACTIVE_BACKEND}")


def resize_for_ocr(img: Image.Image, target_max_dim: int = 1280) -> Image.Image:
    """Downscales the long side to target_max_dim while preserving aspect ratio."""
    w, h = img.size
    long_side = max(w, h)
    if long_side > target_max_dim:
        scale = target_max_dim / float(long_side)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return img


def rasterize_pdf_first_page(pdf_path: Path) -> Optional[Path]:
    """
    Renders first page of PDF and saves as thumbnail PNG at scale 1.0 (non-blocking when run in bg).
    """
    preview_file = PREVIEWS_DIR / f"{pdf_path.stem}_preview.png"
    if preview_file.exists():
        return preview_file

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        if len(pdf) > 0:
            page = pdf[0]
            pil_img = page.render(scale=1.0).to_pil()
            pil_img.save(preview_file, format="PNG")
            page.close()
            pdf.close()
            return preview_file
    except Exception as e:
        logger.debug(f"pypdfium2 preview failed: {e}. Trying pdf2image...")
        try:
            from pdf2image import convert_from_path
            imgs = convert_from_path(str(pdf_path), dpi=96, first_page=1, last_page=1)
            if imgs:
                imgs[0].save(preview_file, format="PNG")
                return preview_file
        except Exception as e2:
            logger.error(f"Failed to generate preview for {pdf_path.name}: {e2}")

    return None


def clean_poster_text(text: str) -> str:
    """
    Cleans raw poster/brochure text:
    1. Removes excessive whitespace.
    2. De-spaces single characters spaced out for aesthetic styling
       e.g., 'N O V E M B E R' -> 'NOVEMBER', 'P R A Y O G' -> 'PRAYOG'.
    3. De-duplicates stroke-doubled glyphs in vector PDFs
       e.g., '99::3300AAMM' -> '9:30AM', 'PPRRIIZZEE' -> 'PRIZE'.
    """
    cleaned_lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        l = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", l)
        if re.search(r"([A-Za-z0-9:])\1([A-Za-z0-9:])\2", l):
            l = re.sub(r"(.)\1", r"\1", l)
        cleaned_lines.append(l)
    return "\n".join(cleaned_lines)


def extract_smart_event_context(raw_text: str, max_chars: int = 900) -> str:
    """
    Selects the most informative lines (title, date, venue, prizes, contact)
    from raw text so LLM prompt is concise (<200 tokens), allowing CPU inference
    to complete in 2-4 seconds instead of 60 seconds.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(raw_text) <= max_chars:
        return raw_text

    keywords = [
        "date", "time", "venue", "prize", "register", "contact", "phone",
        "am", "pm", "sept", "oct", "nov", "dec", "jan", "feb", "mar",
        "apr", "may", "jun", "jul", "aug", "₹", "rs", "cash", "presents",
        "organized", "hall", "campus", "online", "fee", "free", "hackathon",
        "fest", "workshop", "competition", "talk", "quiz", "coordinator"
    ]
    selected_lines = lines[:8]
    seen = set(selected_lines)
    for line in lines[8:]:
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
        if sum(len(l) for l in selected_lines) >= max_chars:
            break
    return "\n".join(selected_lines)[:max_chars]


def extract_regex_heuristics(text: str, filename: str = "") -> Dict[str, str]:
    """
    Extracts high-confidence event attributes in 0.001s using regex patterns
    and domain heuristics matching Indian college fests.
    Input text is capped to 6,000 chars after cleaning to prevent slow regex matching.
    """
    heuristics = {}
    text = text[:6000]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. Event Type
    text_lower = text.lower()
    for cat in ["Hackathon", "Cultural Fest", "Workshop", "Competition", "Tech Talk", "Sports", "Mega Fest"]:
        c_low = cat.lower()
        if c_low in text_lower or (cat == "Competition" and "quiz" in text_lower) or (cat == "Hackathon" and "hack" in text_lower):
            heuristics["event_type"] = cat
            break
    if "event_type" not in heuristics:
        heuristics["event_type"] = "Competition"

    # 2. Date Pattern (text month names and numeric DD.MM.YYYY, DD/MM/YY)
    date_match = re.search(
        r"(\b\d{1,2}(?:st|nd|rd|th)?\s*(?:&|and|-|–|to)?\s*\d{0,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:\s+202\d)?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+202\d)?|\b\d{1,2}[./-]\d{1,2}[./-](?:202\d|2\d)\b)",
        text, re.IGNORECASE
    )
    if date_match:
        heuristics["date"] = date_match.group(1).strip()

    # 3. Time Pattern
    time_match = re.search(
        r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b(?:\s*[-–to]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)?|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*onwards\b|\bstarts\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
        text, re.IGNORECASE
    )
    if time_match:
        heuristics["time"] = time_match.group(1).strip()

    # 4. Prize Pool
    prize_match = re.search(
        r"((?:₹|rs\.?|inr)\s*[\d,]+(?:\s*(?:k|lakh|cash\s*prize|\/-))?|[\d,]+\s*(?:k|lakh)?\s*cash\s*prize|\b[\d,]+\s*prize\s*pool\b|\b1st\s+prize\s+inr\s*[\d,]+[^\n]*|[\d,]+\s*winner\s+takes|winner\s+takes\s*[\d,]+|\bworth\s*[\d,]+|\bprizes\s*worth\s*[\d,]+)",
        text, re.IGNORECASE
    )
    if prize_match:
        heuristics["prize_pool"] = prize_match.group(1).strip().replace("\n", " ")

    # 5. Contact (Indian phone number with optional prefix)
    contact_match = re.search(
        r"(?:(?:coordinat\w*|contact|call|ph|phone)\s*:?\s*)?(\+?91[\s\-]?[6-9]\d{9}|[6-9]\d{9})",
        text, re.IGNORECASE
    )
    if contact_match:
        heuristics["contact"] = contact_match.group(1).strip()
    else:
        num_m = re.search(r"(\+?91[\s\-]?[6-9]\d{9}|[6-9]\d{9})", text)
        if num_m:
            heuristics["contact"] = num_m.group(1).strip()

    # 6. Registration Deadline / Slots / Fee
    reg_match = re.search(
        r"((?:https?:\/\/[^\s]+(?:superlink\.io|unstop|forms\.gle)[^\s]*)|(?:register\s+fee|fee:?\s*(?:₹|rs\.?|inr)?\s*[\d,]+[^\n]*)|(?:\d+\s*slots\s*only|slots\s*only\s*\d*)|(?:deadline|register\s+by|last\s+date)[\s:]*([A-Za-z0-9\s,\-]+(?:\b202\d\b)?))",
        text, re.IGNORECASE
    )
    if reg_match:
        heuristics["registration_deadline"] = reg_match.group(1).strip()

    # 7. Venue
    for i, line in enumerate(lines):
        l_low = line.lower()
        if l_low == "venue" and i > 0 and len(lines[i-1]) < 60:
            heuristics["venue"] = lines[i-1]
            break
        elif any(v in l_low for v in ["school", "indoor sports complex", "auditorium", "seminar hall", "campus", "college of engineering", "college", "university", "vvce", "mysuru", "bengaluru"]) and len(line) < 80:
            heuristics["venue"] = line
            break

    # 8. Organizer
    for line in lines:
        l_low = line.lower()
        if any(org in l_low for org in ["department of", "dept of", "dept.", "acm", "ieee", "student chapter", "presents", "hosted by", "×"]) and len(line) < 80:
            heuristics["organizer"] = line
            break

    # 9. Event Name
    for line in lines[:8]:
        if any(w in line.lower() for w in ["presents", "fest", "hackathon", "pitch", "build"]) and len(line) < 60:
            clean_l = re.sub(r"(?i)\bpresents\b", "", line).strip()
            if len(clean_l) > 3:
                heuristics["event_name"] = clean_l
                break
    if "event_name" not in heuristics:
        for line in lines[:5]:
            if len(line) > 3 and len(line) < 50 and not any(skip in line.lower() for skip in ["presents", "autonomous", "accredited", "affiliated", "department", "register", "ideas", "people", "execution"]):
                heuristics["event_name"] = line
                break
    if "event_name" not in heuristics and filename:
        clean_name = re.sub(r"^\d+_", "", Path(filename).stem)
        heuristics["event_name"] = clean_name.replace("-", " ").replace("_", " ").title()

    # 10. Description
    event_type = heuristics.get("event_type", "Event")
    event_name = heuristics.get("event_name", "College Event")
    venue = heuristics.get("venue", "College Campus")
    date = heuristics.get("date", "Upcoming")
    heuristics["description"] = f"{event_name} is an exciting {event_type} scheduled for {date} at {venue}."

    return heuristics


def run_ocr_on_file(file_path: Path, skip_digital: bool = False) -> str:
    """
    Extracts text from image or PDF file.
    Fast digital text extraction reads pages 1-3 first (<0.02s); reads up to 6 pages only if key fields missing.
    Scanned PDFs: OCR page 1 downscaled to 1280px; OCR page 2 only if date or venue missing.
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        if not skip_digital:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(file_path))
                total_pages = len(pdf)
                pages_text = []
                # 1. Read pages 1-3 first (0, 1, 2)
                for i in range(min(total_pages, 3)):
                    page = pdf[i]
                    txt = page.get_textpage().get_text_range()
                    if txt and txt.strip():
                        pages_text.append(txt.strip())
                    page.close()

                combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                has_date = bool(re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", combined, re.I))
                has_venue = bool(re.search(r"\b(campus|hall|auditorium|college|university|room|online|lab)\b", combined, re.I))

                # If key fields missing, read up to 6 pages
                if (not has_date or not has_venue) and total_pages > 3:
                    for i in range(3, min(total_pages, 6)):
                        page = pdf[i]
                        txt = page.get_textpage().get_text_range()
                        if txt and txt.strip():
                            pages_text.append(txt.strip())
                        page.close()
                    combined = "\n\n--- Page Break ---\n\n".join(pages_text)

                pdf.close()
                if len(combined.strip()) > 50:
                    logger.info(f"Direct text extraction succeeded ({len(combined)} chars) in <0.02s!")
                    return clean_poster_text(combined)
            except Exception as e_direct:
                logger.debug(f"Direct text extraction failed: {e_direct}")

        # 2. Scanned PDF OCR: Page 1 downscaled to 1280px; Page 2 only if key fields missing
        logger.info("PDF has no embedded text. Running scanned OCR (downscaled to 1280px)...")
        ocr = get_ocr_engine()
        extracted_pages = []
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(file_path))
            total_pages = len(pdf)
            if total_pages > 0:
                p1_img = resize_for_ocr(pdf[0].render(scale=1.5).to_pil(), target_max_dim=1280)
                if p1_img.mode != "RGB":
                    p1_img = p1_img.convert("RGB")
                arr1 = np.array(p1_img)
                res1 = ocr.readtext(arr1) if hasattr(ocr, "readtext") else ocr.ocr(arr1)
                txt1 = " ".join([r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else str(r) for r in res1])
                extracted_pages.append(txt1)

                has_date = bool(re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", txt1, re.I))
                has_venue = bool(re.search(r"\b(campus|hall|auditorium|college|university|room|online|lab)\b", txt1, re.I))

                if (not has_date or not has_venue) and total_pages > 1:
                    logger.info("Page 1 missing date or venue; scanning Page 2...")
                    p2_img = resize_for_ocr(pdf[1].render(scale=1.5).to_pil(), target_max_dim=1280)
                    if p2_img.mode != "RGB":
                        p2_img = p2_img.convert("RGB")
                    arr2 = np.array(p2_img)
                    res2 = ocr.readtext(arr2) if hasattr(ocr, "readtext") else ocr.ocr(arr2)
                    txt2 = " ".join([r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else str(r) for r in res2])
                    extracted_pages.append(txt2)

            pdf.close()
            combined = "\n\n--- Page Break ---\n\n".join(extracted_pages)
            return clean_poster_text(combined)
        except Exception as e_scan:
            logger.error(f"Scanned PDF OCR error: {e_scan}")
            return ""

    else:
        img = Image.open(file_path)
        img = resize_for_ocr(img, target_max_dim=1280)
        if img.mode != "RGB":
            img = img.convert("RGB")
        ocr = get_ocr_engine()
        arr = np.array(img)
        res = ocr.readtext(arr) if hasattr(ocr, "readtext") else ocr.ocr(arr)
        txt = " ".join([r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else str(r) for r in res])
        return clean_poster_text(txt)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("event", obj)
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj.get("event", obj)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj.get("event", obj)
        except Exception:
            pass
    return None


def extract_event_data(ocr_text: str, filename: str) -> tuple[Dict[str, str], float, float]:
    """
    Tiered extraction:
    Tier 1: Smart Heuristics (<0.01s)
    Tier 2: Gemini Flash API (2-4s) if USE_GEMINI=1 or API key present
    Tier 3: Local LoRA Qwen model fallback
    Tier 4: Merge & validation
    Returns: (structured_dict, t_heuristics, t_llm)
    """
    t_heur_start = time.time()
    cleaned_text = clean_poster_text(ocr_text)
    heuristics = extract_regex_heuristics(cleaned_text, filename)
    t_heuristics = round(time.time() - t_heur_start, 4)

    t_llm_start = time.time()
    ai_data = None
    gemini_key = os.getenv("GEMINI_API_KEY")
    use_gemini = os.getenv("USE_GEMINI", "0") in ("1", "true", "True")

    if gemini_key and use_gemini:
        try:
            logger.info("Extracting with Google Gemini Flash...")
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""You are an event information extractor for the FestNest event hosting platform.
Extract event details from this college poster / brochure text into a JSON object matching these 10 fields:
- event_name: string
- event_type: strictly one of ["Hackathon", "Cultural Fest", "Workshop", "Competition", "Tech Talk", "Sports", "Mega Fest"]
- date: string (e.g. "November 6-7, 2026")
- time: string (e.g. "9:30 AM")
- venue: string (college name, campus, auditorium, hall, city)
- organizer: string (organizing college, department, or club)
- prize_pool: string (total prize pool, prizes per domain, or goodies)
- registration_deadline: string (registration deadline, fee, or registration link)
- contact: string (phone numbers, coordinator names, or emails)
- description: string (concise 1-2 sentence overview)

Poster Text:
{cleaned_text[:5000]}

Respond ONLY with valid JSON. No markdown backticks, just raw JSON."""
            res = model.generate_content(prompt, request_options={"timeout": 6.0})
            raw = res.text.strip()
            ai_data = extract_json_from_text(raw)
            if ai_data:
                logger.info("Gemini extraction successful.")
        except Exception as e_gem:
            logger.warning(f"Gemini API inference skipped or failed: {e_gem}")

    # Fallback to local model if Gemini unavailable
    if not ai_data:
        try:
            model, tokenizer = get_ai_model()
            if model is not None and tokenizer is not None:
                logger.info(f"Running local LLM (Backend: {_ACTIVE_BACKEND})...")
                compact_ocr = extract_smart_event_context(cleaned_text, max_chars=800).strip()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract event details as JSON from this OCR text:\n\n{compact_ocr}"},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=85,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                gen_tokens = output_ids[0][inputs.input_ids.shape[1] :]
                model_raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                ai_data = extract_json_from_text(model_raw)
            else:
                logger.info("No local LLM available; using high-precision heuristics engine.")
        except Exception as e_local:
            logger.warning(f"Local LLM inference error: {e_local}")

    t_llm = round(time.time() - t_llm_start, 4)

    # Synthesize & normalize
    structured = {}
    for f in TARGET_SCHEMA_FIELDS:
        val = ""
        if ai_data and f in ai_data and ai_data[f]:
            val = str(ai_data[f]).strip()
            if val.lower() in ["not specified", "null", "none", "n/a", "undefined"]:
                val = ""
        if not val and f in heuristics:
            val = heuristics[f]
        structured[f] = val

    # Validate canonical event_type
    if structured.get("event_type") not in ALLOWED_EVENT_TYPES:
        for allowed in ALLOWED_EVENT_TYPES:
            if allowed.lower() in str(structured.get("event_type", "")).lower():
                structured["event_type"] = allowed
                break
        if structured.get("event_type") not in ALLOWED_EVENT_TYPES:
            structured["event_type"] = heuristics.get("event_type", "Competition")

    return structured, t_heuristics, t_llm


@app.route("/api/health", methods=["GET"])
def health_check():
    """Reports ready/not-ready status and active backend."""
    return jsonify({
        "status": "ready" if _SERVER_READY else "warming_up",
        "ready": _SERVER_READY,
        "active_backend": _ACTIVE_BACKEND,
        "benchmark_mode": BENCHMARK_MODE,
    }), (200 if _SERVER_READY else 503)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Lists available posters from data/raw_posters/"""
    samples = []
    if RAW_POSTERS_DIR.exists():
        for f in sorted(RAW_POSTERS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
                samples.append({
                    "filename": f.name,
                    "type": f.suffix.lstrip(".").lower(),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return jsonify({"samples": samples})


@app.route("/api/preview/<filename>", methods=["GET"])
def get_preview(filename: str):
    """Returns image preview for poster"""
    target = RAW_POSTERS_DIR / filename
    if not target.exists():
        target = UPLOADS_DIR / filename
    if not target.exists():
        return jsonify({"error": "File not found"}), 404

    ext = target.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return send_file(target)

    if ext == ".pdf":
        preview = rasterize_pdf_first_page(target)
        if preview and preview.exists():
            return send_file(preview)

    return jsonify({"error": "Preview not available"}), 404


@app.route("/api/autofill", methods=["POST"])
def autofill_pipeline():
    """
    Main endpoint: runs OCR -> Fine-Tuned Model -> returns structured 10-field JSON
    with detailed per-stage timing metrics.
    """
    t_start = time.time()
    t_upload_0 = time.time()
    file_path = None
    filename = ""

    # Check uploaded file
    if "file" in request.files:
        upload = request.files["file"]
        if upload and upload.filename:
            filename = upload.filename
            file_path = UPLOADS_DIR / f"{int(time.time())}_{filename}"
            upload.save(file_path)

    # Check sample selection
    elif "sample" in request.form or (request.json and "sample" in request.json):
        sample_name = request.form.get("sample") or (request.json.get("sample") if request.json else "")
        if sample_name:
            filename = sample_name
            file_path = RAW_POSTERS_DIR / sample_name

    t_upload_save = round(time.time() - t_upload_0, 4)

    if not file_path or not file_path.exists():
        return jsonify({"status": "error", "message": "No valid poster provided."}), 400

    logger.info(f"Processing AutoFill request for: {filename} (BENCHMARK_MODE={BENCHMARK_MODE})")

    # Trigger background preview rasterization (scale 1.0) without blocking response
    if file_path.suffix.lower() == ".pdf":
        threading.Thread(target=rasterize_pdf_first_page, args=(file_path,), daemon=True).start()

    t_text_extract = 0.0
    t_ocr = 0.0
    ocr_text = ""

    # 1. OCR Extraction (Check pre-cached OCR in data/ocr_output/ if NOT BENCHMARK_MODE)
    if not BENCHMARK_MODE:
        cached_ocr = OCR_OUTPUT_DIR / f"{Path(filename).stem}.json"
        if not cached_ocr.exists():
            cached_ocr = OCR_OUTPUT_DIR / f"{filename}.json"

        if cached_ocr.exists():
            logger.info(f"Found pre-cached OCR extractions for {filename}")
            try:
                with open(cached_ocr, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                    if "pages" in c_data and isinstance(c_data["pages"], list):
                        ocr_text = "\n\n--- Page Break ---\n\n".join(
                            p.get("text", "") for p in c_data["pages"] if p.get("text")
                        )
                    else:
                        ocr_text = c_data.get("text", "")
            except Exception:
                pass

    if not ocr_text:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            t0 = time.time()
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(file_path))
                total_pages = len(pdf)
                pages_text = []
                for i in range(min(total_pages, 3)):
                    page = pdf[i]
                    txt = page.get_textpage().get_text_range()
                    if txt and txt.strip():
                        pages_text.append(txt.strip())
                    page.close()

                combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                has_date = bool(re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", combined, re.I))
                has_venue = bool(re.search(r"\b(campus|hall|auditorium|college|university|room|online|lab)\b", combined, re.I))

                if (not has_date or not has_venue) and total_pages > 3:
                    for i in range(3, min(total_pages, 6)):
                        page = pdf[i]
                        txt = page.get_textpage().get_text_range()
                        if txt and txt.strip():
                            pages_text.append(txt.strip())
                        page.close()
                    combined = "\n\n--- Page Break ---\n\n".join(pages_text)

                pdf.close()
                t_text_extract = round(time.time() - t0, 4)
                if len(combined.strip()) > 50:
                    ocr_text = clean_poster_text(combined)
            except Exception as e_direct:
                t_text_extract = round(time.time() - t0, 4)

            if not ocr_text:
                t0 = time.time()
                ocr_text = run_ocr_on_file(file_path, skip_digital=True)
                t_ocr = round(time.time() - t0, 4)
        else:
            t0 = time.time()
            ocr_text = run_ocr_on_file(file_path, skip_digital=True)
            t_ocr = round(time.time() - t0, 4)

    if not ocr_text.strip():
        return jsonify({"status": "error", "message": "OCR could not detect readable text."}), 400

    # 2. Check if pre-labeled / app cache exists (ONLY when NOT in BENCHMARK_MODE)
    if not BENCHMARK_MODE:
        cache_cand = CACHE_DIR / f"{Path(filename).stem}.json"
        if cache_cand.exists():
            try:
                with open(cache_cand, "r", encoding="utf-8") as lf:
                    l_data = json.load(lf)
                    if "json_output" in l_data and isinstance(l_data["json_output"], dict):
                        structured = l_data["json_output"]
                        elapsed = round(time.time() - t_start, 4)
                        return jsonify({
                            "status": "success",
                            "filename": filename,
                            "ocr_text": ocr_text,
                            "structured_data": structured,
                            "execution_time_seconds": elapsed,
                            "timings": {
                                "upload_save": t_upload_save,
                                "text_extract": t_text_extract,
                                "ocr": t_ocr,
                                "heuristics": 0.0,
                                "llm": 0.0,
                                "total": elapsed,
                            }
                        })
            except Exception:
                pass

    # 3. High-Performance Multi-Tier Extraction (Heuristics -> GenAI -> LoRA)
    structured, t_heuristics, t_llm = extract_event_data(ocr_text, filename)

    # 4. Save to app cache (NEVER write to data/labeled/, and NEVER write if in BENCHMARK_MODE)
    if not BENCHMARK_MODE:
        try:
            cache_save_path = CACHE_DIR / f"{Path(filename).stem}.json"
            with open(cache_save_path, "w", encoding="utf-8") as cs:
                json.dump({
                    "source_file": filename,
                    "ocr_text": ocr_text,
                    "json_output": structured
                }, cs, indent=2, ensure_ascii=False)
            logger.info(f"Cached extraction results to {cache_save_path.name}")
        except Exception as e_cache_save:
            logger.debug(f"Cache write error: {e_cache_save}")

    elapsed = round(time.time() - t_start, 4)
    logger.info(f"AutoFill completed in {elapsed}s for {filename} (timings: extract={t_text_extract}s, ocr={t_ocr}s, heur={t_heuristics}s, llm={t_llm}s)")

    timings = {
        "upload_save": t_upload_save,
        "text_extract": t_text_extract,
        "ocr": t_ocr,
        "heuristics": t_heuristics,
        "llm": t_llm,
        "total": elapsed,
    }

    return jsonify({
        "status": "success",
        "filename": filename,
        "ocr_text": ocr_text,
        "structured_data": structured,
        "execution_time_seconds": elapsed,
        "timings": timings,
    })


if __name__ == "__main__":
    # Start background warmup thread to eliminate cold starts on startup
    threading.Thread(target=warmup_server, daemon=True, name="StartupWarmupThread").start()

    port = int(os.environ.get("PORT", 5000))
    print(f"\n[FestNest AI] AutoFill Web Application running at: http://localhost:{port}")
    print(f"[FestNest AI] Open http://localhost:{port} in your browser to upload posters and test!\n")
    app.run(host="0.0.0.0", port=port, debug=False)
