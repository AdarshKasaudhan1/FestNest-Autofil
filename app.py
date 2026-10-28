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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

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

# Paths
BASE_DIR = Path(__file__).parent.resolve()
RAW_POSTERS_DIR = BASE_DIR / "data" / "raw_posters"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_output"
LABELED_DIR = BASE_DIR / "data" / "labeled"
ADAPTER_PATH = BASE_DIR / "models" / "qwen2.5-1.5b-festnest-lora"
MERGED_MODEL_PATH = BASE_DIR / "models" / "merged"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Detect Vercel / AWS Lambda / Serverless read-only environment
IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
if IS_VERCEL:
    CACHE_DIR = Path("/tmp/data/cache")
    PREVIEWS_DIR = Path("/tmp/data/previews")
    UPLOADS_DIR = Path("/tmp/data/uploads")
else:
    CACHE_DIR = BASE_DIR / "data" / "cache"
    PREVIEWS_DIR = BASE_DIR / "data" / "previews"
    UPLOADS_DIR = BASE_DIR / "data" / "uploads"

for p in [PREVIEWS_DIR, UPLOADS_DIR, CACHE_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "0") in ("1", "true", "True")
AUTOFILL_JOBS = {}
import uuid

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
_OCR_ENGINE_NAME = "none"
_MODEL_LOCK = threading.Lock()
_PDFIUM_LOCK = threading.Lock()
_SERVER_READY = False
_ACTIVE_BACKEND = "none"


def get_ocr_engine():
    global _GLOBAL_OCR, _OCR_ENGINE_NAME
    if _GLOBAL_OCR is not None:
        return _GLOBAL_OCR

    # 1. Primary engine: RapidOCR (onnxruntime, PP-OCR mobile models; no paddlepaddle needed)
    try:
        from rapidocr_onnxruntime import RapidOCR
        logger.info("Initializing RapidOCR engine...")
        _GLOBAL_OCR = RapidOCR()
        _OCR_ENGINE_NAME = "rapidocr"
        logger.info("RapidOCR engine loaded successfully.")
        return _GLOBAL_OCR
    except Exception as e_rapid:
        logger.warning(f"RapidOCR unavailable ({e_rapid}). Trying PaddleOCR fallback...")

    # 2. Optional Fallback: PaddleOCR
    try:
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR engine fallback...")
        _GLOBAL_OCR = PaddleOCR(
            lang="en",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
        )
        _OCR_ENGINE_NAME = "paddleocr"
        logger.info("PaddleOCR engine loaded.")
        return _GLOBAL_OCR
    except Exception as e_paddle:
        logger.warning(f"PaddleOCR fallback unavailable: {e_paddle}")

    _OCR_ENGINE_NAME = "none"
    return None


def run_ocr_on_image_array(ocr, arr: np.ndarray) -> str:
    """Invokes the active OCR engine on an image numpy array (RGB)."""
    if ocr is None:
        return ""
    try:
        # RapidOCR callable interface
        if hasattr(ocr, "__call__") and type(ocr).__name__ == "RapidOCR":
            res, _ = ocr(arr)
            if res:
                return " ".join([line[1] for line in res if len(line) > 1 and line[1]])
            return ""
        # PaddleOCR interface
        elif hasattr(ocr, "ocr"):
            res = ocr.ocr(arr)
            if res and res[0]:
                return " ".join([line[1][0] for line in res[0] if len(line) > 1 and isinstance(line[1], (list, tuple))])
            return ""
        # EasyOCR interface
        elif hasattr(ocr, "readtext"):
            res = ocr.readtext(arr)
            if res:
                return " ".join([r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else str(r) for r in res])
            return ""
    except Exception as e:
        logger.error(f"OCR inference execution error: {e}")
    return ""


def get_ai_model():
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _ACTIVE_BACKEND
    if not HAS_TORCH or torch is None:
        _ACTIVE_BACKEND = "gemini_api" if (os.getenv("USE_GEMINI", "0") in ("1", "true", "True") and os.getenv("GEMINI_API_KEY")) else "heuristics"
        return None, None

    with _MODEL_LOCK:
        if _GLOBAL_MODEL is not None:
            return _GLOBAL_MODEL, _GLOBAL_TOKENIZER

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            _ACTIVE_BACKEND = "heuristics"
            return None, None

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
    if not HAS_TORCH or torch is None:
        _ACTIVE_BACKEND = "gemini_api" if (os.getenv("USE_GEMINI", "0") in ("1", "true", "True") and os.getenv("GEMINI_API_KEY")) else "heuristics"
        _SERVER_READY = True
        return

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


def resize_for_ocr(img: Image.Image, target_max_dim: int = 1600) -> Image.Image:
    """Downscales the long side to target_max_dim while preserving aspect ratio."""
    w, h = img.size
    long_side = max(w, h)
    if long_side > target_max_dim:
        scale = target_max_dim / float(long_side)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return img


def rasterize_pdf_first_page(pdf_path: Path) -> Optional[Path]:
    """
    Renders first page of PDF and saves as thumbnail PNG at ~200 dpi (scale 2.0).
    Uses _PDFIUM_LOCK to guarantee safe thread access.
    """
    preview_file = PREVIEWS_DIR / f"{pdf_path.stem}_preview.png"
    if preview_file.exists() and preview_file.stat().st_size > 0:
        return preview_file

    with _PDFIUM_LOCK:
        if preview_file.exists() and preview_file.stat().st_size > 0:
            return preview_file
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(pdf_path))
            if len(pdf) > 0:
                page = pdf[0]
                pil_img = page.render(scale=2.0).to_pil()
                page.close()
                pdf.close()
                pil_img.save(preview_file, format="PNG")
                logger.info(f"Generated preview for {pdf_path.name} -> {preview_file.name}")
                return preview_file
            pdf.close()
        except Exception as e:
            logger.error(f"Failed to generate preview for {pdf_path.name}: {e}", exc_info=True)

    return None


def clean_poster_text(text: str) -> str:
    """
    Cleans raw poster/brochure text:
    1. Removes excessive whitespace.
    2. De-spaces single characters spaced out for aesthetic styling
       e.g., 'N O V E M B E R' -> 'NOVEMBER', 'P R A Y O G' -> 'PRAYOG'.
    3. De-duplicates stroke-doubled letter glyphs in vector PDFs
       e.g., 'PPRRIIZZEE' -> 'PRIZE', but preserves phone numbers and legitimate repeated digits.
    """
    cleaned_lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        l = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", l)
        # Only de-duplicate if at least 3 consecutive character pairs are doubled, and only letters
        if re.search(r"([A-Za-z])\1([A-Za-z])\2([A-Za-z])\3", l):
            l = re.sub(r"([A-Za-z])\1", r"\1", l)
        cleaned_lines.append(l)
    return "\n".join(cleaned_lines)


def extract_smart_event_context(raw_text: str, max_chars: int = 900) -> str:
    """
    Selects the most informative lines (title, date, venue, prizes, contact)
    from raw text so LLM prompt is concise (<200 tokens), allowing CPU inference
    to complete in 2-4 seconds instead of 60 seconds.
    Filters out boilerplate university accreditation blocks.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(raw_text) <= max_chars:
        return raw_text

    keywords = [
        "date", "time", "venue", "prize", "register", "contact", "phone", "mob",
        "am", "pm", "sept", "oct", "nov", "dec", "jan", "feb", "mar",
        "apr", "may", "jun", "jul", "aug", "₹", "rs", "cash", "presents",
        "organise", "organize", "hall", "campus", "online", "fee", "free", "hackathon",
        "fest", "workshop", "competition", "talk", "quiz", "coordinator", "about "
    ]
    skip_boilerplate = ["accredited", "reaccredited", "affiliated", "vision", "carmelite", "patrons", "advisory committee"]

    selected_lines = []
    # Always include top lines
    for l in lines[:5]:
        if not any(sb in l.lower() for sb in skip_boilerplate):
            selected_lines.append(l)

    seen = set(selected_lines)
    for line in lines[5:]:
        line_lower = line.lower()
        if any(sb in line_lower for sb in skip_boilerplate):
            continue
        if any(kw in line_lower for kw in keywords) or re.search(r"[6-9]\d{4}[\s\-]?\d{5}", line):
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
        if sum(len(l) for l in selected_lines) >= max_chars:
            break
    return "\n".join(selected_lines)[:max_chars]


def extract_regex_heuristics(text: str, filename: str = "") -> Dict[str, str]:
    """
    Extracts high-confidence event attributes using robust domain heuristics,
    multi-tier title detection, strict institutional blacklists, and clean date/prize normalization.
    """
    from scripts.pipeline_improved import run_improved_extraction
    res = run_improved_extraction(text, filename)
    data = res["structured_data"]
    # Ensure all TARGET_SCHEMA_FIELDS are present as non-None strings
    for k in TARGET_SCHEMA_FIELDS:
        if k not in data or data[k] is None:
            data[k] = ""
    return data


def run_ocr_on_file(file_path: Path, skip_digital: bool = False) -> tuple[str, Optional[str]]:
    """
    Extracts text from image or PDF file.
    Fast digital text extraction reads pages 1-3 first (<0.02s); reads up to 6 pages only if key fields missing.
    Scanned PDFs: OCR page 1 rendered at ~200 dpi downscaled to max 1600px; OCR page 2 only if date or venue missing.
    Returns (cleaned_text, error_code).
    """
    ocr = get_ocr_engine()
    if ocr is None:
        return "", "ocr_unavailable"

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        if not skip_digital:
            try:
                with _PDFIUM_LOCK:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(str(file_path))
                    total_pages = len(pdf)
                    pages_text = []
                    # Read all pages for digital PDF (up to 30 pages)
                    for i in range(min(total_pages, 30)):
                        page = pdf[i]
                        txt = page.get_textpage().get_text_range()
                        if txt and txt.strip():
                            pages_text.append(txt.strip())
                        page.close()

                    combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                    pdf.close()

                if len(combined.strip()) > 50:
                    logger.info(f"Direct text extraction succeeded ({len(combined)} chars) in <0.02s!")
                    return clean_poster_text(combined), None
            except Exception as e_direct:
                logger.debug(f"Direct text extraction (pypdfium2) failed: {e_direct}")
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path))
                    pages_text = [p.extract_text() or "" for p in reader.pages[:30] if p.extract_text()]
                    combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                    if len(combined.strip()) > 50:
                        logger.info(f"Direct text extraction (pypdf fallback) succeeded ({len(combined)} chars)!")
                        return clean_poster_text(combined), None
                except Exception as e_pypdf:
                    logger.debug(f"Direct text extraction (pypdf) failed: {e_pypdf}")

        # 2. Scanned PDF OCR: Page 1 rendered at ~200 dpi (scale 2.2), capped at 1600px
        logger.info(f"PDF has no embedded text. Running scanned OCR ({_OCR_ENGINE_NAME}, 1600px max)...")
        extracted_pages = []
        try:
            with _PDFIUM_LOCK:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(file_path))
                total_pages = len(pdf)
                if total_pages == 0:
                    pdf.close()
                    return "", "pdf_render_failed"

                # Render page 1 at ~200 dpi (scale 2.2)
                page1 = pdf[0]
                p1_pil = page1.render(scale=2.2).to_pil()
                page1.close()

                # Also save preview immediately if not already present
                preview_file = PREVIEWS_DIR / f"{file_path.stem}_preview.png"
                if not preview_file.exists() or preview_file.stat().st_size == 0:
                    p1_pil.save(preview_file, format="PNG")

                p1_img = resize_for_ocr(p1_pil, target_max_dim=1600)
                if p1_img.mode != "RGB":
                    p1_img = p1_img.convert("RGB")
                arr1 = np.array(p1_img)

                txt1 = run_ocr_on_image_array(ocr, arr1)
                if txt1.strip():
                    extracted_pages.append(txt1)

                has_date = bool(re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", txt1, re.I))
                has_venue = bool(re.search(r"\b(campus|hall|auditorium|college|university|room|online|lab)\b", txt1, re.I))

                if (not has_date or not has_venue) and total_pages > 1:
                    logger.info("Page 1 missing date or venue; scanning Page 2...")
                    page2 = pdf[1]
                    p2_pil = page2.render(scale=2.2).to_pil()
                    page2.close()
                    p2_img = resize_for_ocr(p2_pil, target_max_dim=1600)
                    if p2_img.mode != "RGB":
                        p2_img = p2_img.convert("RGB")
                    arr2 = np.array(p2_img)
                    txt2 = run_ocr_on_image_array(ocr, arr2)
                    if txt2.strip():
                        extracted_pages.append(txt2)

                pdf.close()

            combined = "\n\n--- Page Break ---\n\n".join(extracted_pages)
            cleaned = clean_poster_text(combined)
            if not cleaned.strip():
                return "", "ocr_empty"
            return cleaned, None

        except Exception as e_scan:
            logger.error(f"Scanned PDF OCR error: {e_scan}", exc_info=True)
            return "", "pdf_render_failed"

    else:
        try:
            img = Image.open(file_path)
            img = resize_for_ocr(img, target_max_dim=1600)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            txt = run_ocr_on_image_array(ocr, arr)
            cleaned = clean_poster_text(txt)
            if not cleaned.strip():
                return "", "ocr_empty"
            return cleaned, None
        except Exception as e_img:
            logger.error(f"Image OCR error: {e_img}", exc_info=True)
            return "", "ocr_empty"


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


def evaluate_field_confidence(data: Dict[str, str], raw_text: str) -> Dict[str, str]:
    """
    Evaluates confidence per field: 'high', 'medium', or 'low'.
    Accurately recognizes when optional fields (e.g. deadline, prize) are omitted in the original poster.
    """
    confidence = {}
    raw_lower = raw_text.lower()
    for f in TARGET_SCHEMA_FIELDS:
        val = str(data.get(f, "")).strip()
        if not val or val.lower() in ["not specified", "null", "none", "n/a", "undefined"]:
            if f == "registration_deadline" and not re.search(r"\b(deadline|last date|register before|closes on|due date)\b", raw_lower):
                confidence[f] = "high"
                continue
            if f == "prize_pool" and not re.search(r"\b(prize|cash|pool|₹|rs\.?|inr|awards)\b", raw_lower):
                confidence[f] = "high"
                continue
            confidence[f] = "low"
            continue

        if f == "date":
            confidence[f] = "high" if re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", val, re.I) else "medium"
        elif f == "time":
            confidence[f] = "high" if re.search(r"\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", val, re.I) else "medium"
        elif f == "contact":
            confidence[f] = "high" if re.search(r"[6-9]\d{9}", val) else "medium"
        elif f == "event_type":
            confidence[f] = "high" if val in ALLOWED_EVENT_TYPES else "medium"
        elif f in ("prize_pool", "registration_deadline"):
            confidence[f] = "high" if any(c in val.lower() for c in ["₹", "rs", "inr", "free", "cash", "pool", "slot", "http", "deadline", "last date", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "202"]) else "medium"
        elif f == "venue":
            confidence[f] = "high" if any(v in val.lower() for v in ["hall", "auditorium", "campus", "room", "online", "college", "complex", "ground", "mysuru", "bengaluru", "delhi", "centre", "school"]) else "medium"
        elif f == "organizer":
            confidence[f] = "high" if any(o in val.lower() for o in ["department", "dept", "club", "association", "acm", "ieee", "society", "council", "presents", "college", "university", "institute", "school"]) else "medium"
        elif f == "event_name":
            confidence[f] = "high" if (3 < len(val) < 45 and not any(g in val.lower() for g in ["brochure", "guidelines", "college of", "department of", "presents"])) else "medium"
        elif f == "description":
            confidence[f] = "high" if len(val.split()) >= 6 else "medium"
        else:
            confidence[f] = "medium"

    return confidence


def build_template_description(data: Dict[str, str]) -> str:
    """
    Builds a clean, single-sentence event description capped at ~25 words.
    Never invents placeholder names or locations.
    """
    name = data.get("event_name")
    if not name or name.lower() in ["event", "college event", "null", "none"]:
        return ""
    etype = data.get("event_type") or "event"
    org = data.get("organizer")
    date = data.get("date")
    venue = data.get("venue")

    parts = [f"{name} is an engaging {etype}"]
    if org:
        parts.append(f"organized by {org}")
    if date:
        parts.append(f"scheduled for {date}")
    if venue:
        parts.append(f"at {venue}")
    desc = " ".join(parts).strip() + "."
    words = desc.split()
    if len(words) > 25:
        desc = " ".join(words[:25]) + "."
    return desc


def run_phase_b_llm_refinement(raw_text: str, current_data: Dict[str, str], fields_to_refine: list) -> tuple[Dict[str, str], float]:
    """
    Refines only missing or low-confidence fields using the local LLM.
    Uses C++ early stop on closing brace '}' (token 92) and JSON repair to guarantee fast output.
    """
    t_start = time.time()
    if not fields_to_refine or not HAS_TORCH or torch is None:
        return current_data, 0.0

    model, tokenizer = get_ai_model()
    if model is None or tokenizer is None:
        return current_data, 0.0

    compact_ocr = extract_smart_event_context(raw_text, max_chars=2000).strip()
    keys_str = ", ".join(fields_to_refine)

    system_prompt = f"You are an AI event extractor. Return valid JSON containing only keys: {keys_str}."
    user_prompt = f"Poster Text:\n{compact_ocr}\n\nExtract JSON [{keys_str}]:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{"
    except Exception:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{{"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Max tokens scaled to number of fields (approx 40 tokens per field, capped at 300)
    max_tokens = min(len(fields_to_refine) * 40 + 60, 300)

    try:
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else None
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
        gen_tokens = output_ids[0][prompt_len:]
        raw_output = "{" + tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # JSON repair step: ensure balanced braces
        if raw_output.count("{") > raw_output.count("}"):
            raw_output += "}"

        refined_json = extract_json_from_text(raw_output)
        if refined_json and isinstance(refined_json, dict):
            for k in fields_to_refine:
                if k in refined_json and str(refined_json[k]).strip():
                    val = str(refined_json[k]).strip()
                    if val.lower() not in ["null", "none", "not specified", "n/a", "undefined"]:
                        old_val = str(current_data.get(k, "")).strip()
                        if not old_val or old_val.lower() in ["not specified", "null", "none", "n/a", "undefined"]:
                            current_data[k] = val
                        elif k in ["event_name", "organizer"]:
                            # Only replace if existing heuristic value is generic or noisy
                            if any(bad in old_val.lower() for bad in ["presents", "brochure", "guidelines", "rules", "college of", "department of"]):
                                current_data[k] = val
    except Exception as e_gen:
        logger.warning(f"Phase B LLM refinement warning: {e_gen}")

    t_elapsed = round(time.time() - t_start, 4)
    return current_data, t_elapsed


def extract_event_data(ocr_text: str, filename: str, run_phase_b: bool = True) -> tuple[Dict[str, str], Dict[str, str], float, float]:
    """
    Hybrid Progressive Extraction:
    Phase A: Smart regex heuristics + template description (<0.01s)
    Phase B: Targeted local LLM refinement on low-confidence/missing fields only
    Returns: (structured_dict, confidence_dict, t_heuristics, t_llm)
    """
    t_heur_start = time.time()
    cleaned_text = clean_poster_text(ocr_text)
    heuristics = extract_regex_heuristics(cleaned_text, filename)
    if not heuristics.get("description"):
        heuristics["description"] = build_template_description(heuristics)

    t_heuristics = round(time.time() - t_heur_start, 4)

    # Evaluate confidence
    confidence = evaluate_field_confidence(heuristics, cleaned_text)
    t_llm = 0.0

    # Optional Gemini path (only if USE_GEMINI=1)
    use_gemini = os.getenv("USE_GEMINI", "0") in ("1", "true", "True")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if use_gemini and gemini_key:
        try:
            t0 = time.time()
            logger.info("Extracting with Google Gemini Flash...")
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Extract event details from this poster text as JSON:\n\n{cleaned_text[:3000]}"
            res = model.generate_content(prompt, request_options={"timeout": 6.0})
            ai_data = extract_json_from_text(res.text.strip())
            if ai_data:
                for k, v in ai_data.items():
                    if k in TARGET_SCHEMA_FIELDS and str(v).strip():
                        heuristics[k] = str(v).strip()
            t_llm = round(time.time() - t0, 4)
        except Exception as e_gem:
            logger.warning(f"Gemini API inference skipped: {e_gem}")

    elif run_phase_b:
        # Phase B: only refine fields that are missing or low/medium confidence
        fields_to_refine = [f for f, c in confidence.items() if c == "low" or (c == "medium" and f in ["event_name", "organizer"])]
        if fields_to_refine:
            logger.info(f"Phase B refining fields: {fields_to_refine}...")
            heuristics, t_llm = run_phase_b_llm_refinement(cleaned_text, heuristics, fields_to_refine)
            # Re-evaluate confidence after refinement
            confidence = evaluate_field_confidence(heuristics, cleaned_text)

    # Validate canonical event_type
    if heuristics.get("event_type") not in ALLOWED_EVENT_TYPES:
        for allowed in ALLOWED_EVENT_TYPES:
            if allowed.lower() in str(heuristics.get("event_type", "")).lower():
                heuristics["event_type"] = allowed
                break
        if heuristics.get("event_type") not in ALLOWED_EVENT_TYPES:
            heuristics["event_type"] = "Competition"

    # Dual schema keys (matching both frontend and technical benchmark)
    heuristics["event_category"] = heuristics.get("event_type", "Competition")
    heuristics["event_dates"] = heuristics.get("date", "")
    heuristics["event_time"] = heuristics.get("time", "")
    heuristics["contact_details"] = heuristics.get("contact", "")
    heuristics["event_description"] = heuristics.get("description", "")
    if "registration_link" not in heuristics:
        heuristics["registration_link"] = ""

    # Enforce description <= 35 words
    desc = heuristics.get("description", "")
    if desc:
        words = desc.split()
        if len(words) > 35:
            heuristics["description"] = " ".join(words[:35]) + "."
            heuristics["event_description"] = heuristics["description"]

    return heuristics, confidence, t_heuristics, t_llm


@app.route("/api/health", methods=["GET"])
def health_check():
    """Reports ready/not-ready status, OCR engine, and active backend."""
    ocr = get_ocr_engine()
    ready = True if IS_VERCEL else (_SERVER_READY and (_OCR_ENGINE_NAME != "none") and (ocr is not None))
    return jsonify({
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "active_backend": _ACTIVE_BACKEND,
        "ocr_engine": _OCR_ENGINE_NAME,
        "benchmark_mode": BENCHMARK_MODE,
        "serverless": IS_VERCEL,
    }), (200 if ready else 503)


@app.route("/")
@app.route("/api")
@app.route("/api/index")
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
    """Returns image preview for poster with robust resolution across uploads and samples."""
    clean_name = Path(filename).name
    stem = Path(clean_name).stem
    unprefixed_stem = re.sub(r"^\d+_", "", stem)

    # 1. Check direct preview cache in data/previews/
    exact_preview = PREVIEWS_DIR / f"{stem}_preview.png"
    if exact_preview.exists() and exact_preview.stat().st_size > 0:
        return send_file(exact_preview)

    unprefixed_preview = PREVIEWS_DIR / f"{unprefixed_stem}_preview.png"
    if unprefixed_preview.exists() and unprefixed_preview.stat().st_size > 0:
        return send_file(unprefixed_preview)

    for p_file in PREVIEWS_DIR.glob(f"*{unprefixed_stem}*.png"):
        if p_file.is_file() and p_file.stat().st_size > 0:
            return send_file(p_file)

    # 2. Locate source file in raw_posters or uploads
    target = None
    if (RAW_POSTERS_DIR / clean_name).exists():
        target = RAW_POSTERS_DIR / clean_name
    elif (UPLOADS_DIR / clean_name).exists():
        target = UPLOADS_DIR / clean_name
    else:
        # Search for timestamped uploaded file
        matches = list(UPLOADS_DIR.glob(f"*{clean_name}*"))
        if not matches:
            matches = list(UPLOADS_DIR.glob(f"*{unprefixed_stem}*"))
        if matches:
            target = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    if not target or not target.exists():
        return jsonify({"error": f"Poster file '{filename}' not found"}), 404

    ext = target.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return send_file(target)

    if ext == ".pdf":
        preview = rasterize_pdf_first_page(target)
        if preview and preview.exists() and preview.stat().st_size > 0:
            return send_file(preview)

    return jsonify({"error": "Preview not available"}), 404


@app.route("/api/autofill/poll/<job_id>", methods=["GET"])
def poll_autofill_job(job_id: str):
    """Returns current status and fields for an async AutoFill job."""
    job = AUTOFILL_JOBS.get(job_id)
    if not job:
        return jsonify({"status": "not_found", "message": "Job ID not found"}), 404
    return jsonify(job)


@app.route("/api/autofill/stream/<job_id>", methods=["GET"])
def stream_autofill_job(job_id: str):
    """Server-Sent Events (SSE) stream for progressive AutoFill."""
    from flask import Response

    def event_stream():
        job = AUTOFILL_JOBS.get(job_id)
        if not job:
            yield f"data: {json.dumps({'event': 'not_found'})}\n\n"
            return

        # Phase A event
        yield f"data: {json.dumps({'event': 'phase_a', 'data': job})}\n\n"

        if job.get("status") == "complete":
            yield f"data: {json.dumps({'event': 'phase_b', 'data': job})}\n\n"
            return

        # Poll until complete or 30s timeout
        start = time.time()
        while time.time() - start < 30:
            time.sleep(0.3)
            cur_job = AUTOFILL_JOBS.get(job_id)
            if cur_job and cur_job.get("status") == "complete":
                yield f"data: {json.dumps({'event': 'phase_b', 'data': cur_job})}\n\n"
                return

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/autofill", methods=["POST"])
def autofill_pipeline():
    """
    Main AutoFill endpoint:
    - Phase A: Instant extraction (<1.5s) via digital text & domain heuristics.
    - Phase B: Targeted LLM refinement on missing/low-confidence fields.
    Supports ?sync=1 for benchmarks and async streaming/polling for web UI.
    Protected by 30-second hard timeout for huge/corrupt PDFs.
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
    error_code = None

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
                with _PDFIUM_LOCK:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(str(file_path))
                    total_pages = len(pdf)
                    pages_text = []
                    # Read all pages for digital PDF (up to 30 pages)
                    for i in range(min(total_pages, 30)):
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
                logger.debug(f"Direct text extraction (pypdfium2) error: {e_direct}")
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path))
                    pages_text = [p.extract_text() or "" for p in reader.pages[:30] if p.extract_text()]
                    combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                    if len(combined.strip()) > 50:
                        ocr_text = clean_poster_text(combined)
                except Exception as e_pypdf:
                    logger.debug(f"Direct text extraction (pypdf) error: {e_pypdf}")

            if not ocr_text:
                t0 = time.time()
                ocr_text, error_code = run_ocr_on_file(file_path, skip_digital=True)
                t_ocr = round(time.time() - t0, 4)
        else:
            t0 = time.time()
            ocr_text, error_code = run_ocr_on_file(file_path, skip_digital=True)
            t_ocr = round(time.time() - t0, 4)

    if not ocr_text.strip():
        # Fallback to Gemini Multimodal Vision if configured
        use_gemini = os.getenv("USE_GEMINI", "0") in ("1", "true", "True")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if use_gemini and gemini_key:
            try:
                logger.info(f"Attempting Gemini Multimodal OCR fallback for {filename}...")
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                img_path = None
                if file_path.suffix.lower() == ".pdf":
                    img_path = rasterize_pdf_first_page(file_path)
                elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    img_path = file_path

                if img_path and img_path.exists():
                    p_img = Image.open(img_path)
                    res = model.generate_content([
                        "Read all text on this event poster. Return all readable text word-for-word, preserving headings, dates, venues, contacts, and fees.",
                        p_img
                    ])
                    if res and res.text:
                        ocr_text = clean_poster_text(res.text)
                        error_code = None
            except Exception as e_gem_ocr:
                logger.warning(f"Gemini Multimodal OCR fallback warning: {e_gem_ocr}")

    if not ocr_text.strip():
        code = error_code or "ocr_empty"
        logger.warning(f"No readable text extracted for {filename}. Code: {code}")
        elapsed = round(time.time() - t_start, 4)
        status_http = 503 if code == "ocr_unavailable" else 422
        return jsonify({
            "status": "error",
            "error_code": code,
            "message": f"Failed: No readable text detected ({code}).",
            "filename": filename,
            "preview_url": f"/api/preview/{filename}",
            "ocr_text": "",
            "structured_data": {},
            "execution_time_seconds": elapsed,
            "timings": {
                "upload_save": t_upload_save,
                "text_extract": t_text_extract,
                "ocr": t_ocr,
                "heuristics": 0.0,
                "phase_a": round(t_text_extract + t_ocr, 4),
                "llm": 0.0,
                "total": elapsed,
            }
        }), status_http

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
                            "phase": "cache",
                            "filename": filename,
                            "preview_url": f"/api/preview/{filename}",
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

    # 3. Progressive Hybrid Extraction (Phase A instant; Phase B refined)
    is_sync = BENCHMARK_MODE or request.args.get("sync") == "1" or request.headers.get("X-Sync-Mode") == "1"

    if is_sync:
        # Check hard timeout deadline (protect against freezing on huge PDFs)
        time_left = 30.0 - (time.time() - t_start)
        should_run_b = time_left > 5.0
        structured, confidence, t_heuristics, t_llm = extract_event_data(ocr_text, filename, run_phase_b=should_run_b)
        elapsed = round(time.time() - t_start, 4)

        timings = {
            "upload_save": t_upload_save,
            "text_extract": t_text_extract,
            "ocr": t_ocr,
            "heuristics": t_heuristics,
            "phase_a": round(t_text_extract + t_ocr + t_heuristics, 4),
            "llm": t_llm,
            "total": elapsed,
        }

        # Cache only when NOT in BENCHMARK_MODE
        if not BENCHMARK_MODE:
            try:
                cache_save_path = CACHE_DIR / f"{Path(filename).stem}.json"
                with open(cache_save_path, "w", encoding="utf-8") as cs:
                    json.dump({
                        "source_file": filename,
                        "ocr_text": ocr_text,
                        "json_output": structured
                    }, cs, indent=2, ensure_ascii=False)
            except Exception:
                pass

        logger.info(f"AutoFill complete in {elapsed}s for {filename} (extract={t_text_extract}s, ocr={t_ocr}s, heur={t_heuristics}s, llm={t_llm}s)")

        return jsonify({
            "status": "success",
            "phase": "full",
            "filename": filename,
            "preview_url": f"/api/preview/{filename}",
            "ocr_text": ocr_text,
            "structured_data": structured,
            "confidence": confidence,
            "execution_time_seconds": elapsed,
            "timings": timings,
        })

    else:
        # Web UI mode: Return complete high-quality extraction immediately
        time_left = 25.0 - (time.time() - t_start)
        should_run_b = time_left > 5.0 and _ACTIVE_BACKEND != "none"
        structured, confidence, t_heuristics, t_llm = extract_event_data(ocr_text, filename, run_phase_b=should_run_b)
        elapsed = round(time.time() - t_start, 4)

        timings = {
            "upload_save": t_upload_save,
            "text_extract": t_text_extract,
            "ocr": t_ocr,
            "heuristics": t_heuristics,
            "phase_a": round(t_text_extract + t_ocr + t_heuristics, 4),
            "llm": t_llm,
            "total": elapsed,
        }

        if not BENCHMARK_MODE:
            try:
                cache_save_path = CACHE_DIR / f"{Path(filename).stem}.json"
                with open(cache_save_path, "w", encoding="utf-8") as cs:
                    json.dump({
                        "source_file": filename,
                        "ocr_text": ocr_text,
                        "json_output": structured
                    }, cs, indent=2, ensure_ascii=False)
            except Exception:
                pass

        logger.info(f"UI AutoFill served in {elapsed}s for {filename}")

        return jsonify({
            "status": "success",
            "phase": "full",
            "filename": filename,
            "preview_url": f"/api/preview/{filename}",
            "ocr_text": ocr_text,
            "structured_data": structured,
            "confidence": confidence,
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
