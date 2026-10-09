"""
FestNest AI AutoFill — Phase 7: End-to-End Inference Demo
Takes a poster file (image or PDF), runs the full pipeline:
  1. OCR Extraction (PaddleOCR with robust PDF rasterization)
  2. Information Extraction via Fine-Tuned Qwen2.5-1.5B-Instruct (LoRA)
  3. Structured 10-Field JSON generation matching FestNest's HostEvent schema.
"""

import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Paddle & Torch environment variables for stability on Windows
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "modelscope"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Import torch before Paddle to prevent DLL collision
import torch
import numpy as np
from PIL import Image
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
logger = logging.getLogger("inference_demo")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = Path("models/qwen2.5-1.5b-festnest-lora")

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


def rasterize_pdf_to_images(pdf_path: Path, max_pages: int = 3) -> List[Image.Image]:
    """
    Renders PDF pages into PIL Images up to max_pages.
    Uses pypdfium2 first, falling back to pdf2image if needed.
    """
    images = []
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        pages_to_render = min(len(pdf), max_pages)
        for i in range(pages_to_render):
            page = pdf[i]
            img = page.render(scale=1.5).to_pil()
            images.append(img)
            page.close()
        pdf.close()
        logger.info(f"Rendered {len(images)} page(s) from PDF using pypdfium2.")
        return images
    except Exception as e:
        logger.debug(f"pypdfium2 failed: {e}. Trying pdf2image...")
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=max_pages)
            logger.info(f"Rendered {len(images)} page(s) from PDF using pdf2image.")
            return images
        except Exception as e2:
            raise RuntimeError(f"Could not rasterize PDF {pdf_path.name}: {e2}")


def extract_ocr_from_image(ocr_engine, pil_image: Image.Image) -> str:
    """
    Runs PaddleOCR on a PIL image and returns concatenated text lines.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_np = np.array(pil_image)

    predictions = list(ocr_engine.predict(img_np))
    lines = []
    for pred in predictions:
        if isinstance(pred, dict) or hasattr(pred, "get"):
            texts = pred.get("rec_texts") if "rec_texts" in pred else pred.get("rec_text")
            if texts is not None:
                if isinstance(texts, (list, tuple)):
                    for t in texts:
                        if t and isinstance(t, str) and t.strip():
                            lines.append(t.strip())
                elif isinstance(texts, str) and texts.strip():
                    lines.append(texts.strip())
        elif isinstance(pred, (list, tuple)):
            for item in pred:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    text_info = item[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                        lines.append(str(text_info[0]).strip())
                    elif isinstance(text_info, str):
                        lines.append(text_info.strip())
    return "\n".join(lines)


def run_ocr(file_path: Path) -> str:
    """
    Extracts OCR text from an image or PDF poster file.
    """
    logger.info("Initializing PaddleOCR engine...")
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(
        lang="en",
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
    )

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        images = rasterize_pdf_to_images(file_path)
        page_texts = []
        for i, img in enumerate(images, start=1):
            logger.info(f"Running OCR on page {i}/{len(images)}...")
            txt = extract_ocr_from_image(ocr_engine, img)
            if txt:
                page_texts.append(txt)
        full_text = "\n\n--- Page Break ---\n\n".join(page_texts)
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        logger.info(f"Opening poster image: {file_path.name}")
        img = Image.open(file_path)
        full_text = extract_ocr_from_image(ocr_engine, img)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: PDF, PNG, JPG, JPEG, WEBP.")

    logger.info(f"OCR extraction complete: {len(full_text.splitlines())} lines extracted.")
    return full_text


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON object from model response string.
    """
    text = text.strip()
    if not text:
        return None

    # Direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("event", obj)
    except Exception:
        pass

    # Markdown code fence ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj.get("event", obj)
        except Exception:
            pass

    # Outermost curly brackets {...}
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


def run_model_inference(
    ocr_text: str,
    base_model_name: str,
    adapter_path: Path,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Loads Qwen base model + FestNest LoRA adapter and generates structured JSON.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        cpu_threads = os.cpu_count() or 4
        torch.set_num_threads(cpu_threads)

    logger.info(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading base model {base_model_name} on {device}...")
    dtype = torch.float16 if has_cuda else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.config.use_cache = True

    logger.info(f"Attaching LoRA fine-tuned adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract event details as JSON from this OCR text:\n\n{ocr_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    logger.info("Generating structured event JSON with fine-tuned LoRA model...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen_tokens = output_ids[0][inputs.input_ids.shape[1] :]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    logger.info(f"Raw Model Generation completed ({len(gen_tokens)} tokens).")

    # Parse and structure according to FestNest schema
    parsed = extract_json_from_text(raw_response)
    if parsed is None:
        logger.warning("Could not directly parse valid JSON from output. Building fallback dictionary.")
        parsed = {}

    structured_result = {}
    for f in TARGET_SCHEMA_FIELDS:
        val = parsed.get(f, "")
        if val is None:
            val = ""
        structured_result[f] = str(val).strip() if not isinstance(val, (dict, list)) else val

    # Validate event_type if possible
    if structured_result.get("event_type") not in ALLOWED_EVENT_TYPES:
        for allowed in ALLOWED_EVENT_TYPES:
            if allowed.lower() in str(structured_result.get("event_type", "")).lower():
                structured_result["event_type"] = allowed
                break

    return structured_result, raw_response


def main():
    parser = argparse.ArgumentParser(
        description="FestNest AI AutoFill: End-to-End Inference Demo (Poster -> OCR -> Fine-Tuned Model -> JSON)"
    )
    parser.add_argument(
        "poster",
        nargs="?",
        type=str,
        default=None,
        help="Path to poster image (PNG, JPG) or PDF file. If omitted, uses sample poster.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=str(DEFAULT_ADAPTER_PATH),
        help=f"Path to LoRA adapter (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model identifier (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--ocr_file",
        type=str,
        default=None,
        help="Optional path to existing OCR JSON/text to skip PaddleOCR step.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path to save extracted JSON.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Max generated new tokens (default: 200)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎪 FESTNEST AI AUTOFILL — END-TO-END INFERENCE DEMO")
    print("=" * 70 + "\n")

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"LoRA adapter directory not found at: {adapter_path}")
        logger.error("Please run Phase 5 fine-tuning first.")
        sys.exit(1)

    # Resolve input source
    ocr_text = ""
    poster_path = None

    if args.ocr_file:
        ocr_path = Path(args.ocr_file)
        if not ocr_path.exists():
            logger.error(f"Provided OCR file not found: {ocr_path}")
            sys.exit(1)
        logger.info(f"Loading existing OCR from: {ocr_path}")
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            ocr_text = data.get("text", "") if isinstance(data, dict) else str(data)

    elif args.poster:
        poster_path = Path(args.poster)
        if not poster_path.exists():
            logger.error(f"Poster file not found: {poster_path}")
            sys.exit(1)
    else:
        # Pick default available sample from data/raw_posters
        raw_posters_dir = Path("data/raw_posters")
        candidates = list(raw_posters_dir.glob("*.pdf")) + list(raw_posters_dir.glob("*.png")) + list(raw_posters_dir.glob("*.jpg"))
        if candidates:
            poster_path = candidates[0]
            logger.info(f"No poster specified. Using sample poster: {poster_path.name}")
        else:
            logger.error("No poster file provided and no samples found in data/raw_posters/.")
            sys.exit(1)

    # Step 1: Run OCR if not already provided
    if not ocr_text and poster_path:
        print(f"📄 Step 1: Running OCR on '{poster_path.name}'...")
        ocr_text = run_ocr(poster_path)

    if not ocr_text.strip():
        logger.error("OCR extracted 0 characters. Cannot proceed with model inference.")
        sys.exit(1)

    print("\n" + "-" * 70)
    print("📋 RAW OCR EXTRACTED TEXT (First 300 chars):")
    print("-" * 70)
    print(ocr_text[:300] + ("..." if len(ocr_text) > 300 else ""))
    print("-" * 70 + "\n")

    # Step 2: Fine-Tuned Model Inference
    print("🧠 Step 2: Running Fine-Tuned Qwen2.5-1.5B LoRA Model...")
    structured_json, raw_response = run_model_inference(
        ocr_text=ocr_text,
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    # Step 3: Display Results
    formatted_json_str = json.dumps(structured_json, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("🎯 FINAL STRUCTURED FESTNEST EVENT JSON (10 Fields):")
    print("=" * 70)
    print(formatted_json_str)
    print("=" * 70 + "\n")

    # Save output if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted_json_str)
        logger.info(f"Saved extracted JSON to {out_path}")

    print("✅ Pipeline executed successfully! Ready for FestNest event hosting form auto-fill.\n")


if __name__ == "__main__":
    main()
