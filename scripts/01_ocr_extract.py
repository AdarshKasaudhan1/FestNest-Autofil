"""
FestNest AI AutoFill — Phase 2: OCR Extraction
Extracts raw text and bounding box layout from poster images and PDFs.
Uses PaddleOCR with ModelScope hosting and robust PDF fallback.
Saves outputs as structured JSON in data/ocr_output/<filename>.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure modelscope is used as reliable model hoster and disable problematic mkldnn flags
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "modelscope"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# PyTorch must be imported before Paddle on Windows to avoid DLL collision
import torch
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ocr_extract")

RAW_POSTERS_DIR = Path("data/raw_posters")
OCR_OUTPUT_DIR = Path("data/ocr_output")
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def get_pdf_pages_as_images(pdf_path: Path, max_pages: int = 4) -> List[Image.Image]:
    """
    Renders PDF pages into PIL Images up to max_pages.
    Uses pypdfium2 first (self-contained binaries on Windows), falls back to pdf2image.
    """
    images = []
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        num_pages_to_render = min(len(pdf), max_pages)
        for i in range(num_pages_to_render):
            page = pdf[i]
            pil_img = page.render(scale=1.0).to_pil()
            images.append(pil_img)
            page.close()
        pdf.close()
        return images
    except Exception as e_pdfium:
        logger.debug(f"pypdfium2 error for {pdf_path.name}: {e_pdfium}. Trying pdf2image...")
        try:
            from pdf2image import convert_from_path
            return convert_from_path(str(pdf_path), dpi=100, first_page=1, last_page=max_pages)
        except Exception as e_pdf2img:
            raise RuntimeError(f"Failed to rasterize PDF '{pdf_path.name}': {e_pdf2img}")


def init_ocr_engine():
    """Initializes and returns the optimized PaddleOCR instance."""
    logger.info("Initializing PaddleOCR engine (optimized CPU config)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang="en",
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
    )
    logger.info("PaddleOCR engine initialized successfully.")
    return ocr


def process_image_with_ocr(ocr_engine, image: Image.Image) -> Dict[str, Any]:
    """
    Runs OCR on a single PIL Image.
    Returns extracted blocks and concatenated page text.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)

    predictions = list(ocr_engine.predict(img_np))

    blocks = []
    lines_text = []

    for pred in predictions:
        if isinstance(pred, dict) or hasattr(pred, "get"):
            # Safe retrieval without truth-value testing of numpy arrays
            texts = pred.get("rec_texts") if "rec_texts" in pred else pred.get("rec_text")
            scores = pred.get("rec_scores") if "rec_scores" in pred else pred.get("rec_score")
            boxes = pred.get("rec_boxes") if "rec_boxes" in pred else (
                pred.get("rec_polys") if "rec_polys" in pred else pred.get("dt_polys")
            )

            if texts is not None and len(texts) > 0:
                for idx, t in enumerate(texts):
                    t_str = str(t).strip()
                    if not t_str:
                        continue
                    s = float(scores[idx]) if scores is not None and idx < len(scores) else 1.0
                    b = boxes[idx] if boxes is not None and idx < len(boxes) else []
                    b_list = b.tolist() if hasattr(b, "tolist") else list(b)
                    blocks.append({
                        "text": t_str,
                        "confidence": round(s, 4),
                        "box": b_list,
                    })
                    lines_text.append(t_str)
            else:
                text = pred.get("text") if "text" in pred else pred.get("transcription")
                score = pred.get("score") if "score" in pred else pred.get("confidence", 1.0)
                box = pred.get("box") if "box" in pred else pred.get("points", [])
                if text and str(text).strip():
                    b_list = box.tolist() if hasattr(box, "tolist") else list(box)
                    blocks.append({
                        "text": str(text).strip(),
                        "confidence": round(float(score), 4),
                        "box": b_list,
                    })
                    lines_text.append(str(text).strip())

        elif isinstance(pred, (list, tuple)):
            for item in pred:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    box, text_info = item
                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                        t, s = text_info
                        t_str = str(t).strip()
                        if t_str:
                            b_list = box.tolist() if hasattr(box, "tolist") else list(box)
                            blocks.append({
                                "text": t_str,
                                "confidence": round(float(s), 4),
                                "box": b_list,
                            })
                            lines_text.append(t_str)

    return {
        "text": "\n".join(lines_text),
        "blocks": blocks,
    }


def process_single_poster(ocr_engine, file_path: Path, max_pages: int = 4) -> Dict[str, Any]:
    """Loads a poster (PDF or image), runs OCR, and compiles results."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        pages = get_pdf_pages_as_images(file_path, max_pages=max_pages)
    elif ext in SUPPORTED_IMAGE_EXTS:
        pages = [Image.open(file_path)]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if not pages:
        raise ValueError("File contains no readable pages.")

    pages_output = []
    all_texts = []

    for idx, page_img in enumerate(pages, start=1):
        page_res = process_image_with_ocr(ocr_engine, page_img)
        page_res["page_number"] = idx
        pages_output.append(page_res)
        if page_res["text"]:
            all_texts.append(page_res["text"])

    full_text = "\n\n--- Page Break ---\n\n".join(all_texts) if len(pages_output) > 1 else (all_texts[0] if all_texts else "")

    return {
        "source_file": file_path.name,
        "total_pages": len(pages_output),
        "pages": pages_output,
        "ocr_text": full_text,
    }


def main():
    parser = argparse.ArgumentParser(description="FestNest Poster OCR Extraction")
    parser.add_argument("--max-pages", type=int, default=4, help="Max pages per PDF to process (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing OCR outputs")
    args = parser.parse_args()

    RAW_POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect files
    raw_files = [
        f for f in sorted(RAW_POSTERS_DIR.iterdir())
        if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() == ".pdf" or f.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    ]

    if not raw_files:
        logger.warning(f"No posters or PDFs found in {RAW_POSTERS_DIR}. Please add files and run again.")
        return

    if args.limit:
        raw_files = raw_files[:args.limit]

    logger.info(f"Processing {len(raw_files)} poster/brochure file(s) (max {args.max_pages} pages/PDF)")

    # Initialize OCR
    ocr_engine = init_ocr_engine()

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, file_path in enumerate(raw_files, start=1):
        import re
        safe_stem = re.sub(r'[<>:"/\\|?*]', '_', file_path.stem).strip() or f"poster_{idx}"
        output_file = OCR_OUTPUT_DIR / f"{safe_stem}.json"

        if output_file.exists() and not args.overwrite:
            logger.info(f"[{idx}/{len(raw_files)}] Skipping (already exists): {output_file.name}")
            skipped_count += 1
            success_count += 1
            continue

        logger.info(f"[{idx}/{len(raw_files)}] Processing: {file_path.name}")
        try:
            result = process_single_poster(ocr_engine, file_path, max_pages=args.max_pages)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            total_words = sum(len(p.get("blocks", [])) for p in result["pages"])
            logger.info(
                f"  -> Saved {output_file.name} "
                f"({result['total_pages']} page(s), {total_words} text blocks)"
            )
            success_count += 1

        except Exception as e:
            logger.error(f"  -> Error processing {file_path.name}: {e}")
            failed_count += 1
            continue

    logger.info("=" * 60)
    logger.info(f"OCR Extraction Finished: {success_count} succeeded ({skipped_count} cached), {failed_count} failed.")
    logger.info(f"Outputs saved in: {OCR_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
