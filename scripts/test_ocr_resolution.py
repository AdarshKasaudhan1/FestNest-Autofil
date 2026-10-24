"""
scripts/test_ocr_resolution.py
Compares RapidOCR at 1280px vs 1600px long-side cap across 5 scanned PDF posters.
Measures:
  - time (seconds)
  - non-empty extracted character length
  - key fields detected (date, venue, contact, prize, title)
"""

import time
import re
from pathlib import Path
from PIL import Image
import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

SAMPLES = [
    "IMPERIUM 2.0 - Brochure_.pdf",
    "Aicon'26 Brochure Digital File-Compressed (1).pdf",
    "BUHONO'26 (3).pdf",
    "CHEMOVISTA-2K26-brochure.pdf",
    "KALARAV-2026-brochure.pdf",
]

RAW_POSTERS_DIR = Path("data/raw_posters")


def resize_image(img: Image.Image, target_max_dim: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= target_max_dim:
        return img
    scale = target_max_dim / float(long_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def detect_key_fields(text: str) -> list:
    fields = []
    if re.search(r"\b(202\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[./-]\d{1,2})\b", text, re.I):
        fields.append("date")
    if re.search(r"\b(campus|hall|auditorium|college|university|room|online|lab|bengaluru|mysuru)\b", text, re.I):
        fields.append("venue")
    if re.search(r"(\+91[\s-]?)?[6-9]\d{9}", text):
        fields.append("contact")
    if re.search(r"(₹|rs\.?|inr|cash|prize|pool)", text, re.I):
        fields.append("prize")
    return fields


def main():
    engine = RapidOCR()
    results = {}

    print("=" * 80)
    print("COMPARING RAPIDOCR ON 5 SCANNED POSTERS: 1280px vs 1600px")
    print("=" * 80)

    for sample_name in SAMPLES:
        pdf_path = RAW_POSTERS_DIR / sample_name
        if not pdf_path.exists():
            print(f"File not found: {sample_name}")
            continue

        print(f"\nProcessing {sample_name}...")
        pdf = pdfium.PdfDocument(str(pdf_path))
        page0 = pdf[0]
        # Rasterize at ~200 dpi (scale 2.0-2.5)
        raw_pil = page0.render(scale=2.2).to_pil().convert("RGB")
        page0.close()
        pdf.close()

        # 1. Test 1280px
        im1280 = resize_image(raw_pil, 1280)
        t0 = time.time()
        res1280, _ = engine(np.array(im1280))
        t1280 = round(time.time() - t0, 3)
        txt1280 = " ".join([l[1] for l in res1280]) if res1280 else ""
        f1280 = detect_key_fields(txt1280)

        # 2. Test 1600px
        im1600 = resize_image(raw_pil, 1600)
        t0 = time.time()
        res1600, _ = engine(np.array(im1600))
        t1600 = round(time.time() - t0, 3)
        txt1600 = " ".join([l[1] for l in res1600]) if res1600 else ""
        f1600 = detect_key_fields(txt1600)

        diff_chars = len(txt1600) - len(txt1280)
        diff_time = t1600 - t1280

        results[sample_name] = {
            "1280": {"time": t1280, "chars": len(txt1280), "fields": f1280},
            "1600": {"time": t1600, "chars": len(txt1600), "fields": f1600},
            "diff_chars": diff_chars,
            "diff_time": diff_time,
        }

        print(f"  @ 1280px: {t1280}s | {len(txt1280)} chars | Fields: {f1280}")
        print(f"  @ 1600px: {t1600}s | {len(txt1600)} chars | Fields: {f1600}")
        print(f"  Delta   : {diff_time:+.3f}s | {diff_chars:+d} chars")

    print("\n" + "=" * 95)
    print(f"{'Sample Poster':<35} | {'1280px (s)':<10} | {'1280ch':<8} | {'1600px (s)':<10} | {'1600ch':<8} | {'Delta Chars':<10}")
    print("-" * 95)
    for k, v in results.items():
        print(f"{k[:35]:<35} | {v['1280']['time']:<10.2f} | {v['1280']['chars']:<8} | {v['1600']['time']:<10.2f} | {v['1600']['chars']:<8} | {v['diff_chars']:+d}")
    print("=" * 95)


if __name__ == "__main__":
    main()
