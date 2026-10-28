import sys, os, json, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image

raw_dir = Path('data/raw_posters')
ocr_dir = Path('data/ocr_output')
files = sorted([f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith('.')])

print(f"Total raw PDFs to analyze: {len(files)}")
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr

results = []

for idx, f in enumerate(files, 1):
    stem = f.stem
    cached_ocr_file = ocr_dir / f"{f.name}.json"
    if not cached_ocr_file.exists():
        cached_ocr_file = ocr_dir / f"{stem}.json"
    
    cached_ocr_data = None
    if cached_ocr_file.exists():
        try:
            cached_ocr_data = json.loads(cached_ocr_file.read_text(encoding='utf-8'))
        except Exception:
            cached_ocr_data = None

    doc = pdfium.PdfDocument(str(f))
    total_pages = len(doc)
    doc_info = {
        "filename": f.name,
        "total_pages": total_pages,
        "pages": []
    }
    
    # Check if cached ocr has page-level text
    cached_page_texts = {}
    if cached_ocr_data and "pages" in cached_ocr_data and isinstance(cached_ocr_data["pages"], list):
        for p_entry in cached_ocr_data["pages"]:
            p_no = p_entry.get("page_number") or p_entry.get("page", 0)
            cached_page_texts[p_no] = p_entry.get("text", "")
    
    for p_idx in range(total_pages):
        page = doc[p_idx]
        tp = page.get_textpage()
        native_text = tp.get_text_range().strip()
        tp.close()
        
        ocr_text = ""
        source = "native"
        final_text = native_text

        if len(native_text) < 50:
            # Check cached OCR first
            p_1based = p_idx + 1
            if p_1based in cached_page_texts and len(cached_page_texts[p_1based]) > 10:
                ocr_text = cached_page_texts[p_1based]
                source = "cached_ocr"
                final_text = ocr_text
            elif cached_ocr_data and "text" in cached_ocr_data and total_pages <= 4:
                # If cached text is single blob for whole doc
                ocr_text = cached_ocr_data["text"]
                source = "cached_ocr_blob"
                final_text = ocr_text
            else:
                # Run RapidOCR on demand
                ocr_engine = get_ocr()
                pil_img = page.render(scale=1.5).to_pil()
                res, _ = ocr_engine(np.array(pil_img))
                if res:
                    ocr_text = "\n".join([line[1] for line in res if len(line) > 1 and line[1]])
                source = "rapidocr"
                final_text = ocr_text
        
        page.close()
        
        page_entry = {
            "page_num": p_idx + 1,
            "native_char_count": len(native_text),
            "ocr_char_count": len(ocr_text),
            "text": final_text,
            "source": source
        }
        doc_info["pages"].append(page_entry)
        
    doc.close()
    
    total_text_len = sum(len(p["text"]) for p in doc_info["pages"])
    print(f"[{idx:02d}/{len(files)}] {f.name:50} | Pages: {total_pages:2d} | Chars: {total_text_len:6d}")
    results.append(doc_info)

output_path = Path("data/all_29_extracted_raw.json")
with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, indent=2, ensure_ascii=False)

print(f"\nSaved all raw extractions to {output_path}")
