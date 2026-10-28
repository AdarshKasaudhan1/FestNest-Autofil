import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/all_29_extracted_raw.json", encoding="utf-8") as f:
    docs = {d["filename"]: d for d in json.load(f)}

for target in ["PARIKSHIT 2026 BROCHURE.pdf", "KALARAV-2026-brochure.pdf", "ACM-MarketSphere-2026-brochure.pdf", "IMPERIUM 2.0 - Brochure_.pdf"]:
    doc = docs.get(target, {})
    pages = doc.get("pages", [])
    print(f"=== {target}: {len(pages)} pages ===")
    for p in pages[:4]:
        print(f"--- Page {p.get('page_number')} ({p.get('engine')}) ---")
        txt = p.get("text", "")
        print(txt[:300] if txt else "[NO TEXT]")
        print()
