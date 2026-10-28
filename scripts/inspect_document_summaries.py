import sys, json, re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Loaded {len(docs)} documents.")

for d in docs:
    fname = d["filename"]
    pages = d["pages"]
    tot_pages = d["total_pages"]
    
    # print header text (first page or first two pages)
    p1_text = pages[0]["text"] if len(pages) > 0 else ""
    p2_text = pages[1]["text"] if len(pages) > 1 else ""
    combo_text = p1_text + "\n" + p2_text
    
    # Look for obvious event names or titles
    lines = [l.strip() for l in combo_text.splitlines() if l.strip()]
    first_few = " /// ".join(lines[:6])
    
    print(f"\n==========================================")
    print(f"FILE: {fname} (Pages: {tot_pages})")
    print(f"PREVIEW: {first_few[:200]}")
