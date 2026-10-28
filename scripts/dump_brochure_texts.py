import sys, json, re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

for idx, d in enumerate(docs, 1):
    fname = d["filename"]
    tot_pages = d["total_pages"]
    all_text = "\n--- PAGE BREAK ---\n".join(p["text"] for p in d["pages"])
    
    print("=" * 80)
    print(f"[{idx:02d}/29] FILE: {fname} (Total Pages: {tot_pages})")
    print("=" * 80)
    
    # Print first 1500 chars and last 1000 chars if long
    if len(all_text) <= 2500:
        print(all_text)
    else:
        print(all_text[:1500])
        print("\n... [TRUNCATED " + str(len(all_text) - 2500) + " CHARACTERS] ...\n")
        print(all_text[-1000:])
    print("\n")
