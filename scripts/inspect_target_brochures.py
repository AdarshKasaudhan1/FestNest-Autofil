import sys, json

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    docs = {d["filename"]: d for d in json.load(f)}

target_files = [
    "Metanoia-2-0---The-Rise-of-the-Realm-brochure.pdf",
    "QUIZ DOM 3 page 1.pdf",
    "RNBQ'17_BROCHURE.pdf",
    "Tarang Flyer 2026.pdf",
    "TECH-SPARK .pdf",
    "The Lost Voyage - Event Guidelines.pdf",
    "Shark Tank - Event Guidelines.pdf",
    "VIHAAN'26 - DOOMSDAY Brochure.pdf",
    "KALARAV-2026-brochure.pdf",
    "BUHONO'26 (3).pdf",
    "CHEMOVISTA-2K26-brochure.pdf",
    "INDUX-5-0---2026-brochure.pdf",
    "FEST BROCHURE 2026.PDF",
    "IDEATHON-4-0-brochure.pdf",
    "KAYA-2026---ALL-INDIA-AI-FEST-brochure.pdf"
]

for tf in target_files:
    if tf in docs:
        print("=" * 80)
        print(f"FULL TEXT FOR: {tf}")
        print("=" * 80)
        for p in docs[tf]["pages"]:
            print(f"--- Page {p['page_num']} ---")
            print(p["text"])
        print("\n")
