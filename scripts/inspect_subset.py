import sys, json

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    docs = {d["filename"]: d for d in json.load(f)}

for name in ["DSBA-BATTALIA-2K26-brochure.pdf", "SARANSH 2026 Inter V.1.pdf", "Brochure - AllianceOne 4.0.pdf"]:
    print("=" * 60)
    print("FILE:", name)
    print("=" * 60)
    for p in docs[name]["pages"][:5]:
        print(f"--- Page {p['page_num']} ---")
        print(p["text"])
