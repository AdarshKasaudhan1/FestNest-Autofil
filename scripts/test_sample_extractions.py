import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pipeline_improved import run_improved_extraction

sys.stdout.reconfigure(encoding="utf-8")

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    raw_list = json.load(f)

raw_dict = {}
for doc in raw_list:
    text = "\n\n--- Page Break ---\n\n".join(p.get("text", "") for p in doc.get("pages", []))
    raw_dict[doc["filename"]] = text

test_samples = [
    "BUHONO'26 (3).pdf",
    "IMPERIUM 2.0 - Brochure_.pdf",
    "ACM-MarketSphere-2026-brochure.pdf",
    "CHEMOVISTA-2K26-brochure.pdf",
    "IDEATHON-4-0-brochure.pdf",
    "PARIKSHIT-2026-brochure (1).pdf",
    "KALARAV-2026-brochure.pdf",
]

for sample in test_samples:
    text = raw_dict.get(sample, "")
    res = run_improved_extraction(text, sample)
    sd = res["structured_data"]
    print(f"=== {sample} ===")
    print(f"  Name: {sd['event_name']}")
    print(f"  Cat:  {sd['event_type']}")
    print(f"  Date: {sd['date']}")
    print(f"  Time: {sd['time']}")
    print(f"  Ven:  {sd['venue']}")
    print(f"  Org:  {sd['organizer']}")
    print(f"  Prz:  {sd['prize_pool']}")
    print(f"  Ddl:  {sd['registration_deadline']}")
    print(f"  Lnk:  {sd['registration_link']}")
    print(f"  Cnt:  {sd['contact']}")
    print(f"  Dsc:  {sd['description'][:65]}...")
    print()
