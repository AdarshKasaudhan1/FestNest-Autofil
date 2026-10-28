import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pipeline_improved import extract_event_category

sys.stdout.reconfigure(encoding="utf-8")

with open("data/all_29_extracted_raw.json", encoding="utf-8") as jf:
    docs = {d["filename"]: d for d in json.load(jf)}

for name in ["BUHONO'26 (3).pdf", "PARIKSHIT-2026-brochure (1).pdf", "KALARAV-2026-brochure.pdf", "ACM-MarketSphere-2026-brochure.pdf", "IMPERIUM 2.0 - Brochure_.pdf"]:
    text = "\n".join(p.get("text", "") for p in docs.get(name, {}).get("pages", []))
    cat, prov = extract_event_category(text, name)
    print(f"{name}: {cat} -> {prov}")
