import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/all_29_extracted_raw.json", encoding="utf-8") as jf:
    docs = {d["filename"]: d for d in json.load(jf)}

text = "\n".join(p.get("text", "") for p in docs.get("BUHONO'26 (3).pdf", {}).get("pages", []))
for kw in ["tech talk", "guest lecture", "keynote address", "seminar", "invited talk"]:
    if kw in text.lower():
        print("MATCHED:", kw)
        idx = text.lower().find(kw)
        print("CONTEXT:", text[max(0, idx-30):idx+50])
