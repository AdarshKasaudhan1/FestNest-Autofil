import sys, json

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    raw_docs = {d["filename"]: d for d in json.load(f)}

from pathlib import Path
labeled_dir = Path("data/labeled")

for f in sorted(labeled_dir.iterdir()):
    if f.suffix != ".json":
        continue
    data = json.loads(f.read_text(encoding='utf-8'))
    jo = data.get("json_output", {})
    print(f"\n==========================================")
    print(f"LABELED FILE: {f.name}")
    print(f"  event_name:            {jo.get('event_name')}")
    print(f"  event_type:            {jo.get('event_type')}")
    print(f"  date:                  {jo.get('date')}")
    print(f"  time:                  {jo.get('time')}")
    print(f"  venue:                 {jo.get('venue')}")
    print(f"  organizer:             {jo.get('organizer')}")
    print(f"  prize_pool:            {jo.get('prize_pool')}")
    print(f"  registration_deadline: {jo.get('registration_deadline')}")
    print(f"  contact:               {jo.get('contact')}")
    print(f"  description:           {jo.get('description')[:80] if jo.get('description') else ''}")
