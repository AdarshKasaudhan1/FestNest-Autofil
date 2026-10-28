import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

labeled_dir = Path('data/labeled')
files = sorted([f for f in labeled_dir.iterdir() if f.suffix == '.json'])
print(f"Total labeled json files: {len(files)}")

for f in files:
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        if isinstance(data, list):
            print(f"{f.name:50} -> LIST of {len(data)} events")
        elif 'events' in data and isinstance(data['events'], list):
            print(f"{f.name:50} -> dict with 'events' ({len(data['events'])})")
        else:
            name = data.get('event_name', '')
            etype = data.get('event_type') or data.get('event_category', '')
            date = data.get('date') or data.get('event_dates', '')
            print(f"{f.name:50} -> \"{name}\" | {etype} | Date: {date}")
    except Exception as e:
        print(f"{f.name:50} -> Error: {e}")
