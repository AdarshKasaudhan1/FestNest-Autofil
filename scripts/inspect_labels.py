import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

labeled_dir = Path('data/labeled')
files = sorted([f for f in labeled_dir.iterdir() if f.suffix == '.json'])

print(f"{'Filename':40} | {'Event Name':30} | {'Type':15} | {'Date':20} | {'Prize':15}")
print("-" * 130)

for f in files:
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        jo = data.get('json_output', {})
        name = str(jo.get('event_name', ''))[:30]
        etype = str(jo.get('event_type') or jo.get('event_category', ''))[:15]
        date = str(jo.get('date') or jo.get('event_dates', ''))[:20]
        prize = str(jo.get('prize_pool', ''))[:15]
        print(f"{f.name:40} | {name:30} | {etype:15} | {date:20} | {prize:15}")
    except Exception as e:
        print(f"{f.name:40} | ERROR: {e}")
