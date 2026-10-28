import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, warmup_server

print("=" * 80)
print("TESTING UI BACKEND DIRECTLY VIA FLASK TEST CLIENT")
print("=" * 80)

client = app.test_client()

# 1. Test GET /api/samples
print("\n[1/3] Testing GET /api/samples...")
res = client.get("/api/samples")
assert res.status_code == 200
samples_data = res.get_json()
samples = samples_data.get("samples", [])
print(f"      Found {len(samples)} samples available.")
assert len(samples) >= 28, f"Expected at least 28 samples, got {len(samples)}"

# 2. Test 5 diverse sample extractions
print("\n[2/3] Testing 5 diverse sample extractions via POST /api/autofill...")
test_samples = [
    "BUHONO'26 (3).pdf",
    "IMPERIUM 2.0 - Brochure_.pdf",
    "ACM-MarketSphere-2026-brochure.pdf",
    "CHEMOVISTA-2K26-brochure.pdf",
    "IDEATHON-4-0-brochure.pdf",
]

SCHEMA_KEYS = [
    "event_name",
    "event_type",
    "date",
    "time",
    "venue",
    "organizer",
    "prize_pool",
    "registration_deadline",
    "contact",
    "description",
]

for s in test_samples:
    res = client.post("/api/autofill", data={"sample": s})
    assert res.status_code == 200, f"Failed on {s}: HTTP {res.status_code}"
    data = res.get_json()
    assert data.get("status") == "success"
    struct = data.get("structured_data", {})
    t_sec = data.get("execution_time_seconds", 0.0)
    
    missing_keys = [k for k in SCHEMA_KEYS if k not in struct]
    assert len(missing_keys) == 0, f"Missing keys in {s}: {missing_keys}"
    
    filled = [k for k in SCHEMA_KEYS if str(struct.get(k, "")).strip()]
    print(f"      ✓ {s:35} | {t_sec:>6}s | {len(filled)}/10 filled | {struct.get('event_name')[:25]}")

# 3. Test Preview endpoint
print("\n[3/3] Testing GET /api/preview/<filename>...")
for s in test_samples[:3]:
    res = client.get(f"/api/preview/{s}")
    assert res.status_code == 200, f"Preview failed for {s}: HTTP {res.status_code}"
    print(f"      ✓ Preview for {s:35} | HTTP 200 OK | {len(res.data):>8} bytes")

print("\n" + "=" * 80)
print("ALL UI BACKEND DIRECT TESTS PASSED SUCCESSFULLY!")
print("=" * 80)
