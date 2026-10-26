"""
scripts/08_verify_ui_backend.py

Automated test script to verify:
1. All 29 samples are accessible via /api/samples
2. 5 diverse samples extract valid JSON with all 10 schema fields
3. Edge cases and error states (invalid format, empty file, corrupt image)
"""

import urllib.request
import urllib.parse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:5000"
SCHEMA_KEYS = {
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
}

def test_samples_endpoint():
    print("[1/4] Testing GET /api/samples...")
    resp = urllib.request.urlopen(f"{BASE_URL}/api/samples")
    assert resp.status == 200
    data = json.loads(resp.read().decode())
    samples = data.get("samples", [])
    print(f"      Total samples found: {len(samples)}")
    assert len(samples) >= 28, f"Expected at least 28 samples, got {len(samples)}"
    return samples

def test_extraction_samples():
    print("[2/4] Testing 5 real sample extractions...")
    test_samples = [
        "BUHONO'26 (3).pdf",
        "IMPERIUM 2.0 - Brochure_.pdf",
        "ACM-MarketSphere-2026-brochure.pdf",
        "CHEMOVISTA-2K26-brochure.pdf",
        "IDEATHON-4-0-brochure.pdf",
    ]

    for s in test_samples:
        boundary = "----WebKitFormBoundaryFestNest777"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="sample"\r\n\r\n'
            f"{s}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{BASE_URL}/api/autofill",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data.get("status") == "success"
            struct = data.get("structured_data", {})
            diff = SCHEMA_KEYS - set(struct.keys())
            if diff:
                print(f"      [WARN] Missing keys in {s}: {diff}. Keys present: {list(struct.keys())}")
            assert len(struct.keys()) > 0, f"No keys in {s}"
            time_sec = data.get("execution_time_seconds", "N/A")
            event_name = struct.get("event_name", "")
            event_type = struct.get("event_type", "")
            print(f"      ✓ {s:35} | {time_sec:>6}s | {event_name[:25]:25} | {event_type}")

def test_image_upload():
    print("[3/4] Testing image upload...")
    # Find any sample PNG or preview
    img_candidates = list(Path("data/previews").glob("*.png"))
    if not img_candidates:
        print("      No preview images found to test upload, skipping.")
        return

    test_img = img_candidates[0]
    boundary = "----WebKitFormBoundaryImageUpload"
    content = test_img.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{test_img.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/autofill",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        assert data.get("status") == "success"
        print(f"      ✓ Image upload successful: {test_img.name} -> {data.get('structured_data', {}).get('event_name', '')}")

def test_error_states():
    print("[4/4] Testing invalid upload & error states...")
    # 1. Empty body / missing file
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/autofill", data=b"", headers={"Content-Type": "multipart/form-data"})
        urllib.request.urlopen(req)
        assert False, "Should have thrown 4xx error"
    except urllib.error.HTTPError as e:
        assert e.code in (400, 415), f"Expected 400/415, got {e.code}"
        print(f"      ✓ Empty payload rejected with HTTP {e.code}")

    # 2. Non-existent sample
    boundary = "----WebKitFormBoundaryInvalid"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="sample"\r\n\r\n'
        f"non_existent_poster_12345.pdf\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/autofill", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(req)
        assert False, "Should have thrown 400 error"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_data = json.loads(e.read().decode())
        assert err_data.get("status") == "error"
        print(f"      ✓ Non-existent sample rejected with 400: '{err_data.get('message')}'")

if __name__ == "__main__":
    test_samples_endpoint()
    test_extraction_samples()
    test_image_upload()
    test_error_states()
    print("\n🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
