"""
tests/test_ocr_regression.py

Regression test suite ensuring that:
1. Image-only (scanned) PDFs yield robust, non-empty OCR text.
2. OCR engine is properly reported and healthy.
3. Poster previews are reliably generated and accessible.
4. Clean poster text does not corrupt double digits / phone numbers.
5. Error handling returns explicit reason codes (e.g. ocr_empty, pdf_render_failed)
   instead of silent success with empty fields.
"""

import os
import sys
import pytest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["BENCHMARK_MODE"] = "1"
os.environ["USE_GEMINI"] = "0"

import app as app_mod
from app import (
    app,
    get_ocr_engine,
    run_ocr_on_file,
    clean_poster_text,
    extract_regex_heuristics,
    rasterize_pdf_first_page,
    warmup_server,
)


@pytest.fixture(scope="module")
def warm_client():
    """Initializes and warms up the app test client."""
    warmup_server()
    with app.test_client() as client:
        yield client


def test_ocr_engine_available():
    """Ensures primary OCR engine (RapidOCR or PaddleOCR) initializes successfully."""
    ocr = get_ocr_engine()
    assert ocr is not None, "OCR engine must not be None"
    assert app_mod._OCR_ENGINE_NAME in ("rapidocr", "paddleocr"), (
        f"Unexpected OCR engine: {app_mod._OCR_ENGINE_NAME}"
    )


def test_api_health_endpoint(warm_client):
    """Verifies that /api/health reports ready and exposes active OCR engine."""
    res = warm_client.get("/api/health")
    assert res.status_code == 200, f"Expected HTTP 200, got {res.status_code}"
    data = res.get_json()
    assert data["ready"] is True
    assert data["status"] == "ready"
    assert data["ocr_engine"] in ("rapidocr", "paddleocr")


def test_scanned_pdf_yields_non_empty_ocr():
    """
    REGRESSION TEST:
    Uploading an image-only PDF must run OCR and yield non-empty text.
    Previously failed with 0.3s empty text due to concurrent file lock crash.
    """
    sample_path = PROJECT_ROOT / "data" / "raw_posters" / "BUHONO'26 (3).pdf"
    assert sample_path.exists(), f"Sample poster not found: {sample_path}"

    text, err = run_ocr_on_file(sample_path, skip_digital=True)
    assert err is None, f"Expected no error, got error code: {err}"
    assert len(text.strip()) > 100, f"Expected >100 characters of OCR text, got {len(text)}"
    assert "BUHONO" in text or "MGOCSM" in text or "KOTTAYAM" in text, (
        f"Key poster content missing from OCR output: {text[:200]}"
    )


def test_imperium_forced_scanned_ocr():
    """
    REGRESSION TEST:
    Verifies that IMPERIUM 2.0 forced through scanned OCR extracts valid text.
    """
    sample_path = PROJECT_ROOT / "data" / "raw_posters" / "IMPERIUM 2.0 - Brochure_.pdf"
    assert sample_path.exists(), f"Sample poster not found: {sample_path}"

    text, err = run_ocr_on_file(sample_path, skip_digital=True)
    assert err is None, f"Expected no error, got: {err}"
    assert len(text.strip()) > 300, f"Expected >300 chars, got {len(text)}"
    assert any(term in text for term in ["KRISTU JAYANTI", "IMPERIUM", "RESEARCH", "COMPETITION"]), (
        f"Key content missing from OCR: {text[:200]}"
    )


def test_poster_preview_rasterization():
    """Verifies that first page preview rasterization generates a valid PNG thumbnail."""
    sample_path = PROJECT_ROOT / "data" / "raw_posters" / "IMPERIUM 2.0 - Brochure_.pdf"
    preview = rasterize_pdf_first_page(sample_path)
    assert preview is not None, "Preview generation returned None"
    assert preview.exists(), f"Preview file {preview} does not exist on disk"
    assert preview.stat().st_size > 10000, "Preview file is suspiciously small"


def test_clean_poster_text_preserves_phone_numbers():
    """
    REGRESSION TEST:
    Ensures that clean_poster_text does NOT destructively collapse repeated digits
    in Indian phone numbers (e.g. '9999764768' must remain 10 digits).
    """
    raw_line = "MOB: +91-9999764768, Alternative: 88807 81442"
    cleaned = clean_poster_text(raw_line)
    heuristics = extract_regex_heuristics(cleaned)
    assert heuristics.get("contact") in ("+91-9999764768", "9999764768"), (
        f"Phone number corrupted by cleaning! Extracted: {heuristics.get('contact')}"
    )


def test_autofill_api_returns_explicit_error_on_empty():
    """
    Ensures /api/autofill never returns a success-looking result when OCR yields empty text.
    """
    from PIL import Image
    import io

    # Blank white image (no text)
    blank = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    buf.seek(0)

    with app.test_client() as client:
        res = client.post("/api/autofill?sync=1", data={"file": (buf, "blank_empty.png")})
        assert res.status_code in (422, 503), f"Expected HTTP 422 or 503, got {res.status_code}"
        data = res.get_json()
        assert data["status"] == "error"
        assert data["error_code"] in ("ocr_empty", "ocr_unavailable")
        assert "No readable text" in data["message"]
