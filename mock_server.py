"""Test-only mock of the AutoFill API. NOT part of the integration. Run: python mock_server.py"""
import time, os
from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__, static_folder=None)
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = ["ACM-MarketSphere-2026-brochure.pdf", "cet_techfest_poster.png", "slow_unseen.pdf", "broken_sample.pdf"]
@app.get("/")
def index(): return send_from_directory(HERE, "index.html")
@app.get("/<path:p>")
def static_files(p): return send_from_directory(HERE, p)
@app.get("/api/samples")
def samples(): return jsonify(SAMPLES)
@app.post("/api/extract")
def extract():
    name = request.form.get("sample") or (request.files.get("file").filename if request.files.get("file") else "")
    if not name: return jsonify(error="No file received"), 400
    if "broken" in name or "bad" in name: return jsonify(error="OCR failed: unreadable document"), 500
    time.sleep(float(os.environ.get("MOCK_DELAY", "1.2")))
    if "blank" in name: return jsonify(result={}, ocr_text="")
    return jsonify(result={"event_name":"The Marketsphere Hackathon","event_type":"Hackathon","date":"19th & 20th Sept","time":"",
        "venue":"On-Campus / Online","organizer":"ACM RVCE Student Chapter","prize_pool":"20K Cash Prize","registration_deadline":"",
        "contact":"Rohan Bharadwaj: +91 97415 49360","description":"A finance-focused event where markets meet innovation."}, ocr_text="THE MARKETSPHERE HACKATHON\n19th & 20th Sept\nPrize 20K")
if __name__ == "__main__": app.run(port=5055)
