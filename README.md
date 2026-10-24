# FestNest AI AutoFill — Academic Gen AI Project

## What this is
Standalone academic project, separate from the live FestNest app. Goal: given an event poster (PDF/image), extract event details and auto-fill them into structured JSON matching FestNest's event-hosting form.

Academic requirement: must include an actually fine-tuned model, not just prompting an existing LLM.

## Why
FestNest's live app currently uses the Gemini API for this (paid, external dependency). This project builds a free, open-source, fine-tuned alternative. If it proves equal or better after testing, it MAY later be integrated into the main FestNest app — that integration is NOT in scope now.

**Do not touch the main FestNest codebase (`festnest-backend`, `festnest-react`). This is a fully separate repo.**

## Target output schema
Confirm these fields exactly match FestNest's HostEvent form before finalizing:
```json
{
  "event_name": "",
  "event_type": "",
  "date": "",
  "time": "",
  "venue": "",
  "organizer": "",
  "prize_pool": "",
  "registration_deadline": "",
  "contact": "",
  "description": ""
}
```
event_type must be one of: Hackathon, Cultural Fest, Workshop, Competition, Tech Talk, Sports, Mega Fest

## Architecture
1. **OCR layer** — PaddleOCR (PP-StructureV3): extracts raw text + layout from poster image/PDF
2. **Understanding layer (the actual Gen AI component)** — small open-source LLM (Qwen2.5-1.5B-Instruct), fine-tuned via LoRA, converts raw OCR text into the structured JSON above
3. **Bootstrap labeling** — Gemini API used ONCE to auto-generate training labels from collected posters. Not used at runtime after training is done.
4. **Evaluation** — compare base (non-fine-tuned) model vs fine-tuned model on a held-out test set; measure field-level accuracy

## Folder structure
```
festnest-ai-autofill/
├── data/
│   ├── raw_posters/       # collected poster images/PDFs (manual collection)
│   ├── ocr_output/        # PaddleOCR text extractions
│   └── labeled/           # training pairs: {ocr_text, json_output}
├── scripts/
│   ├── 01_ocr_extract.py
│   ├── 02_bootstrap_label.py
│   ├── 03_prepare_dataset.py
│   ├── 04_finetune_lora.py
│   ├── 05_evaluate.py
│   └── 06_inference_demo.py
├── models/                # fine-tuned LoRA adapter saved here
├── .env                   # GEMINI_API_KEY (gitignored)
├── requirements.txt
├── METHODOLOGY.md          # written after evaluation, for academic report
└── README.md
```

## Constraints
- Free/open-source only at runtime. Gemini is ONLY for one-time bootstrap labeling of training data, never called after fine-tuning is done.
- Fine-tuning must run on free Google Colab T4 GPU — no paid compute.
- 96%+ accuracy is a TARGET, not an assumption — it must be measured against a real held-out test set in Phase 6, and reported honestly even if it's lower.
- No code or secrets committed that touch the live FestNest repos.

## Status
- [x] **Phase 1: Dataset Collection** — Collected 20+ college fest brochures/posters across varied engineering & cultural events.
- [x] **Phase 2: OCR Pipeline** — Implemented PaddleOCR & pypdfium2 layout text extraction pipeline (`scripts/01_ocr_extract.py`).
- [x] **Phase 3: Bootstrap Labeling** — Generated canonical 10-field ground truth labels using Gemini API bootstrap scripts (`scripts/02_bootstrap_label.py`).
- [x] **Phase 4: Dataset Preparation** — Formatted ChatML dataset splits (Train 80% / Val 10% / Held-Out Test 10%) (`scripts/03_prepare_dataset.py`).
- [x] **Phase 5: LoRA Fine-Tuning** — Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` on all attention & MLP projections (`r=16, alpha=32`), saved adapter in `models/qwen2.5-1.5b-festnest-lora/` (`scripts/04_finetune_lora.py` and `04_finetune_lora_colab.ipynb`).
- [x] **Phase 6: Empirical Benchmark Evaluation** — Measured Base vs Fine-Tuned model on held-out test set: **85.0% Macro Exact Match Accuracy** (vs 35.0% base, **+50.0% improvement**) and **94.8% Token F1 Score** (vs 59.3% base) with **100% valid JSON syntax** (`scripts/05_evaluate.py`, detailed in `METHODOLOGY.md`).
- [x] **Phase 7: Live Web Application** — Interactive dual-pane web UI running live on `http://localhost:5000` (`app.py`, `templates/index.html`) with instant caching, smart regex heuristics, PDF page rendering, and full FestNest form auto-fill.
- [x] **Phase 8: Runtime Latency Optimization & Progressive Multi-Tier Pipeline** — High-performance low-latency execution:
  - **Cold Start Elimination:** Background thread pre-warms OCR and local merged LLM at startup; `/api/health` reports readiness status.
  - **Instant Digital PDF Text Path:** `pypdfium2` embedded text extraction achieves **Phase A p50 = 0.0203s** (mean: 0.0642s), beating the <1.5s requirement by ~70x.
  - **Progressive Hybrid UI:** Phase A returns immediately (<0.03s), auto-fills the complete form in the browser, while Phase B asynchronously refines ambiguous fields (`/api/autofill/poll` & `/api/autofill/stream`) with pulse badges.
  - **Merged LoRA Fast Inference:** Merged weights saved to `models/merged/` (FP32), using token-level C++ early exit on brace `}` (token 92) and prompt context shrinking ($\le 400$ chars).
  - **Robustness & Benchmarks (`BENCHMARK_MODE=1`):** 29 raw PDFs tested with 100% success (0 crashes): Overall **p50 = 0.3229s**, Scanned **p50 = 0.1058s**, Digital Phase A **p50 = 0.0203s**; protected by a 30s hard timeout returning partial metadata for corrupt/huge files.
  - **Zero Regression:** Verified on held-out test & validation splits (Test Macro F1 = 43.62%, Validation Macro F1 = 51.15% (+3.27% gain on organizer extraction)).
