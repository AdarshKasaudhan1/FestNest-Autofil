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
Not started. Blocked on poster dataset collection (150-300 real posters) before Phase 1 can produce useful output.
