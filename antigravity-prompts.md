# Antigravity Prompts — FestNest AI AutoFill

Paste these ONE AT A TIME, in order. Don't start the next phase until the current one is working and you've done any manual step listed under it.

---

## Before Phase 1 — you do this yourself, nothing to paste
- [ ] Create a new empty GitHub repo: `festnest-ai-autofill` (separate from festnest-backend / festnest-react)
- [ ] Clone it locally, open the folder in Antigravity, add README.md to the root
- [ ] Get a free Gemini API key: https://aistudio.google.com/apikey
- [ ] Get a free Hugging Face account + access token: https://huggingface.co/settings/tokens
- [ ] Collect 150-300 real event poster images/PDFs (FestNest's own uploaded posters, Instagram/WhatsApp event pages, college notice boards). Put them in a folder — you'll point Antigravity to it in Phase 1.

This data collection step is the actual bottleneck of the whole project. Nothing below works without it.

---

## Phase 1 — Scaffold
```
Read the attached README.md fully before doing anything. Set up the folder structure exactly as described in it. Create requirements.txt with: paddleocr, paddlepaddle, pdf2image, pillow, python-dotenv, google-generativeai, transformers, peft, bitsandbytes, trl, datasets, accelerate, torch. Add .env to .gitignore. Initialize git with an initial commit. Do not build any processing logic yet — just scaffold and confirm the structure matches the README.
```

## Phase 2 — OCR extraction
```
Build scripts/01_ocr_extract.py. It should read every file in data/raw_posters/ (images and PDFs — convert PDF pages to images first with pdf2image), run PaddleOCR with PP-StructureV3 layout detection on each, and save raw extracted text + layout info as .json files in data/ocr_output/, one per poster, named to match the source file. Add error handling for corrupt or unreadable files — skip and log them, don't crash. Run it on the posters in data/raw_posters/ and show me 2-3 sample outputs.
```

## Phase 3 — Bootstrap labeling (one-time Gemini use)
```
Build scripts/02_bootstrap_label.py. For each file in data/ocr_output/, call the Gemini API (load GEMINI_API_KEY from .env) with a prompt that extracts this exact JSON schema from the OCR text:

[paste the schema block from README.md here]

Save each result as a pair in data/labeled/ as {ocr_text, json_output}. Respect free tier rate limits (~10 requests/min, add delay/retry). Run it on all OCR outputs and report how many succeeded/failed.
```
**Manual step after this phase:** open the files in data/labeled/ and manually correct any wrong fields in at least 20-30% of them before training. Gemini's bootstrap output is a starting point, not ground truth — training on uncorrected garbage produces a garbage model.

## Phase 4 — Dataset prep
```
Build scripts/03_prepare_dataset.py. Load all pairs from data/labeled/, split into train/validation/test sets (80/10/10), format each as an instruction-tuning example (instruction: extract event details as JSON from this OCR text; input: the ocr_text; output: the json_output), and save in a Hugging Face datasets-compatible format.
```

## Phase 5 — Fine-tuning
```
Build scripts/04_finetune_lora.py to fine-tune Qwen2.5-1.5B-Instruct using LoRA (use Unsloth if it simplifies this, otherwise PEFT + transformers) on the prepared dataset. Target: completes on a free Colab T4 GPU in under 2 hours. Save the LoRA adapter to models/. Also produce a Colab notebook (.ipynb) version of this script since training will run in Colab, not locally.
```
**Manual step after this phase:** open the generated .ipynb in Google Colab, set runtime to T4 GPU, upload your data/ folder (or mount Drive), run it, and keep the browser tab open/active until it finishes — free Colab disconnects on idle sessions.

## Phase 6 — Evaluation
```
Build scripts/05_evaluate.py. Run both the base Qwen2.5-1.5B-Instruct model and the fine-tuned version on the test set. Compute field-level accuracy (exact match per JSON field, averaged across all fields) for each. Output a before/after comparison table, plus 5 example outputs shown side by side: OCR input, base model output, fine-tuned output, ground truth.
```

## Phase 7 — Demo + report support
```
Build scripts/06_inference_demo.py — takes one new poster file, runs the full pipeline (OCR → fine-tuned model), and prints the final JSON. Then write METHODOLOGY.md summarizing: final dataset size, base model choice and why, fine-tuning method and hyperparameters used, and the Phase 6 evaluation results table. Keep it factual and specific to what was actually built and measured — this will be used in my academic submission, so no invented numbers.
```
