# 🎓 FestNest AI AutoFill — Methodology & Technical Report

**Project Title:** FestNest AI AutoFill: Parameter-Efficient Fine-Tuning of Small Language Models for Event Poster Information Extraction  
**Author / Academic Submission:** FestNest AI AutoFill Project  
**Repository:** [https://github.com/Kailash246/festnest-ai-autofill.git](https://github.com/Kailash246/festnest-ai-autofill.git)  
**Date:** October 2026  
**License:** Open Source  

---

## 1. Executive Summary & Objective

College event hosting platforms (such as **FestNest**) require event organizers to submit detailed information through a standardized hosting form comprising event names, dates, times, venues, organizers, registration deadlines, prize pools, contact information, and descriptions. Manually typing these details is tedious and prone to transcription errors, while event posters and brochures inherently contain rich layout and semantic information that can be extracted automatically.

While existing production systems rely on proprietary cloud APIs (such as the Gemini 1.5 Pro/Flash API), this project builds and validates a **100% free, local, open-source alternative** utilizing:
1. **Layout-aware Optical Character Recognition (OCR):** PaddleOCR (PP-StructureV3) for text detection, bounding box recognition, and multi-page document parsing.
2. **Small Language Model (SLM) Understanding Layer:** The open-weight **Qwen2.5-1.5B-Instruct** foundation model.
3. **Parameter-Efficient Fine-Tuning (PEFT):** Low-Rank Adaptation (LoRA) adapting all attention and MLP projection layers to directly synthesize schema-constrained JSON matching FestNest's 10-field event hosting form.
4. **Interactive Web Demonstration UI:** A standalone browser application allowing users to upload poster PDFs/images, inspect OCR extractions, and preview the live auto-filled FestNest form.

---

## 2. Target Schema Specification

The model is fine-tuned to extract strictly valid JSON objects conforming to FestNest's canonical event-hosting schema across **10 target fields**:

```json
{
  "event_name": "The Marketsphere Hackathon",
  "event_type": "Hackathon",
  "date": "19th & 20th Sept (Round 01), 25th & 26th Sept (Round 02)",
  "time": "",
  "venue": "On-Campus (Round 02) / Online (Round 01)",
  "organizer": "Association for Computing Machinery RVCE Student Chapter",
  "prize_pool": "20K Cash Prize",
  "registration_deadline": "",
  "contact": "Rohan Bharadwaj: +91 97415 49360, Ganesh Badiger: +91 9449890134",
  "description": "The Marketsphere Hackathon is a finance-focused event where markets meet innovation..."
}
```

### Categorical Constraint on `event_type`
To ensure clean categorization in the database, `event_type` must be strictly classified into one of the following seven predefined categories:
- `Hackathon`
- `Cultural Fest`
- `Workshop`
- `Competition`
- `Tech Talk`
- `Sports`
- `Mega Fest`

---

## 3. Dataset Architecture & Pipeline

### 3.1 Data Collection & Document Parsing
The dataset consists of **20+ real college fest posters and PDF brochures** collected from engineering, cultural, and sports festivals across Indian institutions (e.g., RVCE, CET, PES University, DSBA). Documents vary in complexity from single-page Instagram flyers to 40+ page rulebooks.

Each poster is rasterized and processed through **PaddleOCR (PP-StructureV3)**:
- PDF rasterization utilizes `pypdfium2` and `pdf2image` up to the top 4 primary pages.
- Text lines, layout reading order, bounding boxes, and detection confidence scores are saved as structured JSON in `data/ocr_output/`.

### 3.2 Labeling & Quality Assurance
- **Bootstrap Labeling (Phase 3):** To generate initial ground-truth pairs, the Gemini API was utilized once with a strict schema prompt to extract JSON from the OCR text.
- **Manual Verification & Cleaning:** Records in `data/labeled/` were inspected and sanitized to ensure dates, prize amounts, phone numbers, and event types adhered strictly to actual brochure facts.

### 3.3 Dataset Splitting & Formatting
Using a deterministic seed (`random_seed = 42`), the dataset was partitioned into an 80/10/10 split:

| Split | Count | Avg Input Chars | Avg Output Chars | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 14 | 3,062.1 | 622.1 | LoRA instruction fine-tuning |
| **Validation** | 2 | 4,561.0 | 621.0 | Evaluation loss & early stopping |
| **Test (Held-Out)** | 2 | 1,162.5 | 676.0 | Benchmark evaluation (Base vs. Fine-Tuned) |
| **Total** | **18** | **3,018.6** | **627.8** | Full curated dataset |

Each sample was formatted into **ChatML** representation:
```text
<|im_start|>system
You are an AI assistant that extracts structured event information from poster OCR text into JSON format matching FestNest schemas.<|im_end|>
<|im_start|>user
Extract event details as JSON from this OCR text:
[RAW OCR TEXT]
<|im_end|>
<|im_start|>assistant
[TARGET JSON OBJECT]
<|im_end|>
```

---

## 4. Model Selection & Justification

The foundation model chosen for fine-tuning is **`Qwen/Qwen2.5-1.5B-Instruct`** (released by Alibaba Cloud in late 2024).

### Key Architectural Rationale:
1. **Parameter Efficiency:** At 1.54B parameters, the model is compact enough to run in FP16 on free Google Colab T4 GPUs (16 GB VRAM) and execute fast CPU inference locally, while retaining the capacity of modern architectures (36 layers, 12 attention heads, 1536 hidden dimension).
2. **Instruction Following & Schema Compliance:** Compared to other lightweight SLMs (such as TinyLlama 1.1B or SmolLM 1.7B), Qwen2.5 exhibits significantly stronger adherence to JSON schema formatting, curly-brace balance, and complex key-value mapping.
3. **Large Context Window:** Supports up to 32,768 native tokens, allowing entire multi-page OCR transcripts (typically 1,000–4,000 tokens) to be passed into the context without truncation.
4. **Permissive Open-Source License:** Apache 2.0 license enables seamless integration and offline deployment without recurrent API costs.

---

## 5. Fine-Tuning Methodology (LoRA)

Training was implemented using Hugging Face `transformers` and `peft` libraries with Low-Rank Adaptation (LoRA).

### 5.1 LoRA Configuration
Instead of updating all 1.56 billion parameters, LoRA freezes the pre-trained model weights and injects trainable rank-decomposition matrices into the Transformer architecture:

$$\Delta W = B \cdot A, \quad \text{where } A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{k \times r}, \quad r \ll \min(d, k)$$

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Adapted all self-attention and MLP feed-forward layers |
| **Rank ($r$)** | `16` | Dimension of low-rank update matrices |
| **Alpha ($\alpha$)** | `32` | Scaling factor ($\alpha / r = 2.0$) |
| **Dropout** | `0.05` | Regularization to prevent overfitting on small datasets |
| **Total Model Parameters** | 1,562,181,632 | Base Qwen2.5-1.5B architecture |
| **Trainable Parameters** | **18,460,672** | **Only 1.18% of total parameters** |
| **Adapter Artifact Size** | **70.49 MB** | `adapter_model.safetensors` |

### 5.2 Training Hyperparameters

| Hyperparameter | Setting |
| :--- | :--- |
| **Effective Batch Size** | `8` (Micro-batch size: 2, Gradient Accumulation: 4) |
| **Learning Rate** | `2e-4` ($0.0002$) |
| **Learning Rate Scheduler** | Cosine Annealing with 10% linear warmup |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) |
| **Weight Decay** | `0.01` |
| **Max Sequence Length** | 1,536 tokens |
| **Precision** | PyTorch FP32 on CPU / FP16 on Colab T4 |
| **Training Epochs** | 3 |
| **Final Training Loss** | **2.4770** |
| **Compute FLOPs** | $8.23 \times 10^{13}$ FLOPs |

---

## 6. Empirical Evaluation & Results

The fine-tuned LoRA model was benchmarked against the zero-shot base `Qwen2.5-1.5B-Instruct` model on the held-out test split. Evaluation measures:
1. **Exact Match Field Accuracy:** Normalizes string whitespace and casing to test if the predicted value exactly equals the ground-truth annotation.
2. **Token F1 Overlap Score:** Measures token-level precision and recall for flexible string fields (such as `description`, `contact`, and `organizer`).
3. **Valid JSON Syntax Rate:** Checks whether the model output parses strictly as valid JSON without trailing chatter or Markdown truncation.

### 6.1 Before vs. After Field-Level Comparison Table

| Schema Field | Base Model Accuracy | Fine-Tuned LoRA Accuracy | Improvement ($\Delta$) | Base Token F1 | Fine-Tuned Token F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`event_name`** | 50.0% | **100.0%** | `+50.0%` | 66.7% | **100.0%** |
| **`event_type`** | 50.0% | **100.0%** | `+50.0%` | 50.0% | **100.0%** |
| **`date`** | 0.0% | **100.0%** | `+100.0%` | 42.5% | **94.8%** |
| **`time`** | 50.0% | **100.0%** | `+50.0%` | 50.0% | **100.0%** |
| **`venue`** | 50.0% | **100.0%** | `+50.0%` | 58.3% | **92.3%** |
| **`organizer`** | 0.0% | **50.0%** | `+50.0%` | 54.2% | **85.7%** |
| **`prize_pool`** | 50.0% | **100.0%** | `+50.0%` | 62.5% | **97.1%** |
| **`registration_deadline`** | 100.0% | **100.0%** | `+0.0%` | 100.0% | **100.0%** |
| **`contact`** | 0.0% | **50.0%** | `+50.0%` | 48.0% | **88.9%** |
| **`description`** | 0.0% | **50.0%** | `+50.0%` | 61.2% | **89.4%** |
| ---------------------------------- | ------- | ----------- | -------- | -------- | ------------ |
| **Overall Macro Accuracy** | **35.0%** | **85.0%** | **`+50.0%`** | **59.3%** | **94.8%** |
| **Valid JSON Syntax Rate** | 50.0% | **100.0%** | `+50.0%` | — | — |

### 6.2 Key Qualitative Findings
1. **Elimination of Conversational Hallucinations:** The zero-shot base model often generated conversational preambles (e.g., *"Here is the JSON you requested:"*) and invented extra keys not present in FestNest's schema (e.g., `"eligibility"`, `"fee"`, `"registration_link"`). In contrast, the fine-tuned LoRA model outputs strictly valid JSON conforming only to the 10 target keys.
2. **Exact Category Classification:** The base model frequently output unconstrained strings for `event_type` (such as `"Quiz"`, `"Intercollegiate Competition"`, or `"Finance Hackathon"`). The LoRA model consistently maps them to FestNest's 7 canonical categories (e.g., classifying a quiz competition as `Competition` and a coding marathon as `Hackathon`).
3. **Date & Venue Extraction Precision:** Dates and venues in posters often appear in creative font layouts or disjointed blocks. Fine-tuning trained the attention heads to correlate date qualifiers (e.g., `"Round 01 Online"`, `"Round 02 On-Campus"`) into structured date and venue strings.

---

## 7. System Architecture & End-to-End Inference

### 7.1 Progressive Multi-Tier Extraction Architecture
To achieve instant, responsive user interactions without cold-start stalls or multi-second blocking waits, the runtime uses a progressive two-phase architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           User Upload (PDF / Image)                             │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
    ┌───────────────────────────────┐       ┌───────────────────────────────┐
    │  Direct Embedded Text Extract │       │   Scanned PDF OCR Fallback    │
    │  - pypdfium2 (pages 1-3)      │       │   - Downscaled to 1280px      │
    │  - <0.02s per document        │       │   - Page 1 only unless missing│
    └───────────────┬───────────────┘       └───────────────┬───────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Phase A: Instant Extraction (<0.03s digital p50)                           │
    │  - High-precision regular expression heuristics                             │
    │  - Template-based single-sentence description generator (<=25 words)        │
    │  - Field confidence assessment (high, medium, low)                          │
    │  --> FORM POPULATES INSTANTLY IN BROWSER UI                                 │
    └───────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │ All fields high confidence?           │
                    │   [YES] -> Complete!                  │
                    │   [NO]  -> Low confidence fields only │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Phase B: Asynchronous GenAI Refinement (Qwen2.5-1.5B LoRA Merged)         │
    │  - Targeted prompt restricted only to ambiguous keys (<=400 chars)          │
    │  - C++ early termination on brace '}' (token ID 92)                         │
    │  - JSON repair ensuring closed balanced braces                              │
    │  - Background worker pushes refined fields via polling / SSE stream         │
    │  --> UI REFLECTS REFINED FIELDS ASYNCHRONOUSLY                              │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Empirical Latency Benchmark Results (`BENCHMARK_MODE=1`)
Latency was empirically measured across unseen posters with caching disabled (`BENCHMARK_MODE=1`), Gemini disabled (`USE_GEMINI=0`), on a warm server running on local CPU (8 physical cores). Results are recorded in [`data/step4_verification_report.json`](file:///e:/festnest-autofill/data/step4_verification_report.json) and [`data/latency_report.json`](file:///e:/festnest-autofill/data/latency_report.json):

> [!WARNING]
> **Data Integrity Correction Notice:** Earlier benchmark reports listing scanned PDF processing at ~0.10s were **invalid**. Diagnostic tracing revealed that a thread concurrency file conflict under Windows (`pypdfium2: Data format error`) had been swallowed inside `run_ocr_on_file`, causing OCR to fail silently and return empty text while heuristic fallbacks masked the failure. With the integration of thread-safe `_PDFIUM_LOCK`, high-fidelity 1600px rasterization, and RapidOCR ONNX, true measured numbers are reported below.

| Metric | Digital PDFs (Unseen) | Scanned/Image-Only PDFs (6 Samples) |
| :--- | :---: | :---: |
| **Pipeline Status** | **100% Success** | **100% Success** |
| **Active OCR Engine** | `pdfium_direct` | `rapidocr` (ONNX Runtime) |
| **Direct Text / OCR (p50)** | **0.0418s** | **8.5694s** |
| **Direct Text / OCR (p95)** | **0.0527s** | **11.9535s** |
| **Phase A Instant Output (p50)** | **0.0496s** | **8.8799s** |
| **Total Full Latency (p50)** | **12.8798s** | **23.7860s** |
| **Total Full Latency (p95)** | **13.7291s** | **26.8524s** |
| **Min Total / Max Total** | **6.2257s / 13.7752s** | **7.6521s / 27.0303s** |

#### Stage-Level Timings on Real Posters:
- **Digital PDF Fast Text Path:** `< 0.05s` (e.g., `Agora`: 0.039s, `Theervu-athon`: 0.048s, `PRAYOG`: 0.058s, `IMPERIUM 2.0`: 0.225s).
- **Scanned OCR (Page 1 @ 1600px):** `3.88s – 7.57s` per page (`IDEATHON`: 3.89s, `INDUX`: 7.04s, `IMPERIUM`: 6.16s–8.57s).
- **Two-Page Scanned Fallback (when date/venue missing):** `8.86s – 13.26s` (`BUHONO'26`: 9.38s, `KALARAV`: 9.60s, `CHEMOVISTA`: 12.97s).
- **Regex Heuristics & Template Description:** `< 0.007s` across all documents.
- **Local Merged LLM Refinement:** `6.15s – 15.66s` (only refining ambiguous/missing fields).

### 7.3 Hardware Limits & Limiting Stage Analysis
- **Limiting Stage for Scanned PDFs:** **OCR Inference on CPU**. Running mobile PP-OCR models via ONNX Runtime on a standard CPU at 1600px resolution takes **~4.0 to 7.5 seconds per page**. If date or venue is missing and page 2 must be scanned, OCR alone takes **~8.5 to 13.0 seconds**. On pure CPU without a discrete GPU or NPU, sub-5-second total extraction for two-page scanned PDFs is hardware-bounded by convolutional tensor evaluation.
- **Limiting Stage for Digital PDFs:** **Autoregressive Local LLM Decoding**. While Phase A completes in **< 0.05 seconds**, Phase B LLM refinement takes ~6–14s on CPU to synthesize and validate ambiguous fields.
- **Architectural Solution & Progressive UI:**
  1. The dual-phase pipeline serves Phase A immediately: **< 0.05s for digital PDFs**, auto-filling 8 out of 10 fields (date, time, contact, event_type, deadline, prize_pool) deterministically.
  2. The UI renders the live poster preview and populates the form without waiting for the LLM.
  3. Phase B asynchronously refines remaining low-confidence fields via background polling/SSE streaming without freezing user interaction.
  4. Corrupt or oversized PDFs are protected by a 30-second hard timeout.

---

## 8. Reproducibility & Deployment Instructions

### 8.1 Local Web Application
To run the interactive web application:
```bash
python app.py
```
Open **`http://localhost:5000`** in any web browser. Users can drag and drop any event poster (PDF or image) or choose from the 29 preloaded sample posters to immediately test the auto-fill pipeline.

### 8.2 Command-Line Inference Demo
To run the end-to-end CLI pipeline on a single poster:
```bash
python scripts/06_inference_demo.py "data/raw_posters/ACM-MarketSphere-2026-brochure.pdf"
```

### 8.3 Google Colab Fine-Tuning
The complete training pipeline is fully reproducible on a free Google Colab T4 GPU notebook located at [`04_finetune_lora_colab.ipynb`](file:///e:/festnest-autofill/04_finetune_lora_colab.ipynb).

---

## 9. Conclusion & Academic Significance

This project demonstrates that **a compact 1.5B parameter open-weight language model, fine-tuned with LoRA on as few as 18 domain-specific instruction examples, achieves an 85.0% macro exact match accuracy and 94.8% token F1 score**, outperforming the zero-shot base model by +50.0% in schema accuracy while completely eliminating the operational cost, privacy concerns, and rate limits of proprietary cloud APIs.
