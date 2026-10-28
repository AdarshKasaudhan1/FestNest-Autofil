# 🎓 FestNest AutoFill — End-to-End Layout-Aware Event Extraction & Generative AI Auto-Form Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Model: Qwen2.5-1.5B-Instruct](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![PEFT: LoRA (r=16, alpha=32)](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/huggingface/peft)
[![Backend: Flask](https://img.shields.io/badge/Backend-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Zero Cloud API Cost at Runtime](https://img.shields.io/badge/Runtime-100%25%20Offline%20CPU-success.svg)](#architecture)

> **Academic Capstone Project:** Layout-aware document understanding and parameter-efficient fine-tuning (PEFT/LoRA) of Small Language Models (SLMs) for automated college event hosting form generation on [FestNest](https://festnest.in).

📄 **Detailed 60 KB Faculty Viva & Technical Report:** [`FESTNEST_AUTOFILL_TECHNICAL_REPORT.md`](FESTNEST_AUTOFILL_TECHNICAL_REPORT.md)  
📊 **Empirical Evaluation & Methodology Report:** [`METHODOLOGY.md`](METHODOLOGY.md)

---

## 📌 Problem & Motivation

College event discovery platforms like **FestNest** require student organizers to manually submit detailed event hosting forms with **10 complex fields**:
1. `event_name`
2. `event_type` *(strictly categorized into: Hackathon, Cultural Fest, Workshop, Competition, Tech Talk, Sports, Mega Fest)*
3. `date`
4. `time`
5. `venue`
6. `organizer`
7. `prize_pool`
8. `registration_deadline`
9. `contact`
10. `description`

While organizers already produce creative promotional brochures and PDF rulebooks, manually re-typing these details is error-prone, tedious, and causes drop-off. Commercial cloud extraction APIs (like Gemini 1.5 Pro or GPT-4) incur recurring per-token costs, have rate limits, and transmit student event details to third-party cloud servers.

**FestNest AutoFill provides a 100% free, local, open-source alternative** running an instruction-tuned 1.5B parameter language model on commodity hardware with zero cloud API dependencies at runtime.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Ingestion (PDF / Image)                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │  Direct Embedded Text Extract │   │   Scanned PDF OCR Fallback    │
    │  - pypdfium2 (pages 1-3)      │   │   - RapidOCR ONNX Runtime     │
    │  - < 0.02s per document       │   │   - 1600px max long-edge      │
    └───────────────┬───────────────┘   └───────────────┬───────────────┘
                    │                                   │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │  Phase A: Instant Extraction (<0.05s on digital PDFs)             │
    │  - High-precision regular expression heuristics                   │
    │  - Template-based single-sentence description generator (<=25w)   │
    │  - Field confidence assessment (High / Medium / Low)              │
    │  --> Form populates instantly in client browser                   │
    └───────────────────────────────┬───────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ All fields high confidence?   │
                    │   [YES] -> Extraction Done!   │
                    │   [NO]  -> Low confidence only│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │  Phase B: Asynchronous SLM Refinement (Qwen2.5-1.5B LoRA Merged)  │
    │  - Targeted ChatML prompt restricted only to ambiguous keys       │
    │  - Early termination on closing brace '}' (Token ID 92)           │
    │  - JSON syntax repair ensuring balanced braces                    │
    │  - Background worker pushes refined fields via polling / SSE      │
    │  --> Client form reflects refined fields asynchronously           │
    └───────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Fine-Tuning Highlights (PEFT / LoRA)

- **Foundation Backbone:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, 36 layers, 1536 hidden dimension).
- **Target Modules:** All linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **LoRA Hyperparameters:** Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$.
- **Trainable Parameters:** **18,460,672 (only 1.18% of total weights)**.
- **Adapter Checkpoint Size:** **70.49 MB** (`adapter_model.safetensors`).
- **Merged Offline Model:** Fused into `models/merged/model.safetensors` for instant CPU loading without PEFT wrapper overhead.
- **Compute Budget:** Trained in ~15 minutes on a free Google Colab NVIDIA Tesla T4 GPU (`04_finetune_lora_colab.ipynb`).

---

## 📊 Benchmark & Evaluation Results

Tested against the held-out test split comparing zero-shot Base `Qwen2.5-1.5B-Instruct` vs. Fine-Tuned LoRA:

| Target Schema Field | Base Model Accuracy | Fine-Tuned LoRA Accuracy | Improvement ($\Delta$) | Base Token F1 | Fine-Tuned Token F1 |
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
| **Macro Average** | **35.0%** | **85.0%** | **`+50.0%`** | **59.3%** | **94.8%** |
| **Valid JSON Syntax Rate** | 50.0% | **100.0%** | `+50.0%` | — | — |

### Measured Latency Profile (Local CPU)
- **Digital Vector PDFs (p50):** Text extracted in **`0.0418s`**; Phase A form populated in **`0.0496s`**.
- **Scanned / Vector-less Posters (p50):** High-resolution RapidOCR executed in **`8.5694s`**.
- **On-Demand Preview Rasterization:** `< 0.05s` with thread-safe `_PDFIUM_LOCK`.

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/AdarshKasaudhan1/FestNest-Autofil.git
cd FestNest-Autofil

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy the example configuration:

```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
USE_GEMINI=0
ENABLE_GEMINI_RUNTIME=0
BENCHMARK_MODE=0
PORT=5000
```
*(Note: Gemini is for one-time bootstrap labeling only and is disabled at runtime for offline execution).*

### 3. Launch Web Application

```bash
python app.py
```

Open your browser to: **`http://localhost:5000`**

- Drag and drop any fest brochure (PDF or image).
- Or click any of the 29 preloaded college fest brochures in the sample ribbon.
- Inspect the extracted 10 fields and visual confidence badges.

---

## 💻 CLI Tools & Verification Scripts

### Run Standalone CLI Inference Demo

```bash
python scripts/06_inference_demo.py "data/raw_posters/ACM-MarketSphere-2026-brochure.pdf"
```

### Run Model Evaluation (Base vs. Fine-Tuned)

```bash
python scripts/05_evaluate.py
```

### Run Diagnostic Benchmark Across All 29 Posters

```bash
python scripts/07_diagnose.py
```

### Fine-Tune Adapter in Google Colab
Open [`04_finetune_lora_colab.ipynb`](04_finetune_lora_colab.ipynb) directly in Google Colab, select **T4 GPU**, and execute all cells.

---

## 📁 Repository Structure

```
FestNest-Autofil/
├── app.py                                   # Main Flask server, multi-tier engine & REST API
├── 04_finetune_lora_colab.ipynb             # Google Colab T4 GPU fine-tuning notebook
├── FESTNEST_AUTOFILL_TECHNICAL_REPORT.md    # 60 KB Comprehensive Technical & Viva Defense Report
├── METHODOLOGY.md                           # Original academic methodology & empirical benchmarks
├── requirements.txt                         # Exact pinned Python dependencies
├── templates/
│   └── index.html                           # Modern responsive UI with live PDF preview canvas
├── static/
│   ├── css/styles.css                       # FestNest brand theme, CSS variables & animations
│   ├── js/app.js                            # Client logic, drag-and-drop, SSE listener, polling
│   └── img/                                 # Logos, badges, and brand assets
├── data/
│   ├── dataset/                             # ChatML formatted Train, Validation, and Test splits
│   ├── labeled/                             # Ground truth pairs {ocr_text, json_output}
│   ├── ocr_output/                          # Cached layout extractions
│   └── raw_posters/                         # 29 real-world college fest brochure PDFs
├── models/
│   ├── qwen2.5-1.5b-festnest-lora/          # Fine-tuned LoRA adapter & config (70.5 MB)
│   └── merged/                              # Fused FP32 model weights for fast CPU inference
└── scripts/
    ├── 01_ocr_extract.py                    # Layout-aware OCR extraction pipeline
    ├── 02_bootstrap_label.py                # One-time bootstrap labeling via Gemini
    ├── 03_prepare_dataset.py                # Dataset splitter & ChatML formatter
    ├── 04_finetune_lora.py                  # Standalone LoRA training script
    ├── 05_evaluate.py                       # Base vs. Fine-Tuned benchmark script
    ├── 06_inference_demo.py                 # Single-file CLI inference runner
    └── 07_diagnose.py                       # Multi-stage diagnostic framework
```

---

## 🔒 Security & Privacy

- **Zero Cloud Leakage:** All inference runs on-device using local weights; student phone numbers and event data are never sent to external servers.
- **Path Sanitization:** File uploads are timestamped, isolated in `data/uploads/`, and stripped of malicious path traversals.
- **Denial-of-Service Defense:** PDF processing is bounded to 8 pages maximum with a 30-second hard execution deadline.

---

## 📜 Academic Integrity & Citation

This project was built for academic evaluation and open-source contribution to the FestNest platform. 

For full details on the mathematical derivation of LoRA, loss dynamics analysis, and 35+ faculty viva questions, consult [`FESTNEST_AUTOFILL_TECHNICAL_REPORT.md`](FESTNEST_AUTOFILL_TECHNICAL_REPORT.md).
