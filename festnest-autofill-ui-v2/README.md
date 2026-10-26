# FestNest AutoFill UI (v2 Redesign)

A lively, polished, production-grade frontend redesign for **FestNest AutoFill** matching the exact visual identity and UX patterns of FestNest (`festnest.in`).

---

## 🎨 Visual Identity & Brand System

- **Colors:** FestNest Indigo (`#4F46E5`), Hover Indigo (`#4338CA`), Lavender Tints (`#EEF2FF`, `#F5F6FF`), Accent Violet (`#7C3AED`), Slate Inks (`#0F172A`, `#334155`, `#64748B`), and Rupee Gold (`#FEF3C7` / `#B45309`).
- **Typography:** Plus Jakarta Sans & Inter with heavy headlines (`800`), tight tracking (`-0.035em`), and uppercase letter-spaced eyebrow chips. Monospace for raw OCR text and JSON payloads.
- **Components:** 
  - Top navigation bar with logo badge, global search with `/` shortcut, navigation items, and `+ Host Event` button.
  - Compact hero with faint grid background, "NEW AI AutoFill" live eyebrow pill, and floating sample cards.
  - Left column with drag-and-drop zone, screenshot paste (`Ctrl+V`), and lazy-loaded thumbnail gallery with concurrency limiting.
  - Right column with multi-step animated progress stepper, structured 4-section form, staged cascading fill animation, segmented category pills, field-level copy/revert controls, and live card preview.
  - Sticky bottom action bar with Copy JSON, Download JSON, and Check Form.
  - Lightbox modal for full-size poster inspection.
  - Responsive layout verified at 1440px, 1024px, 768px, and 390px.

---

## 🚀 Installation & Quick Start

1. Copy the `templates/` and `static/` directories into your `festnest-autofill/` project root:
   ```bash
   cp -r templates/ <festnest-autofill-root>/templates/
   cp -r static/ <festnest-autofill-root>/static/
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

## 🔌 API Contract Compatibility (Zero Changes to Backend)

- `GET /api/samples` $\rightarrow$ Lists all 29 available sample brochures.
- `GET /api/preview/<filename>` $\rightarrow$ High-resolution poster preview image.
- `POST /api/autofill` $\rightarrow$ Multipart upload (`file` or `sample`) returning the 10 FestNest schema fields.

---

## 📁 Package Structure

```
festnest-autofill-ui-v2/
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── styles.css
    ├── js/
    │   └── app.js
    └── img/
        └── festnest-logo.svg
```
