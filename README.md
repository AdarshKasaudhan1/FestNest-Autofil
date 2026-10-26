# festnest-autofill-ui

Light-theme frontend for FestNest AutoFill. Plain HTML/CSS/JS, no build step.
Works with the existing Flask backend as-is: **no changes to `app.py`, the OCR pipeline or the model.**

## Install
Copy into your project root (`festnest-autofill/`), replacing the old template:

    templates/index.html      (replaces the old one; back it up first)
    static/css/styles.css     (new; Flask serves /static automatically)
    static/js/app.js          (new)

Then run `python app.py` and open http://localhost:5000.

## API it uses (unchanged)
- `GET  /api/samples`            -> `{samples:[{filename,type,size_kb}]}`
- `GET  /api/preview/<filename>` -> poster image (samples only)
- `POST /api/autofill`           -> multipart `file` or `sample`; reads `structured_data`, `ocr_text`, `execution_time_seconds`; errors via `{status:"error", message}`

## Notes
- Brand tokens (accent, ink, line colors, font) are CSS variables at the top of `static/css/styles.css`. The accent is a placeholder; set it to the exact festnest.in color.
- Uploaded PDFs have no preview, because `/api/preview` only finds files by their original name and uploads are saved with a timestamp prefix. Uploaded images preview locally in the browser.
- The old "Model Evaluation" card is condensed into the "How it works" note at the bottom.
