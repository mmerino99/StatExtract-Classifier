# StatExtract-Classifier — Person A: Classification & Vision Engine

> **Role**: Gatekeeper.  Takes a raw PDF or image, extracts all text via OCR, and returns a high-confidence category label + the full text for Person B to mine for dates, totals, and fields.

---

## Project Structure

```
StatExtract-Classifier/
├── src/
│   ├── ocr_engine.py          # Phase 1 – OpenCV preprocessing + Tesseract
│   ├── feature_engineering.py # Phase 2 – text cleaning + TF-IDF vectorisation
│   ├── classifier.py          # Phase 3 – Linear SVM
│   └── pipeline.py            # Phase 4 – full chain + Person B handoff
├── data/
│   ├── raw/
│   │   ├── invoices/          ← put 20 invoice PDFs here
│   │   ├── emails/            ← put 20 email PDFs here
│   │   ├── contracts/         ← put 20 contract PDFs here
│   │   └── reports/           ← put 20 report PDFs here
│   └── processed/             # debug images + JSON outputs
├── models/                    # saved vectorizer.pkl + classifier.pkl
├── train.py                   # run once to build the model
├── predict.py                 # classify a new file
├── demo_ocr.py                # quick OCR test (your "first step")
└── requirements.txt
```

---

## Setup

### 1 — Install Tesseract OCR (Windows)

Download and run the installer from:  
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

If you install it elsewhere, update `TESSERACT_CMD` in `src/ocr_engine.py`.

### 2 — Install Python dependencies

```powershell
pip install -r requirements.txt
```

---

## Workflow

### Step 0 — Verify OCR is working (First Step)

```powershell
python demo_ocr.py data/raw/invoices/invoice_001.pdf
```

This prints the extracted text to the console and saves a debug image of the
binarised page so you can inspect the preprocessing quality.

### Step 1 — Gather training data

Place your documents in the `data/raw/<category>/` folders:

| Folder             | Label    | Suggested count |
|--------------------|----------|-----------------|
| `data/raw/invoices/`  | Invoice  | 20 files        |
| `data/raw/emails/`    | Email    | 20 files        |
| `data/raw/contracts/` | Contract | 20 files        |
| `data/raw/reports/`   | Report   | 20 files        |

Accepted formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

### Step 2 — Train the model

```powershell
python train.py
```

This runs all four phases sequentially and saves the trained models to `models/`.

### Step 3 — Classify a new document

```powershell
python predict.py path/to/your/document.pdf
```

Output example:

```
====================================================
  CLASSIFICATION RESULT  (Person A → Person B)
====================================================
  File       : path/to/your/document.pdf
  Label      : INVOICE
  Confidence : 94.3%
  Summary    : This is an INVOICE.
====================================================

--- RAW TEXT (hand off to Person B for data extraction) ---

INVOICE
Date: 15 March 2025
Invoice No: INV-00412
...
```

A JSON file with the result (excluding raw text) is saved to `data/processed/`.

---

## Technical Justification

> *"We used a Linear SVM because it is mathematically efficient for
> high-dimensional text data.  Combined with TF-IDF, it allows the model to
> ignore common language and focus on the statistical significance of
> category-specific keywords."*

| Component | Why |
|---|---|
| **OpenCV preprocessing** | Grayscale → deskew → Otsu binarisation removes noise and corrects skewed scans, pushing OCR accuracy from ~70 % to ~95 %+ |
| **Stop-word removal + lemmatisation** | Reduces feature noise; "billing / billed / bills" all map to one root token |
| **TF-IDF** | Rewards rare, category-specific words (e.g. "liability" in contracts) and penalises ubiquitous ones |
| **Linear SVM (`kernel='linear'`)** | Best-in-class for high-dimensional sparse text vectors; interpretable and fast |
| **80/20 train-test split** | Standard hold-out validation to measure generalisation |

---

## Handoff Contract (Person A → Person B)

`pipeline.predict_category()` returns:

```python
{
    "file":       str,   # original file path
    "label":      str,   # "INVOICE" | "EMAIL" | "CONTRACT" | "REPORT"
    "confidence": float, # 0.0 – 1.0
    "summary":    str,   # "This is an INVOICE."
    "raw_text":   str,   # every word Tesseract extracted – for Person B to parse
}
```
