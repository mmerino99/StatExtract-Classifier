# StatExtract-Classifier — Person A: Classification & Vision Engine

> **Role**: Gatekeeper.  Takes a raw PDF or image, extracts all text via OCR,
> and returns a high-confidence category label + the full text for Person B to
> mine for dates, totals, and fields.

**Dataset**: [RVL-CDIP Small](https://www.kaggle.com/datasets/uditamin/rvl-cdip-small) — 16 real-world document classes.

---

## 16 Document Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0  | Letter | 8  | File Folder |
| 1  | Form | 9  | News Article |
| 2  | Email | 10 | Budget |
| 3  | Handwritten | 11 | Invoice |
| 4  | Advertisement | 12 | Presentation |
| 5  | Scientific Report | 13 | Questionnaire |
| 6  | Scientific Publication | 14 | Resume |
| 7  | Specification | 15 | Memo |

---

## Project Structure

```
StatExtract-Classifier/
├── src/
│   ├── ocr_engine.py          # Phase 1 – OpenCV preprocessing + Tesseract
│   ├── feature_engineering.py # Phase 2 – text cleaning + TF-IDF vectorisation
│   ├── classifier.py          # Phase 3 – Linear SVM (16 classes)
│   └── pipeline.py            # Phase 4 – full chain + Person B handoff
├── data/
│   └── rvl-cdip-small/        ← created by setup_data.py
│       ├── images/            ← .tif document images
│       └── labels/
│           ├── train.txt      ← "images/.../file.tif <class_id>"
│           ├── val.txt
│           └── test.txt
├── models/                    # saved vectorizer.pkl + classifier.pkl
├── setup_data.py              # download dataset from Kaggle
├── train.py                   # build the model
├── predict.py                 # classify a new file
└── demo_ocr.py                # quick OCR smoke-test
```

---

## Setup

### 1 — Install Tesseract OCR (Windows)

Download and run the installer:  
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`  
If you install elsewhere, update `TESSERACT_CMD` in `src/ocr_engine.py`.

### 2 — Install Python dependencies

```powershell
pip install -r requirements.txt
pip install kaggle          # only needed for setup_data.py
```

### 3 — Configure Kaggle credentials

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to:

```
C:\Users\<your-username>\.kaggle\kaggle.json
```

### 4 — Download the dataset

```powershell
python setup_data.py
```

This downloads the RVL-CDIP Small zip, extracts it to `data/rvl-cdip-small/`,
and verifies that `images/` and `labels/` are in place.

---

## Workflow

### Step 0 — Verify OCR is working

```powershell
python demo_ocr.py data/rvl-cdip-small/images/imagesa/e/f/ef123456/0000001.tif
```

Prints extracted text and saves a binarised debug image to `data/processed/`.

### Step 1 — Quick training run (recommended first)

Cap at 100 images per class to validate the pipeline in minutes:

```powershell
python train.py --limit 100
```

### Step 2 — Full training

```powershell
python train.py
```

Use `--split val` or `--split test` to train on a different split.

### Step 3 — Classify a new document

```powershell
python predict.py path/to/your/document.pdf
python predict.py path/to/your/scan.tif
```

Example output:

```
====================================================
  CLASSIFICATION RESULT  (Person A → Person B)
====================================================
  File       : my_scan.tif
  Label      : INVOICE
  Confidence : 94.3%
  Summary    : This is an INVOICE.
====================================================

--- RAW TEXT (hand off to Person B for data extraction) ---

INVOICE
Date: 15 March 2025  Invoice No: INV-00412 ...
```

A lightweight JSON result is saved to `data/processed/` for Person B.

---

## Technical Justification

> *"We used a Linear SVM because it is mathematically efficient for
> high-dimensional text data. Combined with TF-IDF, it allows the model to
> ignore common language and focus on the statistical significance of
> category-specific keywords."*

| Component | Why |
|---|---|
| **OpenCV preprocessing** | Grayscale → deskew → Otsu binarisation lifts OCR accuracy from ~70% to ~95%+ on scanned documents |
| **Stop-word removal + lemmatisation** | Reduces feature noise; "billing / billed / bills" → single root token "bill" |
| **TF-IDF** | Rewards rare, category-specific words ("liability" in specifications) and penalises ubiquitous ones |
| **Linear SVM (`kernel='linear'`)** | Best-in-class for high-dimensional sparse text; scales linearly with features; interpretable weights |
| **RVL-CDIP Small** | Real-world, 16-class benchmark dataset — standard in document AI research |
| **80/20 stratified split** | Keeps class proportions identical in train and test; required with 16 imbalanced classes |

---

## Handoff Contract (Person A → Person B)

`pipeline.predict_category()` returns:

```python
{
    "file":       str,   # original file path
    "label":      str,   # e.g. "INVOICE", "EMAIL", "SCIENTIFIC REPORT"
    "confidence": float, # 0.0 – 1.0
    "summary":    str,   # "This is an INVOICE."
    "raw_text":   str,   # every word Tesseract extracted – for Person B to parse
}
```
