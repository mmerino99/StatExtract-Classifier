# StatExtract-Classifier

An end-to-end document classification and information extraction system.  
Upload a PDF or image — get back a category label, confidence score, and (for invoices) structured field extraction.

---

## What it does

1. **OCR** — extracts text from any PDF or image using a multi-strategy pipeline:
   - Digital PDFs: uses the native embedded text layer (instant, perfect accuracy)
   - Scanned PDFs / images: Tesseract with advanced OpenCV preprocessing; EasyOCR fallback for handwriting
2. **Classification** — a TF-IDF + structural feature + Linear SVM model classifies the document into one of four categories:

   | Label | Description |
   |---|---|
   | **Invoice** | Commercial invoices, bills, receipts |
   | **Email** | Printed or scanned email threads |
   | **Questionnaire** | Forms, surveys, intake sheets |
   | **Resume** | CVs and professional profiles |

3. **Invoice extraction** — when a document is classified as an Invoice, six structured fields are automatically extracted and displayed:

   | Field | Example |
   |---|---|
   | Invoice number | `INV-0042` |
   | Invoice date | `15/03/2026` |
   | Due date | `Apr 15, 2026` |
   | Issuer name | `Acme Corp` |
   | Recipient name | `John Smith` |
   | Total amount | `1,250.00` |

---

## Project structure

```
StatExtract-Classifier/
│
├── src/
│   ├── ocr_engine.py          # Phase 1 – multi-strategy OCR (native PDF / Tesseract / EasyOCR)
│   ├── feature_engineering.py # Phase 2 – TF-IDF + 20 structural features
│   ├── classifier.py          # Phase 3 – Linear SVM with GridSearchCV
│   ├── pipeline.py            # Phase 4 – full chain: file → enriched result dict
│   └── extractor.py           # Invoice / Email / Questionnaire / Resume field extractors
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── templates/
│   └── index.html             # drag-and-drop web UI
│
├── data/
│   ├── raw/                   # training documents (invoices/ emails/ questionnaires/ resumes/)
│   └── ocr_cache.json         # persists OCR results between training runs
│
├── models/                    # saved vectorizer.pkl + classifier.pkl (created by train.py)
├── debug_ocr/                 # per-upload OCR debug logs (created at runtime)
│
├── app.py                     # Flask web server  →  http://127.0.0.1:5000
├── train.py                   # build and save the classifier
├── predict.py                 # classify a single file from the command line
├── demo_ocr.py                # OCR smoke-test tool
├── setup_data.py              # download training dataset from Kaggle
└── requirements.txt
```

---

## Setup

### 1. Install Tesseract OCR (Windows)

Download and run the installer:  
<https://github.com/UB-Mannheim/tesseract/wiki>

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`  
If you install elsewhere, update `TESSERACT_CMD` in `src/ocr_engine.py`.

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add training data

Place documents in the four category folders:

```
data/raw/
  invoices/
  emails/
  questionnaires/
  resumes/
```

Accepted file types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`

### 4. Train the model

Quick test (10 files per category, takes ~1 minute):

```powershell
python train.py --limit 10
```

Full training:

```powershell
python train.py
```

### 5. Start the web app

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

---

## Command-line tools

### Classify a single file

```powershell
python predict.py path/to/document.pdf
```

### Verify OCR is working

```powershell
python demo_ocr.py path/to/document.pdf
```

Prints the extracted text to the console and saves a preprocessed debug image to `data/processed/`.

### Download training dataset from Kaggle (optional)

If you want to use the RVL-CDIP Small benchmark dataset:

```powershell
pip install kaggle
# Place your kaggle.json at C:\Users\<you>\.kaggle\kaggle.json
python setup_data.py
```

---

## Technical design

| Component | Choice | Reason |
|---|---|---|
| **OCR** | Tesseract + EasyOCR fallback | Tesseract is fast and accurate on printed text; EasyOCR handles handwriting and heavily degraded scans |
| **PDF text** | PyMuPDF native layer first | Digital PDFs already contain embedded text — extracting it directly is 100 % accurate and instant |
| **Preprocessing** | OpenCV pipeline | Upscale → deskew → polarity normalise → bilateral denoise → CLAHE → unsharp mask → Otsu binarise |
| **White text** | Band-based inversion | Detects dark-background regions (headers, footers) and locally inverts them before OCR |
| **Features** | TF-IDF (15k) + 20 structural | Structural features (currency symbols, @ addresses, checkboxes) give the SVM unambiguous class anchors that TF-IDF misses |
| **Classifier** | Linear SVM | Best-in-class for high-dimensional sparse text; scales linearly; interpretable |
| **Tuning** | 5-fold GridSearchCV over C | Automatically selects regularisation strength; avoids overfitting without manual tuning |
| **IE** | Regex + spaCy NER | Rule-based patterns cover structured invoice layouts; spaCy NER extracts party names from free-form text |

---

## API response (POST `/classify`)

```json
{
  "label":             "INVOICE",
  "confidence":        94.3,
  "is_uncertain":      false,
  "summary":           "This is an INVOICE.",
  "all_scores":        [{"label": "Invoice", "confidence": 94.3}, ...],
  "ocr_engine":        "tesseract",
  "word_count":        312,
  "text_preview":      "Acme Corp\nInvoice No: INV-0042 ...",
  "invoice_fields": {
    "invoice_number":  "INV-0042",
    "invoice_date":    "15/03/2026",
    "due_date":        "Apr 15, 2026",
    "issuer_name":     "Acme Corp",
    "recipient_name":  "John Smith",
    "total_amount":    "1,250.00"
  },
  "detected_language": "en",
  "language_name":     "English",
  "is_non_english":    false,
  "warning":           false,
  "elapsed_ms":        1842
}
```

`invoice_fields` is `null` for non-invoice documents.
