"""
train.py – Build and save the TF-IDF + structural vectorizer + Linear SVM.

Usage
-----
  python train.py               # train on everything in data/raw/
  python train.py --limit 50    # cap at 50 files per category (quick test)
  python train.py --no-grid     # skip GridSearchCV (faster, uses C=1.0)

Expected data layout
--------------------
  data/raw/
    invoices/       ← PDF/image invoices
    emails/         ← PDF/image emails
    questionnaires/ ← PDF/image questionnaires
    resumes/        ← PDF/image resumes

Accepted file types: .pdf  .png  .jpg  .jpeg  .tiff  .tif  .bmp

Outputs
-------
  models/vectorizer.pkl   (TF-IDF + structural scaler)
  models/classifier.pkl   (SVM + label encoder)

What improved
-------------
  Phase 2 now builds TF-IDF (8 000 features) + 20 structural features
  (currency symbols → Invoice, @address → Email, checkboxes → Questionnaire,
   education/experience → Resume).  These hard-coded signals give the SVM
  direct, unambiguous anchors that TF-IDF alone can miss.

  Phase 3 runs GridSearchCV over C ∈ [0.01 … 50] using 5-fold cross-
  validation to find the best regularisation strength automatically.
  After evaluation the model is re-fitted on the FULL corpus so no
  training signal is wasted.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.ocr_engine import OCREngine
from src.feature_engineering import TFIDFFeatureExtractor
from src.classifier import DocumentClassifier

DATA_DIR   = Path("data/raw")
MODELS_DIR = Path("models")
CACHE_FILE = Path("data/ocr_cache.json")   # persists OCR results between runs

CATEGORY_MAP = {
    "invoices":       "Invoice",
    "emails":         "Email",
    "questionnaires": "Questionnaire",
    "resumes":        "Resume",
}

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  OCR cache loaded — {len(data)} entries already processed.")
        return data
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def collect_training_data(
    ocr: OCREngine,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    texts:  list[str] = []
    labels: list[str] = []

    cache = load_cache()
    new_entries = 0

    for folder_name, label in CATEGORY_MAP.items():
        folder = DATA_DIR / folder_name
        if not folder.exists():
            print(f"  [WARN] {folder} not found – skipping '{label}'")
            continue

        files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]
        if not files:
            print(f"  [WARN] No supported files in {folder} – skipping '{label}'")
            continue

        if limit:
            files = files[:limit]

        cached_count = sum(1 for f in files if str(f) in cache)
        print(f"\n  [{label}]  {len(files)} file(s)  ({cached_count} cached, {len(files) - cached_count} need OCR) …")

        for fpath in tqdm(files, desc=f"  {label:<10}", unit="file", leave=False):
            key = str(fpath)
            if key in cache:
                text = cache[key]
            else:
                try:
                    text = ocr.process_file(fpath)
                    cache[key] = text
                    new_entries += 1
                    # Save after every 10 new entries so progress survives interruption
                    if new_entries % 10 == 0:
                        save_cache(cache)
                except Exception as exc:
                    print(f"    ✗ {fpath.name}: {exc}")
                    continue

            if text.strip():
                texts.append(text)
                labels.append(label)

    save_cache(cache)
    print(f"\n  Cache updated — {new_entries} new entries written to {CACHE_FILE}")
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the StatExtract-Classifier.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Max files per category (e.g. --limit 10 for a quick smoke-test)",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip GridSearchCV and use C=1.0 (much faster, useful for quick tests)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TRAINING  –  StatExtract Classifier (Person A)")
    if args.limit:
        print(f"  Cap       : {args.limit} files per category")
    print(f"  GridSearch: {'OFF (--no-grid)' if args.no_grid else 'ON'}")
    print("=" * 60)

    # ── Phase 1: OCR ──────────────────────────────────────────────────────
    print("\n[Phase 1] Extracting text via OCR …")
    ocr = OCREngine()
    texts, labels = collect_training_data(ocr, args.limit)

    if len(texts) < 4:
        print(
            f"\n  ERROR: Only {len(texts)} document(s) found.\n"
            "  Add files to data/raw/<category>/ folders and re-run.\n"
        )
        sys.exit(1)

    counts = Counter(labels)
    print(f"\n  Documents collected: {len(texts)}")
    for lbl in sorted(counts):
        print(f"    {lbl:<14} {counts[lbl]}")

    # ── Phase 2: Feature engineering (TF-IDF + structural) ────────────────
    print("\n[Phase 2] Feature engineering (TF-IDF + structural features) …")
    extractor = TFIDFFeatureExtractor()
    X = extractor.fit_transform(texts)
    print(f"  Feature matrix: {X.shape[0]} documents × {X.shape[1]} features")

    # ── Phase 3: Train SVM (with optional GridSearchCV) ───────────────────
    print("\n[Phase 3] Training Linear SVM …")
    clf = DocumentClassifier()
    accuracy = clf.train(X, labels, grid_search=not args.no_grid)

    # Re-fit on full corpus after evaluation so no data is wasted
    if len(set(labels)) > 1 and not (accuracy != accuracy):  # not NaN
        print("  Re-fitting on full corpus (all data) …")
        le_full = LabelEncoder().fit(labels)
        y_full  = le_full.transform(labels)
        clf.model.fit(X, y_full)
        print("  Done — model now trained on 100% of available data.")

    # ── Save models ────────────────────────────────────────────────────────
    print("\n[Saving models]")
    extractor.save(MODELS_DIR / "vectorizer.pkl")
    clf.save(MODELS_DIR / "classifier.pkl")

    if accuracy == accuracy:  # not NaN
        print(f"\n  Training complete.  Test accuracy: {accuracy:.2%}")
    else:
        print("\n  Training complete (accuracy not measurable — add more data).")
    print("  Run  python app.py  to start the web server.\n")


if __name__ == "__main__":
    main()
