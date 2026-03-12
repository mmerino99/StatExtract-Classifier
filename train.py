"""
train.py – Build and save the TF-IDF vectorizer + Linear SVM classifier.

Usage
-----
  python train.py

Expected data layout
--------------------
  data/raw/
    invoices/    ← put 20 PDF/image invoices here
    emails/      ← put 20 PDF/image emails here
    contracts/   ← put 20 PDF/image contracts here
    reports/     ← put 20 PDF/image reports here

Outputs
-------
  models/vectorizer.pkl
  models/classifier.pkl
"""

import sys
from pathlib import Path
from tqdm import tqdm

from src.ocr_engine import OCREngine
from src.feature_engineering import TFIDFFeatureExtractor
from src.classifier import DocumentClassifier

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")

CATEGORY_MAP = {
    "invoices":  "Invoice",
    "emails":    "Email",
    "contracts": "Contract",
    "reports":   "Report",
}

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def collect_training_data(ocr: OCREngine) -> tuple[list[str], list[str]]:
    texts, labels = [], []

    for folder_name, label in CATEGORY_MAP.items():
        folder = DATA_DIR / folder_name
        if not folder.exists():
            print(f"  [WARN] {folder} not found – skipping '{label}'")
            continue

        files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]
        if not files:
            print(f"  [WARN] No supported files in {folder} – skipping '{label}'")
            continue

        print(f"\n  [{label}] processing {len(files)} file(s) …")
        for fpath in tqdm(files, desc=f"  {label}", unit="file", leave=False):
            try:
                text = ocr.process_file(fpath)
                if text.strip():
                    texts.append(text)
                    labels.append(label)
            except Exception as exc:
                print(f"    ✗ {fpath.name}: {exc}")

    return texts, labels


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 52)
    print("  TRAINING  –  StatExtract Classifier (Person A)")
    print("=" * 52)

    # ── Phase 1: OCR ──────────────────────────────────────────────────────
    print("\n[Phase 1] Extracting text via OCR …")
    ocr = OCREngine()
    texts, labels = collect_training_data(ocr)

    if len(texts) < 4:
        print(
            f"\n  ERROR: Only {len(texts)} document(s) found.\n"
            "  Please add files to data/raw/<category>/ folders and re-run.\n"
        )
        sys.exit(1)

    label_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    print(f"\n  Documents collected: {len(texts)}")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl:<12} {cnt}")

    # ── Phase 2: Feature engineering ──────────────────────────────────────
    print("\n[Phase 2] TF-IDF vectorisation …")
    extractor = TFIDFFeatureExtractor()
    X = extractor.fit_transform(texts)
    print(f"  Feature matrix: {X.shape[0]} documents × {X.shape[1]} features")

    # ── Phase 3: Train SVM ─────────────────────────────────────────────────
    print("\n[Phase 3] Training Linear SVM (80 / 20 split) …")
    clf = DocumentClassifier()
    accuracy = clf.train(X, labels)

    # ── Save models ────────────────────────────────────────────────────────
    print("[Saving models]")
    extractor.save(MODELS_DIR / "vectorizer.pkl")
    clf.save(MODELS_DIR / "classifier.pkl")

    print(f"\n  Training complete.  Test accuracy: {accuracy:.2%}")
    print("  Run  python predict.py <file>  to classify a new document.\n")


if __name__ == "__main__":
    main()
