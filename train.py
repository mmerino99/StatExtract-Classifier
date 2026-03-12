"""
train.py – Build and save the TF-IDF vectorizer + Linear SVM classifier.

Usage
-----
  python train.py               # train on everything in data/raw/
  python train.py --limit 50    # cap at 50 files per category (quick test)

Expected data layout
--------------------
  data/raw/
    invoices/    ← 20 PDF/image invoices
    emails/      ← 20 PDF/image emails
    contracts/   ← 20 PDF/image contracts
    reports/     ← 20 PDF/image reports

Accepted file types: .pdf  .png  .jpg  .jpeg  .tiff  .tif  .bmp

Outputs
-------
  models/vectorizer.pkl
  models/classifier.pkl
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from src.ocr_engine import OCREngine
from src.feature_engineering import TFIDFFeatureExtractor
from src.classifier import DocumentClassifier

DATA_DIR   = Path("data/raw")
MODELS_DIR = Path("models")

CATEGORY_MAP = {
    "invoices":  "Invoice",
    "emails":    "Email",
    "contracts": "Contract",
    "reports":   "Report",
}

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def collect_training_data(
    ocr: OCREngine,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    texts:  list[str] = []
    labels: list[str] = []

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

        print(f"\n  [{label}]  {len(files)} file(s) …")
        for fpath in tqdm(files, desc=f"  {label:<10}", unit="file", leave=False):
            try:
                text = ocr.process_file(fpath)
                if text.strip():
                    texts.append(text)
                    labels.append(label)
            except Exception as exc:
                print(f"    ✗ {fpath.name}: {exc}")

    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the StatExtract-Classifier.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Max files per category (e.g. --limit 10 for a quick smoke-test)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 52)
    print("  TRAINING  –  StatExtract Classifier (Person A)")
    if args.limit:
        print(f"  Cap : {args.limit} files per category")
    print("=" * 52)

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

    from collections import Counter
    counts = Counter(labels)
    print(f"\n  Documents collected: {len(texts)}")
    for lbl in sorted(counts):
        print(f"    {lbl:<12} {counts[lbl]}")

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
    print("  Run  python app.py  to start the web server.\n")


if __name__ == "__main__":
    main()
