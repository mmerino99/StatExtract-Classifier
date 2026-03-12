"""
train.py – Build and save the TF-IDF vectorizer + Linear SVM classifier
         using the RVL-CDIP Small Kaggle dataset.

Usage
-----
  python train.py                        # use all available images
  python train.py --limit 150            # cap at 150 images per class (faster)
  python train.py --split val            # train on val split instead of train

Expected data layout (after running setup_data.py or manual Kaggle download)
--------------------
  data/rvl-cdip-small/
    images/          ← .tif document images in sub-folders
    labels/
      train.txt      ← lines: "images/.../file.tif <class_id>"
      val.txt
      test.txt

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
from src.classifier import DocumentClassifier, RVLCDIP_LABELS

DATASET_DIR = Path("data/rvl-cdip-small")
LABELS_DIR  = DATASET_DIR / "labels"
IMAGES_DIR  = DATASET_DIR / "images"
MODELS_DIR  = Path("models")


# ── RVL-CDIP label file loader ────────────────────────────────────────────────

def parse_label_file(label_file: Path) -> list[tuple[Path, str]]:
    """
    Read a RVL-CDIP label file and return (image_path, class_name) pairs.
    Each line has the format:  images/subdir/.../file.tif  <int_label>
    """
    entries: list[tuple[Path, str]] = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel_path, label_id = parts[0], int(parts[1])
            abs_path = DATASET_DIR / rel_path
            class_name = RVLCDIP_LABELS.get(label_id, f"Unknown-{label_id}")
            entries.append((abs_path, class_name))
    return entries


def collect_training_data(
    ocr: OCREngine,
    entries: list[tuple[Path, str]],
    limit: int | None,
) -> tuple[list[str], list[str]]:
    """
    OCR every image in entries (optionally capped at `limit` per class)
    and return (texts, labels).
    """
    # Group by class so we can apply the per-class cap cleanly
    from collections import defaultdict
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path, label in entries:
        by_class[label].append(path)

    texts: list[str] = []
    labels: list[str] = []

    for class_name in sorted(by_class):
        files = by_class[class_name]
        if limit:
            files = files[:limit]

        print(f"\n  [{class_name}]  {len(files)} image(s) …")
        for fpath in tqdm(files, desc=f"  {class_name[:20]:<20}", unit="img", leave=False):
            if not fpath.exists():
                continue
            try:
                text = ocr.process_file(fpath)
                if text.strip():
                    texts.append(text)
                    labels.append(class_name)
            except Exception as exc:
                print(f"    ✗ {fpath.name}: {exc}")

    return texts, labels


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the StatExtract-Classifier on RVL-CDIP Small.")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Which RVL-CDIP split to train on (default: train)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Max images per class (useful for quick experiments, e.g. --limit 100)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 52)
    print("  TRAINING  –  StatExtract Classifier (Person A)")
    print(f"  Dataset : RVL-CDIP Small  |  split: {args.split}")
    if args.limit:
        print(f"  Cap     : {args.limit} images per class")
    print("=" * 52)

    # ── Locate label file ──────────────────────────────────────────────────
    label_file = LABELS_DIR / f"{args.split}.txt"
    if not label_file.exists():
        print(
            f"\n  ERROR: Label file not found: {label_file}\n"
            "  Run  python setup_data.py  to download the dataset first.\n"
        )
        sys.exit(1)

    entries = parse_label_file(label_file)
    print(f"\n  Label file loaded: {len(entries)} entries across 16 classes.")

    # ── Phase 1: OCR ──────────────────────────────────────────────────────
    print("\n[Phase 1] Extracting text via OCR …")
    ocr = OCREngine()
    texts, labels = collect_training_data(ocr, entries, args.limit)

    if len(texts) < 16:
        print(
            f"\n  ERROR: Only {len(texts)} usable document(s) extracted.\n"
            "  Check that the images/ folder is present and Tesseract is installed.\n"
        )
        sys.exit(1)

    from collections import Counter
    counts = Counter(labels)
    print(f"\n  Documents with extracted text: {len(texts)}")
    for cls in sorted(counts):
        print(f"    {cls:<28} {counts[cls]}")

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
