"""
Phase 4 – Full Pipeline & Handoff to Person B
Chains OCR → Feature Engineering → SVM into a single callable.

Output contract (what Person B receives):
  {
    "file"       : str   – original file path
    "label"      : str   – e.g. "INVOICE", "CONTRACT"
    "confidence" : float – model's confidence (0-1)
    "summary"    : str   – human-readable one-liner
    "raw_text"   : str   – every word the OCR extracted (for date/total parsing)
  }
"""

from pathlib import Path

from src.ocr_engine import OCREngine
from src.feature_engineering import TFIDFFeatureExtractor
from src.classifier import DocumentClassifier

_MODEL_PATH = Path("models/classifier.pkl")
_VECTORIZER_PATH = Path("models/vectorizer.pkl")


class ClassificationPipeline:
    """
    End-to-end pipeline:  file path  →  label + raw text
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        vectorizer_path: str | Path = _VECTORIZER_PATH,
        tesseract_cmd: str | None = None,
    ):
        kwargs = {} if tesseract_cmd is None else {"tesseract_cmd": tesseract_cmd}
        self.ocr = OCREngine(**kwargs)
        self.feature_extractor = TFIDFFeatureExtractor()
        self.classifier = DocumentClassifier()
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)

    def load(self) -> None:
        """Load pre-trained vectorizer and classifier from disk."""
        self.feature_extractor.load(self.vectorizer_path)
        self.classifier.load(self.model_path)

    # ── Core prediction ──────────────────────────────────────────────────────

    def predict_category(self, file_path: str | Path) -> dict:
        """
        Full pipeline:  file  →  result dict

        Steps
        -----
        1. OCR engine   → raw_text  (Phase 1)
        2. TF-IDF       → feature vector  (Phase 2)
        3. Linear SVM   → label + confidence  (Phase 3)
        4. Pack result  → handoff dict for Person B  (Phase 4)
        """
        # Phase 1 – Vision
        raw_text: str = self.ocr.process_file(file_path)

        # Phase 2 – Feature engineering
        X = self.feature_extractor.transform([raw_text])

        # Phase 3 – Classification
        labels, confidences = self.classifier.predict(X)
        label: str = labels[0]
        confidence: float = float(confidences[0])

        # Phase 4 – Pack handoff payload
        article = "an" if label[0] in "AEIOU" else "a"
        result = {
            "file": str(file_path),
            "label": label.upper(),
            "confidence": confidence,
            "summary": f"This is {article} {label.upper()}.",
            "raw_text": raw_text,
        }
        return result

    # ── Formatted console handoff ────────────────────────────────────────────

    def handoff_to_person_b(self, result: dict) -> dict:
        """
        Print a clear handoff block so Person B knows exactly what to parse.
        Returns the same result dict for programmatic use.
        """
        bar = "=" * 52
        print(f"\n{bar}")
        print("  CLASSIFICATION RESULT  (Person A → Person B)")
        print(bar)
        print(f"  File       : {result['file']}")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  Summary    : {result['summary']}")
        print(bar)
        print("\n--- RAW TEXT (hand off to Person B for data extraction) ---\n")
        preview = result["raw_text"][:2000]
        print(preview)
        remainder = len(result["raw_text"]) - len(preview)
        if remainder > 0:
            print(f"\n  … [{remainder:,} more characters not shown]")
        print()
        return result
