"""
Phase 4 – Full Pipeline & Handoff to Person B
Chains OCR → Feature Engineering → SVM into a single callable.

Enriched output contract:
  {
    "file"              : str   – original file path
    "label"             : str   – top predicted class (e.g. "INVOICE")
    "confidence"        : float – 0-1 probability of top class
    "is_uncertain"      : bool  – True when confidence < UNCERTAIN_THRESHOLD
    "all_scores"        : list  – all classes ranked highest → lowest
    "summary"           : str   – human-readable one-liner
    "ocr_engine"        : str   – "tesseract" or "easyocr"
    "word_count"        : int   – words extracted by OCR
    "detected_language" : str   – ISO 639-1 code (e.g. "es") or "unknown"
    "language_name"     : str   – human-readable language name
    "is_non_english"    : bool  – True when detected language is not English
    "raw_text"          : str   – full OCR text for Person B
  }
"""

from pathlib import Path

from src.ocr_engine import OCREngine
from src.feature_engineering import TFIDFFeatureExtractor
from src.classifier import DocumentClassifier

try:
    from langdetect import detect as _langdetect
    from langdetect import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ISO 639-1 → human-readable name for the UI
_LANG_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German",  "pt": "Portuguese", "it": "Italian",
    "nl": "Dutch",   "pl": "Polish",    "ru": "Russian",
    "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese",
    "ar": "Arabic",  "ko": "Korean",
}

_MODEL_PATH      = Path("models/classifier.pkl")
_VECTORIZER_PATH = Path("models/vectorizer.pkl")

# Below this confidence the result is flagged as uncertain
UNCERTAIN_THRESHOLD = 0.45


class ClassificationPipeline:
    """End-to-end pipeline:  file path  →  enriched result dict"""

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        vectorizer_path: str | Path = _VECTORIZER_PATH,
        tesseract_cmd: str | None = None,
    ):
        kwargs = {} if tesseract_cmd is None else {"tesseract_cmd": tesseract_cmd}
        self.ocr               = OCREngine(**kwargs)
        self.feature_extractor = TFIDFFeatureExtractor()
        self.classifier        = DocumentClassifier()
        self.model_path        = Path(model_path)
        self.vectorizer_path   = Path(vectorizer_path)

    def load(self) -> None:
        self.feature_extractor.load(self.vectorizer_path)
        self.classifier.load(self.model_path)

    # ── Core prediction ──────────────────────────────────────────────────────

    def predict_category(self, file_path: str | Path) -> dict:
        """
        Full pipeline:  file  →  enriched result dict

        Steps
        -----
        1. OCR engine       → raw_text + engine name + word count  (Phase 1)
        2. TF-IDF           → feature vector                        (Phase 2)
        3. Linear SVM       → all class probabilities               (Phase 3)
        4. Pack result      → handoff dict for Person B             (Phase 4)
        """
        # Phase 1 – Vision
        raw_text: str = self.ocr.process_file(file_path)
        ocr_engine: str = self.ocr.last_engine_used
        word_count: int = self.ocr.last_word_count

        # Language detection (best-effort — graceful fallback)
        detected_language = "unknown"
        if _LANGDETECT_AVAILABLE and len(raw_text.split()) >= 10:
            try:
                detected_language = _langdetect(raw_text)
            except LangDetectException:
                pass
        language_name  = _LANG_NAMES.get(detected_language, detected_language.upper())
        is_non_english = detected_language not in ("en", "unknown")

        # Phase 2 – Feature engineering
        X = self.feature_extractor.transform([raw_text])

        # Phase 3 – Classification (full probability distribution)
        all_scores = self.classifier.predict_all(X)[0]  # first (only) row
        top        = all_scores[0]
        label      = top["label"]
        confidence = top["confidence"]

        # Phase 4 – Pack handoff payload
        article      = "an" if label[0].upper() in "AEIOU" else "a"
        is_uncertain = confidence < UNCERTAIN_THRESHOLD

        return {
            "file":               str(file_path),
            "label":              label.upper(),
            "confidence":         confidence,
            "is_uncertain":       is_uncertain,
            "all_scores":         all_scores,
            "summary":            (
                f"Uncertain — closest match is {label.upper()}."
                if is_uncertain else
                f"This is {article} {label.upper()}."
            ),
            "ocr_engine":         ocr_engine,
            "word_count":         word_count,
            "detected_language":  detected_language,
            "language_name":      language_name,
            "is_non_english":     is_non_english,
            "raw_text":           raw_text,
        }

    # ── Formatted console handoff ────────────────────────────────────────────

    def handoff_to_person_b(self, result: dict) -> dict:
        bar = "=" * 56
        print(f"\n{bar}")
        print("  CLASSIFICATION RESULT  (Person A → Person B)")
        print(bar)
        print(f"  File         : {result['file']}")
        print(f"  Label        : {result['label']}"
              + ("  ⚠ UNCERTAIN" if result["is_uncertain"] else ""))
        print(f"  Confidence   : {result['confidence']:.1%}")
        print(f"  OCR engine   : {result['ocr_engine']}")
        print(f"  Words found  : {result['word_count']}")
        print(f"\n  All scores:")
        for s in result["all_scores"]:
            bar_fill = "█" * int(s["confidence"] * 20)
            print(f"    {s['label']:<18} {s['confidence']:5.1%}  {bar_fill}")
        print(bar)
        print("\n--- RAW TEXT (hand off to Person B) ---\n")
        preview = result["raw_text"][:2000]
        print(preview)
        if len(result["raw_text"]) > 2000:
            print(f"\n  … [{len(result['raw_text'])-2000:,} more characters]")
        print()
        return result
