"""
Phase 2 – Feature Engineering
Converts raw OCR text into a combined numerical feature matrix:

  [  TF-IDF (8 000 features)  |  Structural features (20 features)  ]

TF-IDF captures statistical word patterns across the training corpus.
Structural features capture hard, unambiguous signals that TF-IDF misses
because they are stripped out during text cleaning (symbols, patterns, layout).

Structural feature groups
-------------------------
  Invoice     : currency symbols ($€£), "invoice", "total", "subtotal",
                "amount due", "bill to", "payment", invoice number patterns
  Email       : @ symbol, "dear", "regards", "subject", "from", "cc",
                "forwarded", "reply"
  Resume      : "experience", "education", "skills", "gpa", "objective",
                "references", "employment", bullet-point density
  Questionnaire: checkbox patterns (□ ☐ [ ]), "please circle", "tick",
                 "Q1 / Q2 / Q3", "strongly agree", "rate", "scale",
                 numbered question patterns

Why structural features matter
-------------------------------
  The word "invoice" scores LOW in TF-IDF if it appears in emails
  ("please see the attached invoice").  But the presence of a $ symbol
  followed by a number, combined with "total" and "due date" in the same
  document, is an unambiguous Invoice signal that TF-IDF cannot capture.
  These 20 extra dimensions give the SVM direct, reliable anchors.
"""

import re
import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


# ── Structural signal definitions ────────────────────────────────────────────

_STRUCTURAL_FEATURES: list[tuple[str, str]] = [
    # ── Invoice signals ───────────────────────────────────────────────────────
    ("inv_currency",   r"[\$€£¥]"),
    ("inv_total",      r"\btotal\b"),
    ("inv_amount_due", r"\bamount\s+due\b|\bbalance\s+due\b|\bdue\s+date\b"),
    ("inv_invoice_kw", r"\binvoice\b"),
    ("inv_bill_to",    r"\bbill\s+to\b|\bship\s+to\b|\bsold\s+to\b"),
    ("inv_subtotal",   r"\bsubtotal\b|\bsub-total\b|\btax\b|\bvat\b"),
    ("inv_number",     r"\binv[\s\-#:]+\d+|\binvoice\s*(no|num|number|#)"),

    # ── Email signals ─────────────────────────────────────────────────────────
    ("email_at",       r"@[a-z0-9]+\.[a-z]{2,}"),
    ("email_dear",     r"\bdear\s+\w+"),
    ("email_regards",  r"\bregards\b|\bsincerely\b|\bbest\s+wishes\b|\bthank\s+you\b"),
    ("email_subject",  r"\bsubject\s*:"),
    ("email_from",     r"\bfrom\s*:|\bto\s*:|\bcc\s*:|\bfwd\b|\bforwarded\b"),

    # ── Resume signals ────────────────────────────────────────────────────────
    ("res_experience", r"\bexperience\b|\bemployment\b|\bwork\s+history\b"),
    ("res_education",  r"\beducation\b|\bdegree\b|\buniversity\b|\bcollege\b|\bgpa\b"),
    ("res_skills",     r"\bskills\b|\bproficient\b|\bprogramming\b|\bcertif"),
    ("res_objective",  r"\bobjective\b|\bsummary\b|\bprofile\b|\breferences\b"),

    # ── Questionnaire signals ─────────────────────────────────────────────────
    ("que_checkbox",   r"[\u25a1\u2610\u2611\u2612]|\[\s*\]|\(\s*\)"),
    ("que_scale",      r"\bstrongly\s+(agree|disagree)\b|\brate\b|\bscale\b|\blikert\b"),
    ("que_question",   r"\bq\s*\d+\b|^\s*\d+[\.\)]\s+\w",),
    ("que_circle",     r"\bplease\s+(circle|tick|check|mark|select)\b"),
]

_FEATURE_NAMES = [name for name, _ in _STRUCTURAL_FEATURES]
_PATTERNS      = [re.compile(pat, re.IGNORECASE | re.MULTILINE)
                  for _, pat in _STRUCTURAL_FEATURES]


# ── Text preprocessor ─────────────────────────────────────────────────────────

class TextPreprocessor:
    """Cleans and normalises a raw text string for TF-IDF."""

    def __init__(self):
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in text.split()
            if tok not in self._stop_words and len(tok) > 2
        ]
        return " ".join(tokens)


# ── Structural feature extractor ──────────────────────────────────────────────

class StructuralFeatureExtractor:
    """
    Produces a dense (n_docs × 20) matrix of binary/count structural signals.
    These are computed on the RAW text before cleaning so that symbols like
    $ @ are preserved.
    """

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(_PATTERNS)), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, pat in enumerate(_PATTERNS):
                matches = pat.findall(text)
                # Use log(1 + count) so a document with 10 matches isn't
                # 10× more "invoicy" than one with 1 match
                matrix[i, j] = np.log1p(len(matches))
        return matrix

    @property
    def feature_names(self) -> list[str]:
        return _FEATURE_NAMES


# ── Combined feature extractor ────────────────────────────────────────────────

class TFIDFFeatureExtractor:
    """
    Combines TF-IDF (8 000 features) with structural signals (20 features).
    The structural features are scaled to the same range as TF-IDF weights
    so neither dominates the SVM decision boundary.
    """

    def __init__(self, max_features: int = 15000, ngram_range: tuple = (1, 2)):
        self.preprocessor  = TextPreprocessor()
        self.structural    = StructuralFeatureExtractor()
        self.scaler        = MaxAbsScaler()   # scales each feature to [-1, 1]
        self.vectorizer    = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # 1+log(tf) — reduces dominance of common words
            min_df=3,                   # ignore OCR noise terms seen in < 3 documents
            strip_accents="unicode",
        )

    def _combine(self, tfidf_matrix, raw_texts: list[str], fit: bool):
        struct = self.structural.transform(raw_texts)
        if fit:
            struct_scaled = self.scaler.fit_transform(struct)
        else:
            struct_scaled = self.scaler.transform(struct)
        struct_sparse = sp.csr_matrix(struct_scaled)
        return sp.hstack([tfidf_matrix, struct_sparse])

    def fit_transform(self, texts: list[str]):
        cleaned = [self.preprocessor.clean(t) for t in texts]
        tfidf   = self.vectorizer.fit_transform(cleaned)
        combined = self._combine(tfidf, texts, fit=True)
        n_tfidf  = tfidf.shape[1]
        log.info("Features: %d TF-IDF + %d structural = %d total",
                 n_tfidf, len(_FEATURE_NAMES), combined.shape[1])
        return combined

    def transform(self, texts: list[str]):
        cleaned = [self.preprocessor.clean(t) for t in texts]
        tfidf   = self.vectorizer.transform(cleaned)
        return self._combine(tfidf, texts, fit=False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "scaler": self.scaler}, f)
        log.info("Vectorizer saved → %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Support old format (vectorizer only) and new format (dict)
        if isinstance(data, dict):
            self.vectorizer = data["vectorizer"]
            self.scaler     = data["scaler"]
        else:
            self.vectorizer = data
        log.info("Vectorizer loaded ← %s", path)
