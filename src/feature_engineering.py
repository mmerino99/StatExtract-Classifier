"""
Phase 2 – Feature Engineering
Converts raw extracted text into a TF-IDF numerical matrix that the SVM can learn from.

Pipeline per document:
  1. Lowercase
  2. Remove non-alphabetic characters
  3. Tokenise
  4. Drop stop-words  (words like "the", "is", "at" that carry no category signal)
  5. Lemmatise        ("billing" / "billed" / "bills" → "bill")
  6. TF-IDF           (rewards words that are rare across the corpus = category-specific)
"""

import re
import pickle
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK assets once (silent if already present)
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


class TextPreprocessor:
    """Cleans and normalises a raw text string."""

    def __init__(self):
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)     # keep only letters + spaces
        tokens = text.split()
        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self._stop_words and len(tok) > 2
        ]
        return " ".join(tokens)


class TFIDFFeatureExtractor:
    """
    Wraps sklearn's TfidfVectorizer together with TextPreprocessor.

    Why TF-IDF?
      TF  (term frequency)        – how often a word appears in *this* document
      IDF (inverse doc frequency) – penalises words that appear in *every* document
      The product highlights words that are statistically distinctive to a category
      (e.g. "liability" in contracts, "invoice" in invoices).
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,   # unigrams + bigrams ("due date", "total amount")
            sublinear_tf=True,         # replace tf with 1+log(tf) – reduces dominance of very frequent words
        )

    # ── Fit / transform ──────────────────────────────────────────────────────

    def fit_transform(self, texts: list[str]):
        """Fit the vectorizer on the training corpus and transform it."""
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, texts: list[str]):
        """Transform new texts using the already-fitted vectorizer."""
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"  Vectorizer saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"  Vectorizer loaded ← {path}")
