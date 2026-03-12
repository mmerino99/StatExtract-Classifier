"""
Phase 3 – Statistical Model (Linear SVM)
Trains a Support Vector Machine on TF-IDF feature vectors and predicts
the document category for unseen documents.

Why a Linear SVM?
  Text feature spaces are extremely high-dimensional (thousands of TF-IDF
  features).  In such spaces a linear decision boundary is usually sufficient
  and is mathematically efficient – training is O(n·features) vs O(n²) for
  RBF kernels.  Combined with TF-IDF, the model ignores common language and
  focuses on the *statistical significance* of category-specific keywords.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

CATEGORY_LABELS: list[str] = ["Invoice", "Email", "Questionnaire", "Resume"]


class DocumentClassifier:
    """Linear SVM document classifier with label encoding and persistence."""

    def __init__(self, C: float = 1.0):
        # probability=True enables predict_proba (needed for confidence scores)
        self.model = SVC(kernel="linear", C=C, probability=True, random_state=42)
        self.label_encoder = LabelEncoder()

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X, y: list[str], test_size: float = 0.2) -> float:
        """
        Train on X (sparse TF-IDF matrix) and string labels y.
        Prints a full classification report.
        Returns test-set accuracy.
        """
        y_enc = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=test_size,
            random_state=42,
            stratify=y_enc,        # keep class proportions in both splits
        )

        print(f"  Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples …")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("\n  === Classification Report ===")
        print(
            classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
            )
        )
        print(f"  Overall Accuracy : {acc:.2%}\n")
        return acc

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (labels_array, confidence_array) for each row in X.
        Confidence is the probability assigned to the winning class.
        """
        y_enc = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(y_enc)
        proba = self.model.predict_proba(X)
        confidence = np.max(proba, axis=1)
        return labels, confidence

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "encoder": self.label_encoder}, f)
        print(f"  Classifier saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoder = data["encoder"]
        print(f"  Classifier loaded ← {path}")
