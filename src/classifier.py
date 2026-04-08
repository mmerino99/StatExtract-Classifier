"""
Phase 3 – Statistical Model (Linear SVM)
Trains a Support Vector Machine on the combined TF-IDF + structural feature
matrix and predicts the document category for unseen documents.

Why a Linear SVM?
  Text feature spaces are extremely high-dimensional (thousands of TF-IDF
  features).  In such spaces a linear decision boundary is usually sufficient
  and is mathematically efficient – training is O(n·features) vs O(n²) for
  RBF kernels.  Combined with TF-IDF, the model ignores common language and
  focuses on the *statistical significance* of category-specific keywords.

Improvements over baseline
  • GridSearchCV finds the best regularisation parameter C automatically.
  • Cross-validation (5-fold) gives a reliable accuracy estimate even with
    small datasets (avoids lucky/unlucky 80/20 splits).
  • The combined feature matrix (TF-IDF + structural) is used so the SVM can
    leverage hard signals like $ symbols, @ addresses, checkbox patterns.
"""

import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

CATEGORY_LABELS: list[str] = ["Invoice", "Email", "Questionnaire", "Resume"]

# C values to try during grid search — covers underfitting (0.01) to
# overfitting (100); best value is selected by cross-validation
_C_GRID = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]


class DocumentClassifier:
    """Linear SVM document classifier with label encoding and persistence."""

    def __init__(self, C: float = 1.0):
        # probability=True enables predict_proba (needed for confidence scores)
        self.model = SVC(kernel="linear", C=C, probability=True, random_state=42)
        self.label_encoder = LabelEncoder()

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X, y: list[str], test_size: float = 0.2,
              grid_search: bool = True) -> dict:
        """
        Train on X and string labels y.
        When grid_search=True (default) runs 5-fold CV over _C_GRID to pick
        the best regularisation strength before reporting test-set metrics.

        Returns a metrics dict:
          {
            "accuracy":     float,          overall test-set accuracy (or NaN)
            "cv_accuracy":  float | None,   best CV score from GridSearchCV
            "best_C":       float | None,   regularisation parameter chosen
            "train_samples": int,
            "test_samples":  int,
            "per_class":    {label: {precision, recall, f1_score, support}}
          }
        """
        y_enc = self.label_encoder.fit_transform(y)

        n_classes = len(set(y_enc))
        min_class = min(np.bincount(y_enc))

        _nan_result: dict = {
            "accuracy": float("nan"),
            "cv_accuracy": None,
            "best_C": None,
            "train_samples": len(y_enc),
            "test_samples": 0,
            "per_class": {},
        }

        # Need at least 2 samples per class for stratified split
        if min_class < 2:
            print("  ⚠  Some classes have only 1 sample — skipping train/test split.")
            print("     Add more training documents for reliable evaluation.")
            self.model.fit(X, y_enc)
            return _nan_result

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=test_size,
            random_state=42,
            stratify=y_enc,
        )

        cv_accuracy: float | None = None
        best_C: float | None = None

        if grid_search and X_train.shape[0] >= n_classes * 5:
            # 5-fold CV requires at least 5 samples per class in the training set
            cv_folds = min(5, min(np.bincount(y_train)))
            print(f"  Running GridSearchCV (C ∈ {_C_GRID}, {cv_folds}-fold CV) …")
            gs = GridSearchCV(
                SVC(kernel="linear", probability=True, random_state=42),
                param_grid={"C": _C_GRID},
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring="accuracy",
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            best_C      = float(gs.best_params_["C"])
            cv_accuracy = float(gs.best_score_)
            print(f"  Best C = {best_C}  (CV accuracy = {cv_accuracy:.2%})")
            self.model = SVC(kernel="linear", C=best_C, probability=True, random_state=42)
        else:
            print(f"  Skipping grid search (not enough samples). Using C={self.model.C}")

        print(f"  Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} …")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc    = float(accuracy_score(y_test, y_pred))

        report_str  = classification_report(y_test, y_pred,
                                            target_names=self.label_encoder.classes_)
        report_dict = classification_report(y_test, y_pred,
                                            target_names=self.label_encoder.classes_,
                                            output_dict=True)

        print("\n  === Classification Report ===")
        print(report_str)
        print(f"  Overall Accuracy : {acc:.2%}\n")

        # Keep only per-class rows (drop 'accuracy', 'macro avg', 'weighted avg')
        per_class = {
            label: {
                "precision": round(vals["precision"], 4),
                "recall":    round(vals["recall"], 4),
                "f1_score":  round(vals["f1-score"], 4),
                "support":   int(vals["support"]),
            }
            for label, vals in report_dict.items()
            if label in self.label_encoder.classes_
        }

        return {
            "accuracy":      round(acc, 4),
            "cv_accuracy":   round(cv_accuracy, 4) if cv_accuracy is not None else None,
            "best_C":        best_C,
            "train_samples": int(X_train.shape[0]),
            "test_samples":  int(X_test.shape[0]),
            "per_class":     per_class,
        }

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

    def predict_all(self, X) -> list[list[dict]]:
        """
        Return all class scores for each row in X, sorted highest → lowest.
        Example for one document:
          [
            {"label": "Invoice",       "confidence": 0.82},
            {"label": "Email",         "confidence": 0.10},
            {"label": "Questionnaire", "confidence": 0.05},
            {"label": "Resume",        "confidence": 0.03},
          ]
        """
        proba = self.model.predict_proba(X)
        classes = self.label_encoder.classes_
        results = []
        for row in proba:
            scores = sorted(
                [{"label": classes[i], "confidence": float(row[i])} for i in range(len(classes))],
                key=lambda x: x["confidence"],
                reverse=True,
            )
            results.append(scores)
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "encoder": self.label_encoder}, f)
        log.info("Classifier saved → %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoder = data["encoder"]
        log.info("Classifier loaded ← %s", path)
