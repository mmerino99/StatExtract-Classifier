"""
app.py – Flask web server for the StatExtract-Classifier.

Routes
------
  GET  /           → serves the drag-and-drop UI
  POST /classify   → accepts a file upload, returns enriched classification JSON

Usage
-----
  python app.py   →   open http://127.0.0.1:5000
"""

import os
import time
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.pipeline import ClassificationPipeline
from src.ocr_engine import LowQualityImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Flag result as low-quality when fewer than this many words were extracted
LOW_QUALITY_WORD_THRESHOLD = 30

pipeline = ClassificationPipeline()
_models_loaded = False


def _ensure_models() -> bool:
    global _models_loaded
    if _models_loaded:
        return True
    if not Path("models/classifier.pkl").exists() or \
       not Path("models/vectorizer.pkl").exists():
        return False
    pipeline.load()
    _models_loaded = True
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    # ── Validate upload ────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file included in the request."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{suffix}'. "
                     f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # ── Check models are ready ─────────────────────────────────────────────
    if not _ensure_models():
        return jsonify({
            "error": "Model not trained yet. Run  python train.py  first."
        }), 503

    # ── Save to temp file, classify, clean up ──────────────────────────────
    tmp_path = None
    t_start  = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        result       = pipeline.predict_category(tmp_path)
        elapsed_ms   = round((time.perf_counter() - t_start) * 1000)

        word_count   = result["word_count"]
        low_quality  = word_count < LOW_QUALITY_WORD_THRESHOLD

        # Round all_scores confidences to 1 decimal percent for the UI
        all_scores_pct = [
            {"label": s["label"], "confidence": round(s["confidence"] * 100, 1)}
            for s in result["all_scores"]
        ]

        return jsonify({
            # ── Core result ──────────────────────────────────────────────
            "label":          result["label"],
            "confidence":     round(result["confidence"] * 100, 1),
            "is_uncertain":   result["is_uncertain"],
            "summary":        result["summary"],
            # ── All class probabilities ───────────────────────────────────
            "all_scores":     all_scores_pct,
            # ── OCR diagnostics ───────────────────────────────────────────
            "ocr_engine":     result["ocr_engine"],
            "word_count":     word_count,
            "text_preview":   result["raw_text"][:600].strip(),
            # ── Quality / timing ──────────────────────────────────────────
            "warning":        low_quality,
            "warning_msg": (
                f"Only {word_count} words were extracted. "
                "The document may contain heavy handwriting, noise, or a low-quality "
                "scan — treat this classification with caution."
            ) if low_quality else None,
            "elapsed_ms":     elapsed_ms,
        })

    except LowQualityImageError as exc:
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        app.logger.error("Classification failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
