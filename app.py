"""
app.py – Flask web server for the StatExtract-Classifier.

Routes
------
  GET  /           → serves the drag-and-drop UI
  POST /classify   → accepts a file upload, returns classification JSON

Usage
-----
  python app.py

Then open  http://127.0.0.1:5000  in your browser.

Ensure the model has been trained first:
  python train.py
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.pipeline import ClassificationPipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Load the pipeline once at startup so models stay in memory
pipeline = ClassificationPipeline()
_models_loaded = False


def _ensure_models() -> bool:
    global _models_loaded
    if _models_loaded:
        return True
    model_path = Path("models/classifier.pkl")
    vec_path   = Path("models/vectorizer.pkl")
    if not model_path.exists() or not vec_path.exists():
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
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        result = pipeline.predict_category(tmp_path)

        return jsonify({
            "label":        result["label"],
            "confidence":   round(result["confidence"] * 100, 1),  # percent
            "summary":      result["summary"],
            "text_preview": result["raw_text"][:600].strip(),
        })

    except Exception as exc:
        app.logger.error("Classification failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
