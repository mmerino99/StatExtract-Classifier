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

import json
import os
import re
import time
import tempfile
import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.pipeline import ClassificationPipeline
from src.ocr_engine import LowQualityImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

# Every upload's OCR text is saved here so you can inspect what Tesseract saw
DEBUG_OCR_DIR = Path("debug_ocr")
DEBUG_OCR_DIR.mkdir(exist_ok=True)

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


@app.route("/metrics")
def metrics():
    report_path = Path("models/training_report.json")
    if not report_path.exists():
        report = None
    else:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    return render_template("metrics.html", report=report)


@app.route("/classify", methods=["POST"])
def classify():
    # ── Validate upload ────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file included in the request."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    original_filename = file.filename  # save before request context might close

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

        # ── Save OCR debug log ─────────────────────────────────────────────
        try:
            ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name  = re.sub(r"[^\w\-.]", "_", original_filename)
            debug_file = DEBUG_OCR_DIR / f"{ts}_{safe_name}.txt"
            inv = result.get("invoice_fields") or {}
            lines = [
                "=== OCR DEBUG ===",
                f"File      : {original_filename}",
                f"Engine    : {result['ocr_engine']}",
                f"Words     : {result['word_count']}",
                f"Label     : {result['label']}",
            ]
            if inv:
                lines += [
                    "--- Invoice fields ---",
                    f"  invoice_number  : {inv.get('invoice_number')}",
                    f"  invoice_date    : {inv.get('invoice_date')}",
                    f"  due_date        : {inv.get('due_date')}",
                    f"  issuer_name     : {inv.get('issuer_name')}",
                    f"  recipient_name  : {inv.get('recipient_name')}",
                    f"  total_amount    : {inv.get('total_amount')}",
                ]
            lines += ["=================", "", result["raw_text"]]
            debug_file.write_text("\n".join(lines), encoding="utf-8")
        except Exception as _dbg_exc:
            app.logger.debug("OCR debug log failed: %s", _dbg_exc)

        word_count      = result["word_count"]
        low_quality     = word_count < LOW_QUALITY_WORD_THRESHOLD
        is_non_english  = result.get("is_non_english", False)
        language_name   = result.get("language_name", "Unknown")

        # Round all_scores confidences to 1 decimal percent for the UI
        all_scores_pct = [
            {"label": s["label"], "confidence": round(s["confidence"] * 100, 1)}
            for s in result["all_scores"]
        ]

        # Build warning message
        warning = low_quality or is_non_english
        warning_parts = []
        if low_quality:
            warning_parts.append(
                f"Only {word_count} words were extracted. "
                "The document may contain heavy handwriting, noise, or a low-quality scan."
            )
        if is_non_english:
            warning_parts.append(
                f"⚠ NON-ENGLISH DOCUMENT DETECTED ({language_name}). "
                "This model was trained exclusively on English documents. "
                "The result shown below should NOT be relied upon."
            )

        return jsonify({
            # ── Core result ──────────────────────────────────────────────
            "label":              result["label"],
            "confidence":         round(result["confidence"] * 100, 1),
            "is_uncertain":       result["is_uncertain"],
            "summary":            result["summary"],
            # ── All class probabilities ───────────────────────────────────
            "all_scores":         all_scores_pct,
            # ── OCR diagnostics ───────────────────────────────────────────
            "ocr_engine":         result["ocr_engine"],
            "word_count":         word_count,
            "text_preview":       result["raw_text"].strip(),
            # ── Invoice structured extraction (invoice-only) ──────────────
            "invoice_fields":     result.get("invoice_fields"),
            # ── Language ─────────────────────────────────────────────────
            "detected_language":  result.get("detected_language", "unknown"),
            "language_name":      language_name,
            "is_non_english":     is_non_english,
            # ── Quality / timing ──────────────────────────────────────────
            "warning":            warning,
            "warning_msg":        " ".join(warning_parts) if warning_parts else None,
            "elapsed_ms":         elapsed_ms,
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
    # Disable the Flask auto‑reloader on Windows to avoid constant restarts,
    # which can cause "Failed to fetch" errors in the browser when a request
    # is interrupted mid‑processing.
    app.run(debug=True, port=5000, use_reloader=False)
