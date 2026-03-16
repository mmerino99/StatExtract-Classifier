"""
predict.py – Classify a single document and hand off results to Person B.

Usage
-----
  python predict.py <path_to_file>

Examples
--------
  python predict.py data/raw/invoices/invoice_001.pdf
  python predict.py my_scan.png

Output
------
  Prints the category label, confidence score, and the full extracted text.
  The dict returned by pipeline.predict_category() is the programmatic
  handoff payload for Person B.
"""

import sys
import json
from pathlib import Path

from src.pipeline import ClassificationPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: file not found – {file_path}")
        sys.exit(1)

    pipeline = ClassificationPipeline()

    try:
        pipeline.load()
        print("  Models loaded.")
    except FileNotFoundError:
        print(
            "\n  ERROR: Trained models not found.\n"
            "  Run  python train.py  first to build the classifier.\n"
        )
        sys.exit(1)

    result = pipeline.predict_category(file_path)
    pipeline.handoff_to_person_b(result)

    # Optionally save the handoff payload as JSON for Person B to consume
    out_json = Path("data/processed") / (file_path.stem + "_result.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "raw_text"}  # keep JSON light
    payload["raw_text_preview"] = result["raw_text"][:500]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Result JSON saved → {out_json}")


if __name__ == "__main__":
    main()
