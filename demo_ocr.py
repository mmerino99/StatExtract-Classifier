"""
demo_ocr.py – Your "First Step" script.
Run this BEFORE training to verify that Tesseract + OpenCV are working.

Usage
-----
  python demo_ocr.py <path_to_any_pdf_or_image>

Example
-------
  python demo_ocr.py data/raw/invoices/invoice_001.pdf

What it does
------------
  1. Loads the file
  2. Applies the full OpenCV pre-processing (grayscale → deskew → binarise)
  3. Runs Tesseract
  4. Prints the extracted text to the console
  5. Saves a debug image so you can visually inspect the preprocessing output
"""

import sys
import cv2
from pathlib import Path

from src.ocr_engine import OCREngine


def demo(file_path: str) -> None:
    path = Path(file_path)
    print(f"\n{'='*52}")
    print(f"  OCR DEMO  –  {path.name}")
    print(f"{'='*52}\n")

    ocr = OCREngine()

    # If it's a PDF, process the first page as a preview
    if path.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        page = doc[0]
        img = ocr._pdf_page_to_array(page)
        doc.close()
        print(f"  Page size (pixels): {img.shape[1]} × {img.shape[0]}")

        # Save the preprocessed image for visual inspection
        processed = ocr.preprocess(img)
        debug_path = Path("data/processed") / (path.stem + "_debug_preprocessed.png")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), processed)
        print(f"  Preprocessed image saved → {debug_path}")
        print("  (Open it to verify the binarisation looks clean before OCR)\n")

        text = ocr._extract_text(img)
    else:
        text = ocr.process_file(path)

    print("--- EXTRACTED TEXT ---\n")
    print(text if text.strip() else "[No text extracted – check Tesseract installation]")
    print(f"\n--- END ({len(text)} characters extracted) ---\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_ocr.py <path_to_file>")
        print("Example: python demo_ocr.py data/raw/invoices/invoice_001.pdf")
        sys.exit(0)
    demo(sys.argv[1])
