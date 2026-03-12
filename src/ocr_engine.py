"""
Phase 1 – Vision Engine
Converts a PDF or image file into a clean string of text using a
dual-engine OCR strategy:

  Primary  : Tesseract  – fast, excellent on clean printed text
  Fallback : EasyOCR    – deep-learning model, handles handwriting and
                          mixed printed/handwritten documents that defeat
                          Tesseract

Engine selection logic
----------------------
  Tesseract runs first.  If it extracts fewer than MIN_WORDS_THRESHOLD
  words, the page is assumed to contain significant handwriting (or
  very poor scan quality) and EasyOCR is called instead.

OpenCV preprocessing pipeline (applied before both engines)
-----------------------------------------------------------
  1. Grayscale            – strip colour noise
  2. Upscale              – bring low-res scans up to Tesseract's sweet spot
  3. Deskew               – correct rotated scans
  4. CLAHE                – fix uneven lighting across the page
  5. Unsharp mask         – sharpen blurry letterforms
  6. Adaptive threshold   – robust binarisation under varied brightness
  7. Denoise              – remove salt-and-pepper artefacts

  For EasyOCR the same grayscale+upscale+deskew+CLAHE steps are applied
  but the final hard binarisation is skipped — EasyOCR's internal network
  performs its own thresholding and aggressive binarisation hurts it.
"""

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF
from pathlib import Path

# ── Tesseract path on Windows ────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Upscale images whose width is below this threshold before OCR
MIN_WIDTH = 1400

# If Tesseract extracts fewer words than this, assume handwriting and use EasyOCR
MIN_WORDS_THRESHOLD = 20


class OCREngine:
    """
    Dual-engine OCR: Tesseract for printed text, EasyOCR fallback for handwriting.
    """

    def __init__(self, tesseract_cmd: str = TESSERACT_CMD, dpi: int = 300):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self._easy_reader = None   # lazy-loaded on first use (heavy model)

    # ── Public API ───────────────────────────────────────────────────────────

    def process_file(self, file_path: str | Path) -> str:
        """Auto-detect file type and return the extracted text."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() == ".pdf":
            return self._process_pdf(path)
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            return self._process_image_file(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _process_pdf(self, pdf_path: Path) -> str:
        doc = fitz.open(str(pdf_path))
        page_texts = []
        for page in doc:
            img = self._pdf_page_to_array(page)
            text = self._extract_text(img)
            page_texts.append(text)
        doc.close()
        return "\n\n".join(page_texts)

    def _process_image_file(self, image_path: Path) -> str:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise IOError(f"OpenCV could not read: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._extract_text(rgb)

    def _pdf_page_to_array(self, page: fitz.Page) -> np.ndarray:
        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

    # ── Dual-engine text extraction ──────────────────────────────────────────

    def _extract_text(self, image: np.ndarray) -> str:
        """
        Try Tesseract first.  If it returns too few words (indicating the
        page is mostly handwritten or very degraded), fall back to EasyOCR.
        """
        # Step 1 – shared preprocessing (good for both engines)
        gray = self._preprocess_for_tesseract(image)

        # Step 2 – Tesseract attempt
        config = "--oem 1 --psm 3"   # oem 1 = LSTM only (better on degraded text)
        tess_text = pytesseract.image_to_string(gray, config=config).strip()

        word_count = len(tess_text.split())
        if word_count >= MIN_WORDS_THRESHOLD:
            return tess_text

        # Step 3 – EasyOCR fallback (handwriting / poor scans)
        print(f"    [OCR] Tesseract returned only {word_count} words — "
              "switching to EasyOCR (handwriting mode) …")
        return self._extract_with_easyocr(image)

    # ── Tesseract preprocessing ───────────────────────────────────────────────

    def _preprocess_for_tesseract(self, image: np.ndarray) -> np.ndarray:
        """
        Full pipeline optimised for printed/mixed text:
          grayscale → upscale → deskew → CLAHE → sharpen →
          adaptive threshold → denoise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = self._upscale(gray)
        gray = self._deskew(gray)
        gray = self._clahe(gray)
        gray = self._sharpen(gray)
        binary = self._adaptive_threshold(gray)
        return cv2.fastNlMeansDenoising(binary, h=15)

    def _preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """
        Lighter pipeline for EasyOCR — upscale + deskew + CLAHE only.
        Hard binarisation is deliberately skipped: EasyOCR's neural network
        performs its own internal thresholding and aggressive pre-binarisation
        destroys the stroke-width variation that handwriting recognition relies on.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = self._upscale(gray)
        gray = self._deskew(gray)
        gray = self._clahe(gray)
        # Return as 3-channel image: EasyOCR expects BGR or RGB
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ── EasyOCR engine ────────────────────────────────────────────────────────

    def _get_easy_reader(self):
        """Lazy-load the EasyOCR reader (downloads ~500 MB model on first run)."""
        if self._easy_reader is None:
            try:
                import easyocr
                print("    [EasyOCR] Loading model (first load may take a minute) …")
                # gpu=False ensures it works on machines without CUDA
                self._easy_reader = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                raise ImportError(
                    "EasyOCR is not installed.\n"
                    "Run:  pip install easyocr"
                )
        return self._easy_reader

    def _extract_with_easyocr(self, image: np.ndarray) -> str:
        """Run EasyOCR on the lightly preprocessed image."""
        reader = self._get_easy_reader()
        prepped = self._preprocess_for_easyocr(image)
        results = reader.readtext(prepped, detail=0, paragraph=True)
        return "\n".join(results).strip()

    # ── Individual preprocessing steps ───────────────────────────────────────

    def _upscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w >= MIN_WIDTH:
            return gray
        scale = MIN_WIDTH / w
        return cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 50:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15,
        )

    # ── Legacy public method (used by demo_ocr.py) ────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Expose the Tesseract preprocessing pipeline for debug/demo use."""
        return self._preprocess_for_tesseract(image)
