"""
Phase 1 – Vision Engine
Converts a PDF or image file into a clean string of text using:
  1. PyMuPDF  : renders each PDF page to a high-resolution pixel buffer
  2. OpenCV   : grayscale → deskew → Otsu binarisation → denoise
  3. Tesseract: extracts text from the preprocessed image
"""

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF
from pathlib import Path


# ── Tesseract path on Windows ────────────────────────────────────────────────
# If Tesseract is not on your PATH, set the full path here, e.g.:
#   TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class OCREngine:
    """Handles the full OCR pipeline from raw file to extracted text string."""

    def __init__(self, tesseract_cmd: str = TESSERACT_CMD, dpi: int = 300):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.dpi = dpi

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
        """Render every page of a PDF and concatenate extracted text."""
        doc = fitz.open(str(pdf_path))
        page_texts = []
        for page_index, page in enumerate(doc):
            img = self._pdf_page_to_array(page)
            text = self._extract_text(img)
            page_texts.append(text)
        doc.close()
        return "\n\n".join(page_texts)

    def _process_image_file(self, image_path: Path) -> str:
        """Load an image file and extract text."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise IOError(f"OpenCV could not read: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._extract_text(rgb)

    def _pdf_page_to_array(self, page: fitz.Page) -> np.ndarray:
        """Render a PDF page to an RGB numpy array at self.dpi."""
        zoom = self.dpi / 72          # 72 is the default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img

    # ── Pre-processing pipeline ──────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full OpenCV pipeline that maximises OCR accuracy:
          1. Grayscale       – removes colour noise
          2. Deskew          – corrects rotated scans
          3. Otsu binarise   – hard black-on-white contrast
          4. Fast denoise    – removes salt-and-pepper artefacts
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = self._deskew(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        return denoised

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimate skew angle via minAreaRect on dark-pixel coordinates,
        then rotate back to 0°.  Skips rotation for angles < 0.5°.
        """
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 50:          # too few dark pixels – nothing to measure
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle        # map from (-90,0] to (0,45]

        if abs(angle) < 0.5:          # negligible skew – skip expensive warpAffine
            return gray

        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def _extract_text(self, image: np.ndarray) -> str:
        """Apply preprocessing and run Tesseract on one image."""
        processed = self.preprocess(image)
        # --oem 3  → best available OCR engine (LSTM + legacy)
        # --psm 6  → assume a uniform block of text
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config)
        return text.strip()
