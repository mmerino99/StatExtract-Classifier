"""
Phase 1 – Vision Engine
Converts a PDF or image file into a clean string of text using:
  1. PyMuPDF  : renders each PDF page to a high-resolution pixel buffer
  2. OpenCV   : grayscale → upscale → deskew → CLAHE → sharpen →
                adaptive threshold → denoise
  3. Tesseract: extracts text from the preprocessed image

Preprocessing improvements over v1:
  - Upscaling   : images narrower than MIN_WIDTH are scaled up before OCR.
                  Tesseract accuracy drops sharply below ~150 px per character;
                  upscaling to 2–3× recovers most garbled letters.
  - CLAHE       : Contrast Limited Adaptive Histogram Equalisation normalises
                  uneven lighting (e.g. scanned pages with shadowed corners).
  - Sharpening  : An unsharp-mask kernel makes blurry letterforms crisper.
  - Adaptive    : Adaptive thresholding handles pages where background brightness
    threshold     varies across the image, outperforming global Otsu on scans.
"""

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF
from pathlib import Path


# ── Tesseract path on Windows ────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Upscale images whose shortest dimension is below this threshold (pixels)
MIN_WIDTH = 1400


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
        for page in doc:
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
        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img

    # ── Pre-processing pipeline ──────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full OpenCV pipeline:
          1. Grayscale            – strip colour noise
          2. Upscale              – bring low-res scans up to Tesseract's sweet spot
          3. Deskew               – correct rotated scans
          4. CLAHE                – fix uneven lighting across the page
          5. Unsharp mask         – sharpen blurry letterforms
          6. Adaptive threshold   – robust binarisation under varied brightness
          7. Denoise              – remove salt-and-pepper artefacts
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = self._upscale(gray)
        gray = self._deskew(gray)
        gray = self._clahe(gray)
        gray = self._sharpen(gray)
        binary = self._adaptive_threshold(gray)
        denoised = cv2.fastNlMeansDenoising(binary, h=15)
        return denoised

    # ── Individual steps ─────────────────────────────────────────────────────

    def _upscale(self, gray: np.ndarray) -> np.ndarray:
        """
        Scale up images that are too small for Tesseract to read accurately.
        Tesseract performs best when text is ~30–40 px tall; most scanned
        documents need to be at least 1400 px wide to meet that threshold.
        """
        h, w = gray.shape
        if w >= MIN_WIDTH:
            return gray
        scale = MIN_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimate skew angle via minAreaRect on dark-pixel coordinates,
        then rotate back to 0°. Skips rotation for angles < 0.5°.
        """
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 50:
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.5:
            return gray

        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalisation.
        Divides the image into tiles and equalises each independently,
        which corrects shadowed or unevenly lit scans without blowing out
        bright regions (unlike plain histogram equalisation).
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """
        Unsharp mask: subtract a blurred version from the original to
        amplify high-frequency edges, making blurry letterforms crisper.
        """
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Adaptive (local) thresholding: each pixel's threshold is computed
        from the mean of its neighbourhood (blockSize x blockSize window).
        Handles pages where background brightness varies — much more robust
        than global Otsu on real-world scanned documents.
        """
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,   # neighbourhood window — tune up if text is large
            C=15,           # constant subtracted from mean — tune up to remove noise
        )

    def _extract_text(self, image: np.ndarray) -> str:
        """Apply preprocessing and run Tesseract on one image."""
        processed = self.preprocess(image)
        # --oem 3  → best available engine (LSTM + legacy combined)
        # --psm 3  → fully automatic page segmentation (better than psm 6
        #            for mixed-layout documents like invoices and forms)
        config = "--oem 3 --psm 3"
        text = pytesseract.image_to_string(processed, config=config)
        return text.strip()
