"""
Phase 1 – Vision Engine
Converts a PDF or image file into a clean string of text.

Strategy (in order of preference)
-----------------------------------
1. Native PDF text layer  – if the PDF already has embedded text (digital PDF),
   extract it directly with PyMuPDF.  This is 100 % accurate and instant.
2. Tesseract OCR           – applied to scanned / image-only PDFs and images.
   Excellent on clean printed text.
3. EasyOCR fallback        – deep-learning model activated when Tesseract returns
   too few words.  Handles handwriting and heavily degraded scans.

OpenCV preprocessing pipeline (applied before Tesseract)
---------------------------------------------------------
  1. Grayscale            – strip colour noise
  2. Upscale to ≥1800 px  – bring low-res scans up to Tesseract's sweet spot
  3. Border pad           – prevent edge-clipping artifacts
  4. Deskew               – correct rotated scans (up to ±15°)
  5. Denoise (bilateral)  – smooth noise while preserving edges
  6. CLAHE                – fix uneven lighting across the page
  7. Unsharp mask         – sharpen blurry letterforms
  8. Morphological open   – break apart characters joined by ink smears
  9. Otsu binarisation    – global threshold (works well for clean pages)
     or adaptive thresh   – local threshold (fallback for shadow / gradients)
 10. Final denoise        – remove remaining salt-and-pepper

For EasyOCR the same grayscale+upscale+deskew+CLAHE steps are applied
but the final hard binarisation is skipped — EasyOCR's internal network
performs its own thresholding and aggressive binarisation hurts it.
"""

import re
import logging
import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF
from pathlib import Path

log = logging.getLogger(__name__)

# ── Tesseract path on Windows ────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Upscale images whose shortest side is below this width
MIN_WIDTH = 1800

# Minimum words Tesseract must return before we accept it (else try EasyOCR)
MIN_WORDS_THRESHOLD = 20

# Minimum usable words from either engine before we give up
MIN_USABLE_WORDS = 5

# Native PDF text: if the PDF already has ≥ this many chars, skip OCR entirely
NATIVE_TEXT_MIN_CHARS = 50

# A page is considered blank when this fraction of pixels are near-white/black
BLANK_WHITE_RATIO = 0.998


# ── OCR error corrections ─────────────────────────────────────────────────────

def _fix_ocr_errors(text: str) -> str:
    """
    Correct systematic Tesseract character-substitution errors that
    are consistent across documents and safe to fix globally.

    Key corrections
    ---------------
      |  → I    isolated pipe (sans-serif capital-I confusion)
      rn → m    two-glyph split of 'm' at low resolution
      vv → w    two-glyph split of 'w'
      0  → O    zero in alphabetic context
      l  → 1    lowercase-l in purely numeric strings  (e.g. "l23" → "123")
      S  → $    isolated S before number in currency context
      ©  → (c)  copyright char artefact
      fi/fl ligature normalisations
    """
    # Isolated pipe → capital I
    text = re.sub(r'(?<![A-Za-z0-9])\|(?![A-Za-z0-9])', 'I', text)
    text = re.sub(r'\|([a-z])', r'I\1', text)

    # Two-glyph letter splits
    text = re.sub(r'\brn\b', 'm', text)
    text = re.sub(r'(?<=[a-z])rn(?=[a-z])', 'm', text)
    text = re.sub(r'(?<=[a-z])vv(?=[a-z])', 'w', text)

    # Zero vs O: 0 between letters → O
    text = re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O', text)

    # Lowercase-l inside digit-only strings → 1
    text = re.sub(r'(?<=\d)l(?=\d)', '1', text)
    text = re.sub(r'\bl(\d)', r'1\1', text)

    # Currency: isolated S immediately before digit → $
    # (e.g. "S 1,200.00" → "$ 1,200.00")
    text = re.sub(r'\bS\s(?=\d)', '$ ', text)

    # Ligature normalisations (some fonts produce these as single glyphs)
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

    # Remove stray form-feed characters from PDF rendering
    text = text.replace('\x0c', '\n')

    # Collapse 3+ consecutive blank lines to 2 (keeps structure, reduces noise)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def _native_pdf_text(pdf_path: Path) -> str | None:
    """
    Try to extract the text layer already embedded in a digital PDF.
    Returns None if the PDF appears to be image-only (no embedded text).
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            t = page.get_text("text")  # type: ignore[arg-type]
            pages.append(t)
        doc.close()
        combined = "\n\n".join(pages).strip()
        if len(combined) >= NATIVE_TEXT_MIN_CHARS:
            return _fix_ocr_errors(combined)
    except Exception as exc:
        log.debug("Native PDF text extraction failed: %s", exc)
    return None


class LowQualityImageError(ValueError):
    """Raised when an image is blank, pure noise, or yields too little text."""


class OCREngine:
    """
    Multi-strategy OCR engine:
      1. Native PDF text (instant, perfect for digital PDFs)
      2. Tesseract on heavily preprocessed image
      3. EasyOCR as fallback for handwriting / very degraded scans
    """

    def __init__(self, tesseract_cmd: str = TESSERACT_CMD, dpi: int = 300):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self._easy_reader = None
        self.last_engine_used: str = "tesseract"
        self.last_word_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

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

    # ── PDF handling ──────────────────────────────────────────────────────────

    def _process_pdf(self, pdf_path: Path) -> str:
        # Strategy 1: native text layer (free, perfect)
        native = _native_pdf_text(pdf_path)
        if native:
            log.debug("PDF: using native text layer (%d chars)", len(native))
            self.last_engine_used = "native_pdf"
            self.last_word_count = len(native.split())
            return native

        # Strategy 2+3: render each page to image then OCR
        log.debug("PDF: no native text layer – falling back to OCR")
        doc = fitz.open(str(pdf_path))
        page_texts = []
        for page in doc:
            img = self._pdf_page_to_array(page)
            try:
                text = self._extract_text(img)
            except LowQualityImageError:
                text = ""
            page_texts.append(text)
        doc.close()
        combined = "\n\n".join(t for t in page_texts if t)
        if not combined.strip():
            raise LowQualityImageError(
                "Could not extract any text from the PDF. "
                "The pages may be blank or completely unreadable."
            )
        return combined

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

    # ── Dual-engine text extraction ───────────────────────────────────────────

    def _extract_text(self, image: np.ndarray) -> str:
        gray_raw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        self._check_blank(gray_raw)

        # Run Tesseract with best-quality preprocessing
        preprocessed = self._preprocess_for_tesseract(image)

        # psm 6  = assume a single uniform block of text (better for invoices/forms)
        # psm 3  = fully automatic (good general fallback)
        # We try psm 6 first; if it underperforms, try psm 3
        tess_text = self._run_tesseract(preprocessed, psm=6)
        if len(tess_text.split()) < MIN_WORDS_THRESHOLD:
            tess_text_auto = self._run_tesseract(preprocessed, psm=3)
            if len(tess_text_auto.split()) > len(tess_text.split()):
                tess_text = tess_text_auto

        tess_text = _fix_ocr_errors(tess_text)
        word_count = len(tess_text.split())

        if word_count >= MIN_WORDS_THRESHOLD:
            self.last_engine_used = "tesseract"
            self.last_word_count = word_count
            return tess_text

        # EasyOCR fallback
        log.info("[OCR] Tesseract: %d words – switching to EasyOCR", word_count)
        easy_text = self._extract_with_easyocr(image)
        easy_text = _fix_ocr_errors(easy_text)

        combined = easy_text if len(easy_text.split()) > word_count else tess_text
        final_count = len(combined.split())

        if final_count < MIN_USABLE_WORDS:
            raise LowQualityImageError(
                "Could not extract meaningful text. "
                "The image may be blank, pure noise, or too degraded for OCR."
            )
        self.last_engine_used = "easyocr"
        self.last_word_count = final_count
        return combined

    def _run_tesseract(self, gray: np.ndarray, psm: int = 6) -> str:
        config = f"--oem 1 --psm {psm}"
        return pytesseract.image_to_string(gray, lang="eng", config=config).strip()

    # ── Tesseract preprocessing ───────────────────────────────────────────────

    def _preprocess_for_tesseract(self, image: np.ndarray) -> np.ndarray:
        """
        Comprehensive pipeline optimised for printed invoices/forms:
          grayscale → upscale → pad → deskew → invert dark regions →
          bilateral denoise → CLAHE → unsharp sharpen →
          morphological open → binarise (Otsu or adaptive) → final denoise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        gray = self._upscale(gray)
        gray = self._pad(gray, 20)
        gray = self._deskew(gray)

        # Fix white-on-dark text BEFORE any binarisation step
        gray = self._normalise_text_polarity(gray)

        # Bilateral filter: removes noise without blurring text edges
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        gray = self._clahe(gray)
        gray = self._unsharp_mask(gray)

        # Morphological open: break thin bridges between touching characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Binarise: try Otsu first (cleaner); fall back to adaptive for uneven pages
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = np.sum(otsu == 255) / otsu.size
        if 0.6 <= white_ratio <= 0.97:
            binary = otsu
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25,
                C=11,
            )

        return cv2.fastNlMeansDenoising(binary, h=10)

    def _normalise_text_polarity(self, gray: np.ndarray) -> np.ndarray:
        """
        Tesseract expects dark text on a light background.
        This method handles two common cases where that is NOT true:

        Case 1 – Whole-page inversion (e.g. dark-mode invoice, black background):
            If the overall image is darker than 50% grey on average, the whole
            page is inverted so text becomes dark on white.

        Case 2 – Partial dark bands (e.g. coloured header/footer with white text):
            The page is divided into horizontal bands.  Any band whose mean
            brightness is below the threshold is locally inverted.
            This preserves normal light-background areas while fixing dark ones.

        Both cases are handled automatically without any user input.
        """
        h, w = gray.shape
        mean_brightness = float(np.mean(gray))

        # Case 1: mostly dark page → invert everything
        if mean_brightness < 127:
            return cv2.bitwise_not(gray)

        # Case 2: scan for dark horizontal bands (e.g. header/footer bars)
        # Use bands of ~10% page height so we catch headers, footers, dividers
        band_h = max(1, h // 10)
        result = gray.copy()
        for y in range(0, h, band_h):
            band = gray[y: y + band_h, :]
            if band.size == 0:
                continue
            if float(np.mean(band)) < 100:
                # This band has a dark background — invert it so white text → dark
                result[y: y + band_h, :] = cv2.bitwise_not(band)

        return result

    def _preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """
        Lighter pipeline for EasyOCR — skip hard binarisation.
        EasyOCR's CNN performs its own internal thresholding.
        Polarity normalisation is still applied so white-on-dark regions
        are handled correctly.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        gray = self._upscale(gray)
        gray = self._deskew(gray)
        gray = self._normalise_text_polarity(gray)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        gray = self._clahe(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ── EasyOCR engine ────────────────────────────────────────────────────────

    def _get_easy_reader(self):
        if self._easy_reader is None:
            try:
                import easyocr  # type: ignore
                log.info("[EasyOCR] Loading model …")
                self._easy_reader = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                raise ImportError("EasyOCR is not installed. Run: pip install easyocr")
        return self._easy_reader

    def _extract_with_easyocr(self, image: np.ndarray) -> str:
        reader = self._get_easy_reader()
        prepped = self._preprocess_for_easyocr(image)
        results = reader.readtext(prepped, detail=0, paragraph=True)
        return "\n".join(results).strip()

    # ── Quality guard ─────────────────────────────────────────────────────────

    def _check_blank(self, gray: np.ndarray) -> None:
        total = gray.size
        white_pixels = int(np.sum(gray >= 248))
        black_pixels = int(np.sum(gray <= 7))
        if white_pixels / total >= BLANK_WHITE_RATIO:
            raise LowQualityImageError(
                "The image appears to be blank (near-white). "
                "Please upload a document with visible content."
            )
        if black_pixels / total >= BLANK_WHITE_RATIO:
            raise LowQualityImageError(
                "The image appears to be entirely black or severely underexposed."
            )

    # ── Individual preprocessing steps ───────────────────────────────────────

    def _upscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w >= MIN_WIDTH:
            return gray
        scale = MIN_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _pad(self, gray: np.ndarray, px: int) -> np.ndarray:
        return cv2.copyMakeBorder(gray, px, px, px, px,
                                  cv2.BORDER_CONSTANT, value=255)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct document skew using the minimum-area rectangle
        of all dark pixels.  Only corrects angles up to ±15° to avoid
        accidentally rotating truly landscape-oriented pages.
        """
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 100:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 15 or abs(angle) < 0.3:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        """
        Unsharp mask sharpening: amplifies high-frequency edges.
        amount = 1.5× original − 0.5× blurred
        """
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # ── Legacy public method (used by demo_ocr.py) ────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Expose the Tesseract preprocessing pipeline for debug/demo use."""
        return self._preprocess_for_tesseract(image)
