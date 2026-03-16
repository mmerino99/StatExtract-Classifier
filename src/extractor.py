import re
from typing import Any


# ── Shared helpers ────────────────────────────────────────────────────────────

def _normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _to_float(s: str) -> float | None:
    """Convert a number string like '1,234.56' or '1.234,56' to float."""
    if not s:
        return None
    s = s.strip()
    # European format: "1.234,56"
    if re.match(r"^\d{1,3}(?:\.\d{3})+,\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


# ── Compiled date pattern (reused across methods) ────────────────────────────

# Date pattern WITHOUT an inline (?i) flag so it can be safely embedded
# inside larger compiled patterns. Case-insensitivity is applied by the
# caller via re.IGNORECASE.
_DATE_PAT_STR = (
    # YYYY-MM-DD or YYYY/MM/DD
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|"
    # DD/MM/YYYY or MM/DD/YYYY or DD-MM-YYYY
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
    r"|"
    # "15 March 2026" or "15th March 2026" or "15th of March, 2026"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[a-z]*[\s,]+\d{2,4}"
    r"|"
    # "March 15, 2026" or "March 15 2026"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[a-z]*[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{2,4}"
)
# Pre-compiled version for standalone date searches
_DATE_PAT = re.compile(_DATE_PAT_STR, re.IGNORECASE)

# Keywords that disqualify a line from being a company/issuer name
_ISSUER_SKIP_PAT = re.compile(
    r"(?i)\b(invoice|receipt|bill|statement|date|total|amount|tax|vat"
    r"|due|payment|no\.|number|tel|phone|fax|email|www\.|http|page)\b"
)


class InvoiceExtractor:
    """
    Extracts structured fields from raw OCR text of an invoice.

    Required output fields
    ----------------------
      invoice_number  – e.g. "INV-0042" or "2026-00123"
      invoice_date    – date string as found in the document
      due_date        – payment due date string
      issuer_name     – company / person who issued the invoice
      recipient_name  – company / person being billed
      total_amount    – float total (e.g. 1250.00)
    """

    def __init__(self, raw_text: str, ocr_data_dict: dict | None):
        self.text = raw_text
        self.ocr_data = ocr_data_dict
        self._nlp = None
        self._doc = None

    # ── spaCy (optional, best-effort) ────────────────────────────────────────

    def _ensure_spacy(self):
        if self._doc is not None:
            return
        try:
            import spacy  # type: ignore
            self._nlp = spacy.load("en_core_web_sm")
            self._doc = self._nlp(self.text)
        except Exception:
            self._nlp = None
            self._doc = None

    # ── Invoice number ────────────────────────────────────────────────────────

    def extract_invoice_number(self) -> str | None:
        """
        Matches a wide range of real-world invoice number formats:
          - "Invoice No: INV-0042"
          - "Invoice #: 2026-00123"
          - "Invoice Number 20260315-001"
          - "Invoice ID: A/2026/00042"
          - "Factura No. 0042"          (common in bilingual invoices)
          - "Reference: REF-20260315"
          - "Order No. 98765"
          - "Document No. DOC-001"
        """
        patterns = [
            # Classic "Invoice No / Number / ID / #"
            r"(?i)(?:invoice|inv)[\s\-]*(?:no\.?|num(?:ber)?|#|id)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{1,24})",
            # "Invoice: INV-001"
            r"(?i)\binvoice\s*[:\-]\s*([A-Z0-9][A-Z0-9\-_/\.]{1,24})",
            # Reference / Document No
            r"(?i)(?:reference|ref|document|doc)[\s\-]*(?:no\.?|num(?:ber)?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{1,24})",
            # Order No / PO Number
            r"(?i)(?:order|purchase\s*order|po)[\s\-]*(?:no\.?|num(?:ber)?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{1,24})",
            # Standalone alphanumeric starting with INV / PO / DOC / REF
            r"\b((?:INV|PO|DOC|REF|ORD|BILL)[\-_]?[A-Z0-9]{2,20})\b",
        ]
        for pat in patterns:
            m = re.search(pat, self.text)
            if m:
                candidate = m.group(1).strip().rstrip(".")
                # reject obviously wrong hits (pure date fragments, very short)
                if len(candidate) >= 2 and not re.fullmatch(r"\d{1,2}", candidate):
                    return candidate
        return None

    # ── Date helpers ─────────────────────────────────────────────────────────

    def extract_dates(self) -> list[str]:
        """Return every date string found in the text (deduped, order preserved)."""
        seen: set[str] = set()
        results: list[str] = []
        for m in _DATE_PAT.finditer(self.text):
            val = _normalise_space(m.group(0))
            if val not in seen:
                seen.add(val)
                results.append(val)
        return results

    def _extract_labeled_date(self, label_patterns: list[str]) -> str | None:
        """
        Find a date that immediately follows one of the given label patterns.
        Allows a colon, spaces, or a newline between the label and the date value.
        """
        label_re = "|".join(label_patterns)
        pattern = re.compile(
            rf"(?:{label_re})\s*[:\-]?\s*(?:\n\s*)?({_DATE_PAT_STR})",
            re.IGNORECASE,
        )
        m = pattern.search(self.text)
        return _normalise_space(m.group(1)) if m else None

    def extract_invoice_date(self) -> str | None:
        return self._extract_labeled_date([
            r"invoice\s*date",
            r"date\s*of\s*invoice",
            r"date\s*of\s*issue",
            r"issue\s*date",
            r"billing\s*date",
            r"billed\s*date",
            r"invoice\s*issued",
            r"\bdate\b",           # generic "Date:" — kept last, most ambiguous
        ])

    def extract_due_date(self) -> str | None:
        return self._extract_labeled_date([
            r"due\s*date",
            r"payment\s*due(?:\s*date)?",
            r"pay(?:ment)?\s*by",
            r"payable\s*(?:by|on|before)",
            r"terms?\s*due",
            r"net\s*\d+",          # "Net 30" — next date after this is the due date
            r"please\s*pay\s*by",
        ])

    # ── Total amount ──────────────────────────────────────────────────────────

    def extract_total_amount(self) -> str | None:
        """
        Look for total-like labels and capture the amount that follows.
        Tries more specific labels first (Grand Total) before generic (Total).
        Handles:
          - "Total: $1,250.00"
          - "Amount Due  1250.00"
          - "TOTAL EUR 1,250.00"
          - "Total payable: £ 1 250.00"  (spaces inside number)
          - "Grand Total 1250"            (integer totals, no decimals)
        """
        # Currency symbols/codes optionally preceding or following the number
        currency = r"(?:USD|EUR|GBP|CAD|AUD|CHF|[$€£¥₹])"
        # Amount: digits with optional thousands separator and optional decimals
        amount   = r"([\d][\d\s,\.]{0,15})"  # broad capture; cleaned later

        # Labels from most to least specific
        labels = [
            r"grand\s*total",
            r"total\s*(?:amount\s*)?due",
            r"total\s*payable",
            r"amount\s*(?:due|payable|owed)",
            r"balance\s*(?:due|payable|owed|forward)",
            r"total\s*(?:including|incl\.?)\s*(?:vat|tax)",
            r"total\s*(?:excluding|excl\.?)\s*(?:vat|tax)",
            r"invoice\s*total",
            r"\btotal\b",
        ]

        for label in labels:
            pat = re.compile(
                rf"(?i){label}\s*[:\-]?\s*{currency}?\s*{currency}?\s*{amount}",
                re.IGNORECASE,
            )
            m = pat.search(self.text)
            if m:
                raw = m.group(1).strip()
                # Remove internal spaces (e.g. "1 250.00") and trailing dots
                raw = re.sub(r"\s", "", raw).rstrip(".")
                # Must look like a number
                if re.search(r"\d", raw):
                    return raw
        return None

    def extract_total_spatial(self) -> float | None:
        """
        Spatial fallback: scan Tesseract word-level data for the largest
        currency amount near the bottom of the page.
        """
        if not self.ocr_data:
            return None

        amount_pat = re.compile(r"[$€£]?\s*([\d,]+\.?\d*)")
        candidates: list[dict] = []

        for i, word in enumerate(self.ocr_data.get("text", [])):
            m = amount_pat.search(str(word))
            if m:
                val = _to_float(m.group(1))
                if val and val > 0:
                    candidates.append({"value": val, "y": self.ocr_data["top"][i]})

        if not candidates:
            return None

        # Take the 5 amounts nearest the bottom (highest y) and pick the largest
        candidates.sort(key=lambda x: x["y"], reverse=True)
        return max(candidates[:5], key=lambda x: x["value"])["value"]

    # ── Parties (issuer + recipient) ──────────────────────────────────────────

    def extract_parties(self) -> dict[str, str | None]:
        issuer    = None
        recipient = None

        # ── Recipient ────────────────────────────────────────────────────────
        # Try explicit "Bill To / Sold To / Ship To / Attention" section first
        bill_to_re = re.compile(
            r"(?i)(?:bill(?:ed)?\s*to|sold\s*to|ship(?:ped)?\s*to"
            r"|client|attention|attn\.?|customer)\s*[:\-]?\s*\n?",
        )
        m = bill_to_re.search(self.text)
        if m:
            snippet = self.text[m.end(): m.end() + 300]
            recipient = self._name_from_snippet(snippet)

        # ── Issuer ───────────────────────────────────────────────────────────
        # Try explicit "From / Issued by / Vendor" section
        from_re = re.compile(
            r"(?i)(?:from|issued\s*by|vendor|supplier|service\s*provider"
            r"|seller|company)\s*[:\-]?\s*\n?",
        )
        m = from_re.search(self.text)
        if m:
            snippet = self.text[m.end(): m.end() + 300]
            issuer = self._name_from_snippet(snippet)

        # Fallback issuer: first meaningful short line at the top of the doc
        if not issuer:
            issuer = self._issuer_from_top()

        # Fallback recipient: spaCy ORG/PERSON after any "To:" hint
        if not recipient:
            self._ensure_spacy()
            if self._doc is not None:
                for ent in self._doc.ents:
                    if ent.label_ in ("ORG", "PERSON") and ent.text.strip() != issuer:
                        recipient = ent.text.strip()
                        break

        return {"Issuer": issuer, "Recipient": recipient}

    def _name_from_snippet(self, snippet: str) -> str | None:
        """
        Extract the most likely company/person name from a short snippet
        that immediately follows a label like "Bill To:" or "From:".
        Strategy: try spaCy NER first; fall back to first non-trivial line.
        """
        self._ensure_spacy()
        if self._nlp is not None:
            doc = self._nlp(snippet[:250])
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PERSON"):
                    return ent.text.strip()

        # Line-based fallback
        for line in snippet.splitlines():
            s = _normalise_space(line)
            if not s or len(s) < 2 or len(s) > 80:
                continue
            # Skip lines that are clearly not names
            if re.search(r"(?i)\b(invoice|total|amount|date|due|tax|vat|www\.|http)", s):
                continue
            if re.fullmatch(r"[\d\s\-\+\(\)]{4,}", s):  # pure phone/number lines
                continue
            return s
        return None

    def _issuer_from_top(self) -> str | None:
        """
        Heuristic: on most invoices, the issuing company name is in the
        first few non-empty lines (often the header/logo text).
        Skip lines that look like addresses, dates, or invoice keywords.
        """
        lines_checked = 0
        for line in self.text.splitlines():
            s = _normalise_space(line)
            if not s:
                continue
            lines_checked += 1
            if lines_checked > 20:
                break
            if len(s) < 2 or len(s) > 80:
                continue
            if _ISSUER_SKIP_PAT.search(s):
                continue
            if re.fullmatch(r"[\d\s\-\+\(\)\./,@]{4,}", s):
                continue
            return s
        return None

    # ── Master extraction ─────────────────────────────────────────────────────

    def get_structured_data(self) -> dict[str, Any]:
        """
        Runs all extractors and packages the required invoice fields.
        Gracefully falls back at each step so partial results are always
        returned rather than crashing.
        """
        inv_num   = self.extract_invoice_number()
        inv_date  = self.extract_invoice_date()
        due_date  = self.extract_due_date()
        dates     = self.extract_dates()
        parties   = self.extract_parties()

        # Date fallbacks: use positional dates if labeled extraction failed
        invoice_date = inv_date or (dates[0] if dates else None)
        due_date     = due_date or (dates[1] if len(dates) > 1 else None)

        # Total: spatial first, then text-based
        total_float: float | None = self.extract_total_spatial()
        if total_float is None:
            raw_total = self.extract_total_amount()
            total_float = _to_float(raw_total) if raw_total else None

        # Format total as string for the UI (e.g. "1250.00")
        total_str: str | None = (
            f"{total_float:,.2f}" if total_float is not None else None
        )

        # Optional math validation (subtotal + tax ≈ total)
        math_check: str | bool = "N/A"
        sub_m = re.search(
            r"(?i)(?:sub[\s\-]?total|net\s*amount)[:\s]*[$€£]?\s*([\d,]+\.?\d*)",
            self.text,
        )
        tax_m = re.search(
            r"(?i)(?:tax|vat|gst|hst)[:\s]*[$€£]?\s*([\d,]+\.?\d*)",
            self.text,
        )
        if sub_m and tax_m and total_float is not None:
            sub = _to_float(sub_m.group(1))
            tax = _to_float(tax_m.group(1))
            if sub and tax:
                math_check = abs((sub + tax) - total_float) < 0.05

        return {
            "invoice_number":         inv_num,
            "invoice_date":           invoice_date,
            "due_date":               due_date,
            "issuer_name":            parties.get("Issuer"),
            "recipient_name":         parties.get("Recipient"),
            "total_amount":           total_str,
            "validation_math_passed": math_check,
        }




class EmailExtractor:
    def __init__(self, raw_text: str):
        self.text = raw_text

    def extract_sender(self):
        match = re.search(r"(?i)^From:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_recipient(self):
        match = re.search(r"(?i)^To:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_date(self):
        match = re.search(r"(?i)^Date:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_subject(self):
        match = re.search(r"(?i)^Subject:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_all_email_addresses(self):
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.findall(pattern, self.text)


class QuestionnaireExtractor:
    def __init__(self, raw_text: str):
        self.text = raw_text

    def extract_questions(self):
        """
        Finds likely question prompts in a questionnaire.
        Returns a list of question strings (best-effort).
        """
        # Heuristic 1: lines ending in a question mark
        qmark_lines = re.findall(r"(?m)^\s*(.+?\?)\s*$", self.text)

        # Heuristic 2: numbered prompts like "1. ...", "2) ..."
        numbered = re.findall(r"(?m)^\s*(?:Q\s*)?\d+\s*[\.\)]\s*(.+?)\s*$", self.text)

        # Combine, de-duplicate while keeping order
        seen = set()
        out = []
        for q in (qmark_lines + numbered):
            q = q.strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    def extract_checkboxes(self):
        """
        Finds checkbox-like items such as [ ] Option, [x] Option, ( ) Option, (x) Option.
        Returns a list of dicts: {"label": str, "checked": bool}
        """
        pattern = r"(?mi)^\s*(\[(?:x|X|\s)\]|\((?:x|X|\s)\))\s*(.+?)\s*$"
        matches = re.findall(pattern, self.text)

        results = []
        for box, label in matches:
            checked = "x" in box.lower()
            results.append({"label": label.strip(), "checked": checked})
        return results

    def extract_key_value_answers(self):
        """
        Finds simple "Question: Answer" lines commonly used in forms.
        Returns a list of dicts: {"question": str, "answer": str}
        """
        # Non-greedy question up to ":" or "-" delimiter, then capture answer
        pattern = r"(?m)^\s*([A-Za-z0-9][^:\n]{2,}?)\s*[:\-]\s*(.+?)\s*$"
        matches = re.findall(pattern, self.text)

        results = []
        for q, a in matches:
            q = q.strip()
            a = a.strip()
            if not q or not a:
                continue
            results.append({"question": q, "answer": a})
        return results


class ResumeExtractor:
    def __init__(self, raw_text: str):
        self.text = raw_text

    def extract_email(self):
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", self.text)
        return match.group(0) if match else None

    def extract_phone_number(self):
        # Best-effort phone matcher: supports +country, separators, parentheses
        pattern = (
            r"(?x)"
            r"(\+?\d{1,3}[\s\.-]?)?"      # optional country code
            r"(\(?\d{2,4}\)?[\s\.-]?)"    # area code
            r"\d{3,4}[\s\.-]?\d{3,4}"     # local number
        )
        match = re.search(pattern, self.text)
        return match.group(0).strip() if match else None

    def extract_links(self):
        # finds common profile links (LinkedIn, GitHub, portfolios)
        pattern = r"(?i)\bhttps?://[^\s\)\]]+\b"
        return re.findall(pattern, self.text)

    def extract_name(self):
        """
        Tries to get the candidate name (usually first non-empty line).
        Skips lines that look like contact info.
        """
        for line in self.text.splitlines():
            s = line.strip()
            if not s:
                continue
            # skip obvious non-name lines
            if "@" in s or re.search(r"\d{3}[\s\.-]?\d{3}", s) or s.lower().startswith(("resume", "curriculum vitae", "cv")):
                continue
            # if it's too long, it is probably a header sentence
            if len(s) > 60:
                continue
            return s
        return None

    def extract_sections(self):
        """
        Splits resume into common sections using heading lines.
        Returns a dict like {"EDUCATION": "...", "EXPERIENCE": "..."}.
        """
        # common headings, allow variations like "Work Experience"
        headings = [
            "summary", "professional summary", "objective",
            "experience", "work experience", "employment",
            "education", "skills", "projects", "certifications",
            "languages", "awards", "publications",
        ]

        heading_re = r"|".join([re.escape(h) for h in headings])
        pattern = rf"(?mi)^\s*({heading_re})\s*[:\-]?\s*$"

        matches = list(re.finditer(pattern, self.text))
        if not matches:
            return {}

        sections = {}
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.text)
            title = m.group(1).strip().upper()
            body = self.text[start:end].strip()
            if body:
                sections[title] = body
        return sections


# ── Smoke-tests (run with: python -m src.extractor) ──────────────────────────

if __name__ == "__main__":
    _invoice_text = """
    Acme Corp
    Invoice Number: INV-89475
    Date: 15/03/2026
    Due Date: Apr 15, 2026

    Services Rendered .... $500.00
    Tax .................. $50.00
    Total: $550.00
    """
    inv = InvoiceExtractor(_invoice_text, None)
    print("=== Invoice ===")
    print("Number :", inv.extract_invoice_number())
    print("Dates  :", inv.extract_dates())
    print("Total  :", inv.extract_total_amount())
    print("Full   :", inv.get_structured_data())

    _email_text = """
    From: Jane Doe <jane.doe@example.com>
    To: John Smith <jsmith@corp.net>
    Date: October 12, 2025 10:30 AM
    Subject: Q4 Project Update

    Hi John, please find the attached invoice.
    Contact support@example.com with any questions.
    """
    em = EmailExtractor(_email_text)
    print("\n=== Email ===")
    print("Sender  :", em.extract_sender())
    print("Subject :", em.extract_subject())
    print("Emails  :", em.extract_all_email_addresses())

    _questionnaire_text = """
    CUSTOMER INTAKE QUESTIONNAIRE
    Name: John Smith
    Email: john.smith@example.com

    1) Do you have any allergies?
    Yes - peanuts

    2. Preferred contact method:
    [x] Email
    [ ] Phone
    [ ] SMS
    """
    qu = QuestionnaireExtractor(_questionnaire_text)
    print("\n=== Questionnaire ===")
    print("Questions :", qu.extract_questions())
    print("Checkboxes:", qu.extract_checkboxes())

    _resume_text = """
    Jane A. Doe
    janedoe@gmail.com  |  +1 (555) 123-4567
    https://www.linkedin.com/in/jane-doe

    EDUCATION
    BSc Computer Science — Example University (2018)

    EXPERIENCE
    Data Analyst — Example Corp (2020–2025)
    """
    re_ = ResumeExtractor(_resume_text)
    print("\n=== Resume ===")
    print("Name     :", re_.extract_name())
    print("Email    :", re_.extract_email())
    print("Sections :", list(re_.extract_sections().keys()))
