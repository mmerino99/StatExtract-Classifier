import re 


class InvoiceExtractor:
    def __init__(self, raw_text: str):

        self.text = raw_text #initializing the raw text 

    def extract_invoice_number(self):
        # (?i) makes it case-insensitive we want it like that so 
        # that it doesnt matter if its all caps or not 
        # (?: ... ) is a non-capturing group for the label
        # finds invoice number, gets the NUMBER and dumps 
        # the "invoice number" string part
        # ([A-Z0-9\-]+) capturing group that captures actual invoice number
        pattern = r"(?i)(?:Invoice\s*(?:No\.?|Number)|Inv\s*#)[:\s]*([A-Z0-9\-]+)"
        match = re.search(pattern, self.text)
        
        return match.group(1) if match else None
    


    def extract_dates(self):
        """
        Finds dates in formats like DD/MM/YYYY, MM-DD-YYYY, or Jan 15, 2026.
        Returns a list of all dates found.
        """
        # Pattern 1: 10/01/2026 or 10-01-2026
        # Pattern 2: Jan 10, 2026 or January 10, 2026
        pattern = (
            r"(?i)(" 
            # (?i) case insensitive
            # Format 1: 10/10/2004 or 10-10-2004
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|" 
            
            # (?:st|nd|rd|th) non capturing group gets 1st, 2nd, etc
            # (?:of\s+) optional "of" followed by space
            # Format 2: 10th Oct 2004 or 10th of October, 2004
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{2,4}|" 
            
            # [\s,]+ allows any combination od spaces and commas
            # Format 3: October 10th 2004 or Oct 10th, 2004
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{2,4}"
            r")"
        )
        matches = re.findall(pattern, self.text)
        
        return matches if matches else []

    

    def extract_total_amount(self):
        # Looks for 'Total' or 'Amount Due', ignores spaces/colons, 
        # allows optional $ or €, and captures the formatted number.
        pattern = r"(?i)(?:Total|Amount\s*Due)[:\s]*[$€]?\s*([\d,]+\.\d{2})"
        match = re.search(pattern, self.text)
        
        return match.group(1) if match else None
    

# Test invoice
if __name__ == "__main__":
    sample_ocr_text = """
    Acme Corp
    Invoice Number: INV-89475
    Date: 15/03/2026
    Due Date: Apr 15, 2026
    
    Services Rendered .... $500.00
    Tax .................. $50.00
    Total: $550.00
    """

    extractor = InvoiceExtractor(sample_ocr_text)
    print("Invoice Number:", extractor.extract_invoice_number())
    print("All Dates Found:", extractor.extract_dates())
    print("Total Amount:", extractor.extract_total_amount())





class EmailExtractor:
    def __init__(self, raw_text: str):
        self.text = raw_text

    def extract_sender(self):
        # non capturing group finds From and gets what comes after
        match = re.search(r"(?i)^From:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_recipient(self):
        # non capturing group finds To and gets what comes after
        match = re.search(r"(?i)^To:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_date(self):
        # non capturing group finds Date and gets what comes after
        match = re.search(r"(?i)^Date:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_subject(self):
        # non capturing group finds Subject and gets what comes after
        match = re.search(r"(?i)^Subject:\s*(.+)", self.text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_all_email_addresses(self):
        # finds all email addresses in the text
        # you dont need a capturing group () because 
        # re.findall() will return a list of all matches
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.findall(pattern, self.text)

# Test email
if __name__ == "__main__":
    sample_email_text = """
    From: Jane Doe <jane.doe@example.com>
    To: John Smith <jsmith@corp.net>
    Date: October 12, 2025 10:30 AM
    Subject: Q4 Project Update
    
    Hi John,
    Please find the attached invoice for the Q4 project. If you have any questions, 
    contact support@example.com.
    
    Best,
    Jane
    """
    
    email_ext = EmailExtractor(sample_email_text)
    print("Sender:", email_ext.extract_sender())
    print("Subject:", email_ext.extract_subject())
    print("All Email Addresses Found:", email_ext.extract_all_email_addresses())


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


# Test questionare
if __name__ == "__main__":
    sample_questionnaire_text = """
    CUSTOMER INTAKE QUESTIONNAIRE
    Name: John Smith
    Email: john.smith@example.com

    1) Do you have any allergies?
    Yes - peanuts

    2. Preferred contact method:
    [x] Email
    [ ] Phone
    [ ] SMS

    What is your availability? 
    Mon-Fri after 5pm
    """

    q_ext = QuestionnaireExtractor(sample_questionnaire_text)
    print("\nQuestions:", q_ext.extract_questions())
    print("Checkboxes:", q_ext.extract_checkboxes())
    print("Key/Value:", q_ext.extract_key_value_answers())


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


# Test resume
if __name__ == "__main__":
    sample_resume_text = """
    Jane A. Doe
    janedoe@gmail.com  |  +1 (555) 123-4567
    https://www.linkedin.com/in/jane-doe  https://github.com/janedoe

    SUMMARY
    Data analyst with 5+ years of experience in Python, SQL, and dashboards.

    SKILLS
    Python, Pandas, SQL, Tableau, PowerBI

    EDUCATION
    BSc Computer Science — Example University (2018)

    EXPERIENCE
    Data Analyst — Example Corp (2020–2025)
    - Built ETL pipelines and automated reporting.
    """

    r_ext = ResumeExtractor(sample_resume_text)
    print("\nName:", r_ext.extract_name())
    print("Email:", r_ext.extract_email())
    print("Phone:", r_ext.extract_phone_number())
    print("Links:", r_ext.extract_links())
    print("Sections:", list(r_ext.extract_sections().keys()))