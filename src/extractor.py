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