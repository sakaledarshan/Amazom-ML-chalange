"""
Text preprocessing and normalization for Business Entity Resolution.
Handles multilingual business names, legal suffixes, accents,
and country-specific address standardizations (US, India, France).
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Decompose Unicode characters and remove accent marks (crucial for French addresses/names)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Legal and company suffixes across US, India, and France
LEGAL_SUFFIX_MAP = {
    r"\bcorp(?:oration)?\b": "corp",
    r"\binc(?:orporated)?\b": "inc",
    r"\bltd\b|\blimited\b": "ltd",
    r"\bpvt\b|\bprivate\b": "pvt",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bplc\b": "plc",
    r"\bco\b|\bcompany\b": "co",
    r"\bsarl\b": "sarl",  # French
    r"\bsas\b": "sas",    # French
    r"\bsa\b": "sa",      # French
    r"\beurl\b": "eurl",  # French
    r"\bgmbh\b": "gmbh",
}

# Street and address component abbreviations
STREET_ABBREVIATIONS = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bcourt\b": "ct",
    r"\bcircle\b": "cir",
    r"\bplace\b": "pl",
    r"\bsquare\b": "sq",
    r"\bhighway\b": "hwy",
    r"\bapartment\b|\bapt\b": "apt",
    r"\bsuite\b|\bste\b": "ste",
    r"\bfloor\b|\bfl\b": "fl",
    # French street words
    r"\brue\b": "st",
    r"\ballee\b": "allee",
    r"\bchemin\b": "chemin",
    r"\bimpasse\b": "impasse",
    # Indian address words
    r"\bmarg\b": "rd",
    r"\brasta\b": "rd",
    r"\bnagar\b": "nagar",
    r"\bcolony\b": "colony",
    r"\bchowk\b": "chowk",
    r"\bbazar\b|\bbazaar\b": "bazaar",
}

# Compile patterns for fast repeated execution
COMPILED_LEGAL = [(re.compile(p, re.IGNORECASE), repl) for p, repl in LEGAL_SUFFIX_MAP.items()]
COMPILED_STREET = [(re.compile(p, re.IGNORECASE), repl) for p, repl in STREET_ABBREVIATIONS.items()]

PIN_REGEX_INDIA = re.compile(r"\b[1-9]\d{5}\b")
ZIP_REGEX_US = re.compile(r"\b\d{5}(?:-\d{4})?\b")
POSTAL_REGEX_FRANCE = re.compile(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b")


def clean_basic_text(text: Optional[str]) -> str:
    """Lowercases, normalizes Unicode accents, replaces symbols with words, strips excess whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_business_name(name: Optional[str]) -> str:
    """Normalizes business name by expanding symbols and standardizing legal forms."""
    cleaned = clean_basic_text(name)
    if not cleaned:
        return ""
    for pattern, repl in COMPILED_LEGAL:
        cleaned = pattern.sub(repl, cleaned)
    return " ".join(cleaned.split())


def get_name_tokens_core(normalized_name: str) -> List[str]:
    """Extracts non-generic distinctive name tokens (skipping pure legal forms)."""
    skip_tokens = {"corp", "inc", "ltd", "pvt", "llc", "llp", "plc", "co", "and", "the", "sarl", "sas", "sa"}
    tokens = normalized_name.split()
    return [t for t in tokens if t not in skip_tokens and len(t) > 1]


def normalize_address(address: Optional[str]) -> str:
    """Normalizes address by standardizing street suffixes and common terms."""
    cleaned = clean_basic_text(address)
    if not cleaned:
        return ""
    for pattern, repl in COMPILED_STREET:
        cleaned = pattern.sub(repl, cleaned)
    return " ".join(cleaned.split())


def extract_postal_code(address: str, country: str) -> Optional[str]:
    """Extracts postal code based on country."""
    if not address:
        return None
    if country == "India":
        m = PIN_REGEX_INDIA.search(address)
        return m.group(0) if m else None
    elif country == "US":
        m = ZIP_REGEX_US.search(address)
        return m.group(0)[:5] if m else None
    elif country == "France":
        m = POSTAL_REGEX_FRANCE.search(address)
        return m.group(0) if m else None
    return None


def extract_numbers(text: str) -> Set[str]:
    """Extracts all numeric sequences (house numbers, suite numbers, pincodes)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


if __name__ == "__main__":
    # Test cases
    print("Testing preprocessing...")
    sample_name = "Kroger Corp. & Co., Pvt. Ltd."
    norm_name = normalize_business_name(sample_name)
    print(f"Original: {sample_name} -> Normalized: {norm_name}")
    print(f"Core tokens: {get_name_tokens_core(norm_name)}")

    sample_addr = "123 Elm Street, Suite 4B, New Delhi - 110001"
    norm_addr = normalize_address(sample_addr)
    pin = extract_postal_code(sample_addr, "India")
    print(f"Original: {sample_addr} -> Normalized: {norm_addr} | Postal: {pin}")
    print(f"Numbers: {extract_numbers(sample_addr)}")
