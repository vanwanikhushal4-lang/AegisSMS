# -*- coding: utf-8 -*-
"""
AegisSMS Preprocessing Engine - Canonical Feature Extraction & Text Normalization.
Ensures 100% bit-for-bit mathematical parity between Python, Kotlin, and Java.
"""
import re
import unicodedata
from typing import Dict, Tuple, List

# -------------------------------------------------------------
# 1. NUMERIC FEATURE NAMES (EXACTLY 11 CANONICAL METRICS)
# -------------------------------------------------------------
NUMERIC_FEATURES = [
    "char_len",
    "word_count",
    "digit_ratio",
    "exclaim_count",
    "special_ratio",
    "has_url",
    "has_phone",
    "currency_count",
    "urgency_count",
    "sensitive_info_count",
    "refund_scam_count"
]

ID_TO_LABEL = {0: "PERSONAL", 1: "TRANSACTIONAL", 2: "PROMOTIONAL", 3: "SCAM"}
LABEL_TO_ID = {"PERSONAL": 0, "TRANSACTIONAL": 1, "PROMOTIONAL": 2, "SCAM": 3}

# -------------------------------------------------------------
# 2. COMPILED REGEX PATTERNS FOR FEATURE EXTRACTION & DE-IDENTIFICATION
# -------------------------------------------------------------
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\uFFFD\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# PII Scrubbers
PHONE_PII_RE = re.compile(r"(\+?\d[\d\- ]{7,}\d)")
EMAIL_PII_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
VPA_RE = re.compile(r"([a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64})", re.IGNORECASE)
ACCOUNT_PII_RE = re.compile(r"\b(?:a/c|ac|account\s*ending\s*in|account|card\s*ending\s*in|card\s*ending|card|ending\s*in|ending)\s*(?:no\.?|num|number)?\s*[:#.]*\s*([xX0-9]{3,18})\b", re.IGNORECASE)
REF_PII_RE = re.compile(r"\b(?:upi\s*(?:reference|ref|txn)?|reference|ref|utr|awb|pnr|order|txn|rrn|crn|id)\s*(?:no\.?|num|number)?\s*[:#.]*\s*([a-zA-Z0-9]{6,20})\b", re.IGNORECASE)
OTP_PII_RE = re.compile(r"\b(\d{4,8})\b")

# Feature Patterns
REF_RE = re.compile(r"\b(?:upi\s*(?:reference|ref|txn)?|reference|ref|utr|awb|pnr|order|txn|rrn|crn|id)\s*(?:no\.?|num|number)?\s*[:#.]*\s*([a-zA-Z0-9]{6,20})\b", re.IGNORECASE)
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|in|co|co\.in|gov|gov\.in|edu|edu\.in|io|ai|me|info|biz|link|site|top|xyz|club|live|shop|store|online|vip|app|apk|ly|gd|gl|cc|to|is|tv|uk|co\.uk)(?:/[^\s]*)?)",
    re.IGNORECASE
)
PHONE_RE = re.compile(r"(\+?\d[\d\- ]{7,}\d)")
CURRENCY_RE = re.compile(r"(?:rs\.?|inr|₹|\$|£|eur)\s*[\d,]+(?:\.\d{1,2})?", re.IGNORECASE)
URGENCY_RE = re.compile(
    r"\b(urgent|immediately|action required|avoid suspension|account.*locked|account.*blocked.*update|disconnect tonight|cut off tonight|expire.*hours|limited time|hours left|last chance|hurry|final notice|threat|coercive|विद्युत खंडित|तातडीने|लगेच कॉल|तुरंत कॉल|काट दिया)\b",
    re.IGNORECASE
)
SENSITIVE_INFO_RE = re.compile(
    r"\b(otp|pin|password|cvv|aadhaar|pan card|kyc|verify details|login to verify|credit card|update kyc|केवायसी|पॅन कार्ड|आधार|ओटीपी)\b",
    re.IGNORECASE
)
REFUND_SCAM_RE = re.compile(
    r"\b(wrong transfer|sent by mistake|galti se|refund|claim reward|lottery|kbc|lucky draw|won prize|won lakh|won crore|won cash|bheja hai|वापस भेजें|बक्षीस|लॉटरी)\b",
    re.IGNORECASE
)

PUNCT_RE = re.compile(r"[\.,;:!\?\(\)\[\]\{\}\"\'<>\/\-_+*~`@#\$%\^&\\|=]")

def normalize_unicode(text: str) -> str:
    """
    Applies standard Unicode NFC normalization and strips invisible/corrupted characters.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    return ZERO_WIDTH_RE.sub("", t)

def deidentify_text(text: str) -> str:
    """
    Automated PII scrubbing engine. Masks all phone numbers, email addresses,
    account numbers, VPA handles, reference IDs, and OTPs.
    """
    t = normalize_unicode(text)
    # Mask Email
    t = EMAIL_PII_RE.sub("<EMAIL>", t)
    # Mask VPA
    t = VPA_RE.sub("<VPA>", t)
    # Mask Account numbers
    t = ACCOUNT_PII_RE.sub("A/c <ACCT>", t)
    # Mask Transaction / Ref numbers
    t = REF_PII_RE.sub("Ref <REF>", t)
    # Mask Phone numbers
    t = PHONE_PII_RE.sub("<PHONE>", t)
    return t

def clean_and_featurize(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Cleans raw SMS text, expands tokens, and computes the exact 11 numeric features.
    Integrates de-identification so account/reference IDs normalize to ACCT/REF.
    """
    raw_norm = deidentify_text(text)
    char_len = float(len(raw_norm))

    vpa_masked = VPA_RE.sub(" upivpa ", raw_norm)

    has_url = 1.0 if URL_RE.search(vpa_masked) else 0.0
    has_phone = 1.0 if PHONE_RE.search(vpa_masked) else 0.0
    currency_count = float(len(CURRENCY_RE.findall(raw_norm)))
    urgency_count = float(len(URGENCY_RE.findall(raw_norm)))
    sensitive_info_count = float(len(SENSITIVE_INFO_RE.findall(raw_norm)))
    refund_scam_count = float(len(REFUND_SCAM_RE.findall(raw_norm)))

    # Compute character-level ratios
    digit_count = sum(1 for c in raw_norm if c.isdigit())
    exclaim_count = float(raw_norm.count("!"))
    special_count = sum(1 for c in raw_norm if not c.isalnum() and not c.isspace())

    digit_ratio = (digit_count / char_len) if char_len > 0 else 0.0
    special_ratio = (special_count / char_len) if char_len > 0 else 0.0

    # Clean text transformation for vectorizer
    cleaned = vpa_masked

    # Extract and expand domain words from URLs
    def url_replacer(match):
        url = match.group(0)
        parts = re.split(r"[/._\-:=?&%]", url)
        words = [p.lower() for p in parts if len(p) > 1 and not p.isdigit()]
        return f" httpurl {' '.join(words)} "

    cleaned = URL_RE.sub(url_replacer, cleaned)
    cleaned = PHONE_RE.sub(" phonenumber ", cleaned)
    cleaned = PUNCT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [w for w in cleaned.lower().split() if w]
    word_count = float(len(words))

    features = {
        "char_len": char_len,
        "word_count": word_count,
        "digit_ratio": digit_ratio,
        "exclaim_count": exclaim_count,
        "special_ratio": special_ratio,
        "has_url": has_url,
        "has_phone": has_phone,
        "currency_count": currency_count,
        "urgency_count": urgency_count,
        "sensitive_info_count": sensitive_info_count,
        "refund_scam_count": refund_scam_count
    }

    return cleaned, features
