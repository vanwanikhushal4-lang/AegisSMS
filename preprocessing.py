# -*- coding: utf-8 -*-
"""
AegisSMS Enterprise Preprocessing & Feature Engineering Contract
Shared between Python and Kotlin/Java implementations with 100% mathematical parity.
"""
import re
import unicodedata
from typing import Tuple, Dict, Any

# 1. Regex Patterns
VPA_RE = re.compile(r"\b[a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-_]+\b")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|in|org|net|co|gov|edu|io|ai|xyz|top|site|online|apk|app|live|me|ly|link|info)(?:/\S*)?)", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\- ]{7,}\d)")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u00AD]")

# 2. Urgency & Threat Keywords
URGENCY_KEYWORDS = [
    "urgent", "immediately", "block", "suspend", "verify now", "verify immediately",
    "kyc", "winner", "lucky draw", "claim now", "act now", "penalty",
    "congratulations you", "confirm now", "update now", "turant verify",
    "will be blocked", "will be suspended", "will be disconnected",
    "will be terminated", "will be discontinued", "will be cut off",
    "connection will be", "pending verification", "meter verification",
    "service will be", "ready for transfer", "final notice", "account locked",
    "pay immediately", "disconnect tonight", "call immediately",
    "pending challan", "challan", "legal action", "court notice", "traffic fine",
    "parivahan", "avoid legal action", "avoid legal disputes", "penalty",
    "तुरंत", "ब्लॉक हो जाएगा", "सत्यापित करें", "जीत चुका", "केवाईसी",
    "निलंबित", "अभी वेरीफाई", "अभिनंदन! आप", "लगेच केवायसी",
    "समाप्त कर दिया जाएगा", "बंद केले जाईल", "काट दी जाएगी",
    "खाते बंद", "तात्काळ", "निलंबित केले जाईल", "विद्युत पुरवठा खंडित"
]

# 3. Sensitive Credentials / Phishing Keywords
SENSITIVE_INFO_KEYWORDS = [
    "bank details", "bank account details", "passbook", "account number",
    "share your otp", "share the otp", "send your otp", "share your pin",
    "share your cvv", "aadhaar number", "pan number", "upi pin",
    "debit card number", "credit card number", "submit your bank",
    "submit your aadhaar", "send your bank", "share your bank",
    "share your details", "share your account", "provide your bank",
    "bank passbook", "enter your pin", "provide your otp",
    "बैंक विवरण", "पासबुक", "खाता संख्या", "आधार नंबर",
    "पैन नंबर", "ओटीपी साझा करें", "सीवीवी", "बँक तपशील", "खाते क्रमांक",
    "आधार क्रमांक", "ओटीपी शेअर करा", "पिन प्रविष्ट करा", "गुप्त क्रमांक"
]

# 4. Refund / Wrong Transfer Scam Keywords
REFUND_SCAM_KEYWORDS = [
    "accidentally sent", "sent by mistake", "by mistake", "wrong transfer",
    "wrongly transferred", "transferred to the wrong", "please refund",
    "refund it to this upi", "refund kar dijiye", "refund kijiye",
    "galti se bheja", "galti se transfer", "galti se", "galat transfer",
    "गलती से भेजा", "गलती से ट्रांसफर", "गलती से", "चुकून पाठवले", "चुकून ट्रान्सफर",
    "परत करा", "वापस भेजें", "वापस कर दीजिए"
]

# 5. Currency Markers
CURRENCY_MARKERS = ["rs.", "rs ", "₹", "inr", "$", "eur", "usd", "£"]

# 6. Feature List
NUMERIC_FEATURES = [
    "char_len", "word_count", "digit_ratio", "exclaim_count", "special_ratio",
    "has_url", "has_phone", "currency_count", "urgency_count",
    "sensitive_info_count", "refund_scam_count"
]

# 7. 4-Way Intent & Threat Label Mapping
LABEL_TO_ID = {
    "PERSONAL": 0,
    "TRANSACTIONAL": 1,
    "PROMOTIONAL": 2,
    "SCAM": 3
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Operating threshold
IS_SCAM_OPERATING_THRESHOLD = 0.50

def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = ZERO_WIDTH_RE.sub("", text)
    return text

def urlwords(url: str) -> str:
    u = url.lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    parts = re.split(r"[^a-z0-9]+", u)
    return " ".join(p for p in parts if p)

def clean_and_featurize(text: str) -> Tuple[str, Dict[str, float]]:
    text = normalize_unicode(text)
    orig_len = len(text)

    # 1. Mask UPI VPAs before URL extraction to avoid false domain matches (e.g. name.cf@axisbank)
    vpas = VPA_RE.findall(text)
    cleaned = text
    for v in vpas:
        cleaned = cleaned.replace(v, " upivpa ")

    # 2. Extract and expand URLs
    urls = URL_RE.findall(cleaned)
    has_url = 1.0 if len(urls) > 0 else 0.0
    for u in urls:
        cleaned = cleaned.replace(u, " " + urlwords(u) + " ")

    # 3. Mask Phones
    phones = PHONE_RE.findall(cleaned)
    has_phone = 1.0 if len(phones) > 0 else 0.0
    for p in phones:
        cleaned = cleaned.replace(p, " phonenumber ")

    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 4. Compute 11 Numeric Threat & Structure Features
    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = float(digit_count) / max(orig_len, 1)
    exclaim_count = float(text.count("!"))
    special_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    special_ratio = float(special_count) / max(orig_len, 1)
    words = re.findall(r"(?u)\b\w+\b", cleaned)
    word_count = float(len(words))

    lower_full = text.lower()
    urgency_count = float(sum(lower_full.count(k) for k in URGENCY_KEYWORDS))
    currency_count = float(sum(lower_full.count(k) for k in CURRENCY_MARKERS))
    sensitive_info_count = float(sum(lower_full.count(k) for k in SENSITIVE_INFO_KEYWORDS))
    refund_scam_count = float(sum(lower_full.count(k) for k in REFUND_SCAM_KEYWORDS))

    features = {
        "char_len": float(orig_len),
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
