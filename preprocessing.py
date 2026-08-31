# -*- coding: utf-8 -*-
"""
Shared text cleaning / feature engineering for the multilingual SMS
Ham/Spam classifier. Pure stdlib (no pandas/sklearn/tensorflow) so it can
be imported by the lightweight production API as well as the training
pipeline, keeping preprocessing logic in exactly one place.
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{8,}\d)")

# General urgency / threat framing. Deliberately excludes ambiguous single
# words that show up in ordinary reassuring/negated ham text too (e.g. "otp",
# "alert", "cashback", "due"/"bakaya" all appear in legitimate bank
# messages), keeping mostly multi-word phrases that are near-exclusively
# used in scam framing. Broadened (v2) to cover threats that don't use the
# word "disconnect" (e.g. "terminated", "cut off", "discontinued") and softer
# scheme-phishing framing ("ready for transfer", "selected for"), since a
# purely narrow keyword list was shown to miss real scam phrasing entirely.
URGENCY_KEYWORDS = [
    # english / hinglish
    "urgent", "immediately", "block", "suspend", "verify now", "verify immediately",
    "kyc", "winner", "lucky draw", "claim now", "act now", "penalty",
    "congratulations you", "confirm now", "update now", "turant verify",
    "will be blocked", "will be suspended", "will be disconnected",
    "will be terminated", "will be discontinued", "will be cut off",
    "connection will be", "pending verification", "meter verification",
    "service will be", "ready for transfer", "final notice",
    # hindi / marathi (devanagari substrings)
    "तुरंत", "ब्लॉक हो जाएगा", "सत्यापित करें", "जीत चुका", "केवाईसी",
    "निलंबित", "अभी वेरीफाई", "अभिनंदन! आप", "लगेच केवायसी",
    "समाप्त कर दिया जाएगा", "बंद केले जाईल", "काट दी जाएगी",
]

# Requests for financial/personal credentials -- the hallmark of phishing
# regardless of whether the message also uses urgent/threatening language.
# A legitimate institution essentially never asks you to text/reply with
# these details, so this is a strong standalone signal.
SENSITIVE_INFO_KEYWORDS = [
    "bank details", "bank account details", "passbook", "account number",
    "share your otp", "share the otp", "send your otp", "share your pin",
    "share your cvv", "aadhaar number", "pan number", "upi pin",
    "debit card number", "credit card number", "submit your bank",
    "submit your aadhaar", "send your bank", "share your bank",
    "share your details", "share your account", "provide your bank",
    "bank passbook", "बैंक विवरण", "पासबुक", "खाता संख्या", "आधार नंबर",
    "पैन नंबर", "ओटीपी साझा करें", "सीवीवी", "बँक तपशील", "खाते क्रमांक",
    "आधार क्रमांक", "ओटीपी शेअर करा",
]

# The classic "wrongly sent you money, please refund" social-engineering
# pattern -- exploits the fact that no money actually changed hands.
REFUND_SCAM_KEYWORDS = [
    "accidentally sent", "sent by mistake", "by mistake", "wrong transfer",
    "wrongly transferred", "transferred to the wrong", "please refund",
    "refund it to this upi", "galti se bheja", "galti se transfer",
    "गलती से भेजा", "गलती से ट्रांसफर", "चुकून पाठवले", "चुकून ट्रान्सफर",
]

CURRENCY_MARKERS = ["rs.", "rs ", "₹", "inr", "$"]

NUMERIC_FEATURES = [
    "char_len", "word_count", "digit_ratio", "exclaim_count", "special_ratio",
    "has_url", "has_phone", "currency_count", "urgency_count",
    "sensitive_info_count", "refund_scam_count",
]


def urlwords(url):
    """Break a URL into its meaningful lexical parts (no hardcoded domain
    whitelist) so the model can learn which *words* correlate with spam
    (e.g. 'verify', 'kyc', 'block', 'secure') from co-occurrence statistics,
    the same way it learns from the rest of the message -- instead of
    memorizing a finite set of "trusted" domains, which can never cover the
    real world and made an earlier version of the model flag any
    unfamiliar-but-legitimate domain as spam.
    """
    u = url.lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    parts = re.split(r"[^a-z0-9]+", u)
    return " ".join(p for p in parts if p)


def clean_and_featurize(text):
    text = str(text)
    orig_len = len(text)
    urls = URL_RE.findall(text)
    has_url = 1 if urls else 0
    cleaned = text
    for u in urls:
        cleaned = cleaned.replace(u, " " + urlwords(u) + " ")

    phones = PHONE_RE.findall(cleaned)
    has_phone = 1 if phones else 0
    for p in phones:
        cleaned = cleaned.replace(p, " phonenumber ")

    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = digit_count / max(orig_len, 1)
    exclaim_count = text.count("!")
    special_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    special_ratio = special_count / max(orig_len, 1)
    word_count = len(cleaned.split())
    lower_full = text.lower()
    urgency_count = sum(lower_full.count(k) for k in URGENCY_KEYWORDS)
    currency_count = sum(lower_full.count(k) for k in CURRENCY_MARKERS)
    sensitive_info_count = sum(lower_full.count(k) for k in SENSITIVE_INFO_KEYWORDS)
    refund_scam_count = sum(lower_full.count(k) for k in REFUND_SCAM_KEYWORDS)

    features = {
        "char_len": orig_len,
        "word_count": word_count,
        "digit_ratio": digit_ratio,
        "exclaim_count": exclaim_count,
        "special_ratio": special_ratio,
        "has_url": has_url,
        "has_phone": has_phone,
        "currency_count": currency_count,
        "urgency_count": urgency_count,
        "sensitive_info_count": sensitive_info_count,
        "refund_scam_count": refund_scam_count,
    }
    return cleaned, features
