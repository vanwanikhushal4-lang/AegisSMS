# AegisSMS Dataset De-Identification, Privacy, and Consent Protocol

**Document Version**: 2.3.0-P5  
**Classification**: Public Research & Production Release  
**Last Updated**: 2026-08-31  

---

## 1. Scope & Objective

The AegisSMS dataset comprises 34,140 verified authentic mobile SMS messages collected across Indian telecom networks and international smishing research datasets. This document outlines the rigorous de-identification, anonymization, and consent compliance standards enforced across all ingested training, validation, and evaluation corpora.

---

## 2. De-Identification & Privacy Scrubbing

Prior to tokenization and feature extraction, every SMS record undergoes multi-stage automated PII scrubbing to prevent the persistence, leak, or exposure of personal or financial identifiers:

| Identifier Type | Pattern Masked | Replacement Token | Purpose |
| :--- | :--- | :--- | :--- |
| **Phone Numbers** | 8 to 15 digit mobile numbers, international dial codes | `<PHONE>` / `phonenumber` | Eliminates subscriber mobile identifiers |
| **UPI VPAs** | Handle handles (e.g. `user@okhdfcbank`, `user@paytm`) | `<VPA>` / `<upivpa>` | Prevents financial virtual payment address exposure |
| **Account Numbers** | Suffixes and full bank accounts (`A/c XXXXXX7406`) | `<ACCT>` | Protects banking customer account numbers |
| **Transaction / Reference IDs** | UTR, PNR, AWB, UPI Ref numbers | `<REF>` | Prevents financial transaction traceability |
| **One-Time Passwords (OTPs)** | 4 to 8 digit authentication codes | `<OTP>` | Prevents credential replay or exposure |
| **Monetary Amounts** | Exact currency figures | `<AMT>` | Normalizes financial impact while removing private data |

---

## 3. Data Sources & Consent Framework

All 34,140 records are derived strictly from consented, open-access, or regulatory threat-monitoring repositories:

1. **Enterprise DLT Telemetry (`user_dataset_384`)**:
   - Ingested from registered Indian enterprise traffic compliant with Telecom Regulatory Authority of India (TRAI) Distributed Ledger Technology (DLT) regulations.
   - Senders are verified corporate entity headers (e.g., `HDFCBK`, `SBIINB`, `MSEDCL`).
2. **Honeypot Phishing Captures (`electricsheep_africa_smishing`, `dataset_5971_real`)**:
   - Ingested from automated mobile threat honeypots designed to intercept smishing campaigns.
   - Attackers have no expectation of privacy when transmitting malicious links or malware distribution payloads.
3. **Open Research Repositories (`uci_sms_spam_collection`, `codesignal_sms_spam`, `cloveai_india_spam_sms`)**:
   - Sourced under Creative Commons Attribution 4.0 International (CC-BY-4.0) and academic open data licenses.
   - Contributors provided voluntary submissions for spam and security benchmarking.

---

## 4. Compliance Verification

- **0% Synthetic Records**: The corpus contains zero synthetic, LLM-generated, or back-translated noise.
- **Zero Raw PII Persistence**: No unmasked phone numbers, bank account numbers, or personal email addresses are retained in model weights, vectorizer vocabularies, or published CSV splits.
