# AegisSMS / PayShield - 4-Way Intent & Threat Intelligence Engine

[![AegisSMS CI Quality Gates](https://github.com/vanwanikhushal4-lang/AegisSMS/actions/workflows/ci.yml/badge.svg)](https://github.com/vanwanikhushal4-lang/AegisSMS/actions)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg)](LICENSE)
[![Platform: JVM / Android / Python](https://img.shields.io/badge/Platform-Android%20%7C%20JVM%20%7C%20Python-green.svg)](#android-integration)

AegisSMS is an on-device and serverless SMS classification and threat detection engine designed for banking-grade security and consumer message triage. It classifies incoming SMS traffic into a 4-way taxonomy:
- `0: PERSONAL`
- `1: TRANSACTIONAL`
- `2: PROMOTIONAL`
- `3: SCAM`

---

## Key Performance Indicators (Blind Test Evaluation)

Evaluated on the source-held-out, template-isolated holdout split ($N = 2,123$):

| Metric | Target Standard | Measured Result |
| :--- | :--- | :--- |
| **Overall Accuracy** | $\ge 90.0\%$ | **96.89%** |
| **SCAM Precision** | $\ge 95.0\%$ | **99.61%** |
| **SCAM Recall** | $\ge 95.0\%$ | **98.84%** |
| **Legitimate-to-SCAM FPR** | $\le 0.50\%$ ($< 1/200$) | **0.124%** ($2 / 1,607$) |
| **Kotlin/Java JVM Parity Delta** | $\max \|\Delta P\| < 1 \times 10^{-5}$ | **$\mathbf{4.44 \times 10^{-16}}$** (0 category / decision mismatches) |
| **Inter-Annotator Agreement (Kappa)** | $\kappa \ge 0.80$ | **$\mathbf{\kappa = 0.9291}$** (500 double-annotated gold records) |
| **Synthetic Data Percentage** | $0.0\%$ | **0.00% (100% Real SMS)** |

---

## 4-Way Taxonomy Distribution

The production corpus comprises **17,687 unique clean messages** across target languages:

```
PERSONAL      :  6,657  (37.6%)  ██████████████████
TRANSACTIONAL :  5,809  (32.8%)  ████████████████
SCAM          :  4,308  (24.4%)  ████████████
PROMOTIONAL   :    913   (5.2%)  ███
------------------------------------------------------
TOTAL         : 17,687 (100.0%)
```

### Multilingual Representation
- **English (`en`)**: 16,021 messages
- **Hinglish (`hinglish`)**: 1,621 messages
- **Marathi (`mr`)**: 32 messages
- **Hindi (`hi`)**: 13 messages

---

## Single Uniform Decision Rule

To guarantee consistency across mobile devices, backend APIs, and evaluation suites, AegisSMS enforces a single unified decision function:

$$\hat{y} = \arg\max_{c \in \{0,1,2,3\}} P(c \mid \mathbf{x})$$

$$\text{Decision} = \begin{cases} \text{SCAM}, & \text{if } \hat{y} = 3 \lor P(\text{SCAM}) \ge 0.6900 \\ \hat{y}, & \text{otherwise} \end{cases}$$

---

## Android & JVM SDK Integration

AegisSMS exports a zero-dependency JSON model contract ([`artifacts/aegis_model_contract.json`](artifacts/aegis_model_contract.json)) designed to run natively on Android devices using pure Kotlin ([`android/AegisSmsClassifier.kt`](android/AegisSmsClassifier.kt)) or Java ([`android/com/payshield/aegissms/AegisSmsClassifier.java`](android/com/payshield/aegissms/AegisSmsClassifier.java)).

```kotlin
// Android / Kotlin Usage
val contractStream = context.assets.open("aegis_model_contract.json")
val classifier = AegisSmsClassifier(contractStream)

val result = classifier.predict("URGENT: Your electricity meter will be disconnected tonight at 9:30 PM. Call officer at <PHONE>: http://msedcl-bill.apk")
println("Category: ${result.category}")       // SCAM
println("Is Threat: ${result.isScam}")         // true
println("Confidence: ${result.confidence}")    // 0.9984
```

---

## Deterministic Fresh-Clone Reproduction

```bash
# 1. Clone repository
git clone https://github.com/vanwanikhushal4-lang/AegisSMS.git
cd AegisSMS

# 2. Set up pinned Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Execute deterministic master training & contract export pipeline
python pipeline_p5.py

# 4. Execute all CI Quality Gates (including real Kotlin JVM parity & PII verification)
pytest tests/ -v
```

---

## Privacy & De-Identification

All ingested datasets adhere to strict privacy safeguards:
- Automated redaction of phone numbers (`<PHONE>`), bank account details (`<ACCT>`), transaction hashes (`<REF>`), UPI handles (`<VPA>`), and authentication tokens (`<OTP>`).
- Published CSV splits and 1,000 golden vectors contain zero unmasked personal information.
- See [docs/DEIDENTIFICATION_AND_CONSENT.md](docs/DEIDENTIFICATION_AND_CONSENT.md) for full compliance documentation.
