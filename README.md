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

Evaluated on the source-held-out, template-isolated holdout split ($N = 3,727$):

| Metric | Target Standard | Measured Result |
| :--- | :--- | :--- |
| **Overall Accuracy** | $\ge 90.0\%$ | **97.53%** |
| **SCAM Precision** | $\ge 95.0\%$ | **98.70%** |
| **SCAM Recall** | $\ge 95.0\%$ | **99.02%** |
| **Legitimate-to-SCAM FPR** | $\le 0.50\%$ ($< 1/200$) | **0.428%** ($12 / 2,805$) |
| **Kotlin/Java JVM Parity Delta** | $\max \|\Delta P\| < 1 \times 10^{-5}$ | **$\mathbf{5.30 \times 10^{-7}}$** (0 category / decision mismatches) |
| **Synthetic Data Percentage** | $0.0\%$ | **0.00% (100% Real SMS)** |

---

## 4-Way Taxonomy Distribution

The production corpus comprises **34,140 unique clean messages** across target languages:

```
TRANSACTIONAL : 15,080  (44.2%)  ██████████████████████
PERSONAL      : 10,121  (29.6%)  ███████████████
SCAM          :  6,205  (18.2%)  █████████
PROMOTIONAL   :  2,734   (8.0%)  ████
------------------------------------------------------
TOTAL         : 34,140 (100.0%)
```

### Multilingual Representation
- **English (`en`)**: 31,736 messages
- **Hinglish (`hinglish`)**: 2,345 messages
- **Marathi (`mr`)**: 46 messages
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

val result = classifier.predict("URGENT: Your electricity meter will be disconnected tonight at 9:30 PM. Call officer at 9876543210: http://msedcl-bill.apk")
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

# 4. Execute all 13 CI Quality Gates (including real Kotlin JVM parity)
pytest tests/ -v
```

---

## Privacy & De-Identification

All ingested datasets adhere to strict privacy safeguards:
- Automated redaction of phone numbers (`<PHONE>`), bank account details (`<ACCT>`), transaction hashes (`<REF>`), UPI handles (`<VPA>`), and authentication tokens (`<OTP>`).
- See [docs/DEIDENTIFICATION_AND_CONSENT.md](docs/DEIDENTIFICATION_AND_CONSENT.md) for full compliance documentation.
