# 🛡️ AegisSMS: 4-Way SMS Intent & Threat Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-2.3.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python)](https://python.org)
[![Zero Leakage](https://img.shields.io/badge/Data%20Leakage-0.00%25%20Verified-brightgreen.svg)]()
[![100% Real Data](https://img.shields.io/badge/Dataset-34%2C762%20Real%20SMS%20(21K%20Scams)-success.svg)]()
[![Scam Recall](https://img.shields.io/badge/Scam%20Recall-95.67%25-red.svg)]()
[![Scam Precision](https://img.shields.io/badge/Scam%20Precision-97.79%25-brightgreen.svg)]()

**AegisSMS** is an enterprise-grade multilingual SMS intent and threat intelligence engine trained on **100% Real SMS Traffic** (0% synthetic data) across **English**, **Hinglish**, **Hindi**, and **Marathi**.

It performs end-to-end 4-way classification:
1. 🟢 **`PERSONAL`** (P2P conversations, informal chats, casual greetings, family/friend messaging)
2. 🔵 **`TRANSACTIONAL`** (Legitimate banking debits/credits, OTPs, utility bills, courier delivery updates, account balance alerts, e-challans)
3. 🟡 **`PROMOTIONAL`** (Legitimate marketing campaigns, discount coupons, telecom recharge offers, sales advertisements, brand campaigns)
4. 🚨 **`SCAM`** (Phishing URLs, APK malware distribution, fake electricity disconnection threats, KYC/PAN card locks, lottery/prize scams, refund traps)

---

## 🌟 4-Way Intent & Threat Taxonomy

| Class | Description | Real Dataset Count | Examples |
| :--- | :--- | :---: | :--- |
| 🟢 **`PERSONAL`** | Direct peer-to-peer social messaging, casual chats, and family check-ins. | **6,000** | *"Hey buddy are you coming to play football?"*, *"Call me when you reach home"* |
| 🔵 **`TRANSACTIONAL`** | Time-critical, automated service alerts, OTP verifications, bank debits/credits, delivery status, and utility notices. | **4,500** | *"Sent Rs.50.00 from Kotak Bank A/c X2056..."*, *"Your Zepto order OTP is 4829"*, *"Power supply disconnected due to arrears..."* |
| 🟡 **`PROMOTIONAL`** | Legitimate commercial broadcasts, discount vouchers, sales announcements, and telecom recharge upselling. | **3,000** | *"FLAT 25% OFF on Tata CLiQ Luxury with code LUXE25"*, *"Recharge with Rs.348 for Unlimited 5G on Airtel"* |
| 🚨 **`SCAM`** | Malicious phishing links, fake utility disconnection threats, APK trojans, lottery fraud, and credential harvesting. | **21,262** | *"Electricity will be cut off tonight, call 9876543210 or click msedcl-pay.apk"*, *"WINNER! You won 25 Lakh in KBC lottery"*, *"SBI account locked, update PAN at http://..."* |
| **TOTAL** | **Curated Master Training Dataset** | **34,762** | **100% Real Data (0% Synthetic)** |

---

## 📊 Blind Real Test Set Evaluation (5,215 Holdout Samples)

| Category | Precision | Recall | F1-Score | Support (Real Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **`PERSONAL`** | **80.00%** | **84.44%** | **82.16%** | 900 |
| 🔵 **`TRANSACTIONAL`** | **71.17%** | **69.48%** | **70.31%** | 675 |
| 🟡 **`PROMOTIONAL`** | **79.18%** | **85.33%** | **82.14%** | 450 |
| 🚨 **`SCAM`** | **97.79%** | **95.67%** | **96.72%** | **3,190** |
| **Overall Accuracy** | **89.45%** | - | - | **5,215** |

### Confusion Matrix
```
                     Predicted Personal   Predicted Transactional   Predicted Promotional   Predicted Scam
Actual Personal:            760                     112                        8                  20
Actual Transactional:       137                     469                       35                  34
Actual Promotional:          19                      32                      384                  15
Actual Scam:                 34                      46                       58                3052
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/vanwanikhushal4-lang/AegisSMS.git
cd AegisSMS
pip install -r requirements.txt
```

### 2. Interactive CLI Predictor

```bash
python interactive_predict.py "Dear customer, your electricity power will be disconnected tonight at 9:30 PM. Call 08634017553 or click http://bit.ly/msedcl-pay.apk"
```

**Output**:
```
======================================================================
INPUT SMS: Dear customer, your electricity power will be disconnected tonight...
======================================================================
VERDICT:     🚨 SCAM / PHISHING (Malicious / Fraud Threat)
CONFIDENCE:  100.00%

4-WAY PROBABILITIES:
  🟢 [PERSONAL]      (Peer-to-peer / Chat):      0.00%
  🔵 [TRANSACTIONAL] (Banking / OTP / Alerts):   0.00%
  🟡 [PROMOTIONAL]   (Offers / Sales / Ads):     0.00%
  🚨 [SCAM]          (Phishing / Fraud):       100.00%
======================================================================
```

### 3. Launch Inference API

```bash
python main.py
```

### 4. API Endpoint (`POST /predict`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! You have won 25 Lakh in KBC lottery. Call 9876543210 immediately."}'
```

**Response**:
```json
{
  "text": "WINNER! You have won 25 Lakh in KBC lottery. Call 9876543210 immediately.",
  "category": "SCAM",
  "is_scam": true,
  "confidence": 0.9997,
  "probabilities": {
    "PERSONAL": 0.0000,
    "TRANSACTIONAL": 0.0000,
    "PROMOTIONAL": 0.0003,
    "SCAM": 0.9997
  },
  "signals": {
    "has_url": false,
    "has_phone": true,
    "urgency_detected": true,
    "credentials_requested": false,
    "refund_phrases": false
  }
}
```

---

## 🧪 Quality Gates & CI Pipeline

Run all automated quality gates (leakage validation, metrics gates, 1,000 golden vectors, API concurrency, adversarial test suite):

```bash
pytest tests/ -v
```
