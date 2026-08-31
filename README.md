# 🛡️ AegisSMS: Multilingual SMS Intent & Threat Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-2.1.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python)](https://python.org)
[![Zero Leakage](https://img.shields.io/badge/Data%20Leakage-0.00%25%20Verified-brightgreen.svg)]()
[![100% Real Data](https://img.shields.io/badge/Dataset-100%25%20Real%20SMS-success.svg)]()
[![Overall Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.04%25-brightgreen.svg)]()

**AegisSMS** is an enterprise-grade, high-throughput, multilingual SMS intent classification engine trained on **100% Real Indian SMS Traffic** across **English**, **Hinglish**, **Hindi**, and **Marathi**.

It categorizes every incoming SMS into three distinct operational intents:
* **`PERSONAL`** (P2P conversations, informal chats, family/friends messaging)
* **`TRANSACTIONAL`** (Bank debits/credits, OTPs, utility bills, delivery updates, account balance alerts)
* **`PROMOTIONAL`** (Marketing campaigns, discount coupons, telecom recharge offers, loan/credit promotions)

---

## 🌟 3-Way Intent Taxonomy

| Intent Class | Description | Examples |
| :--- | :--- | :--- |
| **`PERSONAL`** | Direct peer-to-peer social messaging, casual chats, and family check-ins. | *"Hey are you free tonight?"*, *"Call me when you reach home"*, *"Mom is asking what time you'll be back"* |
| **`TRANSACTIONAL`** | Time-critical, automated service alerts, OTP verifications, bank ledger updates, courier delivery status, and utility outage notices. | *"Sent Rs.50.00 from Kotak Bank A/c X2056..."*, *"Your OTP for Zepto order is 9979"*, *"Power supply disconnected due to arrears..."*, *"Blue Dart shipment delivered"* |
| **`PROMOTIONAL`** | Commercial broadcasts, discount vouchers, sales announcements, telecom recharge upselling, and loan offers. | *"FLAT 15% OFF on Tata CLiQ Luxury"*, *"Recharge with Rs.348 for Unlimited 5G on Airtel"*, *"Voltas AC on Easy EMIs with 0 downpayment"* |

---

## 📊 Evaluation on Untouched 100% Real Blind Test Holdout (915 Samples)

| Intent Category | Precision | Recall | F1-Score | Support (Real Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **`PERSONAL`** | **91.67%** | **99.07%** | **95.22%** | 644 |
| **`TRANSACTIONAL`** | **86.79%** | **63.45%** | **73.31%** | 145 |
| **`PROMOTIONAL`** | **91.15%** | **81.75%** | **86.19%** | 126 |
| **Overall Accuracy** | **91.04%** | - | - | **915** |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/vanwanikhushal4-lang/AegisSMS.git
cd AegisSMS
pip install -r requirements.txt
```

### 2. Interactive CLI Predictor

Test any SMS text directly from the command line:

```bash
python interactive_predict.py "Sent Rs.50.00 from Kotak Bank A/c X2056 to VETAIL on 31-08-26. UPI Ref 624311493216."
```

### 3. Launch FastAPI Server

```bash
python main.py
```

### 4. API Usage

#### Single SMS Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your OTP for HDFC Bank login is 482910. Do not share with anyone."}'
```

**Response**:
```json
{
  "text": "Your OTP for HDFC Bank login is 482910. Do not share with anyone.",
  "category": "TRANSACTIONAL",
  "confidence": 0.9412,
  "probabilities": {
    "PERSONAL": 0.0124,
    "TRANSACTIONAL": 0.9412,
    "PROMOTIONAL": 0.0464
  },
  "signals": {
    "has_url": false,
    "has_phone": false,
    "urgency_detected": false,
    "credentials_requested": false,
    "refund_phrases": false
  }
}
```

#### Batch SMS Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "Hey bro, are we meeting at 5 PM today?",
      "An amount of INR 500.00 has been debited from your Kotak Bank A/c X2056.",
      "50% off on all Myntra orders today with code FESTIVE50: https://myntra.com/sale"
    ]
  }'
```

---

## 🧪 Quality Gates & CI Pipeline

Run all automated quality gates:

```bash
pytest tests/ -v
```
