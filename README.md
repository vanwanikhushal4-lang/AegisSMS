# 🛡️ AegisSMS: Enterprise SMS Intent & Threat Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-2.1.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python)](https://python.org)
[![Zero Leakage](https://img.shields.io/badge/Data%20Leakage-0.00%25%20Verified-brightgreen.svg)]()
[![100% Real Data](https://img.shields.io/badge/Dataset-14%2C478%20Real%20SMS-success.svg)]()
[![Overall Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.47%25-brightgreen.svg)]()

**AegisSMS** is an enterprise-grade, high-throughput, multilingual SMS intent classification engine trained on **14,478 100% Real SMS Messages** across **English**, **Hinglish**, **Hindi**, and **Marathi** with **zero synthetic data**.

The model classifies every SMS into three distinct operational intents:
* **`PERSONAL`** (P2P conversations, informal chat, casual greetings, family/friend check-ins)
* **`TRANSACTIONAL`** (Bank debits/credits, OTPs, utility bills, courier tracking, e-challans, account alerts)
* **`PROMOTIONAL`** (Marketing campaigns, discount coupons, telecom recharge offers, sales advertisements, loan/credit promotions)

---

## 🌟 3-Way Intent Taxonomy & Class Distribution

The dataset was curated and deduplicated from multi-source real-world SMS collections:

| Intent Category | Real Dataset Count | Percentage | Primary Examples |
| :--- | :---: | :---: | :--- |
| **`PERSONAL`** | **8,544** | **59.0%** | *"Hey buddy are you coming to play football?"*, *"Call me when you get home"*, *"Mom said dinner is ready"* |
| **`TRANSACTIONAL`** | **4,226** | **29.2%** | *"Sent Rs.50.00 from Kotak Bank A/c X2056..."*, *"Your Zepto order OTP is 4829"*, *"Power supply disconnected due to arrears..."*, *"Challan issued for vehicle DL01AB1234"* |
| **`PROMOTIONAL`** | **1,708** | **11.8%** | *"FLAT 25% OFF on Tata CLiQ Luxury with code LUXE25"*, *"Recharge with Rs.348 for Unlimited 5G on Airtel"*, *"0 downpayment on Voltas AC"* |
| **Total Real Dataset** | **14,478** | **100.0%** | **100% Real Traffic (0% Synthetic Data)** |

---

## 📊 Evaluation on Untouched 100% Real Blind Test Holdout (2,175 Samples)

| Intent Category | Precision | Recall | F1-Score | Support (Real Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **`PERSONAL`** | **95.35%** | **95.87%** | **95.61%** | 1,283 |
| **`TRANSACTIONAL`** | **89.30%** | **90.71%** | **90.00%** | 635 |
| **`PROMOTIONAL`** | **94.58%** | **88.33%** | **91.35%** | 257 |
| **Overall Accuracy** | **93.47%** | - | - | **2,175** |

### Confusion Matrix (Test Split)
```
                Predicted Personal   Predicted Transactional   Predicted Promotional
Actual Personal:       1,230                   48                         5
Actual Transactional:     51                  576                         8
Actual Promotional:        9                   21                       227
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
python interactive_predict.py "Sent Rs.50.00 from Kotak Bank A/c X2056 to VETAIL on 31-08-26. UPI Ref 624311493216."
```

### 3. Launch Inference API

```bash
python main.py
```

### 4. API Usage

#### Single SMS Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your Zepto order OTP is 4829. Share with delivery partner."}'
```

**Response**:
```json
{
  "text": "Your Zepto order OTP is 4829. Share with delivery partner.",
  "category": "TRANSACTIONAL",
  "confidence": 0.9973,
  "probabilities": {
    "PERSONAL": 0.0006,
    "TRANSACTIONAL": 0.9973,
    "PROMOTIONAL": 0.0021
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

---

## 🧪 Quality Gates & CI Pipeline

Run all automated quality gates (leakage check, metric gates, 1,000 parity vectors, API concurrency, adversarial suite):

```bash
pytest tests/ -v
```
