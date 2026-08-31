# -*- coding: utf-8 -*-
"""
Entry point for the SMS Spam/Ham detection API.

Usage:
    python main.py

Then POST SMS text to http://localhost:8000/predict, e.g.:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"Your account will be blocked, verify now: http://bit.ly/xyz\"}"

Interactive API docs are served at http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
