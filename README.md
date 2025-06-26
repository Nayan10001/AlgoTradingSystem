# AlgoTradingSystem

AlgoTradingSystem is a modular, end-to-end algorithmic trading platform built using Python. It combines technical analysis, machine learning, Google Sheets integration, and Telegram alerting, all accessible through a Streamlit-based user interface.

## Features

- RSI + Moving Average backtesting strategy
- Machine Learning predictions using Logistic Regression and Random Forest
- Google Sheets integration for logging trades, model summaries, and performance metrics
- Telegram alerts with optional AI-generated summaries (via Gemini)
- Streamlit UI for interactive monitoring and control
- Modular code structure for easy maintenance and extension

## Machine Learning Models

- Logistic Regression for binary prediction of next-day movement
- Random Forest for improved accuracy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nayan10001/AlgoTradingSystem.git
cd AlgoTradingSystem

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run src/app.py
