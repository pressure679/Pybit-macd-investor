# 📈 Pybit MACD Investor

A Python trading bot for **Bybit futures** that enters trades based on the **MACD (Moving Average Convergence Divergence)** indicator, with a basic backtesting script included.

This project is for learning, strategy testing, and automation experimentation using the Bybit API via **Pybit**.

---

## 🧠 Overview

This repository contains two primary scripts:

- `pybit-bot.py` — A live Bybit trading bot that uses the MACD indicator to open and manage positions. :contentReference[oaicite:1]{index=1}  
- `pybit-bot-backtest.py` — A backtesting script to simulate the MACD strategy on historical price data. :contentReference[oaicite:2]{index=2}

The strategy enters long/short positions based on MACD crossovers and is intended for futures trading on Bybit.

---

## ⚙️ Requirements

You’ll need:

- Python 3.7+  
- The **Pybit** API client  
- Technical indicator library (e.g., TA-Lib or equivalent)  
- A Bybit API Key & Secret (for live trading)  
- Historical data for backtesting

Example installation:

```bash
pip install pybit pandas ta-lib
```

---

## 🚀 Running the Live Bot

1. Clone the repository:

```bash
git clone https://github.com/pressure679/Pybit-macd-investor.git
cd Pybit-macd-investor
```

2. Configure your **API credentials** securely:

Edi api key and secret line at line 27-28

3. Run the bot:

```bash
python pybit-bot.py
```

The bot will connect to Bybit and place orders based on MACD crossovers.

> ⚠️ Make sure to test with leverage and position size settings carefully — live trading involves risk.

---

## 📊 Backtesting

You can simulate the MACD strategy using historical data:

```bash
python pybit-bot-backtest.py
```

Provide your own historical price CSV or dataset to evaluate performance over time.

This script allows you to:

- Test entries/exits
- Compare return performance
- Validate strategy behavior

---

## 📈 Strategy Logic

The MACD strategy typically works like this:

1. Calculate:
   - MACD Line
   - Signal Line
   - Histogram
2. **Buy signal** when MACD crosses above the signal line
3. **Sell signal** when MACD crosses below the signal line  
4. Optionally include stop loss / take profit logic

Implementations may vary depending on your risk profile.

---

## 🧪 Tips Before Live Trading

- Always backtest on historical data first  
- Use Bybit **Testnet** before deploying real capital  
- Start with small position sizes  
- Consider slippage and fees in performance

---

## 📍 Notes

- This bot is **educational** and should be adapted to your risk tolerance  
- Not production-grade — lacks advanced risk management and safety checks

---

## 🛡️ Disclaimer

Trading cryptocurrencies is **high risk**. This code is provided for educational purposes only. Use at your own responsibility.

---

## 📜 License

Distributed under the MIT License (or add your chosen license here).

---

⭐ If you find this project useful, consider giving it a star!

