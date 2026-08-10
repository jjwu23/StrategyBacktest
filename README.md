# Signal Lab

A Streamlit app for backtesting simple long-only strategies described in natural language. The default test uses up to ten years of daily SPY OHLCV data from Yahoo Finance and displays wins, losses, win rate, return, drawdown, a candlestick chart, and entry/exit markers.

Supported language patterns include SMA/EMA crossovers, RSI thresholds, volume above its moving average, maximum holding periods, stop losses, and take-profit targets.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit. Choose a ticker and history length in the sidebar, describe a strategy, and select **Run backtest**.

The app uses Yahoo Finance through `yfinance`. Signals are generated from the daily close and filled at the next day's open to reduce look-ahead bias. This is an educational backtester, not investment advice.
