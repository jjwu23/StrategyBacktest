"""Natural-language, long-only trading strategy backtester."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Signal Lab", page_icon="◒", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv(symbol: str, years: int) -> pd.DataFrame:
    """Download daily OHLCV data, capped at the last ten years."""
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365 * min(max(years, 1), 10))
    frame = yf.download(
        symbol, start=start, end=end, interval="1d", auto_adjust=False,
        progress=False, group_by="column", threads=False,
    )
    if frame.empty:
        raise ValueError(f"No daily data was found for {symbol}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"The data source did not return: {', '.join(missing)}.")
    frame = frame[needed].dropna().copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def number_in(text: str, pattern: str, default: int) -> int:
    match = re.search(pattern, text, flags=re.I)
    return int(match.group(1)) if match else default


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    change = series.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def crossover(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left > right) & (left.shift(1) <= right.shift(1))


def crossunder(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left < right) & (left.shift(1) >= right.shift(1))


@dataclass
class ParsedStrategy:
    entry: pd.Series
    exit: pd.Series
    description: list[str]
    max_hold_days: Optional[int]
    stop_loss: Optional[float]
    take_profit: Optional[float]


def parse_strategy(text: str, data: pd.DataFrame) -> ParsedStrategy:
    """Translate common trading language into vectorized boolean signals."""
    raw = " ".join(text.lower().split())
    close = data["Close"]
    entry = pd.Series(False, index=data.index)
    exit_signal = pd.Series(False, index=data.index)
    descriptions: list[str] = []

    sma_periods = sorted({int(value) for value in re.findall(r"(?:sma|simple moving average)[ -]?(\d+)", raw)})
    ema_periods = sorted({int(value) for value in re.findall(r"(?:ema|exponential moving average)[ -]?(\d+)", raw)})
    for period in sma_periods:
        data[f"SMA {period}"] = close.rolling(period).mean()
    for period in ema_periods:
        data[f"EMA {period}"] = close.ewm(span=period, adjust=False).mean()

    # Find an explicit exit clause; otherwise use the same crossover in reverse.
    exit_words = re.search(r"(?:exit|sell|close|short)\s+(?:when|if|on)?\s*(.*)$", raw)
    entry_clause = raw[: exit_words.start()] if exit_words else raw
    exit_clause = exit_words.group(1) if exit_words else ""

    def crossover_signal(clause: str, direction: str) -> tuple[pd.Series, Optional[str]]:
        ma = re.search(
            r"(?:above|over|below|under|cross(?:es|ing)?\s+(?:above|below))\s+(?:the\s+)?(?:(sma|ema|simple moving average|exponential moving average)[ -]?(\d+)|(\d+)[ -]?(?:day|days?)[ -]?(sma|ema|simple moving average|exponential moving average))",
            clause,
        )
        if not ma:
            return None, None
        kind = ma.group(1) or ma.group(4)
        period = int(ma.group(2) or ma.group(3))
        key = f"{'SMA' if 'sma' in kind or 'simple' in kind else 'EMA'} {period}"
        if key not in data:
            data[key] = close.rolling(period).mean() if key.startswith("SMA") else close.ewm(span=period, adjust=False).mean()
        series = data[key]
        above = any(word in clause for word in ["above", "over"]) and "below" not in clause
        signal = crossover(close, series) if above else crossunder(close, series)
        return signal, f"Close crosses {'above' if above else 'below'} {key}"

    enter, enter_desc = crossover_signal(entry_clause, "entry")
    leave, leave_desc = crossover_signal(exit_clause, "exit") if exit_clause else (None, None)

    entry_rsi = re.search(r"rsi(?:\s*\(?\s*(\d+)\s*\)?)?\s*(?:is\s*)?(below|under|less than|above|over|greater than)\s*(\d+)", entry_clause)
    exit_rsi = re.search(r"rsi(?:\s*\(?\s*(\d+)\s*\)?)?\s*(?:is\s*)?(below|under|less than|above|over|greater than)\s*(\d+)", exit_clause)

    def rsi_signal(match: re.Match[str]) -> tuple[pd.Series, str]:
        period = int(match.group(1) or 14)
        threshold = float(match.group(3))
        values = rsi(close, period)
        is_below = match.group(2) in {"below", "under", "less than"}
        return ((values < threshold) if is_below else (values > threshold), f"RSI({period}) {'<' if is_below else '>'} {threshold:g}")

    if entry_rsi:
        enter, enter_desc = rsi_signal(entry_rsi)
    if exit_rsi:
        leave, leave_desc = rsi_signal(exit_rsi)

    if enter is None:
        period = number_in(raw, r"(?:sma|simple moving average)[ -]?(\d+)", 50)
        key = f"SMA {period}"
        if key not in data:
            data[key] = close.rolling(period).mean()
        enter, enter_desc = crossover(close, data[key]), f"Close crosses above {key} (default)"
    if leave is None:
        period = number_in(raw, r"(?:sma|simple moving average)[ -]?(\d+)", 50)
        key = f"SMA {period}"
        if key not in data:
            data[key] = close.rolling(period).mean()
        leave, leave_desc = crossunder(close, data[key]), f"Close crosses below {key} (default)"
    descriptions.extend([f"Enter: {enter_desc}", f"Exit: {leave_desc}"])

    volume_match = re.search(r"volume\s+(?:is\s+)?(?:above|over|greater than)\s+(?:its\s+)?(?:average|avg)(?:\s+volume)?(?:\s+(\d+)[ -]?day)?", raw)
    if volume_match:
        period = int(volume_match.group(1) or 20)
        entry &= data["Volume"] > data["Volume"].rolling(period).mean()
        descriptions.append(f"Filter: Volume > {period}-day average")

    hold_match = re.search(r"(?:hold|exit after|sell after)\s+(\d+)\s+days?", raw)
    stop_match = re.search(r"(?:stop loss|stop-loss)\s*(?:at|of)?\s*(\d+(?:\.\d+)?)\s*%", raw)
    profit_match = re.search(r"(?:take profit|profit target)\s*(?:at|of)?\s*(\d+(?:\.\d+)?)\s*%", raw)
    max_hold = int(hold_match.group(1)) if hold_match else None
    stop_loss = float(stop_match.group(1)) / 100 if stop_match else None
    take_profit = float(profit_match.group(1)) / 100 if profit_match else None
    if max_hold: descriptions.append(f"Time exit: {max_hold} trading days")
    if stop_loss: descriptions.append(f"Risk exit: {stop_loss:.1%} stop loss")
    if take_profit: descriptions.append(f"Profit exit: {take_profit:.1%} take profit")
    return ParsedStrategy(enter.fillna(False), leave.fillna(False), descriptions, max_hold, stop_loss, take_profit)


def backtest(data: pd.DataFrame, strategy: ParsedStrategy) -> tuple[dict, pd.DataFrame]:
    trades: list[dict] = []
    equity = pd.Series(1.0, index=data.index)
    position = None
    entry_price = None
    entry_date = None
    days_held = 0
    for i in range(len(data)):
        if position is None and i < len(data) - 1 and strategy.entry.iloc[i]:
            position, entry_price, entry_date, days_held = i + 1, float(data["Open"].iloc[i + 1]), data.index[i + 1], 0
        elif position is not None:
            days_held += 1
            current = float(data["Close"].iloc[i])
            pnl = current / entry_price - 1
            timed = strategy.max_hold_days is not None and days_held >= strategy.max_hold_days
            risk = strategy.stop_loss is not None and pnl <= -strategy.stop_loss
            target = strategy.take_profit is not None and pnl >= strategy.take_profit
            if strategy.exit.iloc[i] or timed or risk or target or i == len(data) - 1:
                exit_price = current if i == len(data) - 1 else float(data["Open"].iloc[i + 1])
                result = exit_price / entry_price - 1
                trades.append({"Entry": entry_date, "Exit": data.index[i] if i == len(data) - 1 else data.index[i + 1], "Entry price": entry_price, "Exit price": exit_price, "Return": result})
                equity.iloc[i:] *= 1 + result
                position = None
    if trades:
        trade_frame = pd.DataFrame(trades)
        wins = int((trade_frame["Return"] > 0).sum())
        losses = int((trade_frame["Return"] <= 0).sum())
        win_rate = wins / len(trade_frame)
        total_return = float(equity.iloc[-1] - 1)
    else:
        trade_frame = pd.DataFrame(columns=["Entry", "Exit", "Entry price", "Exit price", "Return"])
        wins = losses = 0
        win_rate = total_return = 0.0
    curve = equity.ffill()
    drawdown = curve / curve.cummax() - 1
    metrics = {"wins": wins, "losses": losses, "win_rate": win_rate, "return": total_return, "drawdown": float(drawdown.min()), "trades": len(trade_frame)}
    return metrics, trade_frame


def price_chart(data: pd.DataFrame, trades: pd.DataFrame, symbol: str) -> go.Figure:
    figure = go.Figure(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name=symbol))
    if not trades.empty:
        figure.add_trace(go.Scatter(x=trades["Entry"], y=trades["Entry price"], mode="markers", name="Entry", marker=dict(symbol="triangle-up", size=11, color="#18c29c")))
        figure.add_trace(go.Scatter(x=trades["Exit"], y=trades["Exit price"], mode="markers", name="Exit", marker=dict(symbol="triangle-down", size=11, color="#ff6b6b")))
    figure.update_layout(height=560, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    return figure


st.title("Signal Lab")
st.caption("Describe a long-only trading strategy in plain English, then test it on daily market data.")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", "SPY", max_chars=12).strip().upper()
    years = st.slider("Years of daily history", 1, 10, 10)
    st.caption("Yahoo Finance data · up to 10 years · trades fill at the next day's open")

strategy_text = st.text_area(
    "Strategy in natural language",
    value="Buy when close crosses above the 50-day SMA. Sell when close crosses below the 50-day SMA.",
    height=100,
    help="Examples: 'Buy when RSI is below 30, sell when RSI is above 70.' Add 'hold for 10 days', 'stop loss 5%', or 'take profit 12%'.",
)
run = st.button("Run backtest", type="primary", use_container_width=True)

if run or "backtest_data" not in st.session_state:
    if not ticker or not ticker.replace("-", "").replace(".", "").isalnum():
        st.error("Enter a valid ticker symbol, such as SPY or BRK-B.")
        st.stop()
    with st.spinner(f"Downloading {ticker} data and running the backtest..."):
        try:
            market_data = get_ohlcv(ticker, years)
            parsed = parse_strategy(strategy_text, market_data)
            metrics, trades = backtest(market_data, parsed)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()
    st.session_state.backtest_data = (market_data, parsed, metrics, trades, ticker, years)

market_data, parsed, metrics, trades, ticker, years = st.session_state.backtest_data
st.write(f"**{ticker}** · {market_data.index[0]:%b %d, %Y} – {market_data.index[-1]:%b %d, %Y} · {len(market_data):,} sessions")

metric_columns = st.columns(6)
metric_columns[0].metric("Wins", f"{metrics['wins']:,}")
metric_columns[1].metric("Losses", f"{metrics['losses']:,}")
metric_columns[2].metric("Win rate", f"{metrics['win_rate']:.1%}")
metric_columns[3].metric("Total return", f"{metrics['return']:+.1%}")
metric_columns[4].metric("Max drawdown", f"{metrics['drawdown']:.1%}")
metric_columns[5].metric("Trades", f"{metrics['trades']:,}")

with st.expander("Interpreted strategy", expanded=True):
    for description in parsed.description:
        st.write("• " + description)

st.plotly_chart(price_chart(market_data, trades, ticker), use_container_width=True)
if trades.empty:
    st.info("No trades matched the strategy in this period. Try a shorter moving average or different RSI thresholds.")
else:
    display_trades = trades.copy()
    display_trades["Return"] = display_trades["Return"].map(lambda value: f"{value:+.2%}")
    st.subheader("Trade log")
    st.dataframe(display_trades, use_container_width=True, hide_index=True)

st.caption("Educational use only. Historical results are hypothetical and do not guarantee future performance.")
