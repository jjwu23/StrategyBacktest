"""Signal Lab — S&P 500 stock screener and natural-language backtester."""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

SEC_HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "Signal Lab research contact@example.com",
    ),
    "Accept-Encoding": "gzip, deflate",
}

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

FINNHUB_BASE = "https://finnhub.io/api/v1"

SEC_COMPANY_TICKERS_URL = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)

PURCHASE_CODES = {"P", "P4"}
SALE_CODES = {"S", "S4"}

LONG_SCORE_MAX = 100
SHORT_SCORE_MAX = 100

st.set_page_config(
    page_title="Signal Lab",
    page_icon="◒",
    layout="wide",
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value) -> float:
    """Convert a value to float, returning NaN when unavailable."""
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def number_in(text: str, pattern: str, default: int) -> int:
    match = re.search(pattern, text, flags=re.I)
    return int(match.group(1)) if match else default


# ============================================================
# PRICE DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv(symbol: str, years: int) -> pd.DataFrame:
    """Download daily OHLCV data, capped at the last ten years."""

    end = date.today() + timedelta(days=1)
    start = end - timedelta(
        days=365 * min(max(years, 1), 10)
    )

    frame = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if frame.empty:
        raise ValueError(
            f"No daily data was found for {symbol}."
        )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    needed = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in needed
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "The data source did not return: "
            + ", ".join(missing)
        )

    frame = frame[needed].dropna().copy()

    frame.index = pd.to_datetime(
        frame.index
    ).tz_localize(None)

    return frame


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    change = series.diff()

    gain = change.clip(
        lower=0
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    loss = (
        -change.clip(upper=0)
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    return 100 - (
        100
        / (
            1
            + gain
            / loss.replace(0, np.nan)
        )
    )


def crossover(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    return (
        (left > right)
        & (left.shift(1) <= right.shift(1))
    )


def crossunder(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    return (
        (left < right)
        & (left.shift(1) >= right.shift(1))
    )


# ============================================================
# STRATEGY PARSER
# ============================================================

@dataclass
class ParsedStrategy:
    entry: pd.Series
    exit: pd.Series
    description: list[str]
    max_hold_days: Optional[int]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    atr_profit: Optional[float]
    atr_loss: Optional[float]
    atr_period: int


def parse_strategy(
    text: str,
    data: pd.DataFrame,
) -> ParsedStrategy:
    """Translate common trading language into signals."""

    raw = " ".join(text.lower().split())

    close = data["Close"]

    entry = pd.Series(
        False,
        index=data.index,
    )

    exit_signal = pd.Series(
        False,
        index=data.index,
    )

    descriptions: list[str] = []

    sma_periods = sorted(
        {
            int(value)
            for value in re.findall(
                r"(?:sma|simple moving average)[ -]?(\d+)",
                raw,
            )
        }
    )

    ema_periods = sorted(
        {
            int(value)
            for value in re.findall(
                r"(?:ema|exponential moving average)[ -]?(\d+)",
                raw,
            )
        }
    )

    for period in sma_periods:
        data[f"SMA {period}"] = (
            close.rolling(period).mean()
        )

    for period in ema_periods:
        data[f"EMA {period}"] = (
            close.ewm(
                span=period,
                adjust=False,
            ).mean()
        )

    exit_words = re.search(
        r"\b(?:exit|sell|short)\b\s*(?:when|if|on)?\s*(.*)$",
        raw,
    )

    if exit_words is None:
        exit_words = re.search(
            r"\bclose\s+(?:(?:the\s+)?(?:position|trade)|when|if)\s*(.*)$",
            raw,
        )

    entry_clause = (
        raw[:exit_words.start()]
        if exit_words
        else raw
    )

    exit_clause = (
        exit_words.group(1)
        if exit_words
        else ""
    )

    def moving_average_signals(
        clause: str,
    ) -> tuple[
        Optional[pd.Series],
        list[str],
    ]:

        pattern = re.compile(
            r"(?:(?P<relation>cross(?:es|ing)?\s+above|"
            r"cross(?:es|ing)?\s+below|above|over|below|"
            r"under|at|near|touch(?:es)?)\s+(?:the\s+)?)*"
            r"(?:(?P<kind>sma|ema|simple moving average|"
            r"exponential moving average)[ -]?(?P<period>\d+)|"
            r"(?P<period2>\d+)[ -]?(?P<unit>day|days|"
            r"week|weeks)[ -]?(?P<kind2>sma|ema|moving average|ma))",
            flags=re.I,
        )

        signals = []
        labels = []

        for match in pattern.finditer(clause):

            kind = (
                match.group("kind")
                or match.group("kind2")
                or "sma"
            ).lower()

            period = int(
                match.group("period")
                or match.group("period2")
            )

            unit = (
                match.group("unit")
                or "day"
            ).lower()

            daily_period = (
                period * 5
                if unit.startswith("week")
                else period
            )

            is_ema = (
                "ema" in kind
                or "exponential" in kind
            )

            key = (
                f"{'EMA' if is_ema else 'SMA'} "
                f"{daily_period}"
            )

            if key not in data:

                if is_ema:
                    data[key] = close.ewm(
                        span=daily_period,
                        adjust=False,
                    ).mean()
                else:
                    data[key] = close.rolling(
                        daily_period
                    ).mean()

            average = data[key]

            relation = (
                match.group("relation")
                or "at"
            ).lower()

            if relation in {
                "at",
                "near",
                "touch",
                "touches",
            }:

                signal = (
                    (data["Low"] <= average)
                    & (data["High"] >= average)
                )

                action = "touches"

            elif (
                "below" in relation
                or relation == "under"
            ):

                signal = crossunder(
                    close,
                    average,
                )

                action = "crosses below"

            else:

                signal = crossover(
                    close,
                    average,
                )

                action = "crosses above"

            signals.append(
                signal.fillna(False)
            )

            original_period = (
                f"{period}-{unit.rstrip('s')}"
            )

            labels.append(
                f"Close {action} "
                f"{original_period} "
                f"{'EMA' if is_ema else 'moving average'}"
            )

        if not signals:
            return None, []

        combined = signals[0]

        for signal in signals[1:]:
            combined |= signal

        return combined, labels

    enter, enter_labels = moving_average_signals(
        entry_clause
    )

    leave, leave_labels = (
        moving_average_signals(exit_clause)
        if exit_clause
        else (None, [])
    )

    enter_desc = (
        " or ".join(enter_labels)
        if enter_labels
        else None
    )

    leave_desc = (
        " or ".join(leave_labels)
        if leave_labels
        else None
    )

    entry_rsi = re.search(
        r"rsi(?:\s*\(?\s*(\d+)\s*\)?)?\s*"
        r"(?:is\s*)?(below|under|less than|above|"
        r"over|greater than)\s*(\d+)",
        entry_clause,
    )

    exit_rsi = re.search(
        r"rsi(?:\s*\(?\s*(\d+)\s*\)?)?\s*"
        r"(?:is\s*)?(below|under|less than|above|"
        r"over|greater than)\s*(\d+)",
        exit_clause,
    )

    def rsi_signal(
        match: re.Match[str],
    ) -> tuple[pd.Series, str]:

        period = int(
            match.group(1) or 14
        )

        threshold = float(
            match.group(3)
        )

        values = rsi(
            close,
            period,
        )

        is_below = match.group(2) in {
            "below",
            "under",
            "less than",
        }

        return (
            (
                values < threshold
                if is_below
                else values > threshold
            ),
            (
                f"RSI({period}) "
                f"{'<' if is_below else '>'} "
                f"{threshold:g}"
            ),
        )

    if entry_rsi:
        enter, enter_desc = rsi_signal(
            entry_rsi
        )

    if exit_rsi:
        leave, leave_desc = rsi_signal(
            exit_rsi
        )

    if enter is None:

        period = number_in(
            raw,
            r"(?:sma|simple moving average)[ -]?(\d+)",
            50,
        )

        key = f"SMA {period}"

        if key not in data:
            data[key] = close.rolling(
                period
            ).mean()

        enter = crossover(
            close,
            data[key],
        )

        enter_desc = (
            f"Close crosses above {key} "
            "(default)"
        )

    if (
        leave is None
        and not re.search(
            r"\batr\b|average true range",
            exit_clause,
        )
    ):

        period = number_in(
            raw,
            r"(?:sma|simple moving average)[ -]?(\d+)",
            50,
        )

        key = f"SMA {period}"

        if key not in data:
            data[key] = close.rolling(
                period
            ).mean()

        leave = crossunder(
            close,
            data[key],
        )

        leave_desc = (
            f"Close crosses below {key} "
            "(default)"
        )

    elif leave is None:

        leave = pd.Series(
            False,
            index=data.index,
        )

        leave_desc = "ATR thresholds"

    descriptions.extend(
        [
            f"Enter: {enter_desc}",
            f"Exit: {leave_desc}",
        ]
    )

    volume_match = re.search(
        r"volume\s+(?:is\s+)?(?:above|over|greater than)\s+"
        r"(?:its\s+)?(?:average|avg)(?:\s+volume)?"
        r"(?:\s+(\d+)[ -]?day)?",
        raw,
    )

    if volume_match:

        period = int(
            volume_match.group(1) or 20
        )

        entry &= (
            data["Volume"]
            > data["Volume"].rolling(
                period
            ).mean()
        )

        descriptions.append(
            f"Filter: Volume > "
            f"{period}-day average"
        )

    hold_match = re.search(
        r"(?:hold|exit after|sell after)\s+(\d+)\s+days?",
        raw,
    )

    stop_match = re.search(
        r"(?:stop loss|stop-loss)\s*(?:at|of)?\s*"
        r"(\d+(?:\.\d+)?)\s*%",
        raw,
    )

    profit_match = re.search(
        r"(?:take profit|profit target)\s*(?:at|of)?\s*"
        r"(\d+(?:\.\d+)?)\s*%",
        raw,
    )

    atr_matches = re.findall(
        r"([+-]?)\s*(\d+(?:\.\d+)?)\s*atr"
        r"(?:\s*\(?\s*(win|loss|profit|target|stop)\s*\)?)?",
        exit_clause,
    )

    atr_profit = next(
        (
            float(value)
            for sign, value, label
            in atr_matches
            if sign == "+"
            or label in {
                "win",
                "profit",
                "target",
            }
        ),
        None,
    )

    atr_loss = next(
        (
            float(value)
            for sign, value, label
            in atr_matches
            if sign == "-"
            or label in {
                "loss",
                "stop",
            }
        ),
        None,
    )

    atr_period = number_in(
        raw,
        r"(?:atr|average true range)\s*\(?\s*(\d+)\s*\)?",
        14,
    )

    max_hold = (
        int(hold_match.group(1))
        if hold_match
        else None
    )

    stop_loss = (
        float(stop_match.group(1)) / 100
        if stop_match
        else None
    )

    take_profit = (
        float(profit_match.group(1)) / 100
        if profit_match
        else None
    )

    if max_hold:
        descriptions.append(
            f"Time exit: "
            f"{max_hold} trading days"
        )

    if stop_loss:
        descriptions.append(
            f"Risk exit: "
            f"{stop_loss:.1%} stop loss"
        )

    if take_profit:
        descriptions.append(
            f"Profit exit: "
            f"{take_profit:.1%} take profit"
        )

    if atr_profit is not None:
        descriptions.append(
            f"ATR profit exit: "
            f"+{atr_profit:g} × ATR({atr_period})"
        )

    if atr_loss is not None:
        descriptions.append(
            f"ATR loss exit: "
            f"-{atr_loss:g} × ATR({atr_period})"
        )

    return ParsedStrategy(
        entry=enter.fillna(False),
        exit=leave.fillna(False),
        description=descriptions,
        max_hold_days=max_hold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr_profit=atr_profit,
        atr_loss=atr_loss,
        atr_period=atr_period,
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(
    data: pd.DataFrame,
    strategy: ParsedStrategy,
) -> tuple[dict, pd.DataFrame]:

    trades: list[dict] = []

    equity = pd.Series(
        1.0,
        index=data.index,
    )

    position = None
    entry_price = None
    entry_date = None
    days_held = 0

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (
                data["High"]
                - data["Close"].shift()
            ).abs(),
            (
                data["Low"]
                - data["Close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        strategy.atr_period
    ).mean()

    for i in range(len(data)):

        if (
            position is None
            and i < len(data) - 1
            and strategy.entry.iloc[i]
        ):

            position = i + 1
            entry_price = float(
                data["Open"].iloc[i + 1]
            )
            entry_date = data.index[i + 1]
            days_held = 0

        elif position is not None:

            days_held += 1

            current = float(
                data["Close"].iloc[i]
            )

            pnl = (
                current / entry_price
                - 1
            )

            current_atr = (
                float(atr.iloc[i])
                if pd.notna(atr.iloc[i])
                else 0.0
            )

            timed = (
                strategy.max_hold_days is not None
                and days_held
                >= strategy.max_hold_days
            )

            risk = (
                strategy.stop_loss is not None
                and pnl <= -strategy.stop_loss
            )

            target = (
                strategy.take_profit is not None
                and pnl >= strategy.take_profit
            )

            atr_risk = (
                strategy.atr_loss is not None
                and current_atr > 0
                and current
                <= entry_price
                - strategy.atr_loss
                * current_atr
            )

            atr_target = (
                strategy.atr_profit is not None
                and current_atr > 0
                and current
                >= entry_price
                + strategy.atr_profit
                * current_atr
            )

            if (
                strategy.exit.iloc[i]
                or timed
                or risk
                or target
                or atr_risk
                or atr_target
                or i == len(data) - 1
            ):

                exit_price = (
                    current
                    if i == len(data) - 1
                    else float(
                        data["Open"].iloc[i + 1]
                    )
                )

                result = (
                    exit_price
                    / entry_price
                    - 1
                )

                trades.append(
                    {
                        "Entry": entry_date,
                        "Exit": (
                            data.index[i]
                            if i == len(data) - 1
                            else data.index[i + 1]
                        ),
                        "Entry price": entry_price,
                        "Exit price": exit_price,
                        "Return": result,
                        "Outcome": (
                            "Win"
                            if result > 0
                            else "Loss"
                        ),
                    }
                )

                equity.iloc[i:] *= (
                    1 + result
                )

                position = None

    if trades:

        trade_frame = pd.DataFrame(
            trades
        )

        wins = int(
            (
                trade_frame["Return"] > 0
            ).sum()
        )

        losses = int(
            (
                trade_frame["Return"] <= 0
            ).sum()
        )

        win_rate = (
            wins / len(trade_frame)
        )

        total_return = float(
            equity.iloc[-1] - 1
        )

    else:

        trade_frame = pd.DataFrame(
            columns=[
                "Entry",
                "Exit",
                "Entry price",
                "Exit price",
                "Return",
                "Outcome",
            ]
        )

        wins = 0
        losses = 0
        win_rate = 0.0
        total_return = 0.0

    curve = equity.ffill()

    drawdown = (
        curve / curve.cummax()
        - 1
    )

    metrics = {
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "return": total_return,
        "drawdown": float(
            drawdown.min()
        ),
        "trades": len(trade_frame),
    }

    return metrics, trade_frame


# ============================================================
# PRICE CHART
# ============================================================

def signal_chart(
    data: pd.DataFrame,
    symbol: str,
    insider_transactions: Optional[pd.DataFrame] = None,
    long_signals: Optional[pd.Series] = None,
    short_signals: Optional[pd.Series] = None,
) -> go.Figure:

    frame = add_technical_features(
        data
    )

    figure = go.Figure()

    # --------------------------------------------------------
    # Candlesticks
    # --------------------------------------------------------

    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name=symbol,
        )
    )

    # --------------------------------------------------------
    # 8 EMA — cyan
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["EMA 8"],
            mode="lines",
            name="8 EMA",
            line=dict(
                color="#00FFFF",
                width=1.5,
            ),
        )
    )

    # --------------------------------------------------------
    # 20 EMA — orange
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["EMA 20"],
            mode="lines",
            name="20 EMA",
            line=dict(
                color="#FF8C00",
                width=1.8,
            ),
        )
    )

    # --------------------------------------------------------
    # 200 EMA — yellow
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["EMA 200"],
            mode="lines",
            name="200 EMA",
            line=dict(
                color="#FFD700",
                width=2,
            ),
        )
    )

    # --------------------------------------------------------
    # 200 WEEK EMA — thick yellow
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["EMA 200 Week"],
            mode="lines",
            name="200 Week EMA",
            line=dict(
                color="#FFD700",
                width=4,
            ),
        )
    )

    # --------------------------------------------------------
    # Insider purchases — purple dots
    # --------------------------------------------------------

    if (
        insider_transactions is not None
        and not insider_transactions.empty
    ):

        insider = (
            insider_transactions
            .copy()
        )

        insider["Transaction Date"] = (
            pd.to_datetime(
                insider["Transaction Date"],
                errors="coerce",
            )
        )

        insider = insider.dropna(
            subset=["Transaction Date"]
        )

        for _, row in insider.iterrows():

            transaction_date = (
                row["Transaction Date"]
            )

            matching_dates = frame.index[
                frame.index.normalize()
                == transaction_date.normalize()
            ]

            if len(matching_dates) == 0:
                continue

            chart_date = matching_dates[0]

            # Price level is the actual
            # Form 4 transaction price.
            purchase_price = float(
                row["Price"]
            )

            market_cap = np.nan

            pct_market_cap = np.nan

            figure.add_trace(
                go.Scatter(
                    x=[chart_date],
                    y=[purchase_price],
                    mode="markers",
                    name="Insider Purchase",
                    marker=dict(
                        color="#B000FF",
                        size=10,
                        symbol="circle",
                    ),
                    customdata=[[
                        row.get(
                            "Insider",
                            "Unknown",
                        ),
                        row.get(
                            "Value",
                            np.nan,
                        ),
                        pct_market_cap,
                    ]],
                    hovertemplate=(
                        "<b>Insider Purchase</b><br>"
                        "Insider: %{customdata[0]}<br>"
                        "Price: $%{y:.2f}<br>"
                        "Purchase Value: $%{customdata[1]:,.0f}"
                        "<br>"
                        "Market Cap: %{customdata[2]:.3%}"
                        "<extra></extra>"
                    ),
                )
            )

    # --------------------------------------------------------
    # Long signals — green upward triangles
    # --------------------------------------------------------

    if long_signals is not None:

        long_dates = frame.index[
            long_signals.reindex(
                frame.index,
                fill_value=False,
            )
        ]

        if len(long_dates):

            figure.add_trace(
                go.Scatter(
                    x=long_dates,
                    y=frame.loc[
                        long_dates,
                        "Low",
                    ] * 0.985,
                    mode="markers",
                    name="Long Signal",
                    marker=dict(
                        symbol="triangle-up",
                        color="#00E676",
                        size=13,
                    ),
                )
            )

    # --------------------------------------------------------
    # Short signals — red downward triangles
    # --------------------------------------------------------

    if short_signals is not None:

        short_dates = frame.index[
            short_signals.reindex(
                frame.index,
                fill_value=False,
            )
        ]

        if len(short_dates):

            figure.add_trace(
                go.Scatter(
                    x=short_dates,
                    y=frame.loc[
                        short_dates,
                        "High",
                    ] * 1.015,
                    mode="markers",
                    name="Short Signal",
                    marker=dict(
                        symbol="triangle-down",
                        color="#FF5252",
                        size=13,
                    ),
                )
            )

    figure.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10,
        ),
        legend=dict(
            orientation="h",
            y=1.02,
        ),
    )

    return figure


# ============================================================
# S&P 500 CONSTITUENTS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_listed_universe() -> pd.DataFrame:
    """
    Load U.S.-listed companies from SEC ticker/exchange data.

    Restrict the universe to NYSE and NASDAQ.
    """

    response = requests.get(
        SEC_COMPANY_TICKERS_URL,
        headers=SEC_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    if "data" not in payload:
        raise ValueError(
            "SEC ticker/exchange file did not contain data."
        )

    frame = pd.DataFrame(
        payload["data"],
        columns=[
            "CIK",
            "Symbol",
            "Company",
            "Exchange",
        ],
    )

    frame["Symbol"] = (
        frame["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )

    frame["Company"] = (
        frame["Company"]
        .astype(str)
        .str.strip()
    )

    frame["Exchange"] = (
        frame["Exchange"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    frame["CIK"] = (
        pd.to_numeric(
            frame["CIK"],
            errors="coerce",
        )
        .astype("Int64")
    )

    frame = frame[
        frame["Exchange"].isin(
            ["NYSE", "NASDAQ"]
        )
    ].copy()

    # Remove obvious non-operating securities.
    excluded = frame["Symbol"].str.contains(
        r"[\^/]",
        regex=True,
        na=False,
    )

    frame = frame.loc[~excluded]

    frame = (
        frame
        .dropna(subset=["CIK", "Symbol"])
        .drop_duplicates("Symbol")
        .sort_values("Symbol")
        .reset_index(drop=True)
    )

    return frame


# ============================================================
# INSIDER ACTIVITY
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_sec_submissions(cik: int) -> dict:
    cik_string = f"{int(cik):010d}"

    url = (
        "https://data.sec.gov/submissions/"
        f"CIK{cik_string}.json"
    )

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def get_recent_form4_filings(
    cik: int,
    days: int = 90,
) -> pd.DataFrame:

    payload = get_sec_submissions(cik)

    recent = payload.get(
        "filings",
        {},
    ).get(
        "recent",
        {},
    )

    if not recent:
        return pd.DataFrame()

    frame = pd.DataFrame(recent)

    if frame.empty:
        return frame

    frame["filingDate"] = pd.to_datetime(
        frame["filingDate"],
        errors="coerce",
    )

    cutoff = (
        pd.Timestamp.utcnow().tz_localize(None)
        - pd.Timedelta(days=days)
    )

    frame = frame[
        (frame["form"] == "4")
        & (frame["filingDate"] >= cutoff)
    ].copy()

    return frame


def parse_form4_xml(
    xml_text: str,
) -> list[dict]:

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    namespace = ""

    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"

    rows = []

    issuer = root.find(
        f".//{namespace}issuer",
    )

    issuer_symbol = ""

    if issuer is not None:
        symbol_node = issuer.find(
            f"{namespace}issuerTradingSymbol"
        )

        if symbol_node is not None:
            issuer_symbol = (
                symbol_node.text or ""
            ).strip().upper()

    reporting = root.find(
        f".//{namespace}reportingOwner"
    )

    insider_name = ""

    if reporting is not None:
        name_node = reporting.find(
            f".//{namespace}rptOwnerName"
        )

        if name_node is not None:
            insider_name = (
                name_node.text or ""
            ).strip()

    for transaction in root.findall(
        f".//{namespace}nonDerivativeTable/"
        f"{namespace}nonDerivativeTransaction"
    ):

        code_node = transaction.find(
            f".//{namespace}transactionCoding/"
            f"{namespace}transactionCode"
        )

        if code_node is None:
            continue

        code = (
            code_node.text or ""
        ).strip().upper()

        date_node = transaction.find(
            f".//{namespace}transactionDate/"
            f"{namespace}value"
        )

        shares_node = transaction.find(
            f".//{namespace}transactionAmounts/"
            f"{namespace}transactionShares/"
            f"{namespace}value"
        )

        price_node = transaction.find(
            f".//{namespace}transactionAmounts/"
            f"{namespace}transactionPricePerShare/"
            f"{namespace}value"
        )

        acquired_disposed = transaction.find(
            f".//{namespace}transactionAmounts/"
            f"{namespace}transactionAcquiredDisposedCode/"
            f"{namespace}value"
        )

        transaction_date = (
            pd.to_datetime(
                date_node.text,
                errors="coerce",
            )
            if date_node is not None
            else pd.NaT
        )

        shares = (
            safe_float(shares_node.text)
            if shares_node is not None
            else np.nan
        )

        price = (
            safe_float(price_node.text)
            if price_node is not None
            else np.nan
        )

        acquired = (
            acquired_disposed is not None
            and (
                acquired_disposed.text or ""
            ).upper() == "A"
        )

        if (
            code in PURCHASE_CODES
            and acquired
            and pd.notna(shares)
            and pd.notna(price)
        ):

            rows.append(
                {
                    "Symbol": issuer_symbol,
                    "Insider": insider_name,
                    "Transaction Date": transaction_date,
                    "Code": code,
                    "Shares": shares,
                    "Price": price,
                    "Value": shares * price,
                }
            )

    return rows


def get_insider_purchase_data(
    cik: int,
    market_cap: float,
) -> dict:

    result = {
        "Insider Buy Count": 0,
        "Insider Buy Value": 0.0,
        "Insider Buyers": 0,
        "Insider Cluster": False,
        "Insider Buy % Market Cap": np.nan,
        "Insider Transactions": pd.DataFrame(),
    }

    try:

        filings = get_recent_form4_filings(
            cik,
            days=90,
        )

        if filings.empty:
            return result

        purchases = []

        for _, filing in filings.iterrows():

            accession = str(
                filing["accessionNumber"]
            ).replace(
                "-",
                "",
            )

            primary_document = str(
                filing["primaryDocument"]
            )

            archive_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/"
                f"{accession}/"
                f"{primary_document}"
            )

            try:

                response = requests.get(
                    archive_url,
                    headers=SEC_HEADERS,
                    timeout=20,
                )

                if response.status_code != 200:
                    continue

                xml_rows = parse_form4_xml(
                    response.text
                )

                purchases.extend(xml_rows)

            except Exception:
                continue

            time.sleep(0.12)

        if not purchases:
            return result

        purchases = pd.DataFrame(
            purchases
        )

        purchases["Transaction Date"] = (
            pd.to_datetime(
                purchases["Transaction Date"],
                errors="coerce",
            )
        )

        purchases = purchases.dropna(
            subset=["Transaction Date"]
        )

        if purchases.empty:
            return result

        result[
            "Insider Transactions"
        ] = purchases

        result["Insider Buy Count"] = int(
            len(purchases)
        )

        result["Insider Buy Value"] = float(
            purchases["Value"].sum()
        )

        result["Insider Buyers"] = int(
            purchases["Insider"].nunique()
        )

        result["Insider Cluster"] = (
            result["Insider Buyers"] >= 2
        )

        if (
            pd.notna(market_cap)
            and market_cap > 0
        ):

            result[
                "Insider Buy % Market Cap"
            ] = (
                result["Insider Buy Value"]
                / market_cap
            )

    except Exception:
        pass

    return result










def finnhub_get(
    endpoint: str,
    params: Optional[dict] = None,
) -> Optional[object]:

    if not FINNHUB_API_KEY:
        return None

    params = dict(params or {})
    params["token"] = FINNHUB_API_KEY

    try:

        response = requests.get(
            f"{FINNHUB_BASE}/{endpoint}",
            params=params,
            timeout=20,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def get_analyst_data(
    symbol: str,
) -> dict:

    result = {
        "Analyst Rating": "",
        "Buy %": np.nan,
        "Sell %": np.nan,
        "Buy % YoY": np.nan,
        "Sell % YoY": np.nan,
        "Target Price": np.nan,
        "Target Price YoY": np.nan,
        "Analyst Count": np.nan,
    }

    recommendations = finnhub_get(
        "stock/recommendation",
        {"symbol": symbol},
    )

    if isinstance(
        recommendations,
        list,
    ) and recommendations:

        frame = pd.DataFrame(
            recommendations
        )

        frame["period"] = pd.to_datetime(
            frame["period"],
            errors="coerce",
        )

        frame = frame.sort_values(
            "period"
        ).dropna(
            subset=["period"]
        )

        if not frame.empty:

            latest = frame.iloc[-1]

            total = (
                latest.get("buy", 0)
                + latest.get("strongBuy", 0)
                + latest.get("hold", 0)
                + latest.get("sell", 0)
                + latest.get("strongSell", 0)
            )

            if total > 0:

                buy_count = (
                    latest.get("buy", 0)
                    + latest.get(
                        "strongBuy",
                        0,
                    )
                )

                sell_count = (
                    latest.get("sell", 0)
                    + latest.get(
                        "strongSell",
                        0,
                    )
                )

                result["Buy %"] = (
                    buy_count / total
                )

                result["Sell %"] = (
                    sell_count / total
                )

                result[
                    "Analyst Count"
                ] = total

                if (
                    buy_count / total
                    >= 0.50
                ):
                    result[
                        "Analyst Rating"
                    ] = "Buy"

                elif (
                    sell_count / total
                    >= 0.50
                ):
                    result[
                        "Analyst Rating"
                    ] = "Sell"

                else:
                    result[
                        "Analyst Rating"
                    ] = "Hold"

            # Approximately 12 months earlier.
            target_date = (
                latest["period"]
                - pd.DateOffset(months=12)
            )

            prior_idx = (
                frame["period"]
                - target_date
            ).abs().idxmin()

            prior = frame.loc[
                prior_idx
            ]

            prior_total = (
                prior.get("buy", 0)
                + prior.get("strongBuy", 0)
                + prior.get("hold", 0)
                + prior.get("sell", 0)
                + prior.get("strongSell", 0)
            )

            if prior_total > 0:

                prior_buy = (
                    prior.get("buy", 0)
                    + prior.get(
                        "strongBuy",
                        0,
                    )
                )

                prior_sell = (
                    prior.get("sell", 0)
                    + prior.get(
                        "strongSell",
                        0,
                    )
                )

                result["Buy % YoY"] = (
                    result["Buy %"]
                    - prior_buy / prior_total
                )

                result["Sell % YoY"] = (
                    result["Sell %"]
                    - prior_sell / prior_total
                )

    target = finnhub_get(
        "stock/price-target",
        {"symbol": symbol},
    )

    if isinstance(target, dict):

        target_mean = safe_float(
            target.get("targetMean")
        )

        result[
            "Target Price"
        ] = target_mean

        result[
            "Analyst Count"
        ] = safe_float(
            target.get("numberAnalysts")
        )

        # Finnhub's current endpoint does not
        # always expose historical target values.
        #
        # Leave this NaN rather than pretending
        # current target data is historical data.

    earnings = finnhub_get(
        "stock/earnings",
        {
            "symbol": symbol,
            "limit": 4,
        },
    )

    if isinstance(
        earnings,
        list,
    ) and earnings:

        earnings_frame = pd.DataFrame(
            earnings
        )

        result[
            "Earnings Beats"
        ] = int(
            (
                pd.to_numeric(
                    earnings_frame["actual"],
                    errors="coerce",
                )
                >
                pd.to_numeric(
                    earnings_frame["estimate"],
                    errors="coerce",
                )
            ).sum()
        )

        result[
            "Earnings Quarters"
        ] = len(
            earnings_frame
        )

    else:

        result[
            "Earnings Beats"
        ] = np.nan

        result[
            "Earnings Quarters"
        ] = np.nan

    return result









def get_institutional_data(
    symbol: str,
) -> dict:

    result = {
        "Institution Net Purchase": np.nan,
        "Institution Net Value": np.nan,
        "Institutional Buying": False,
    }

    data = finnhub_get(
        "institutional/ownership",
        {
            "symbol": symbol,
        },
    )

    if data is None:
        return result

    rows = []

    if isinstance(data, dict):
        rows = data.get(
            "data",
            data.get(
                "ownership",
                [],
            ),
        )

    elif isinstance(data, list):
        rows = data

    if not rows:
        return result

    frame = pd.DataFrame(rows)

    if frame.empty:
        return result

    if "change" in frame.columns:

        changes = pd.to_numeric(
            frame["change"],
            errors="coerce",
        ).dropna()

        if not changes.empty:

            net_purchase = float(
                changes.sum()
            )

            result[
                "Institution Net Purchase"
            ] = net_purchase

            result[
                "Institutional Buying"
            ] = (
                net_purchase > 0
            )

    if "value" in frame.columns:

        values = pd.to_numeric(
            frame["value"],
            errors="coerce",
        ).dropna()

        if not values.empty:

            result[
                "Institution Net Value"
            ] = float(
                values.sum()
            )

    return result








def score_directional_stocks(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    data = frame.copy()

    # ========================================================
    # LONG SCORE
    # ========================================================

    data["Long Revenue Score"] = np.where(
        data["Revenue Growth YoY"] >= 0.15,
        10,
        0,
    )

    data["Long EPS Score"] = np.where(
        data["EPS Growth YoY"] >= 0.15,
        10,
        0,
    )

    data["Long Earnings Beat Score"] = np.where(
        data["Earnings Beat"],
        10,
        0,
    )

    data["Long FCF Score"] = np.where(
        data["FCF Growth YoY"] >= 0.15,
        10,
        0,
    )

    data["Long Positive Margin Score"] = np.where(
        (
            (data["Gross Margin"] > 0)
            & (data["Operating Margin"] > 0)
            & (data["Net Margin"] > 0)
        ),
        10,
        0,
    )

    data["Long Expanding Margin Score"] = np.where(
        (
            (data["Gross Margin YoY"] > 0)
            & (data["Operating Margin YoY"] > 0)
            & (data["Net Margin YoY"] > 0)
        ),
        10,
        0,
    )

    data["Long Analyst Score"] = np.where(
        data["Analyst Rating"].isin(
            ["Buy", "Hold"]
        ),
        10,
        0,
    )

    data["Long Buy Trend Score"] = np.where(
        data["Buy % YoY"] > 0,
        10,
        0,
    )

    data["Long Target Trend Score"] = np.where(
        data["Target Price YoY"] > 0,
        10,
        0,
    )

    data["Long Target Upside Score"] = np.where(
        data["Target Upside"] >= 0.15,
        10,
        0,
    )

    data["Long Insider Score"] = np.where(
        data["Insider Cluster"],
        10,
        0,
    )

    data["Long Institution Score"] = np.where(
        data["Institutional Buying"],
        10,
        0,
    )

    long_columns = [
        "Long Revenue Score",
        "Long EPS Score",
        "Long Earnings Beat Score",
        "Long FCF Score",
        "Long Positive Margin Score",
        "Long Expanding Margin Score",
        "Long Analyst Score",
        "Long Buy Trend Score",
        "Long Target Trend Score",
        "Long Target Upside Score",
        "Long Insider Score",
        "Long Institution Score",
    ]

    data["Long Score"] = data[
        long_columns
    ].sum(axis=1)

    # ========================================================
    # SHORT SCORE
    # ========================================================

    data["Short Revenue Score"] = np.where(
        data["Revenue Growth YoY"] < 0,
        10,
        0,
    )

    data["Short EPS Score"] = np.where(
        data["EPS Growth YoY"] < 0,
        10,
        0,
    )

    data["Short Earnings Score"] = np.where(
        ~data["Earnings Beat"],
        10,
        0,
    )

    data["Short FCF Score"] = np.where(
        data["FCF Growth YoY"] < 0,
        10,
        0,
    )

    data["Short Positive Margin Score"] = np.where(
        (
            (data["Gross Margin"] < 0)
            | (data["Operating Margin"] < 0)
            | (data["Net Margin"] < 0)
        ),
        10,
        0,
    )

    data["Short Contracting Margin Score"] = np.where(
        (
            (data["Gross Margin YoY"] < 0)
            & (data["Operating Margin YoY"] < 0)
            & (data["Net Margin YoY"] < 0)
        ),
        10,
        0,
    )

    data["Short Analyst Score"] = np.where(
        data["Analyst Rating"].isin(
            ["Hold", "Sell"]
        ),
        10,
        0,
    )

    data["Short Sell Trend Score"] = np.where(
        data["Sell % YoY"] > 0,
        10,
        0,
    )

    data["Short Target Trend Score"] = np.where(
        data["Target Price YoY"] < 0,
        10,
        0,
    )

    data["Short Target Downside Score"] = np.where(
        data["Target Upside"] <= -0.15,
        10,
        0,
    )

    data["Short Insider Score"] = np.where(
        ~data["Insider Cluster"],
        10,
        0,
    )

    data["Short Institution Score"] = np.where(
        data["Institution Net Purchase"] < 0,
        10,
        0,
    )

    short_columns = [
        "Short Revenue Score",
        "Short EPS Score",
        "Short Earnings Score",
        "Short FCF Score",
        "Short Positive Margin Score",
        "Short Contracting Margin Score",
        "Short Analyst Score",
        "Short Sell Trend Score",
        "Short Target Trend Score",
        "Short Target Downside Score",
        "Short Insider Score",
        "Short Institution Score",
    ]

    data["Short Score"] = data[
        short_columns
    ].sum(axis=1)

    return data










def latest_yoy(
    series: pd.Series,
) -> float:

    if series is None:
        return np.nan

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(series) < 2:
        return np.nan

    current = float(series.iloc[0])
    prior = float(series.iloc[1])

    if prior == 0:
        return np.nan

    return (
        current / prior
        - 1
    )


def margin_yoy(
    income: pd.DataFrame,
    numerator_name: str,
) -> tuple[float, float]:

    if income is None or income.empty:
        return np.nan, np.nan

    if (
        numerator_name not in income.index
        or "Total Revenue" not in income.index
    ):
        return np.nan, np.nan

    revenue = pd.to_numeric(
        income.loc["Total Revenue"],
        errors="coerce",
    )

    numerator = pd.to_numeric(
        income.loc[numerator_name],
        errors="coerce",
    )

    common = pd.concat(
        [numerator, revenue],
        axis=1,
    ).dropna()

    if len(common) < 2:
        return np.nan, np.nan

    current_margin = (
        common.iloc[0, 0]
        / common.iloc[0, 1]
    )

    prior_margin = (
        common.iloc[1, 0]
        / common.iloc[1, 1]
    )

    return (
        current_margin,
        current_margin - prior_margin,
    )


def calculate_financial_metrics(
    ticker: yf.Ticker,
) -> dict:

    result = {
        "Revenue Growth YoY": np.nan,
        "EPS Growth YoY": np.nan,
        "FCF Growth YoY": np.nan,
        "Gross Margin": np.nan,
        "Operating Margin": np.nan,
        "Net Margin": np.nan,
        "Gross Margin YoY": np.nan,
        "Operating Margin YoY": np.nan,
        "Net Margin YoY": np.nan,
    }

    try:

        income = ticker.income_stmt

    except Exception:
        income = pd.DataFrame()

    try:

        cashflow = ticker.cashflow

    except Exception:
        cashflow = pd.DataFrame()

    if (
        income is None
        or income.empty
    ):
        return result

    revenue = (
        income.loc["Total Revenue"]
        if "Total Revenue" in income.index
        else pd.Series(dtype=float)
    )

    result[
        "Revenue Growth YoY"
    ] = latest_yoy(revenue)

    if "Diluted EPS" in income.index:
        result[
            "EPS Growth YoY"
        ] = latest_yoy(
            income.loc["Diluted EPS"]
        )

    elif "Basic EPS" in income.index:
        result[
            "EPS Growth YoY"
        ] = latest_yoy(
            income.loc["Basic EPS"]
        )

    gross, gross_yoy = margin_yoy(
        income,
        "Gross Profit",
    )

    operating, operating_yoy = margin_yoy(
        income,
        "Operating Income",
    )

    net, net_yoy = margin_yoy(
        income,
        "Net Income",
    )

    result[
        "Gross Margin"
    ] = gross

    result[
        "Gross Margin YoY"
    ] = gross_yoy

    result[
        "Operating Margin"
    ] = operating

    result[
        "Operating Margin YoY"
    ] = operating_yoy

    result[
        "Net Margin"
    ] = net

    result[
        "Net Margin YoY"
    ] = net_yoy

    if (
        cashflow is not None
        and not cashflow.empty
    ):

        if (
            "Free Cash Flow"
            in cashflow.index
        ):

            result[
                "FCF Growth YoY"
            ] = latest_yoy(
                cashflow.loc[
                    "Free Cash Flow"
                ]
            )

        elif (
            "Operating Cash Flow"
            in cashflow.index
            and "Capital Expenditure"
            in cashflow.index
        ):

            fcf = (
                cashflow.loc[
                    "Operating Cash Flow"
                ]
                + cashflow.loc[
                    "Capital Expenditure"
                ]
            )

            result[
                "FCF Growth YoY"
            ] = latest_yoy(fcf)

    return result












# ============================================================
# INDIVIDUAL STOCK FUNDAMENTALS
# ============================================================

def get_stock_screen_data(
    symbol: str,
    company: str,
    cik: int,
) -> dict:

    base = {
        "Symbol": symbol,
        "Company": company,
        "CIK": cik,
        "Price": np.nan,
        "Market Cap": np.nan,

        "Revenue Growth YoY": np.nan,
        "EPS Growth YoY": np.nan,
        "FCF Growth YoY": np.nan,

        "Gross Margin": np.nan,
        "Operating Margin": np.nan,
        "Net Margin": np.nan,

        "Gross Margin YoY": np.nan,
        "Operating Margin YoY": np.nan,
        "Net Margin YoY": np.nan,

        "Earnings Beat": False,
        "Earnings Beats": np.nan,
        "Earnings Quarters": np.nan,

        "Analyst Rating": "",
        "Buy %": np.nan,
        "Sell %": np.nan,
        "Buy % YoY": np.nan,
        "Sell % YoY": np.nan,

        "Target Price": np.nan,
        "Target Price YoY": np.nan,
        "Target Upside": np.nan,

        "Insider Buy Count": 0,
        "Insider Buy Value": 0.0,
        "Insider Buyers": 0,
        "Insider Cluster": False,
        "Insider Buy % Market Cap": np.nan,

        "Institution Net Purchase": np.nan,
        "Institution Net Value": np.nan,
        "Institutional Buying": False,
    }

    try:

        ticker = yf.Ticker(symbol)

        info = ticker.info or {}

        base["Price"] = safe_float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
        )

        base["Market Cap"] = safe_float(
            info.get("marketCap")
        )

        financials = (
            calculate_financial_metrics(
                ticker
            )
        )

        base.update(financials)

        analyst = get_analyst_data(
            symbol
        )

        base.update(analyst)

        if (
            pd.notna(base["Target Price"])
            and pd.notna(base["Price"])
            and base["Price"] > 0
        ):

            base["Target Upside"] = (
                base["Target Price"]
                / base["Price"]
                - 1
            )

        insider = (
            get_insider_purchase_data(
                cik,
                base["Market Cap"],
            )
        )

        base.update(
            {
                key: value
                for key, value in insider.items()
                if key != "Insider Transactions"
            }
        )

        institution = (
            get_institutional_data(
                symbol
            )
        )

        base.update(institution)

        return base

    except Exception as exc:

        base["Error"] = str(exc)

        return base


# ============================================================
# DOWNLOAD S&P 500 DATA
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False,
)
def get_us_stock_fundamentals(
    symbols: tuple[str, ...],
    companies: tuple[str, ...],
    ciks: tuple[int, ...],
) -> pd.DataFrame:

    jobs = list(
        zip(
            symbols,
            companies,
            ciks,
        )
    )

    records = []

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = {
            executor.submit(
                get_stock_screen_data,
                symbol,
                company,
                cik,
            ): symbol
            for symbol, company, cik
            in jobs
        }

        for future in as_completed(
            futures
        ):

            try:

                records.append(
                    future.result()
                )

            except Exception:
                pass

    if not records:
        raise ValueError(
            "No stock data was returned."
        )

    frame = pd.DataFrame(
        records
    )

    numeric_columns = [
        "Price",
        "Market Cap",
        "Revenue Growth YoY",
        "EPS Growth YoY",
        "FCF Growth YoY",
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "Gross Margin YoY",
        "Operating Margin YoY",
        "Net Margin YoY",
        "Buy %",
        "Sell %",
        "Buy % YoY",
        "Sell % YoY",
        "Target Price",
        "Target Price YoY",
        "Target Upside",
        "Insider Buy Count",
        "Insider Buy Value",
        "Insider Buyers",
        "Insider Buy % Market Cap",
        "Institution Net Purchase",
        "Institution Net Value",
    ]

    for column in numeric_columns:

        if column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return frame









def add_technical_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    frame = data.copy()

    close = frame["Close"]

    frame["EMA 8"] = (
        close.ewm(
            span=8,
            adjust=False,
        ).mean()
    )

    frame["EMA 20"] = (
        close.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    frame["EMA 200"] = (
        close.ewm(
            span=200,
            adjust=False,
        ).mean()
    )

    # 200-week EMA must be calculated from
    # weekly closes, not approximated as SMA 1000.
    weekly = (
        frame["Close"]
        .resample("W-FRI")
        .last()
        .dropna()
    )

    weekly_ema = (
        weekly
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    frame["EMA 200 Week"] = (
        weekly_ema
        .reindex(
            frame.index,
            method="ffill",
        )
    )

    frame["Volume Average 20"] = (
        frame["Volume"]
        .rolling(20)
        .mean()
    )

    frame["Volume Ratio"] = (
        frame["Volume"]
        / frame["Volume Average 20"]
    )

    frame["Horizontal Support"] = (
        frame["Low"]
        .rolling(60)
        .min()
        .shift(1)
    )

    frame["Horizontal Resistance"] = (
        frame["High"]
        .rolling(60)
        .max()
        .shift(1)
    )

    return frame


def within_one_percent(
    price: float,
    level: float,
) -> bool:

    if (
        pd.isna(price)
        or pd.isna(level)
        or level == 0
    ):
        return False

    return (
        abs(price / level - 1)
        <= 0.01
    )


def detect_channel(
    data: pd.DataFrame,
    direction: str,
) -> bool:

    if len(data) < 60:
        return False

    recent = data.tail(60).copy()

    x = np.arange(
        len(recent)
    )

    high_slope = np.polyfit(
        x,
        recent["High"].values,
        1,
    )[0]

    low_slope = np.polyfit(
        x,
        recent["Low"].values,
        1,
    )[0]

    price = recent["Close"].iloc[-1]

    normalized_high_slope = (
        high_slope / price
    )

    normalized_low_slope = (
        low_slope / price
    )

    if direction == "up":
        return (
            normalized_high_slope > 0.0005
            and normalized_low_slope > 0.0003
        )

    return (
        normalized_high_slope < -0.0005
        and normalized_low_slope < -0.0003
    )


def detect_wedge(
    data: pd.DataFrame,
    direction: str,
) -> bool:

    if len(data) < 80:
        return False

    recent = data.tail(80)

    x = np.arange(
        len(recent)
    )

    high_slope = np.polyfit(
        x,
        recent["High"].values,
        1,
    )[0]

    low_slope = np.polyfit(
        x,
        recent["Low"].values,
        1,
    )[0]

    spread_start = (
        recent["High"].iloc[:20].mean()
        - recent["Low"].iloc[:20].mean()
    )

    spread_end = (
        recent["High"].iloc[-20:].mean()
        - recent["Low"].iloc[-20:].mean()
    )

    contracting = (
        spread_end < spread_start * 0.80
    )

    if not contracting:
        return False

    if direction == "up":
        return (
            high_slope > 0
            and low_slope > 0
        )

    return (
        high_slope < 0
        and low_slope < 0
    )


def technical_signal(
    data: pd.DataFrame,
    direction: str,
) -> dict:

    frame = add_technical_features(
        data
    )

    latest = frame.iloc[-1]

    price = float(
        latest["Close"]
    )

    ema20 = within_one_percent(
        price,
        latest["EMA 20"],
    )

    ema200 = within_one_percent(
        price,
        latest["EMA 200"],
    )

    ema200w = within_one_percent(
        price,
        latest["EMA 200 Week"],
    )

    if direction == "long":

        horizontal = within_one_percent(
            price,
            latest[
                "Horizontal Support"
            ],
        )

        channel = detect_channel(
            frame,
            "up",
        )

        wedge = detect_wedge(
            frame,
            "up",
        )

    else:

        horizontal = within_one_percent(
            price,
            latest[
                "Horizontal Resistance"
            ],
        )

        channel = detect_channel(
            frame,
            "down",
        )

        wedge = detect_wedge(
            frame,
            "down",
        )

    high_volume = (
        latest["Volume Ratio"] >= 2.0
    )

    near_level = (
        ema20
        or ema200
        or ema200w
        or horizontal
    )

    pattern = (
        channel
        or wedge
        or high_volume
    )

    return {
        "Technical Pass": (
            near_level
            and pattern
        ),
        "Near 20 EMA": ema20,
        "Near 200 EMA": ema200,
        "Near 200 Week EMA": ema200w,
        "Near Horizontal Level": horizontal,
        "Channel Up": (
            channel
            if direction == "long"
            else False
        ),
        "Channel Down": (
            channel
            if direction == "short"
            else False
        ),
        "Wedge Up": (
            wedge
            if direction == "long"
            else False
        ),
        "Wedge Down": (
            wedge
            if direction == "short"
            else False
        ),
        "High Volume 2x": high_volume,
        "Volume Ratio": float(
            latest["Volume Ratio"]
        ),
    }




# ============================================================
# SCREENER SCORE
# ============================================================

def score_sp500_stocks(
    frame: pd.DataFrame,
    min_roe: float,
    min_revenue_growth: float,
    min_earnings_growth: float,
    max_pe: float,
    max_forward_pe: float,
    max_debt_equity: float,
    min_upside: float,
    require_insider_buy: bool,
) -> pd.DataFrame:

    data = frame.copy()

    # --------------------------------------------------------
    # Fundamental quality — 35 points
    # --------------------------------------------------------

    roe_score = (
        (
            data["ROE"]
            >= min_roe
        )
        .fillna(False)
        .astype(int)
        * 10
    )

    revenue_score = (
        (
            data["Revenue Growth"]
            >= min_revenue_growth
        )
        .fillna(False)
        .astype(int)
        * 7
    )

    earnings_score = (
        (
            data["Earnings Growth"]
            >= min_earnings_growth
        )
        .fillna(False)
        .astype(int)
        * 8
    )

    margin_score = (
        (
            data["Profit Margin"]
            > 0
        )
        .fillna(False)
        .astype(int)
        * 5
    )

    debt_score = (
        (
            data["Debt / Equity"]
            <= max_debt_equity
        )
        .fillna(False)
        .astype(int)
        * 5
    )

    data["Fundamental Score"] = (
        roe_score
        + revenue_score
        + earnings_score
        + margin_score
        + debt_score
    )

    # --------------------------------------------------------
    # Valuation — 25 points
    # --------------------------------------------------------

    pe_score = (
        (
            (data["P/E"] > 0)
            & (data["P/E"] <= max_pe)
        )
        .fillna(False)
        .astype(int)
        * 10
    )

    forward_pe_score = (
        (
            (data["Forward P/E"] > 0)
            & (
                data["Forward P/E"]
                <= max_forward_pe
            )
        )
        .fillna(False)
        .astype(int)
        * 10
    )

    peg_score = (
        (
            (data["PEG"] > 0)
            & (data["PEG"] <= 2.0)
        )
        .fillna(False)
        .astype(int)
        * 5
    )

    data["Valuation Score"] = (
        pe_score
        + forward_pe_score
        + peg_score
    )

    # --------------------------------------------------------
    # Analyst upside — 25 points
    # --------------------------------------------------------

    upside_score = pd.Series(
        0.0,
        index=data.index,
    )

    upside_score += (
        (
            data["Upside"]
            >= min_upside
        )
        .fillna(False)
        .astype(int)
        * 15
    )

    upside_score += (
        (
            data["Upside"]
            >= min_upside + 0.10
        )
        .fillna(False)
        .astype(int)
        * 5
    )

    upside_score += (
        (
            data["Upside"]
            >= min_upside + 0.20
        )
        .fillna(False)
        .astype(int)
        * 5
    )

    data["Upside Score"] = (
        upside_score.clip(
            upper=25
        )
    )

    # --------------------------------------------------------
    # Insider buying — 15 points
    # --------------------------------------------------------

    data["Insider Score"] = np.where(
        data["Insider buys"].fillna(0) > 0,
        15,
        0,
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    data["Score"] = (
        data["Fundamental Score"]
        + data["Valuation Score"]
        + data["Upside Score"]
        + data["Insider Score"]
    )

    # --------------------------------------------------------
    # Hard filters
    # --------------------------------------------------------

    mask = (
        (data["ROE"] >= min_roe)
        & (
            data["Revenue Growth"]
            >= min_revenue_growth
        )
        & (
            data["Earnings Growth"]
            >= min_earnings_growth
        )
        & (data["P/E"] > 0)
        & (data["P/E"] <= max_pe)
        & (
            data["Forward P/E"] > 0
        )
        & (
            data["Forward P/E"]
            <= max_forward_pe
        )
        & (
            data["Debt / Equity"]
            <= max_debt_equity
        )
        & (
            data["Upside"]
            >= min_upside
        )
    )

    if require_insider_buy:
        mask &= (
            data["Insider buys"]
            .fillna(0)
            > 0
        )

    data = data.loc[
        mask
    ].copy()

    return data.sort_values(
        [
            "Score",
            "Upside",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# FINANCIAL STATEMENTS
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False,
)
def get_selected_stock_financials(
    symbol: str,
) -> dict:

    ticker = yf.Ticker(symbol)

    info = ticker.info or {}

    try:
        income = ticker.income_stmt
    except Exception:
        income = pd.DataFrame()

    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    return {
        "info": info,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }


def format_financial_statement(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if (
        frame is None
        or frame.empty
    ):
        return pd.DataFrame()

    result = frame.copy()

    result = result.iloc[:, :4]

    new_columns = []

    for column in result.columns:

        try:
            new_columns.append(
                pd.to_datetime(
                    column
                ).strftime("%Y")
            )
        except Exception:
            new_columns.append(
                str(column)
            )

    result.columns = new_columns

    for row in result.index:

        result.loc[row] = pd.to_numeric(
            result.loc[row],
            errors="coerce",
        ) / 1_000_000

    return result.round(1)


# ============================================================
# PEER IDENTIFICATION
# ============================================================

def find_closest_peers(
    selected_symbol: str,
    screen_data: pd.DataFrame,
    number_of_peers: int = 2,
) -> pd.DataFrame:

    if screen_data.empty:
        return pd.DataFrame()

    selected_rows = screen_data[
        screen_data["Symbol"]
        == selected_symbol
    ]

    if selected_rows.empty:
        return pd.DataFrame()

    selected = selected_rows.iloc[0]

    candidates = screen_data[
        screen_data["Symbol"]
        != selected_symbol
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    selected_sector = selected.get(
        "Sector",
        "",
    )

    selected_industry = selected.get(
        "Industry",
        "",
    )

    selected_market_cap = selected.get(
        "Market Cap",
        np.nan,
    )

    candidates["Peer Score"] = 0.0

    candidates["Peer Score"] += np.where(
        candidates["Industry"]
        == selected_industry,
        100,
        0,
    )

    candidates["Peer Score"] += np.where(
        candidates["Sector"]
        == selected_sector,
        30,
        0,
    )

    if (
        pd.notna(selected_market_cap)
        and selected_market_cap > 0
    ):

        candidates[
            "Market Cap Distance"
        ] = (
            np.log(
                candidates[
                    "Market Cap"
                ].clip(lower=1)
            )
            - np.log(
                selected_market_cap
            )
        ).abs()

        candidates["Peer Score"] += (
            30
            / (
                1
                + candidates[
                    "Market Cap Distance"
                ]
            )
        )

    else:

        candidates[
            "Market Cap Distance"
        ] = np.nan

    return candidates.sort_values(
        "Peer Score",
        ascending=False,
    ).head(
        number_of_peers
    )


# ============================================================
# PEER RELATIVE ANALYSIS
# ============================================================

def add_peer_relative_metrics(
    selected_symbol: str,
    comparison: pd.DataFrame,
) -> pd.DataFrame:

    result = comparison.copy()

    if (
        selected_symbol not in result.index
        or len(result) < 2
    ):
        return result

    peer_rows = result.drop(
        index=selected_symbol
    )

    higher_is_better = [
        "ROE",
        "ROA",
        "Gross Margin",
        "Operating Margin",
        "Profit Margin",
        "Revenue Growth",
        "Earnings Growth",
        "Free Cash Flow",
        "Upside",
        "Score",
    ]

    lower_is_better = [
        "P/E",
        "Forward P/E",
        "PEG",
        "EV / EBITDA",
        "Debt / Equity",
    ]

    for metric in (
        higher_is_better
        + lower_is_better
    ):

        if metric not in result.columns:
            continue

        peer_median = pd.to_numeric(
            peer_rows[metric],
            errors="coerce",
        ).median()

        result.loc[
            selected_symbol,
            f"{metric} Peer Median",
        ] = peer_median

        selected_value = pd.to_numeric(
            result.loc[
                selected_symbol,
                metric,
            ],
            errors="coerce",
        )

        if (
            pd.isna(selected_value)
            or pd.isna(peer_median)
            or selected_value == 0
            or peer_median == 0
        ):

            relative = np.nan

        elif metric in higher_is_better:

            relative = (
                selected_value
                / peer_median
                - 1
            )

        else:

            relative = (
                peer_median
                / selected_value
                - 1
            )

        result.loc[
            selected_symbol,
            f"{metric} vs Peers",
        ] = relative

    # --------------------------------------------------------
    # Relative score — maximum 50
    # --------------------------------------------------------

    relative_score = 0.0

    for metric, weight in [
        ("ROE", 8),
        ("Revenue Growth", 5),
        ("Earnings Growth", 5),
        ("Profit Margin", 3),
        ("Free Cash Flow", 4),
    ]:

        column = f"{metric} vs Peers"

        if column in result.columns:

            value = result.loc[
                selected_symbol,
                column,
            ]

            if pd.notna(value):

                if value > 0.20:
                    relative_score += weight

                elif value > 0.10:
                    relative_score += (
                        weight * 0.75
                    )

                elif value > 0:
                    relative_score += (
                        weight * 0.50
                    )

    for metric, weight in [
        ("P/E", 5),
        ("Forward P/E", 5),
        ("EV / EBITDA", 3),
    ]:

        column = f"{metric} vs Peers"

        if column in result.columns:

            value = result.loc[
                selected_symbol,
                column,
            ]

            if pd.notna(value):

                if value > 0.20:
                    relative_score += weight

                elif value > 0.10:
                    relative_score += (
                        weight * 0.75
                    )

                elif value > 0:
                    relative_score += (
                        weight * 0.50
                    )

    if "Upside vs Peers" in result.columns:

        value = result.loc[
            selected_symbol,
            "Upside vs Peers",
        ]

        if pd.notna(value):

            if value > 0.20:
                relative_score += 7

            elif value > 0.10:
                relative_score += 5

            elif value > 0:
                relative_score += 3

    if (
        "Debt / Equity vs Peers"
        in result.columns
    ):

        value = result.loc[
            selected_symbol,
            "Debt / Equity vs Peers",
        ]

        if pd.notna(value):

            if value > 0.20:
                relative_score += 3

            elif value > 0.10:
                relative_score += 2

            elif value > 0:
                relative_score += 1

    result.loc[
        selected_symbol,
        "Peer Relative Score",
    ] = min(
        relative_score,
        50,
    )

    return result

# ============================================================
# SESSION STATE
# ============================================================

if "backtest_ticker" not in st.session_state:
    st.session_state["backtest_ticker"] = "SPY"

if "screener_raw_data" not in st.session_state:
    st.session_state["screener_raw_data"] = None

if "screener_results" not in st.session_state:
    st.session_state["screener_results"] = None

if "selected_screen_symbol" not in st.session_state:
    st.session_state[
        "selected_screen_symbol"
    ] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Signal Screener"
)

screen_direction = st.sidebar.segmented_control(
    "Candidates",
    options=[
        "Both",
        "Long",
        "Short",
    ],
    default="Both",
    help=(
        "Long finds bullish candidates, "
        "Short finds bearish candidates, "
        "Both shows both lists."
    ),
)

# If your Streamlit version does not have segmented_control, use:
#screen_direction = st.sidebar.radio(
#    "Candidates",
#    ["Both", "Long", "Short"],
#    horizontal=True,
#)



# ============================================================
# PAGE HEADER
# ============================================================

st.title("Signal Lab")

st.caption(
    "Screen the S&P 500 for attractive fundamentals, "
    "valuation, analyst upside and insider buying — "
    "then investigate the company, compare it with "
    "its closest peers and backtest a strategy."
)


# ============================================================
# S&P 500 SCREENER
# ============================================================

st.header(
    "NYSE / NASDAQ Long & Short Screener"
)

st.caption(
    "Fundamental/valuation/ownership score: "
    "120 points. Technical analysis is applied "
    "as a second-stage filter."
)

col1, col2, col3 = st.columns(3)

with col1:

    minimum_long_score = st.slider(
        "Minimum Long score",
        0,
        120,
        70,
        5,
    )

with col2:

    minimum_short_score = st.slider(
        "Minimum Short score",
        0,
        120,
        70,
        5,
    )

with col3:

    technical_lookback_years = st.slider(
        "Technical history",
        1,
        5,
        2,
    )

run_screener = st.button(
    "Run NYSE / NASDAQ Screener",
    type="primary",
    use_container_width=True,
)

if run_screener:

    if not FINNHUB_API_KEY:

        st.warning(
            "FINNHUB_API_KEY is not configured. "
            "The screener requires analyst/earnings/institutional "
            "data for the full scoring model."
        )

    with st.spinner(
        "Loading NYSE/NASDAQ universe and calculating "
        "fundamental, analyst, insider and institutional scores..."
    ):

        universe = get_us_listed_universe()

        # Start with a liquidity filter so that the
        # application doesn't attempt thousands of
        # micro-cap securities on every run.
        #
        # The threshold is configurable below.
        liquidity_columns = st.columns(2)

        raw = get_us_stock_fundamentals(
            tuple(
                universe["Symbol"]
            ),
            tuple(
                universe["Company"]
            ),
            tuple(
                universe["CIK"].astype(int)
            ),
        )

        scored = score_directional_stocks(
            raw
        )

        st.session_state[
            "screener_raw_data"
        ] = scored

screened_data = st.session_state.get(
    "screener_raw_data"
)

if (
    screened_data is not None
    and not screened_data.empty
):

    long_candidates = screened_data[
        screened_data["Long Score"]
        >= minimum_long_score
    ].copy()

    short_candidates = screened_data[
        screened_data["Short Score"]
        >= minimum_short_score
    ].copy()

    @st.cache_data(
        ttl=3600,
        show_spinner=False,
    )
    def get_technical_candidate_data(
        symbols: tuple[str, ...],
        years: int,
        direction: str,
    ) -> pd.DataFrame:

        records = []

        for symbol in symbols:

            try:

                data = get_ohlcv(
                    symbol,
                    years,
                )

                signal = technical_signal(
                    data,
                    direction,
                )

                signal["Symbol"] = symbol

                records.append(
                    signal
                )

            except Exception:
                continue

        return pd.DataFrame(
            records
        )

    if (
        screen_direction
        in ["Both", "Long"]
        and not long_candidates.empty
    ):

        long_technical = (
            get_technical_candidate_data(
                tuple(
                    long_candidates["Symbol"]
                ),
                technical_lookback_years,
                "long",
            )
        )

        long_final = long_candidates.merge(
            long_technical,
            on="Symbol",
            how="inner",
        )

        long_final = long_final[
            long_final["Technical Pass"]
        ].sort_values(
            "Long Score",
            ascending=False,
        )

    else:

        long_final = pd.DataFrame()

    if (
        screen_direction
        in ["Both", "Short"]
        and not short_candidates.empty
    ):

        short_technical = (
            get_technical_candidate_data(
                tuple(
                    short_candidates["Symbol"]
                ),
                technical_lookback_years,
                "short",
            )
        )

        short_final = short_candidates.merge(
            short_technical,
            on="Symbol",
            how="inner",
        )

        short_final = short_final[
            short_final["Technical Pass"]
        ].sort_values(
            "Short Score",
            ascending=False,
        )

    else:

        short_final = pd.DataFrame()

def display_candidate_table(
    frame: pd.DataFrame,
    direction: str,
):

    if frame.empty:

        st.info(
            f"No {direction.lower()} candidates "
            "passed the current criteria."
        )

        return

    score_column = (
        "Long Score"
        if direction == "Long"
        else "Short Score"
    )

    display_columns = [
        "Symbol",
        "Company",
        score_column,
        "Revenue Growth YoY",
        "EPS Growth YoY",
        "FCF Growth YoY",
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "Analyst Rating",
        "Buy %",
        "Sell %",
        "Target Price",
        "Target Upside",
        "Insider Buy Value",
        "Insider Buyers",
        "Institution Net Purchase",
        "Volume Ratio",
        "Near 20 EMA",
        "Near 200 EMA",
        "Near 200 Week EMA",
        "Near Horizontal Level",
        "Channel Up",
        "Channel Down",
        "Wedge Up",
        "Wedge Down",
        "High Volume 2x",
    ]

    available = [
        column
        for column in display_columns
        if column in frame.columns
    ]

    display = frame[
        available
    ].copy()

    percent_columns = [
        "Revenue Growth YoY",
        "EPS Growth YoY",
        "FCF Growth YoY",
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "Gross Margin YoY",
        "Operating Margin YoY",
        "Net Margin YoY",
        "Buy %",
        "Sell %",
        "Buy % YoY",
        "Sell % YoY",
        "Target Upside",
        "Insider Buy % Market Cap",
    ]

    for column in percent_columns:

        if column in display.columns:

            display[column] = display[
                column
            ].map(
                lambda x: (
                    f"{x:+.1%}"
                    if pd.notna(x)
                    else "N/A"
                )
            )

    if (
        "Insider Buy Value"
        in display.columns
    ):

        display[
            "Insider Buy Value"
        ] = display[
            "Insider Buy Value"
        ].map(
            lambda x: (
                f"${x:,.0f}"
                if pd.notna(x)
                else "N/A"
            )
        )

    if (
        "Institution Net Purchase"
        in display.columns
    ):

        display[
            "Institution Net Purchase"
        ] = display[
            "Institution Net Purchase"
        ].map(
            lambda x: (
                f"{x:,.0f}"
                if pd.notna(x)
                else "N/A"
            )
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=600,
    )


if screen_direction == "Both":

    long_tab, short_tab = st.tabs(
        [
            "🟢 Long Candidates",
            "🔴 Short Candidates",
        ]
    )

    with long_tab:

        st.subheader(
            f"{len(long_final)} Long candidates"
        )

        display_candidate_table(
            long_final,
            "Long",
        )

    with short_tab:

        st.subheader(
            f"{len(short_final)} Short candidates"
        )

        display_candidate_table(
            short_final,
            "Short",
        )

elif screen_direction == "Long":

    st.subheader(
        f"{len(long_final)} Long candidates"
    )

    display_candidate_table(
        long_final,
        "Long",
    )

else:

    st.subheader(
        f"{len(short_final)} Short candidates"
    )

    display_candidate_table(
        short_final,
        "Short",
    )










st.divider()

st.header(
    "Daily Technical Chart"
)

all_chart_symbols = sorted(
    set(
        (
            long_final["Symbol"].tolist()
            if not long_final.empty
            else []
        )
        +
        (
            short_final["Symbol"].tolist()
            if not short_final.empty
            else []
        )
    )
)

if all_chart_symbols:

    chart_symbol = st.selectbox(
        "Chart candidate",
        all_chart_symbols,
    )

    chart_data = get_ohlcv(
        chart_symbol,
        technical_lookback_years,
    )

    chart_row = screened_data[
        screened_data["Symbol"]
        == chart_symbol
    ]

    insider_transactions = None

    if not chart_row.empty:

        cik = int(
            chart_row.iloc[0]["CIK"]
        )

        insider_data = (
            get_insider_purchase_data(
                cik,
                safe_float(
                    chart_row.iloc[0][
                        "Market Cap"
                    ]
                ),
            )
        )

        insider_transactions = (
            insider_data[
                "Insider Transactions"
            ]
        )

    chart_features = (
        add_technical_features(
            chart_data
        )
    )

    long_signal = pd.Series(
        False,
        index=chart_features.index,
    )

    short_signal = pd.Series(
        False,
        index=chart_features.index,
    )

    # Long technical signal:
    # near a major EMA/support + bullish pattern/volume.
    long_signal = (
        (
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 20"]
                    - 1
                ) <= 0.01
            )
            |
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 200"]
                    - 1
                ) <= 0.01
            )
            |
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 200 Week"]
                    - 1
                ) <= 0.01
            )
        )
        &
        (
            chart_features["Volume Ratio"]
            >= 2
        )
    )

    short_signal = (
        (
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 20"]
                    - 1
                ) <= 0.01
            )
            |
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 200"]
                    - 1
                ) <= 0.01
            )
            |
            (
                abs(
                    chart_features["Close"]
                    / chart_features["EMA 200 Week"]
                    - 1
                ) <= 0.01
            )
        )
        &
        (
            chart_features["Volume Ratio"]
            >= 2
        )
    )

    st.plotly_chart(
        signal_chart(
            chart_data,
            chart_symbol,
            insider_transactions,
            long_signal,
            short_signal,
        ),
        use_container_width=True,
    )

else:

    st.info(
        "Run the screener to populate the technical chart."
    )



# ============================================================
# BACKTESTER
# ============================================================

st.divider()

st.header("Natural-Language Backtester")

if not ticker:

    st.error(
        "Enter a valid ticker symbol."
    )

else:

    valid_ticker = (
        ticker
        .replace("-", "")
        .replace(".", "")
        .isalnum()
    )

    if not valid_ticker:

        st.error(
            "Enter a valid ticker symbol, "
            "such as SPY or BRK-B."
        )

    else:

        if (
            run
            or "backtest_data"
            not in st.session_state
        ):

            with st.spinner(
                f"Downloading {ticker} data "
                "and running the backtest..."
            ):

                try:

                    market_data = get_ohlcv(
                        ticker,
                        years,
                    )

                    parsed = parse_strategy(
                        strategy_text,
                        market_data,
                    )

                    metrics, trades = (
                        backtest(
                            market_data,
                            parsed,
                        )
                    )

                except Exception as exc:

                    st.error(
                        f"Backtest failed: {exc}"
                    )

                    st.stop()

            st.session_state[
                "backtest_data"
            ] = (
                market_data,
                parsed,
                metrics,
                trades,
                ticker,
                years,
            )

        (
            market_data,
            parsed,
            metrics,
            trades,
            ticker,
            years,
        ) = st.session_state[
            "backtest_data"
        ]

        st.write(
            f"**{ticker}** · "
            f"{market_data.index[0]:%b %d, %Y}"
            f" – "
            f"{market_data.index[-1]:%b %d, %Y}"
            f" · "
            f"{len(market_data):,} sessions"
        )

        # ----------------------------------------------------
        # Backtest metrics
        # ----------------------------------------------------

        metric_columns = st.columns(6)

        metric_columns[0].metric(
            "Wins",
            f"{metrics['wins']:,}",
        )

        metric_columns[1].metric(
            "Losses",
            f"{metrics['losses']:,}",
        )

        metric_columns[2].metric(
            "Win rate",
            f"{metrics['win_rate']:.1%}",
        )

        metric_columns[3].metric(
            "Total return",
            f"{metrics['return']:+.1%}",
        )

        metric_columns[4].metric(
            "Max drawdown",
            f"{metrics['drawdown']:.1%}",
        )

        metric_columns[5].metric(
            "Trades",
            f"{metrics['trades']:,}",
        )

        # ----------------------------------------------------
        # Interpreted strategy
        # ----------------------------------------------------

        with st.expander(
            "Interpreted strategy",
            expanded=True,
        ):

            for description in (
                parsed.description
            ):

                st.write(
                    "• " + description
                )

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        st.plotly_chart(
            price_chart(
                market_data,
                trades,
                ticker,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Trade log
        # ----------------------------------------------------

        if trades.empty:

            st.info(
                "No trades matched the strategy "
                "in this period. Try a shorter "
                "moving average or different RSI "
                "thresholds."
            )

        else:

            display_trades = (
                trades
                .drop(
                    columns=["Outcome"]
                )
                .copy()
            )

            display_trades[
                "Return"
            ] = display_trades[
                "Return"
            ].map(
                lambda value:
                f"{value:+.2%}"
            )

            st.subheader(
                "Trade log"
            )

            st.dataframe(
                display_trades,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Educational use only. Historical results are hypothetical "
    "and do not guarantee future performance. Fundamental, "
    "analyst and insider data are sourced from Yahoo Finance "
    "and may be incomplete, delayed or inconsistent."
)
