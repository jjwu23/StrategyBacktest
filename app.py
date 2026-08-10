"""Signal Lab — S&P 500 stock screener and natural-language backtester."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


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

def price_chart(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
) -> go.Figure:

    figure = go.Figure()

    close = data["Close"]

    sma_200 = close.rolling(200).mean()

    sma_200w = close.rolling(
        1000
    ).mean()

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=sma_200,
            mode="lines",
            name="200 SMA",
            line=dict(
                color="#FFD700",
                width=1.5,
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=sma_200w,
            mode="lines",
            name="200 WMA",
            line=dict(
                color="#FFD700",
                width=3.5,
            ),
        )
    )

    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
        )
    )

    if not trades.empty:

        for _, trade in trades.iterrows():

            entry_x = trade["Entry"]
            exit_x = trade["Exit"]

            entry_p = trade["Entry price"]
            exit_p = trade["Exit price"]

            is_win = (
                trade["Return"] > 0
            )

            fill_color = (
                "rgba(0, 230, 118, 0.15)"
                if is_win
                else "rgba(255, 82, 82, 0.15)"
            )

            figure.add_shape(
                type="rect",
                x0=entry_x,
                x1=exit_x,
                y0=min(
                    entry_p,
                    exit_p,
                ),
                y1=max(
                    entry_p,
                    exit_p,
                ),
                fillcolor=fill_color,
                layer="below",
                line=dict(width=0),
            )

        wins_df = trades[
            trades["Return"] > 0
        ]

        losses_df = trades[
            trades["Return"] <= 0
        ]

        if not wins_df.empty:

            figure.add_trace(
                go.Scatter(
                    x=wins_df["Entry"],
                    y=wins_df["Entry price"],
                    mode="markers",
                    name="Long Entry (Win)",
                    marker=dict(
                        symbol="triangle-up",
                        size=13,
                        color="#00E676",
                        line=dict(
                            width=1,
                            color="#000",
                        ),
                    ),
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=wins_df["Exit"],
                    y=wins_df["Exit price"],
                    mode="markers",
                    name="Target Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=13,
                        color="#00E676",
                        line=dict(
                            width=1,
                            color="#000",
                        ),
                    ),
                )
            )

        if not losses_df.empty:

            figure.add_trace(
                go.Scatter(
                    x=losses_df["Entry"],
                    y=losses_df["Entry price"],
                    mode="markers",
                    name="Long Entry (Loss)",
                    marker=dict(
                        symbol="triangle-up",
                        size=13,
                        color="#FF5252",
                        line=dict(
                            width=1,
                            color="#000",
                        ),
                    ),
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=losses_df["Exit"],
                    y=losses_df["Exit price"],
                    mode="markers",
                    name="Stop Loss Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=13,
                        color="#FF5252",
                        line=dict(
                            width=1,
                            color="#000",
                        ),
                    ),
                )
            )

    figure.update_layout(
        height=560,
        template="plotly_dark",
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        xaxis_rangeslider_visible=False,
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
def get_sp500_constituents() -> pd.DataFrame:
    """Load the current S&P 500 constituents."""

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    tables = pd.read_html(url)

    if not tables:
        raise ValueError(
            "Could not load the S&P 500 constituent list."
        )

    constituents = tables[0][
        [
            "Symbol",
            "Security",
            "GICS Sector",
        ]
    ].copy()

    constituents["Symbol"] = (
        constituents["Symbol"]
        .astype(str)
        .str.replace(
            ".",
            "-",
            regex=False,
        )
        .str.strip()
    )

    return constituents


# ============================================================
# INSIDER ACTIVITY
# ============================================================

def get_insider_purchase_data(
    ticker: yf.Ticker,
) -> dict:
    """
    Estimate recent insider purchases.

    Yahoo Finance transaction fields can vary by company,
    so this intentionally treats insider buying as a signal
    rather than a perfect accounting measure.
    """

    result = {
        "Insider buys": 0,
        "Insider buy value": 0.0,
        "Recent insider purchase": False,
    }

    try:

        transactions = (
            ticker.insider_transactions
        )

        if (
            transactions is None
            or transactions.empty
        ):
            return result

        frame = transactions.copy()

        frame.columns = [
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            for column in frame.columns
        ]

        text_columns = [
            column
            for column in frame.columns
            if any(
                word in column
                for word in [
                    "transaction",
                    "type",
                    "description",
                    "action",
                    "text",
                ]
            )
        ]

        if not text_columns:
            return result

        text = (
            frame[text_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )

        purchase_mask = (
            text.str.contains(
                r"purchase|buy",
                regex=True,
                na=False,
            )
            & ~text.str.contains(
                r"sale|sell",
                regex=True,
                na=False,
            )
        )

        purchases = frame.loc[
            purchase_mask
        ]

        if purchases.empty:
            return result

        result["Insider buys"] = int(
            len(purchases)
        )

        result[
            "Recent insider purchase"
        ] = True

        value_columns = [
            column
            for column in frame.columns
            if any(
                word in column
                for word in [
                    "value",
                    "transaction_value",
                    "total_value",
                ]
            )
        ]

        if value_columns:

            values = pd.to_numeric(
                purchases[
                    value_columns[0]
                ],
                errors="coerce",
            ).dropna()

            if not values.empty:
                result[
                    "Insider buy value"
                ] = float(
                    values.sum()
                )

    except Exception:
        pass

    return result


# ============================================================
# INDIVIDUAL STOCK FUNDAMENTALS
# ============================================================

def get_stock_screen_data(
    symbol: str,
    company: str,
    sector: str,
) -> dict:
    """Download fundamental and valuation data."""

    try:

        ticker = yf.Ticker(symbol)

        info = ticker.info or {}

        current_price = safe_float(
            info.get("currentPrice")
            or info.get(
                "regularMarketPrice"
            )
        )

        market_cap = safe_float(
            info.get("marketCap")
        )

        try:

            fast_info = ticker.fast_info

            if pd.isna(current_price):
                current_price = safe_float(
                    fast_info.get(
                        "lastPrice"
                    )
                )

            if pd.isna(market_cap):
                market_cap = safe_float(
                    fast_info.get(
                        "marketCap"
                    )
                )

        except Exception:
            pass

        target_price = safe_float(
            info.get("targetMeanPrice")
        )

        upside = np.nan

        if (
            pd.notna(current_price)
            and current_price > 0
            and pd.notna(target_price)
        ):

            upside = (
                target_price
                / current_price
                - 1
            )

        insider = (
            get_insider_purchase_data(
                ticker
            )
        )

        return {
            "Symbol": symbol,
            "Company": company,
            "Sector": sector,
            "Industry": info.get(
                "industry",
                "",
            ),
            "Market Cap": market_cap,

            # Price / valuation
            "Price": current_price,
            "P/E": safe_float(
                info.get("trailingPE")
            ),
            "Forward P/E": safe_float(
                info.get("forwardPE")
            ),
            "PEG": safe_float(
                info.get("pegRatio")
            ),
            "Price / Book": safe_float(
                info.get("priceToBook")
            ),
            "EV / EBITDA": safe_float(
                info.get(
                    "enterpriseToEbitda"
                )
            ),

            # Profitability
            "ROE": safe_float(
                info.get(
                    "returnOnEquity"
                )
            ),
            "ROA": safe_float(
                info.get(
                    "returnOnAssets"
                )
            ),
            "Profit Margin": safe_float(
                info.get(
                    "profitMargins"
                )
            ),
            "Operating Margin": safe_float(
                info.get(
                    "operatingMargins"
                )
            ),
            "Gross Margin": safe_float(
                info.get(
                    "grossMargins"
                )
            ),

            # Growth
            "Revenue Growth": safe_float(
                info.get(
                    "revenueGrowth"
                )
            ),
            "Earnings Growth": safe_float(
                info.get(
                    "earningsGrowth"
                )
            ),
            "Earnings Quarterly Growth": safe_float(
                info.get(
                    "earningsQuarterlyGrowth"
                )
            ),

            # Balance sheet / cash flow
            "Debt / Equity": safe_float(
                info.get(
                    "debtToEquity"
                )
            ),
            "Current Ratio": safe_float(
                info.get(
                    "currentRatio"
                )
            ),
            "Free Cash Flow": safe_float(
                info.get(
                    "freeCashflow"
                )
            ),
            "Operating Cash Flow": safe_float(
                info.get(
                    "operatingCashflow"
                )
            ),

            # Analyst expectations
            "Target Price": target_price,
            "Upside": upside,
            "Analyst Rating": info.get(
                "recommendationKey",
                "",
            ),

            # Insider activity
            "Insider buys": insider[
                "Insider buys"
            ],
            "Insider buy value": insider[
                "Insider buy value"
            ],
            "Recent insider purchase": insider[
                "Recent insider purchase"
            ],
        }

    except Exception as exc:

        return {
            "Symbol": symbol,
            "Company": company,
            "Sector": sector,
            "Industry": "",
            "Market Cap": np.nan,
            "Error": str(exc),
        }


# ============================================================
# DOWNLOAD S&P 500 DATA
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False,
)
def get_sp500_fundamentals(
    symbols: tuple[str, ...],
    companies: tuple[str, ...],
    sectors: tuple[str, ...],
) -> pd.DataFrame:
    """Download S&P 500 data using a small thread pool."""

    records = []

    jobs = list(
        zip(
            symbols,
            companies,
            sectors,
        )
    )

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                get_stock_screen_data,
                symbol,
                company,
                sector,
            ): symbol
            for symbol, company, sector
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
            "No S&P 500 fundamental data was returned."
        )

    frame = pd.DataFrame(
        records
    )

    numeric_columns = [
        "Price",
        "Market Cap",
        "P/E",
        "Forward P/E",
        "PEG",
        "Price / Book",
        "EV / EBITDA",
        "ROE",
        "ROA",
        "Profit Margin",
        "Operating Margin",
        "Gross Margin",
        "Revenue Growth",
        "Earnings Growth",
        "Earnings Quarterly Growth",
        "Debt / Equity",
        "Current Ratio",
        "Free Cash Flow",
        "Operating Cash Flow",
        "Target Price",
        "Upside",
        "Insider buys",
        "Insider buy value",
    ]

    for column in numeric_columns:

        if column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return frame


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

st.sidebar.header("Signal Lab")

show_screener = st.sidebar.toggle(
    "S&P 500 Stock Screener",
    value=True,
    help=(
        "Show or hide the S&P 500 "
        "fundamental stock screener."
    ),
)

st.sidebar.divider()

st.sidebar.header(
    "Backtest settings"
)

ticker = st.sidebar.text_input(
    "Ticker",
    max_chars=12,
    key="backtest_ticker",
).strip().upper()

years = st.sidebar.slider(
    "Years of daily history",
    1,
    10,
    10,
)

st.sidebar.caption(
    "Yahoo Finance data · up to 10 years · "
    "trades fill at the next day's open"
)

strategy_text = st.sidebar.text_area(
    "Strategy in natural language",
    value=(
        "Buy when close crosses above "
        "the 50-day SMA. Sell when close "
        "crosses below the 50-day SMA."
    ),
    height=120,
    help=(
        "Examples: 'Buy when RSI is below 30, "
        "sell when RSI is above 70.' Add "
        "'hold for 10 days', 'stop loss 5%', "
        "or 'take profit 12%'."
    ),
)

run = st.sidebar.button(
    "Run backtest",
    type="primary",
    use_container_width=True,
)


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

if show_screener:

    st.header(
        "S&P 500 Fundamental Screener"
    )

    st.caption(
        "100-point score: 35 fundamental quality + "
        "25 valuation + 25 analyst upside + "
        "15 insider activity."
    )

    with st.expander(
        "Screener settings",
        expanded=True,
    ):

        screen_columns = st.columns(4)

        with screen_columns[0]:

            min_roe = st.slider(
                "Minimum ROE",
                -0.20,
                0.50,
                0.12,
                0.01,
                format="%.0f%%",
            )

            min_revenue_growth = st.slider(
                "Minimum revenue growth",
                -0.20,
                0.50,
                0.05,
                0.01,
                format="%.0f%%",
            )

        with screen_columns[1]:

            min_earnings_growth = st.slider(
                "Minimum earnings growth",
                -0.50,
                1.00,
                0.05,
                0.01,
                format="%.0f%%",
            )

            max_debt_equity = st.slider(
                "Maximum debt / equity",
                0.0,
                300.0,
                150.0,
                10.0,
            )

        with screen_columns[2]:

            max_pe = st.slider(
                "Maximum P/E",
                5.0,
                100.0,
                30.0,
                1.0,
            )

            max_forward_pe = st.slider(
                "Maximum forward P/E",
                5.0,
                100.0,
                25.0,
                1.0,
            )

        with screen_columns[3]:

            min_upside = st.slider(
                "Minimum analyst upside",
                -0.20,
                1.00,
                0.15,
                0.05,
                format="%.0f%%",
            )

            require_insider_buy = st.checkbox(
                "Require recent insider purchase",
                value=False,
            )

        run_screener = st.button(
            "Run S&P 500 Screener",
            type="primary",
            use_container_width=True,
        )

    if run_screener:

        with st.spinner(
            "Downloading S&P 500 fundamentals, "
            "valuations and insider activity..."
        ):

            try:

                constituents = (
                    get_sp500_constituents()
                )

                raw_screen = (
                    get_sp500_fundamentals(
                        tuple(
                            constituents[
                                "Symbol"
                            ]
                        ),
                        tuple(
                            constituents[
                                "Security"
                            ]
                        ),
                        tuple(
                            constituents[
                                "GICS Sector"
                            ]
                        ),
                    )
                )

                screened = score_sp500_stocks(
                    raw_screen,
                    min_roe=min_roe,
                    min_revenue_growth=(
                        min_revenue_growth
                    ),
                    min_earnings_growth=(
                        min_earnings_growth
                    ),
                    max_pe=max_pe,
                    max_forward_pe=(
                        max_forward_pe
                    ),
                    max_debt_equity=(
                        max_debt_equity
                    ),
                    min_upside=min_upside,
                    require_insider_buy=(
                        require_insider_buy
                    ),
                )

                st.session_state[
                    "screener_raw_data"
                ] = raw_screen

                st.session_state[
                    "screener_results"
                ] = screened

                if not screened.empty:

                    st.session_state[
                        "selected_screen_symbol"
                    ] = screened.iloc[
                        0
                    ]["Symbol"]

            except Exception as exc:

                st.error(
                    f"S&P 500 screener failed: {exc}"
                )

    raw_screen = st.session_state[
        "screener_raw_data"
    ]

    screened = st.session_state[
        "screener_results"
    ]

    if (
        raw_screen is not None
        and screened is not None
    ):

        st.write(
            f"**{len(screened)} stocks** passed "
            f"the current filters out of "
            f"**{len(raw_screen)}** S&P 500 constituents."
        )

        if screened.empty:

            st.warning(
                "No stocks passed the current criteria. "
                "Try lowering the upside requirement "
                "or relaxing the valuation filters."
            )

        else:

            # ------------------------------------------------
            # Screener summary metrics
            # ------------------------------------------------

            metric_columns = st.columns(5)

            metric_columns[0].metric(
                "Stocks passing",
                f"{len(screened):,}",
            )

            metric_columns[1].metric(
                "Average score",
                f"{screened['Score'].mean():.1f}/100",
            )

            metric_columns[2].metric(
                "Average upside",
                f"{screened['Upside'].mean():+.1%}",
            )

            metric_columns[3].metric(
                "Average P/E",
                f"{screened['P/E'].mean():.1f}",
            )

            insider_count = int(
                (
                    screened[
                        "Insider buys"
                    ].fillna(0)
                    > 0
                ).sum()
            )

            metric_columns[4].metric(
                "With insider buying",
                f"{insider_count:,}",
            )

            # ------------------------------------------------
            # Screener table
            # ------------------------------------------------

            st.subheader(
                "Top S&P 500 candidates"
            )

            display_screen = screened[
                [
                    "Symbol",
                    "Company",
                    "Sector",
                    "Score",
                    "Fundamental Score",
                    "Valuation Score",
                    "Upside Score",
                    "Insider Score",
                    "Price",
                    "P/E",
                    "Forward P/E",
                    "ROE",
                    "Revenue Growth",
                    "Earnings Growth",
                    "Debt / Equity",
                    "Target Price",
                    "Upside",
                    "Insider buys",
                    "Insider buy value",
                ]
            ].copy()

            for column in [
                "ROE",
                "Revenue Growth",
                "Earnings Growth",
                "Upside",
            ]:

                display_screen[
                    column
                ] = display_screen[
                    column
                ].map(
                    lambda value: (
                        f"{value:+.1%}"
                        if pd.notna(value)
                        else "N/A"
                    )
                )

            for column in [
                "Price",
                "Target Price",
                "P/E",
                "Forward P/E",
                "Debt / Equity",
            ]:

                display_screen[
                    column
                ] = display_screen[
                    column
                ].map(
                    lambda value: (
                        f"{value:.2f}"
                        if pd.notna(value)
                        else "N/A"
                    )
                )

            display_screen[
                "Insider buy value"
            ] = display_screen[
                "Insider buy value"
            ].map(
                lambda value: (
                    f"${value:,.0f}"
                    if (
                        pd.notna(value)
                        and value > 0
                    )
                    else "N/A"
                )
            )

            st.dataframe(
                display_screen,
                use_container_width=True,
                hide_index=True,
                height=600,
            )

            st.download_button(
                "Download screener results CSV",
                data=screened.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "sp500_screener.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            # =================================================
            # SELECTED STOCK
            # =================================================

            st.divider()

            st.header(
                "Selected Stock Analysis"
            )

            available_symbols = (
                screened[
                    "Symbol"
                ].tolist()
            )

            if (
                st.session_state[
                    "selected_screen_symbol"
                ]
                not in available_symbols
            ):

                st.session_state[
                    "selected_screen_symbol"
                ] = available_symbols[0]

            selected_symbol = st.selectbox(
                "Select a stock to analyze",
                available_symbols,
                key="selected_screen_symbol",
                format_func=lambda symbol: (
                f"{symbol} — {screened.loc[screened['Symbol'] == symbol, 'Company'].iloc[0]}"
                ),
            )

            selected_row = screened[
                screened["Symbol"]
                == selected_symbol
            ].iloc[0]

            st.subheader(
                f"{selected_symbol} — "
                f"{selected_row['Company']}"
            )

            # ------------------------------------------------
            # Company overview
            # ------------------------------------------------

            overview = st.columns(6)

            overview[0].metric(
                "Price",
                (
                    f"${selected_row['Price']:,.2f}"
                    if pd.notna(
                        selected_row["Price"]
                    )
                    else "N/A"
                ),
            )

            overview[1].metric(
                "Market Cap",
                (
                    f"${selected_row['Market Cap'] / 1e9:.1f}B"
                    if pd.notna(
                        selected_row["Market Cap"]
                    )
                    else "N/A"
                ),
            )

            overview[2].metric(
                "P/E",
                (
                    f"{selected_row['P/E']:.1f}"
                    if pd.notna(
                        selected_row["P/E"]
                    )
                    else "N/A"
                ),
            )

            overview[3].metric(
                "ROE",
                (
                    f"{selected_row['ROE']:.1%}"
                    if pd.notna(
                        selected_row["ROE"]
                    )
                    else "N/A"
                ),
            )

            overview[4].metric(
                "Analyst upside",
                (
                    f"{selected_row['Upside']:+.1%}"
                    if pd.notna(
                        selected_row["Upside"]
                    )
                    else "N/A"
                ),
            )

            overview[5].metric(
                "Screener score",
                f"{selected_row['Score']:.0f}/100",
            )

            # ------------------------------------------------
            # Fundamental snapshot
            # ------------------------------------------------

            st.markdown(
                "#### Fundamental snapshot"
            )

            fundamental = st.columns(6)

            fundamental[0].metric(
                "Revenue growth",
                (
                    f"{selected_row['Revenue Growth']:+.1%}"
                    if pd.notna(
                        selected_row[
                            "Revenue Growth"
                        ]
                    )
                    else "N/A"
                ),
            )

            fundamental[1].metric(
                "Earnings growth",
                (
                    f"{selected_row['Earnings Growth']:+.1%}"
                    if pd.notna(
                        selected_row[
                            "Earnings Growth"
                        ]
                    )
                    else "N/A"
                ),
            )

            fundamental[2].metric(
                "Profit margin",
                (
                    f"{selected_row['Profit Margin']:.1%}"
                    if pd.notna(
                        selected_row[
                            "Profit Margin"
                        ]
                    )
                    else "N/A"
                ),
            )

            fundamental[3].metric(
                "Debt / Equity",
                (
                    f"{selected_row['Debt / Equity']:.1f}"
                    if pd.notna(
                        selected_row[
                            "Debt / Equity"
                        ]
                    )
                    else "N/A"
                ),
            )

            fundamental[4].metric(
                "Free cash flow",
                (
                    f"${selected_row['Free Cash Flow'] / 1e9:.2f}B"
                    if pd.notna(
                        selected_row[
                            "Free Cash Flow"
                        ]
                    )
                    else "N/A"
                ),
            )

            fundamental[5].metric(
                "Insider purchases",
                f"{int(selected_row['Insider buys'])}"
                if pd.notna(
                    selected_row[
                        "Insider buys"
                    ]
                )
                else "0",
            )

            # ------------------------------------------------
            # Financial statements
            # ------------------------------------------------

            st.markdown(
                "#### Financial statements"
            )

            with st.spinner(
                f"Loading {selected_symbol} "
                "financial statements..."
            ):

                financials = (
                    get_selected_stock_financials(
                        selected_symbol
                    )
                )

            statement_tabs = st.tabs(
                [
                    "Income Statement",
                    "Balance Sheet",
                    "Cash Flow",
                ]
            )

            with statement_tabs[0]:

                income = (
                    format_financial_statement(
                        financials["income"]
                    )
                )

                if income.empty:

                    st.info(
                        "Income statement data unavailable."
                    )

                else:

                    st.caption(
                        "Values shown in $ millions."
                    )

                    st.dataframe(
                        income,
                        use_container_width=True,
                    )

            with statement_tabs[1]:

                balance = (
                    format_financial_statement(
                        financials["balance"]
                    )
                )

                if balance.empty:

                    st.info(
                        "Balance sheet data unavailable."
                    )

                else:

                    st.caption(
                        "Values shown in $ millions."
                    )

                    st.dataframe(
                        balance,
                        use_container_width=True,
                    )

            with statement_tabs[2]:

                cashflow = (
                    format_financial_statement(
                        financials["cashflow"]
                    )
                )

                if cashflow.empty:

                    st.info(
                        "Cash-flow statement data unavailable."
                    )

                else:

                    st.caption(
                        "Values shown in $ millions."
                    )

                    st.dataframe(
                        cashflow,
                        use_container_width=True,
                    )

            # ------------------------------------------------
            # Closest peers
            # ------------------------------------------------

            st.markdown(
                "#### Closest S&P 500 peers"
            )

            peers = find_closest_peers(
                selected_symbol,
                raw_screen,
                number_of_peers=2,
            )

            if peers.empty:

                st.info(
                    "Unable to identify suitable peers."
                )

            else:

                comparison = pd.concat(
                    [
                        screened[
                            screened[
                                "Symbol"
                            ]
                            == selected_symbol
                        ],
                        peers,
                    ],
                    ignore_index=True,
                )

                comparison = (
                    comparison.drop_duplicates(
                        subset=["Symbol"]
                    )
                )

                comparison = (
                    comparison.set_index(
                        "Symbol"
                    )
                )

                comparison = (
                    add_peer_relative_metrics(
                        selected_symbol,
                        comparison,
                    )
                )

                # ------------------------------------------------
                # Peer-relative summary
                # ------------------------------------------------

                peer_relative_score = (
                    comparison.loc[
                        selected_symbol,
                        "Peer Relative Score",
                    ]
                    if "Peer Relative Score"
                    in comparison.columns
                    else np.nan
                )

                if pd.notna(
                    peer_relative_score
                ):

                    if peer_relative_score >= 35:
                        relative_label = (
                            "Strongly better than peers"
                        )

                    elif peer_relative_score >= 25:
                        relative_label = (
                            "Better than peers"
                        )

                    elif peer_relative_score >= 15:
                        relative_label = (
                            "Mixed vs peers"
                        )

                    else:
                        relative_label = (
                            "Weaker vs peers"
                        )

                    st.markdown(
                        f"### Relative investment case: "
                        f"**{relative_label}**"
                    )

                    relative_columns = (
                        st.columns(5)
                    )

                    relative_columns[0].metric(
                        "Peer-relative score",
                        f"{peer_relative_score:.0f}/50",
                    )

                    for idx, metric in enumerate(
                        [
                            "ROE",
                            "Revenue Growth",
                            "P/E",
                            "Upside",
                        ],
                        start=1,
                    ):

                        value = comparison.loc[
                            selected_symbol,
                            f"{metric} vs Peers",
                        ]

                        if pd.isna(value):
                            formatted = "N/A"
                        else:
                            formatted = (
                                f"{value:+.1%}"
                            )

                        labels = {
                            "ROE": "ROE vs peers",
                            "Revenue Growth": (
                                "Growth vs peers"
                            ),
                            "P/E": (
                                "P/E advantage"
                            ),
                            "Upside": (
                                "Upside vs peers"
                            ),
                        }

                        relative_columns[
                            idx
                        ].metric(
                            labels[metric],
                            formatted,
                        )

                # ------------------------------------------------
                # Comparison table
                # ------------------------------------------------

                comparison_metrics = [
                    ("Company", "Company"),
                    ("Industry", "Industry"),
                    ("Market Cap", "Market Cap"),

                    ("P/E", "P/E"),
                    (
                        "Peer median P/E",
                        "P/E Peer Median",
                    ),
                    (
                        "P/E vs peers",
                        "P/E vs Peers",
                    ),

                    (
                        "Forward P/E",
                        "Forward P/E",
                    ),
                    (
                        "Peer median forward P/E",
                        "Forward P/E Peer Median",
                    ),
                    (
                        "Forward P/E vs peers",
                        "Forward P/E vs Peers",
                    ),

                    ("PEG", "PEG"),
                    ("EV / EBITDA", "EV / EBITDA"),

                    ("ROE", "ROE"),
                    (
                        "Peer median ROE",
                        "ROE Peer Median",
                    ),
                    (
                        "ROE vs peers",
                        "ROE vs Peers",
                    ),

                    (
                        "Revenue Growth",
                        "Revenue Growth",
                    ),
                    (
                        "Peer median revenue growth",
                        "Revenue Growth Peer Median",
                    ),
                    (
                        "Revenue growth vs peers",
                        "Revenue Growth vs Peers",
                    ),

                    (
                        "Earnings Growth",
                        "Earnings Growth",
                    ),
                    (
                        "Peer median earnings growth",
                        "Earnings Growth Peer Median",
                    ),

                    (
                        "Profit Margin",
                        "Profit Margin",
                    ),
                    (
                        "Peer median profit margin",
                        "Profit Margin Peer Median",
                    ),

                    (
                        "Debt / Equity",
                        "Debt / Equity",
                    ),
                    (
                        "Peer median debt / equity",
                        "Debt / Equity Peer Median",
                    ),
                    (
                        "Debt / Equity vs peers",
                        "Debt / Equity vs Peers",
                    ),

                    (
                        "Free Cash Flow",
                        "Free Cash Flow",
                    ),

                    (
                        "Target Price",
                        "Target Price",
                    ),

                    ("Upside", "Upside"),
                    (
                        "Peer median upside",
                        "Upside Peer Median",
                    ),
                    (
                        "Upside vs peers",
                        "Upside vs Peers",
                    ),

                    (
                        "Insider buys",
                        "Insider buys",
                    ),

                    ("Score", "Score"),

                    (
                        "Peer Relative Score",
                        "Peer Relative Score",
                    ),
                ]

                comparison_display = pd.DataFrame(
                    index=[
                        label
                        for label, _
                        in comparison_metrics
                    ],
                    columns=comparison.index,
                )

                for label, column in comparison_metrics:

                    if column not in comparison.columns:
                        continue

                    for symbol in comparison.index:

                        value = comparison.loc[
                            symbol,
                            column,
                        ]

                        if pd.isna(value):

                            formatted = "N/A"

                        elif column == "Market Cap":

                            formatted = (
                                f"${value / 1e9:.1f}B"
                            )

                        elif column == "Free Cash Flow":

                            formatted = (
                                f"${value / 1e9:.2f}B"
                            )

                        elif column == "Target Price":

                            formatted = (
                                f"${value:,.2f}"
                            )

                        elif column == "Peer Relative Score":

                            formatted = (
                                f"{value:.0f}/50"
                            )

                        elif column == "Score":

                            formatted = (
                                f"{value:.0f}/100"
                            )

                        elif (
                            "vs Peers"
                            in column
                            or "Margin"
                            in column
                            or "Growth"
                            in column
                            or column in {
                                "ROE",
                                "ROA",
                                "Upside",
                            }
                        ):

                            formatted = (
                                f"{value:+.1%}"
                            )

                        elif column in {
                            "P/E",
                            "Forward P/E",
                            "PEG",
                            "EV / EBITDA",
                            "Debt / Equity",
                        }:

                            formatted = (
                                f"{value:.1f}"
                            )

                        else:

                            formatted = str(
                                value
                            )

                        comparison_display.loc[
                            label,
                            symbol,
                        ] = formatted

                st.dataframe(
                    comparison_display,
                    use_container_width=True,
                )

                peer_names = [
                    f"{row['Symbol']} "
                    f"({row['Company']})"
                    for _, row
                    in peers.iterrows()
                ]

                st.caption(
                    "Peers selected using "
                    "industry, sector and "
                    "market-cap similarity: "
                    + " | ".join(
                        peer_names
                    )
                )

            # ------------------------------------------------
            # Load into backtester
            # ------------------------------------------------

            st.markdown(
                "#### Backtest this stock"
            )

            if st.button(
                f"Load {selected_symbol} into backtester",
                type="primary",
                use_container_width=True,
            ):

                st.session_state[
                    "backtest_ticker"
                ] = selected_symbol

                # Remove the previous result so the
                # selected stock is actually backtested.
                st.session_state.pop(
                    "backtest_data",
                    None,
                )

                st.success(
                    f"{selected_symbol} loaded into "
                    "the backtester."
                )

                st.rerun()


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
