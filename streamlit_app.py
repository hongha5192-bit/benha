from __future__ import annotations

from typing import Tuple

import pandas as pd
import streamlit as st

from app import (
    FACTOR_OPTIONS,
    MarkovConfig,
    build_candlestick_figure,
    build_performance_figure,
    load_price_data,
    prepare_markov_dataframe,
)


@st.cache_resource
def load_base_data() -> pd.DataFrame:
    return load_price_data("df_vnindex.csv")


def _summarize_trades(trades: pd.DataFrame, group_key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])
    grouped = (
        trades.groupby(group_key)
        .agg(
            Trades=("win", "size"),
            AvgReturn=("ret", "mean"),
            WinRate=("win", "mean"),
        )
        .sort_index()
    )
    grouped["Trades"] = grouped["Trades"].astype(int)
    return grouped


def compute_strategy_performance(
    frame: pd.DataFrame, long_threshold: float, short_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = frame.dropna(subset=["prob_bull", "prob_bear"]).copy()
    if trades.empty:
        empty = pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])
        return empty, empty

    trades["month"] = trades["date"].dt.strftime("%Y-%m")

    long_trades = trades[trades["prob_bear"] < long_threshold].copy()
    if long_trades.empty:
        long_summary = pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])
    else:
        long_trades["win"] = (long_trades["close"] > long_trades["open"]).astype(int)
        long_trades["ret"] = (
            long_trades["close"] - long_trades["open"]
        ) / long_trades["open"]
        long_summary = _summarize_trades(long_trades, "month")

    short_trades = trades[trades["prob_bull"] < short_threshold].copy()
    if short_trades.empty:
        short_summary = pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])
    else:
        short_trades["win"] = (short_trades["close"] < short_trades["open"]).astype(int)
        short_trades["ret"] = (
            short_trades["open"] - short_trades["close"]
        ) / short_trades["open"]
        short_summary = _summarize_trades(short_trades, "month")

    return long_summary, short_summary


def compute_yearly_strategy_performance(
    frame: pd.DataFrame, long_threshold: float, short_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = frame.dropna(subset=["prob_bull", "prob_bear"]).copy()
    if trades.empty:
        empty = pd.DataFrame(columns=["Year", "Trades", "AvgReturn", "WinRate"])
        return empty, empty

    trades["Year"] = trades["date"].dt.year

    long_trades = trades[trades["prob_bear"] < long_threshold].copy()
    if not long_trades.empty:
        long_trades["win"] = (long_trades["close"] > long_trades["open"]).astype(int)
        long_trades["ret"] = (
            long_trades["close"] - long_trades["open"]
        ) / long_trades["open"]
        long_summary = (
            long_trades.groupby("Year")
            .agg(
                Trades=("win", "size"),
                AvgReturn=("ret", "mean"),
                WinRate=("win", "mean"),
            )
            .reset_index()
            .sort_values("Year")
        )
        long_summary["Trades"] = long_summary["Trades"].astype(int)
    else:
        long_summary = pd.DataFrame(
            columns=["Year", "Trades", "AvgReturn", "WinRate"]
        )

    short_trades = trades[trades["prob_bull"] < short_threshold].copy()
    if not short_trades.empty:
        short_trades["win"] = (short_trades["close"] < short_trades["open"]).astype(int)
        short_trades["ret"] = (
            short_trades["open"] - short_trades["close"]
        ) / short_trades["open"]
        short_summary = (
            short_trades.groupby("Year")
            .agg(
                Trades=("win", "size"),
                AvgReturn=("ret", "mean"),
                WinRate=("win", "mean"),
            )
            .reset_index()
            .sort_values("Year")
        )
        short_summary["Trades"] = short_summary["Trades"].astype(int)
    else:
        short_summary = pd.DataFrame(
            columns=["Year", "Trades", "AvgReturn", "WinRate"]
        )

    return long_summary, short_summary


st.set_page_config(
    page_title="VNINDEX Markov Regime — Streamlit",
    layout="wide",
    page_icon="📈",
)

st.title("VNINDEX Markov Regime Dashboard")
st.markdown(
    "Interactive view of Pine-style Markov regimes. Adjust model and strategy parameters "
    "to explore how probabilities and trade performance change."
)

base_data = load_base_data()
years = sorted(base_data["date"].dt.year.unique().tolist())
year_options = ["All Years"] + [str(year) for year in years]
default_config = MarkovConfig()

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    selected_year = st.selectbox("Year", year_options, index=len(year_options) - 1)

with col2:
    factor_key = st.selectbox(
        "Factor",
        options=list(FACTOR_OPTIONS.keys()),
        format_func=lambda key: FACTOR_OPTIONS[key],
        index=0,
    )

with col3:
    st.markdown("**Model Parameters**")
    lookback = st.number_input(
        "Lookback (bars)",
        min_value=5,
        max_value=500,
        value=default_config.lookback,
        step=1,
    )
    threshold = st.number_input(
        "Bull/Bear Z-score threshold",
        min_value=0.01,
        max_value=3.0,
        value=float(default_config.bull_threshold),
        step=0.01,
        format="%.4f",
    )
    st.markdown("**Strategy Thresholds**")
    long_threshold = st.number_input(
        "Enter long when Bear probability is below",
        min_value=0.05,
        max_value=0.9,
        value=0.30,
        step=0.01,
        format="%.2f",
    )
    short_threshold = st.number_input(
        "Enter short when Bull probability is below",
        min_value=0.05,
        max_value=0.9,
        value=0.30,
        step=0.01,
        format="%.2f",
    )

config = MarkovConfig(
    lookback=int(lookback),
    bull_threshold=float(threshold),
    bear_threshold=-float(threshold),
)

processed = prepare_markov_dataframe(base_data, config=config, mode=factor_key)

if selected_year != "All Years":
    subset = processed[processed["date"].dt.year == int(selected_year)].copy()
    title_suffix = selected_year
else:
    subset = processed.copy()
    title_suffix = "All Years"

if subset.empty:
    st.warning("No data available for the selected period.")
    st.stop()

candle_fig = build_candlestick_figure(
    subset.sort_values("date"), FACTOR_OPTIONS[factor_key]
)
st.plotly_chart(candle_fig, use_container_width=True)

monthly_long, monthly_short = compute_strategy_performance(
    subset, long_threshold, short_threshold
)

if monthly_long.empty and monthly_short.empty:
    st.info(
        "No qualifying trades for the selected thresholds within the chosen period."
    )
else:
    perf_fig = build_performance_figure(
        monthly_long, monthly_short, FACTOR_OPTIONS[factor_key], title_suffix
    )
    if perf_fig:
        st.plotly_chart(perf_fig, use_container_width=True)

st.markdown("### Monthly Trade Performance")
col_long, col_short = st.columns(2)

with col_long:
    st.markdown("**Long Strategy**")
    if monthly_long.empty:
        st.write("No long trades for the selected thresholds.")
    else:
        idx_name = monthly_long.index.name or "Month"
        display_long = monthly_long.reset_index().rename(columns={idx_name: "Month"})
        st.dataframe(
            display_long.style.format(
                {"AvgReturn": "{:.2%}", "WinRate": "{:.2%}", "Trades": "{:d}"}
            ),
            use_container_width=True,
        )

with col_short:
    st.markdown("**Short Strategy**")
    if monthly_short.empty:
        st.write("No short trades for the selected thresholds.")
    else:
        idx_name = monthly_short.index.name or "Month"
        display_short = monthly_short.reset_index().rename(columns={idx_name: "Month"})
        st.dataframe(
            display_short.style.format(
                {"AvgReturn": "{:.2%}", "WinRate": "{:.2%}", "Trades": "{:d}"}
            ),
            use_container_width=True,
        )

yearly_long, yearly_short = compute_yearly_strategy_performance(
    processed, long_threshold, short_threshold
)

st.markdown("### Yearly Performance (2023-2025)")
col_year_long, col_year_short = st.columns(2)

with col_year_long:
    st.markdown("**Long Strategy (Yearly)**")
    if yearly_long.empty:
        st.write("No long trades across the full dataset.")
    else:
        st.dataframe(
            yearly_long.style.format(
                {"AvgReturn": "{:.2%}", "WinRate": "{:.2%}", "Trades": "{:d}"}
            ),
            use_container_width=True,
        )

with col_year_short:
    st.markdown("**Short Strategy (Yearly)**")
    if yearly_short.empty:
        st.write("No short trades across the full dataset.")
    else:
        st.dataframe(
            yearly_short.style.format(
                {"AvgReturn": "{:.2%}", "WinRate": "{:.2%}", "Trades": "{:d}"}
            ),
            use_container_width=True,
        )
