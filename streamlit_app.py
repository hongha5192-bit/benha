from __future__ import annotations

import streamlit as st

from app import (
    FACTOR_OPTIONS,
    MarkovConfig,
    build_candlestick_figure,
    build_performance_figure,
    compute_trade_performance,
    compute_yearly_performance,
    load_price_data,
    prepare_markov_dataframe,
)


@st.cache_resource
def load_base_data():
    return load_price_data("df_vnindex.csv")


st.set_page_config(
    page_title="VNINDEX Markov Regime — Streamlit",
    layout="wide",
    page_icon="📈",
)

st.title("VNINDEX Markov Regime Dashboard")
st.markdown(
    "Interactive view of Pine-style Markov regimes with configurable parameters. "
    "Charts and performance stats update automatically."
)

base_data = load_base_data()
years = sorted(base_data["date"].dt.year.unique().tolist())
year_options = ["All Years"] + [str(year) for year in years]

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
        value=MarkovConfig.lookback,
        step=1,
    )
    threshold = st.number_input(
        "Bull/Bear Z-score threshold",
        min_value=0.01,
        max_value=3.0,
        value=float(MarkovConfig.bull_threshold),
        step=0.01,
        format="%.4f",
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

candlestick_fig = build_candlestick_figure(subset.sort_values("date"), FACTOR_OPTIONS[factor_key])
st.plotly_chart(candlestick_fig, use_container_width=True)

long_summary, short_summary = compute_trade_performance(subset)
perf_fig = build_performance_figure(
    long_summary, short_summary, FACTOR_OPTIONS[factor_key], title_suffix
)
if perf_fig:
    st.plotly_chart(perf_fig, use_container_width=True)
else:
    st.info("No qualifying trades for monthly performance with the current parameters.")

yearly_long, yearly_short = compute_yearly_performance(processed)

st.subheader("Yearly Performance (2023-2025)")
col_long, col_short = st.columns(2)

with col_long:
    if yearly_long.empty:
        st.write("No long trades.")
    else:
        st.dataframe(yearly_long.style.format({"AvgReturn": "{:.2%}", "WinRate": "{:.2%}"}))

with col_short:
    if yearly_short.empty:
        st.write("No short trades.")
    else:
        st.dataframe(yearly_short.style.format({"AvgReturn": "{:.2%}", "WinRate": "{:.2%}"}))
