from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Union

import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string, request


LOOKBACK_DEFAULT = 50
BULL_THRESHOLD = 1 / 3
BEAR_THRESHOLD = -1 / 3
STATE_LABELS = {0: "Bull", 1: "Bear", 2: "Neutral"}
FACTOR_OPTIONS = {
    "price": "Price Z-Score (returns)",
    "value": "Trading Value + Session Return filter",
}
SESSION_RETURN_THRESHOLD = 0.005  # 0.5%


@dataclass
class MarkovConfig:
    lookback: int = LOOKBACK_DEFAULT
    bull_threshold: float = BULL_THRESHOLD
    bear_threshold: float = BEAR_THRESHOLD


def load_price_data(csv_path: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["day"], format="%Y_%m_%d")
    df = df.sort_values("date").reset_index(drop=True)
    df["session_return"] = df["close"].pct_change()
    df["trading_value"] = df["volume"]
    return df


def _compute_z_score(series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = series.rolling(lookback).mean()
    rolling_stdev = series.rolling(lookback).std(ddof=0)
    z = np.where(
        rolling_stdev > 0, (series - rolling_mean) / rolling_stdev, np.nan
    )
    return pd.Series(z, index=series.index)


def _determine_states(
    df: pd.DataFrame, config: MarkovConfig, mode: Literal["price", "value"]
) -> Tuple[pd.Series, np.ndarray]:
    returns = df["session_return"]

    if mode == "value":
        value_z = _compute_z_score(df["trading_value"], config.lookback)
        states: List[int] = []
        for vz, ret in zip(value_z, returns):
            if np.isnan(vz) or np.isnan(ret):
                states.append(0)
                continue
            if abs(vz) <= config.bull_threshold:
                states.append(0)
                continue
            if ret > SESSION_RETURN_THRESHOLD:
                states.append(1)
            elif ret < -SESSION_RETURN_THRESHOLD:
                states.append(-1)
            else:
                states.append(0)
        return value_z, np.array(states, dtype=int)

    price_z = _compute_z_score(returns, config.lookback)
    states = np.where(
        price_z > config.bull_threshold,
        1,
        np.where(price_z < config.bear_threshold, -1, 0),
    )
    states = np.nan_to_num(states, nan=0.0).astype(int)
    return price_z, states


def prepare_markov_dataframe(
    data: Union[pd.DataFrame, Path, str],
    config: MarkovConfig = MarkovConfig(),
    mode: Literal["price", "value"] = "price",
) -> pd.DataFrame:
    """Compute regime states and next-bar probabilities for the selected factor."""
    base_df = (
        load_price_data(data)
        if isinstance(data, (str, Path))
        else data.copy(deep=True)
    )

    z_score, states = _determine_states(base_df, config, mode)
    state_indices = pd.Series(states).replace({1: 0, -1: 1, 0: 2}).astype(int)

    probabilities = np.full((len(base_df), 3), np.nan, dtype=float)
    row_counts = np.zeros(len(base_df), dtype=int)

    recent_states: deque[int] = deque()
    for idx in range(len(base_df)):
        if idx > 0:
            recent_states.append(int(state_indices.iloc[idx - 1]))
            if len(recent_states) > config.lookback + 1:
                recent_states.popleft()

        if len(recent_states) >= 2:
            transition_counts = np.zeros((3, 3), dtype=int)
            rlist = list(recent_states)
            for prev, curr in zip(rlist[:-1], rlist[1:]):
                transition_counts[prev, curr] += 1

            current_state_idx = recent_states[-1]
            current_counts = transition_counts[current_state_idx]
            denom = current_counts.sum() + 3  # Laplace smoothing
            probabilities[idx] = (current_counts + 1) / denom
            row_counts[idx] = current_counts.sum()

    result = base_df.copy()
    result["z_score"] = z_score
    result["state"] = states
    result["state_label"] = [
        STATE_LABELS[state_indices.iloc[i]] for i in range(len(base_df))
    ]
    result["date_str"] = result["date"].dt.strftime("%Y-%m-%d")
    result[["prob_bull", "prob_bear", "prob_neutral"]] = probabilities
    result["transition_count_sum"] = row_counts
    return result


def build_candlestick_figure(
    frame: pd.DataFrame, factor_label: str
) -> go.Figure:
    """Return a Plotly candlestick figure with hover probabilities."""
    hover_text = []
    for row in frame.itertuples():
        date_str = row.date.strftime("%Y-%m-%d")
        prob_bull = row.prob_bull
        prob_bear = row.prob_bear
        prob_neutral = row.prob_neutral
        row_count = row.transition_count_sum
        if np.isnan(prob_bull):
            prob_section = "Probabilities unavailable (warm-up window)"
        else:
            prob_section = (
                f"P→Bull: {prob_bull:.2%}<br>"
                f"P→Bear: {prob_bear:.2%}<br>"
                f"P→Neutral: {prob_neutral:.2%}<br>"
                f"Row count: {int(row_count)} transitions"
            )
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{date_str}</b>",
                    f"Open: {row.open:.2f}",
                    f"High: {row.high:.2f}",
                    f"Low: {row.low:.2f}",
                    f"Close: {row.close:.2f}",
                    f"Regime (t): {row.state_label}",
                    f"Factor: {factor_label}",
                    prob_section,
                ]
            )
        )

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=frame["date_str"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",
                decreasing_fillcolor="#ef5350",
                text=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    fig.update_xaxes(type="category")
    return fig


def compute_trade_performance(
    frame: pd.DataFrame, prob_threshold: float = 0.30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = frame.dropna(subset=["prob_bull", "prob_bear"]).copy()
    if trades.empty:
        return (
            pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"]),
            pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"]),
        )

    trades["month"] = trades["date"].dt.strftime("%Y-%m")

    long_trades = trades[trades["prob_bear"] < prob_threshold].copy()
    if not long_trades.empty:
        long_trades["win"] = (long_trades["close"] > long_trades["open"]).astype(
            int
        )
        long_trades["ret"] = (
            long_trades["close"] - long_trades["open"]
        ) / long_trades["open"]
        long_summary = (
            long_trades.groupby("month")
            .agg(
                Trades=("win", "size"),
                AvgReturn=("ret", "mean"),
                WinRate=("win", "mean"),
            )
            .sort_index()
        )
    else:
        long_summary = pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])

    short_trades = trades[trades["prob_bull"] < prob_threshold].copy()
    if not short_trades.empty:
        short_trades["win"] = (short_trades["close"] < short_trades["open"]).astype(
            int
        )
        short_trades["ret"] = (
            short_trades["open"] - short_trades["close"]
        ) / short_trades["open"]
        short_summary = (
            short_trades.groupby("month")
            .agg(
                Trades=("win", "size"),
                AvgReturn=("ret", "mean"),
                WinRate=("win", "mean"),
            )
            .sort_index()
        )
    else:
        short_summary = pd.DataFrame(columns=["Trades", "AvgReturn", "WinRate"])

    return long_summary, short_summary


def build_performance_figure(
    long_summary: pd.DataFrame,
    short_summary: pd.DataFrame,
    factor_label: str,
    title_suffix: str,
) -> go.Figure | None:
    if long_summary.empty and short_summary.empty:
        return None

    long_summary = long_summary.copy()
    short_summary = short_summary.copy()
    long_summary.index = long_summary.index.astype(str)
    short_summary.index = short_summary.index.astype(str)
    months = sorted(set(long_summary.index) | set(short_summary.index))
    if not months:
        return None

    long_avg = [
        (long_summary.loc[m, "AvgReturn"] * 100) if m in long_summary.index else None
        for m in months
    ]
    short_avg = [
        (short_summary.loc[m, "AvgReturn"] * 100)
        if m in short_summary.index
        else None
        for m in months
    ]
    long_win = [
        (long_summary.loc[m, "WinRate"] * 100) if m in long_summary.index else None
        for m in months
    ]
    short_win = [
        (short_summary.loc[m, "WinRate"] * 100)
        if m in short_summary.index
        else None
        for m in months
    ]
    long_trades = [
        int(long_summary.loc[m, "Trades"]) if m in long_summary.index else None
        for m in months
    ]
    short_trades = [
        int(short_summary.loc[m, "Trades"]) if m in short_summary.index else None
        for m in months
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Long Avg Return",
            x=months,
            y=long_avg,
            marker_color="#26a69a",
            text=long_trades,
            textposition="outside",
            texttemplate="Trades:%{text}",
            hovertemplate=(
                "Month: %{x}<br>Long Avg Return: %{y:.2f}%<br>"
                "Trades: %{text}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            name="Short Avg Return",
            x=months,
            y=short_avg,
            marker_color="#ef5350",
            text=short_trades,
            textposition="outside",
            texttemplate="Trades:%{text}",
            hovertemplate=(
                "Month: %{x}<br>Short Avg Return: %{y:.2f}%<br>"
                "Trades: %{text}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Long Win Rate",
            x=months,
            y=long_win,
            mode="lines+markers",
            line=dict(color="#81c784", width=2),
            hovertemplate="Month: %{x}<br>Long Win Rate: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            name="Short Win Rate",
            x=months,
            y=short_win,
            mode="lines+markers",
            line=dict(color="#ff8a80", width=2),
            hovertemplate="Month: %{x}<br>Short Win Rate: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        title=f"Monthly Performance ({title_suffix}) — {factor_label}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(
        title_text="Average 1-Bar Return (%)", secondary_y=False, tickformat=".1f"
    )
    fig.update_yaxes(
        title_text="Win Rate (%)", secondary_y=True, range=[0, 100], tickformat=".0f"
    )
    return fig


def compute_yearly_performance(
    frame: pd.DataFrame,
    prob_threshold: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = frame.dropna(subset=["prob_bull", "prob_bear"]).copy()
    if trades.empty:
        empty = pd.DataFrame(columns=["Year", "Trades", "AvgReturn", "WinRate"])
        return empty, empty

    trades["Year"] = trades["date"].dt.year

    def summarize(trade_df: pd.DataFrame, win_condition) -> pd.DataFrame:
        if trade_df.empty:
            return pd.DataFrame(columns=["Year", "Trades", "AvgReturn", "WinRate"])
        trade_df = trade_df.copy()
        trade_df["win"] = win_condition(trade_df).astype(int)
        grouped = (
            trade_df.groupby("Year")
            .agg(
                Trades=("win", "size"),
                AvgReturn=("ret", "mean"),
                WinRate=("win", "mean"),
            )
            .reset_index()
            .sort_values("Year")
        )
        return grouped

    long_trades = trades[trades["prob_bear"] < prob_threshold].copy()
    if not long_trades.empty:
        long_trades["ret"] = (long_trades["close"] - long_trades["open"]) / long_trades[
            "open"
        ]
        long_summary = summarize(long_trades, lambda df: df["close"] > df["open"])
    else:
        long_summary = pd.DataFrame(
            columns=["Year", "Trades", "AvgReturn", "WinRate"]
        )

    short_trades = trades[trades["prob_bull"] < prob_threshold].copy()
    if not short_trades.empty:
        short_trades["ret"] = (short_trades["open"] - short_trades["close"]) / short_trades[
            "open"
        ]
        short_summary = summarize(short_trades, lambda df: df["close"] < df["open"])
    else:
        short_summary = pd.DataFrame(
            columns=["Year", "Trades", "AvgReturn", "WinRate"]
        )

    return long_summary, short_summary


def dataframe_to_html(df: pd.DataFrame, percentage_cols: List[str]) -> str:
    if df.empty:
        return ""
    formatted = df.copy()
    for col in percentage_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col] * 100
    return formatted.to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}",
        classes="perf-table",
        border=0,
    )


def filter_by_year(frame: pd.DataFrame, year: str | int) -> pd.DataFrame:
    if year == "All":
        return frame
    return frame[frame["date"].dt.year == int(year)]


def create_flask_app(base_data: pd.DataFrame) -> Flask:
    app = Flask(__name__)
    app.logger.disabled = True
    logging.getLogger("werkzeug").disabled = True

    years: List[str] = sorted(
        {str(year) for year in base_data["date"].dt.year.unique()}
    )
    default_year = "2025" if "2025" in years else "All"
    default_factor = "price"
    default_config = MarkovConfig()
    year_options = ["All"] + years
    factor_options = list(FACTOR_OPTIONS.keys())

    template = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>VNINDEX Markov Regime Dashboard</title>
        <style>
            body { background-color: #111; color: #eee; font-family: Arial, sans-serif; margin: 0; padding: 0; }
            header { padding: 1.5rem 2rem; }
            main { padding: 0 2rem 2rem; }
            a.year-link { color: #9bd; margin-right: 0.75rem; text-decoration: none; }
            a.year-link.active { font-weight: bold; color: #fff; text-decoration: underline; }
            .param-box { border: 1px solid #333; padding: 1rem; margin-top: 1rem; max-width: 360px; background-color: #1a1a1a; }
            .param-box label { display: block; margin-bottom: 0.25rem; font-weight: bold; }
            .param-box input { width: 100%; padding: 0.4rem; margin-bottom: 0.75rem; background-color: #222; border: 1px solid #444; color: #eee; }
            .param-box .note { font-size: 0.8rem; margin-bottom: 0.5rem; color: #ccc; }
            .param-box button { padding: 0.5rem 1rem; background-color: #26a69a; border: none; color: #fff; cursor: pointer; }
            .param-box button:hover { background-color: #2bbbad; }
            table.perf-table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
            table.perf-table th, table.perf-table td { border: 1px solid #333; padding: 0.4rem 0.6rem; text-align: center; }
            table.perf-table th { background-color: #1f1f1f; }
        </style>
    </head>
    <body>
        <header>
            <h1>VNINDEX Markov Regime Dashboard</h1>
            <p>Select a year to filter the candlestick chart. Hover candles to view transition probabilities available at the next bar.</p>
            <div>
                {% for option in year_options %}
                    <a class="year-link {% if option == selected_year %}active{% endif %}" href="/?year={{ option }}&factor={{ selected_factor }}&lookback={{ current_lookback }}&threshold={{ current_threshold }}">{{ "All Years" if option == "All" else option }}</a>
                {% endfor %}
            </div>
            <div style="margin-top: 1rem;">
                {% for option in factor_options %}
                    <a class="year-link {% if option == selected_factor %}active{% endif %}" href="/?year={{ selected_year }}&factor={{ option }}&lookback={{ current_lookback }}&threshold={{ current_threshold }}">{{ factor_labels[option] }}</a>
                {% endfor %}
            </div>
            <div class="param-box">
                <form method="get" action="/">
                    <input type="hidden" name="year" value="{{ selected_year }}" />
                    <input type="hidden" name="factor" value="{{ selected_factor }}" />
                    <label for="lookback-input">Lookback (bars)</label>
                    <input id="lookback-input" type="number" name="lookback" min="5" max="500" value="{{ current_lookback }}" />
                    <div class="note">Default: {{ default_lookback }}</div>
                    <label for="threshold-input">Bull/Bear Z-score threshold</label>
                    <input id="threshold-input" type="number" step="any" min="0.01" max="3.0" name="threshold" value="{{ current_threshold }}" />
                    <div class="note">Bear threshold is applied automatically as −threshold. Default: {{ default_threshold }}</div>
                    <button type="submit">Update Parameters</button>
                </form>
            </div>
        </header>
        <main>
            {% if figure_html %}
                {{ figure_html | safe }}
            {% else %}
                <p>No data available for the selected period.</p>
            {% endif %}
            {% if performance_html %}
                <section style="margin-top: 2rem;">
                    {{ performance_html | safe }}
                </section>
            {% endif %}
        </main>
        {% if yearly_long_html or yearly_short_html %}
        <section style="padding: 0 2rem 2rem;">
            <h2>Yearly Performance (2023-2025)</h2>
            {% if yearly_long_html %}
                <h3>Long Signals</h3>
                {{ yearly_long_html | safe }}
            {% else %}
                <p>No long trades for yearly summary.</p>
            {% endif %}
            {% if yearly_short_html %}
                <h3>Short Signals</h3>
                {{ yearly_short_html | safe }}
            {% else %}
                <p>No short trades for yearly summary.</p>
            {% endif %}
        </section>
        {% endif %}
    </body>
    </html>
    """

    @app.route("/")
    def index() -> str:
        selected_year = request.args.get("year", default_year)
        selected_factor = request.args.get("factor", default_factor)
        try:
            lookback = int(request.args.get("lookback", default_config.lookback))
            if lookback < 5:
                lookback = 5
        except (TypeError, ValueError):
            lookback = default_config.lookback

        try:
            threshold = float(
                request.args.get("threshold", default_config.bull_threshold)
            )
            if threshold <= 0:
                threshold = default_config.bull_threshold
        except (TypeError, ValueError):
            threshold = default_config.bull_threshold

        if selected_year not in year_options:
            selected_year = default_year
        if selected_factor not in factor_options:
            selected_factor = default_factor

        config = MarkovConfig(
            lookback=lookback,
            bull_threshold=threshold,
            bear_threshold=-threshold,
        )
        current_threshold_str = f"{threshold:.4f}"
        current_lookback_str = str(lookback)
        default_threshold_str = f"{default_config.bull_threshold:.4f}"
        default_lookback_str = str(default_config.lookback)

        processed = prepare_markov_dataframe(
            base_data, config=config, mode=selected_factor
        )
        subset = filter_by_year(processed, selected_year)
        figure_html = ""
        performance_html = ""
        yearly_long_html = ""
        yearly_short_html = ""
        if not subset.empty:
            fig = build_candlestick_figure(
                subset.sort_values("date"),
                FACTOR_OPTIONS[selected_factor],
            )
            figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

            long_summary, short_summary = compute_trade_performance(subset)
            perf_fig = build_performance_figure(
                long_summary,
                short_summary,
                FACTOR_OPTIONS[selected_factor],
                selected_year,
            )
            if perf_fig is not None:
                performance_html = perf_fig.to_html(
                    full_html=False, include_plotlyjs=False
                )

        # Yearly summaries always use the selected factor but for all available years.
        yearly_long, yearly_short = compute_yearly_performance(processed)
        yearly_long_html = dataframe_to_html(yearly_long, ["AvgReturn", "WinRate"])
        yearly_short_html = dataframe_to_html(yearly_short, ["AvgReturn", "WinRate"])

        return render_template_string(
            template,
            figure_html=figure_html,
            performance_html=performance_html,
            year_options=year_options,
            selected_year=selected_year,
            factor_options=factor_options,
            selected_factor=selected_factor,
            factor_labels=FACTOR_OPTIONS,
            yearly_long_html=yearly_long_html,
            yearly_short_html=yearly_short_html,
            current_lookback=current_lookback_str,
            current_threshold=current_threshold_str,
            default_lookback=default_lookback_str,
            default_threshold=default_threshold_str,
        )

    return app


def main() -> None:
    base_data = load_price_data("df_vnindex.csv")
    app = create_flask_app(base_data)
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, use_reloader=False, port=port)


if __name__ == "__main__":
    main()
