"""Explainable forecasting with confidence bands."""
import altair as alt
import pandas as pd
import streamlit as st

from bap import forecast, llm, ui

ui.page("Forecasting", "📈")
st.title("📈 Forecasting")
st.caption("Trend + seasonality model fitted in milliseconds on a CPU. Shaded area = 90% prediction interval. "
           "Anomalous transactions are excluded before fitting.")

df = ui.ledger()
s_df, src = ui.sec_data(ui.sec_version())

c1, c2, c3 = st.columns([2, 2, 1])
source = c1.radio("Dataset", ["Internal ledger (monthly)", "Public peers - SEC (quarterly)"], horizontal=False)
if source.startswith("Internal"):
    options = {
        "Total revenue": ("Revenue", None, None),
        "Total expenses": ("Expense", None, None),
        "Cloud & Software spend": ("Expense", "Cloud & Software", None),
        "Professional Services spend": ("Expense", "Professional Services", None),
        **{f"Revenue - {bu}": ("Revenue", None, bu) for bu in sorted(df["business_unit"].unique())[:8]},
    }
    choice = c2.selectbox("Series", list(options))
    series = forecast.ledger_monthly(df, *options[choice])
    period, unit, horizon_max, default_h = 12, "EUR", 18, 6
else:
    comp = c2.selectbox("Company", sorted(s_df["company"].unique()))
    sub = s_df[s_df["company"] == comp].sort_values("period_end")
    series = pd.Series(sub["revenue_usd"].to_numpy(), index=pd.PeriodIndex(sub["fiscal_quarter"], freq="Q").to_timestamp())
    choice, period, unit, horizon_max, default_h = f"{comp} quarterly revenue", 4, "USD", 8, 4
    st.caption(f"Source: {src}")
horizon = c3.slider("Horizon", 1, horizon_max, default_h)

fc = forecast.fit_forecast(series, horizon, period)
mape = forecast.backtest_mape(series, holdout=min(6, max(2, len(series) // 6)), period=period)
plot = fc.reset_index().rename(columns={"index": "date"})

base = alt.Chart(plot).encode(x=alt.X("date:T", title=None))
band = base.mark_area(opacity=0.18, color="#2a78d6").encode(y=alt.Y("lower:Q", title=unit, axis=alt.Axis(format="~s")), y2="upper:Q")
actual = base.mark_line(color="#1f2937", strokeWidth=2).encode(y="actual:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("actual:Q", format=",.0f")])
pred = base.mark_line(color="#2a78d6", strokeDash=[6, 4], strokeWidth=2.5).encode(y="forecast:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("forecast:Q", format=",.0f")])
st.altair_chart((band + actual + pred).properties(height=380), width="stretch")

fut = fc[fc["forecast"].notna()]
last_actual = series.iloc[-period:].sum() if len(series) >= period else series.sum()
next_sum = fut["forecast"].iloc[:period].sum() if len(fut) >= period else float("nan")
m1, m2, m3 = st.columns(3)
m1.metric(f"Next {len(fut)} periods (forecast total)", f"{unit} {fut['forecast'].sum() / 1e6:,.2f}M")
m2.metric("Backtest error (MAPE)", f"{mape:.1f}%" if mape == mape else "n/a", help="Mean absolute % error on a held-out recent window.")
if next_sum == next_sum:
    m3.metric("Next 12 months vs last 12" if period == 12 else "Next 4 quarters vs last 4", f"{100 * (next_sum / last_actual - 1):+.1f}%")

with st.expander("Forecast table"):
    st.dataframe(fut.round(0), width="stretch")

if st.button("AI commentary for the CFO"):
    facts = (f"Series: {choice}. Last 6 actuals: {', '.join(f'{v:,.0f}' for v in series.tail(6))} {unit}. "
             f"Forecast next {len(fut)}: {', '.join(f'{v:,.0f}' for v in fut['forecast'])} {unit} "
             f"(90% band +/- {(fut['upper'] - fut['forecast']).mean():,.0f}). Backtest MAPE {mape:.1f}%.")
    if ui.use_llm():
        try:
            g = llm.generate(facts, system="You are an FP&A analyst. Write 3 crisp sentences for the CFO: trend, risk/uncertainty, one recommended action.", max_tokens=200)
            st.success(g["text"]); ui.mode_badge("gemini (cached)" if g["cached"] else "gemini")
        except llm.LLMUnavailable as e:
            st.warning(f"Gemini unavailable: {e}")
    else:
        trend = "upward" if fut["forecast"].iloc[-1] > series.tail(period).mean() else "flat to downward"
        st.success(f"The model projects a {trend} trend for {choice.lower()} over the next {len(fut)} periods, "
                   f"with a backtest error of {mape:.1f}%. Plan budgets against the lower band to stay conservative.")
        ui.mode_badge("offline")
