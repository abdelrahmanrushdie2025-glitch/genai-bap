"""GenAI Business Analytics Platform - Executive Overview (home page).

Run locally:  streamlit run app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

from bap import COMPANY, sec, ui

ui.page("Executive Overview", "📊")

st.title("GenAI Business Analytics Platform")
st.markdown(f"**{COMPANY}** · enterprise data synthesis, conversational querying and sustainable cloud decision analytics")

df = ui.ledger()
years = sorted(df["txn_date"].dt.year.unique())
year = st.selectbox("Reporting year", years[::-1], index=1 if len(years) > 1 else 0,
                    help="2026 contains data up to September (year to date).")


def totals(y: int, max_month: int = 12) -> tuple[float, float]:
    d = df[(df["txn_date"].dt.year == y) & (df["txn_date"].dt.month <= max_month)]
    r = d.loc[d["account_type"] == "Revenue", "amount_eur"].sum()
    e = d.loc[d["account_type"] == "Expense", "amount_eur"].sum()
    return r, e


last_month = int(df.loc[df["txn_date"].dt.year == year, "txn_date"].dt.month.max())
rev, exp = totals(year, last_month)
prev_rev, prev_exp = totals(year - 1, last_month) if year - 1 in years else (0, 0)


def delta(a, b):
    return f"{100 * (a - b) / b:+.1f}% YoY" if b else None


label = "" if last_month == 12 else f" (Jan-{pd.Timestamp(2000, last_month, 1):%b})"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue" + label, ui.eur(rev), delta(rev, prev_rev))
c2.metric("Operating expenses" + label, ui.eur(exp), delta(exp, prev_exp), delta_color="inverse")
c3.metric("Operating profit", ui.eur(rev - exp), delta(rev - exp, prev_rev - prev_exp))
c4.metric("Operating margin", f"{100 * (rev - exp) / rev:.1f}%",
          f"{100 * ((rev - exp) / rev - (prev_rev - prev_exp) / prev_rev):+.1f} pp" if prev_rev else None)

st.subheader("Monthly revenue vs. expenses")
monthly = df.groupby(["month", "account_type"])["amount_eur"].sum().reset_index()
chart = alt.Chart(monthly).mark_line(point=False, strokeWidth=2.5).encode(
    x=alt.X("month:T", title=None),
    y=alt.Y("amount_eur:Q", title="EUR", axis=alt.Axis(format="~s")),
    color=alt.Color("account_type:N", title=None, scale=alt.Scale(domain=["Revenue", "Expense"], range=["#2a78d6", "#e8803a"])),
    tooltip=[alt.Tooltip("month:T", format="%b %Y"), "account_type", alt.Tooltip("amount_eur:Q", format=",.0f")],
).properties(height=320)
st.altair_chart(chart, use_container_width=True)

ydf = df[df["txn_date"].dt.year == year]
left, right = st.columns(2)
with left:
    st.subheader("Revenue by business unit")
    bu = ydf[ydf["account_type"] == "Revenue"].groupby("business_unit")["amount_eur"].sum().reset_index()
    st.altair_chart(alt.Chart(bu).mark_bar(color="#2a78d6").encode(
        x=alt.X("amount_eur:Q", title="EUR", axis=alt.Axis(format="~s")),
        y=alt.Y("business_unit:N", sort="-x", title=None),
        tooltip=["business_unit", alt.Tooltip("amount_eur:Q", format=",.0f")]).properties(height=300),
        use_container_width=True)
with right:
    st.subheader("Expenses by category")
    cat = ydf[ydf["account_type"] == "Expense"].groupby("category")["amount_eur"].sum().reset_index()
    st.altair_chart(alt.Chart(cat).mark_bar(color="#e8803a").encode(
        x=alt.X("amount_eur:Q", title="EUR", axis=alt.Axis(format="~s")),
        y=alt.Y("category:N", sort="-x", title=None),
        tooltip=["category", alt.Tooltip("amount_eur:Q", format=",.0f")]).properties(height=300),
        use_container_width=True)

st.subheader("What this platform does")
a, b, c, d = st.columns(4)
a.markdown("**💬 Ask the Data**  \nPlain-English questions become safe, read-only SQL with an executive summary.")
b.markdown("**📄 Document Insights**  \nRetrieval-augmented answers over reports, memos and regulation - with sources.")
c.markdown("**🚨 Anomaly Detection**  \nRules + Isolation Forest screen every transaction and explain each alert.")
d.markdown("**📈 Forecasting & 🌱 Green Cloud**  \nExplainable forecasts with confidence bands; live energy and CO2 meter.")

with st.expander("Data sources & pipeline status"):
    s, src = ui.sec_data(ui.sec_version())
    st.markdown(
        f"- **Internal ledger (structured):** {len(df):,} transactions, "
        f"{df['txn_date'].min():%b %Y} - {df['txn_date'].max():%b %Y} (synthetic, fictional company)\n"
        f"- **Documents (unstructured):** {len(ui.retriever().chunks)} indexed passages from 8 internal reports\n"
        f"- **Public peer financials:** {s['ticker'].nunique()} companies, {len(s):,} quarters - source: *{src}*"
    )
    if st.button("🔄 Refresh peer data from SEC EDGAR"):
        with st.spinner("Calling the SEC XBRL API ..."):
            out, errors = sec.fetch_all()
        if out.empty:
            st.error("Could not reach SEC EDGAR. The demo sample stays active. " + "; ".join(errors[:3]))
        else:
            st.session_state["sec_version"] = ui.sec_version() + 1
            st.success(f"Loaded {len(out):,} quarters for {out['ticker'].nunique()} companies.")
            if errors:
                st.warning("Some companies failed: " + "; ".join(errors))
            st.rerun()
