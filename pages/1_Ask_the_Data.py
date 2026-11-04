"""Conversational querying: natural language -> SQL -> answer."""
import altair as alt
import pandas as pd
import streamlit as st

from bap import nl2sql, ui

ui.page("Ask the Data", "💬")
st.title("💬 Ask the Data")
st.caption("Ask questions about the ledger and public peer financials in plain English. "
           "Every answer shows the SQL it ran - read-only and fully auditable.")

if "chat" not in st.session_state:
    st.session_state.chat = []


def auto_chart(df: pd.DataFrame):
    if df.empty or df.shape[1] < 2 or len(df) < 2:
        return None
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    if not num or not cat:
        return None
    x, y = cat[0], num[-1] if len(num) == 1 else num[0]
    is_time = x in ("month", "year", "fiscal_quarter", "period_end", "quarter") or df[x].astype(str).str.match(r"^\d{4}").all()
    if len(num) > 1 and is_time:
        long = df.melt(id_vars=[x], value_vars=num[:3], var_name="metric", value_name="value")
        return alt.Chart(long).mark_line(point=True).encode(x=alt.X(f"{x}:O", title=None),
                                                           y=alt.Y("value:Q", axis=alt.Axis(format="~s"), title=None),
                                                           color="metric:N").properties(height=300)
    if is_time:
        return alt.Chart(df).mark_line(point=True, color="#2a78d6").encode(
            x=alt.X(f"{x}:O", title=None), y=alt.Y(f"{y}:Q", axis=alt.Axis(format="~s"))).properties(height=300)
    return alt.Chart(df.head(25)).mark_bar(color="#2a78d6").encode(
        x=alt.X(f"{y}:Q", axis=alt.Axis(format="~s")), y=alt.Y(f"{x}:N", sort="-x", title=None),
        tooltip=list(df.columns)).properties(height=min(40 + 28 * min(len(df), 25), 600))


def render(res: nl2sql.QueryResult):
    st.markdown(res.summary)
    if not res.df.empty:
        ch = auto_chart(res.df)
        if ch is not None:
            st.altair_chart(ch, width="stretch")
        with st.expander(f"Result table ({len(res.df)} rows)"):
            st.dataframe(res.df, width="stretch", hide_index=True)
    if res.sql:
        with st.expander("SQL executed (read-only)"):
            st.code(res.sql, language="sql")
    for n in res.notes:
        st.info(n)
    ui.mode_badge(res.mode)


st.markdown("**Try a question:**")
cols = st.columns(3)
clicked = None
for i, q in enumerate(nl2sql.SUGGESTED_QUESTIONS):
    if cols[i % 3].button(q, key=f"sugg{i}", width="stretch"):
        clicked = q

for item in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(item.question)
    with st.chat_message("assistant"):
        render(item)

question = st.chat_input("e.g. Which business unit had the highest expenses in 2025?") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            res = nl2sql.ask(question, ui.connection(ui.sec_version()), use_llm=ui.use_llm())
        render(res)
    st.session_state.chat.append(res)

if st.session_state.chat and st.button("Clear conversation"):
    st.session_state.chat = []
    st.rerun()
