"""Automated anomaly detection on the general ledger."""
import altair as alt
import streamlit as st

from bap import anomaly, llm, ui

ui.page("Anomaly Detection", "🚨")
st.title("🚨 Anomaly Detection")
st.caption("Every expense transaction is screened by audit control rules plus an Isolation Forest model. "
           "Each alert explains why it was raised, so a human reviewer can decide quickly.")


@st.cache_data(show_spinner="Scoring transactions ...")
def scored():
    return anomaly.detect(ui.ledger())


sc = scored()
ev = anomaly.evaluate(sc)
flags = sc[sc["flagged"] == 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions screened", f"{ev['reviewed_pool']:,}")
c2.metric("Alerts raised", f"{ev['flagged']:,}", f"{100 * ev['flagged'] / ev['reviewed_pool']:.1f}% need review", delta_color="off")
c3.metric("Recall (known issues found)", f"{ev['recall']:.1%}")
c4.metric("Precision (alerts that are real)", f"{ev['precision']:.1%}")
st.caption("Recall/precision are measured against anomalies deliberately injected into the synthetic ledger "
           "(duplicates, split payments, off-hours transfers, extreme amounts) - labels the model never sees.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Alerts by reason")
    reasons = flags["reasons"].str.split("; ").explode().value_counts().reset_index()
    reasons.columns = ["reason", "alerts"]
    st.altair_chart(alt.Chart(reasons).mark_bar(color="#d64545").encode(
        x="alerts:Q", y=alt.Y("reason:N", sort="-x", title=None), tooltip=["reason", "alerts"]).properties(height=260),
        use_container_width=True)
with right:
    st.subheader("Detection rate by issue type")
    rt = [{"issue type": k, "detected %": v} for k, v in ev["recall_by_type"].items()]
    st.dataframe(rt, hide_index=True, use_container_width=True)

st.subheader("Review queue")
f1, f2 = st.columns(2)
reason_opts = sorted(reasons["reason"])
sel = f1.multiselect("Filter by reason", reason_opts)
min_score = f2.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
q = flags[flags["risk_score"] >= min_score]
if sel:
    q = q[q["reasons"].apply(lambda r: any(s in r for s in sel))]
show = ["txn_id", "txn_date", "business_unit", "category", "counterparty", "amount_eur", "payment_method", "risk_score", "reasons"]
st.dataframe(q[show], use_container_width=True, hide_index=True,
             column_config={"risk_score": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f"),
                            "amount_eur": st.column_config.NumberColumn("amount (EUR)", format="%.2f")})
st.download_button("Download review queue (CSV)", q[show].to_csv(index=False), "anomaly_review_queue.csv", "text/csv")

st.subheader("Explain an alert")
txn = st.selectbox("Transaction", q["txn_id"].head(200).tolist())
if txn:
    row = q[q["txn_id"] == txn].iloc[0]
    peers = sc[(sc["category"] == row["category"]) & (sc["flagged"] == 0)]["amount_eur"]
    facts = (f"Transaction {row.txn_id} on {row.txn_date:%Y-%m-%d %H:%M} ({row.txn_date:%A}), "
             f"{row.business_unit}, category {row.category}, vendor {row.counterparty}, "
             f"EUR {row.amount_eur:,.2f} via {row.payment_method}. Typical (median) amount in this category: "
             f"EUR {peers.median():,.2f}. Vendor used {int(row.vendor_freq)} times. Risk score {row.risk_score:.2f}. "
             f"Triggered: {row.reasons}.")
    st.markdown(facts)
    if st.button("Generate auditor explanation"):
        if ui.use_llm():
            try:
                g = llm.generate(facts, system=("You are an internal auditor. In max 4 sentences explain why this "
                                                "transaction is suspicious, the likely risk (error, fraud, control "
                                                "bypass) and the concrete next verification step."), max_tokens=220)
                st.success(g["text"])
                ui.mode_badge("gemini (cached)" if g["cached"] else "gemini")
            except llm.LLMUnavailable as e:
                st.warning(f"Gemini unavailable: {e}")
        else:
            steps = {"Duplicate": "check whether the vendor invoiced twice and request a refund or credit note.",
                     "split": "check for a single underlying invoice and whether dual approval was bypassed.",
                     "off-hours": "verify the vendor's master data, bank account and who released the payment.",
                     "Rarely": "confirm the vendor's onboarding and KYC documentation.",
                     "norm": "check the invoice for a data-entry error (extra digits, wrong currency)."}
            nxt = next((v for k, v in steps.items() if k in row.reasons), "review the supporting documents.")
            st.success(f"This payment was flagged because: {row.reasons.lower()}. Recommended next step: {nxt}")
            ui.mode_badge("offline")
