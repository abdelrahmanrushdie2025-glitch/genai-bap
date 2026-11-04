"""Green Cloud dashboard: live AI footprint + what-if scenario."""
import pandas as pd
import streamlit as st

from bap import green, llm, ui

ui.page("Green Cloud", "🌱")
st.title("🌱 Green Cloud Dashboard")
st.caption("Measures the footprint of this platform's AI usage and models the savings of the green design choices.")

m = green.load()
st.subheader("Live usage (this deployment)")
c = st.columns(5)
c[0].metric("LLM API calls", f"{m.llm_calls:,}")
c[1].metric("Cache hits", f"{m.cache_hits:,}", f"{m.cache_hit_rate:.0%} hit rate", delta_color="off")
c[2].metric("Offline answers", f"{m.offline_answers:,}", help="Answered deterministically without any LLM call")
c[3].metric("Tokens processed", f"{m.total_tokens:,}", f"{m.tokens_saved_by_cache:,} saved by cache", delta_color="off")
c[4].metric("Est. energy", f"{m.energy_wh():.2f} Wh", f"-{m.saved_wh():.2f} Wh via cache", delta_color="off")

st.subheader("What-if: enterprise rollout")
s1, s2, s3 = st.columns(3)
qpd = s1.slider("AI queries per day", 100, 50_000, 5_000, 100)
tok = s2.slider("Average tokens per query", 300, 4_000, 1_200, 100)
cache = s3.slider("Cache hit rate", 0.0, 0.9, 0.4, 0.05)
r = green.scenario(qpd, tok, cache)
k = st.columns(3)
k[0].metric("Baseline: large model, no cache, avg. grid", f"{r['baseline_kwh']:,.0f} kWh/yr", f"{r['baseline_kg_co2']:,.0f} kg CO2e", delta_color="off")
k[1].metric("Green design: light model + cache + green region", f"{r['green_kwh']:,.0f} kWh/yr", f"{r['green_kg_co2']:,.0f} kg CO2e", delta_color="off")
k[2].metric("CO2e reduction", f"{r['reduction_pct']:.1f}%")
st.bar_chart(pd.DataFrame({"kg CO2e per year": [r["baseline_kg_co2"], r["green_kg_co2"]]},
                          index=["Baseline", "Green design"]), height=260)

st.caption(f"Assumptions (illustrative, configurable in bap/green.py): {green.WH_PER_1K_TOKENS_LARGE} Wh per 1k tokens for a "
           f"large model vs {green.WH_PER_1K_TOKENS_LIGHT} Wh for a Flash-Lite class model; grid intensity "
           f"{green.GRID_KG_CO2_PER_KWH} kg CO2e/kWh (EU average) vs {green.GREEN_REGION_KG_CO2_PER_KWH} for a "
           "renewable-heavy region. Published estimates vary widely, so treat results as order-of-magnitude.")

st.subheader("Green design principles implemented")
st.table(pd.DataFrame([
    ["Right-sized model", "Flash-Lite class Gemini model by default; capped output tokens", "bap/llm.py"],
    ["Response caching", "SHA-256 keyed disk cache - repeated questions cost 0 tokens", "bap/llm.py"],
    ["Offline-first logic", "Deterministic SQL templates & extractive answers when AI is not needed", "bap/nl2sql.py, bap/rag.py"],
    ["CPU-only analytics", "TF-IDF retrieval, Isolation Forest, least-squares forecasting - no GPU", "bap/rag.py, anomaly.py, forecast.py"],
    ["Small prompts", "Only top-k passages and 25 result rows are sent to the model", "bap/rag.py, nl2sql.py"],
    ["Scale-to-zero hosting", "Serverless/sleeping app when idle (Streamlit Cloud, Azure Container Apps)", "deployment"],
    ["Carbon-aware regions", "Production design places batch jobs in low-carbon regions (e.g. Sweden Central)", "architecture"],
    ["Quantisation (roadmap)", "8/4-bit quantised open models for self-hosted inference", "roadmap"],
], columns=["Principle", "How", "Where"]))

b1, b2 = st.columns(2)
if b1.button("Reset meter"):
    green.reset(); st.rerun()
if b2.button("Clear LLM cache"):
    llm.clear_cache(); st.success("Cache cleared.")
