"""Shared Streamlit helpers: cached loaders, sidebar, formatting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import COMPANY, data, green, llm, rag


@st.cache_data(show_spinner="Loading ledger ...")
def ledger() -> pd.DataFrame:
    return data.load_ledger()


@st.cache_data(show_spinner=False)
def sec_data(_version: int = 0) -> tuple[pd.DataFrame, str]:
    return data.load_sec()


@st.cache_resource(show_spinner=False)
def connection(_version: int = 0):
    s, _ = data.load_sec()
    return data.sqlite_connection(data.load_ledger(), s)


@st.cache_resource(show_spinner="Indexing documents ...")
def retriever() -> rag.Retriever:
    return rag.Retriever(data.load_documents())


def sec_version() -> int:
    return st.session_state.get("sec_version", 0)


def use_llm() -> bool:
    return st.session_state.get("use_llm", True) and llm.is_configured()


def eur(x: float) -> str:
    if abs(x) >= 1e6:
        return f"EUR {x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"EUR {x / 1e3:,.1f}k"
    return f"EUR {x:,.0f}"


def mode_badge(mode: str) -> None:
    if mode.startswith("gemini"):
        st.caption(f"Answered by Gemini ({llm.model_name()})" + (" - served from green cache, 0 new tokens" if "cached" in mode else ""))
    else:
        st.caption("Answered in offline mode (deterministic, no LLM call)")


def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {COMPANY}")
        st.caption("GenAI Business Analytics Platform - prototype")
        if llm.is_configured():
            st.session_state.setdefault("use_llm", True)
            st.toggle("Use Gemini (GenAI mode)", key="use_llm")
            st.success(f"Gemini connected - {llm.model_name()}", icon="✅")
        else:
            st.session_state["use_llm"] = False
            st.warning("No GEMINI_API_KEY found - running in offline demo mode.", icon="⚠️")
        m = green.load()
        st.markdown("**Green AI meter**")
        c1, c2 = st.columns(2)
        c1.metric("Tokens used", f"{m.total_tokens:,}")
        c2.metric("Cache hit rate", f"{m.cache_hit_rate:.0%}")
        st.caption(f"Estimated energy: {m.energy_wh():.2f} Wh - saved by cache: {m.saved_wh():.2f} Wh")
        st.divider()
        st.caption("Big Data Challenge - university prototype. Company and ledger are fictional; "
                   "peer data comes from the public SEC EDGAR API.")


def page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} - GenAI BAP", page_icon=icon, layout="wide")
    sidebar()
